"""Train ST-GCN on fall 'event' labels (OmniFall segments) + shortcut-check evaluation.

  python train/train_event.py --train OF-Syn --run A --seed 0
  python train/train_event.py --train OF-Syn FallVision --run A --seed 0
  python train/train_event.py --eval-only results/stgcn/runA_T16_s0     # evaluate an existing FallVision-trained model on the same criteria
  python train/train_event.py --train OF-Syn --filter camera_elevation=high,top   # viewpoint shift: train on high viewpoints only
  python train/train_event.py --mode skel_tokmean --train OF-Syn FallVision --seed 0  # VLM / fusion (models/fusion.py)
  python train/train_event.py --train OF-Syn --label video        # control: same data with video labels (all clips of a fall video = 1)
  python train/train_event.py --train OF-Syn --label multiclass   # OmniFall 16-action softmax. alarm score = p(fall) or p(fall)+p(fallen)
mode: stgcn (default), hidden / tokmean (VLM features only -> MLP), skel_tokmean (ST-GCN + VLM concat fusion),
  skel_xattn / skel_xattn_sm (cross-attention; patch tokens {split}_tokens.npy read via memmap, moved to GPU per batch).
  VLM features = tokmean from vlm_tokens.py (hidden = its second half = token right before the answer); motion-removed eval uses vlm_T16_last features.
Train labels: external (OmniFall) datasets use clip y_event (1 if overlapping a fall segment by >= 0.3 s, ambiguous excluded),
           FallVision uses video labels (no segment labels). Model selection: OF-Syn val clip AUROC.
Eval (evaluate_all): per dataset, clip AUROC (y_seg: overlaps fall segment), video AUROC (clip max),
  clip / video AUROC with motion removed (last frame repeated), per-action Fall rate (threshold = OF-Syn val recall 0.9).
Output: results/event/[{mode}/]{train name}_run{run}_T{T}_s{seed}/{best.pt, log.csv, tb/, metrics.json} (no mode folder for stgcn)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, threshold_for_recall, video_level  # noqa: E402
import torch.nn as nn  # noqa: E402
from models.fusion import build_model  # noqa: E402
from models.stgcn import STGCN  # noqa: E402
from train.train_fusion import hflip  # noqa: E402
from train.train_stgcn import LOG_FIELDS, augment, tb_log, weighted_bce  # noqa: E402

ACTIONS = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other",
           "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump"]
MODES = ("stgcn", "hidden", "tokmean", "skel_tokmean", "skel_xattn", "skel_xattn_sm", "zeroshot")
EVAL_SETS = [("OF-Syn", "test"), ("GMDCSA24", "test"), ("CAUCAFall", "test"), ("FallVision", "test")]
_OFSYN_META = None


def ofsyn_attr(video_ids, key):
    """Video metadata for OF-Syn clips (camera_elevation etc.). video_id 'fall/fall_ch_001.mp4' -> path in label csv."""
    global _OFSYN_META
    if _OFSYN_META is None:
        _OFSYN_META = pd.read_csv(ROOT / "data/raw/OF-Syn/meta/labels/of-syn.csv").drop_duplicates("path").set_index("path")
    return _OFSYN_META.loc[[str(Path(v).with_suffix("")) for v in video_ids], key].to_numpy()


def apply_filter(d, flt):
    """flt = 'key=v1,v2' -> keep only clips from matching OF-Syn videos."""
    if not flt:
        return d
    key, vals = flt.split("=")
    m = np.isin(ofsyn_attr(d["video_id"], key), vals.split(","))
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in d.items()}


class ZeroShot(nn.Module):
    """Training-free VLM zero-shot: clip Yes/No p(Fall) used directly as score."""
    use_skel = False

    def forward(self, x, emb, pfall):
        return torch.logit(pfall.clamp(1e-6, 1 - 1e-6))


FALL, FALLEN = 1, 2
SCORES = {"fall": (FALL,), "fall+fallen": (FALL, FALLEN)}  # alarm score definitions for multiclass models


class AlarmScore(nn.Module):
    """Multiclass model -> binary alarm logit = logit(sum of softmax probs of chosen classes). Reuses the binary-model eval code."""

    def __init__(self, inner, classes):
        super().__init__()
        self.inner, self.classes = inner, list(classes)
        self.use_skel = getattr(inner, "use_skel", True)

    def forward(self, *a):
        p = self.inner(*a).softmax(-1)[:, self.classes].sum(1)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def feat_of(mode):
    return {"hidden": "hidden", "tokmean": "tokmean", "skel_tokmean": "tokmean", "zeroshot": "hidden",
            "skel_xattn": "tokens", "skel_xattn_sm": "tokens"}.get(mode)


def load_vlm(name, split, T, feat, frames="normal"):
    """(emb, pfall), or None if no features. Row order same as run{A,B}_T{T}/{split}.npz."""
    v = ROOT / "data/processed" / ("" if name == "FallVision" else f"ext_{name}") / \
        (f"vlm_T{T}" + ("" if frames == "normal" else f"_{frames}"))
    f = v / f"{split}_{'tokens' if feat == 'tokens' else 'tokmean'}.npy"
    if not f.exists():
        return None
    e = np.load(f, mmap_mode="r" if feat == "tokens" else None)  # tokens (N, 73, 3584) fp16 via memmap
    pf = v / f"{split}_pfall.npy"  # original FallVision uses p_fall from vlm_captions.py (same question, same computation)
    pf = np.load(pf) if pf.exists() else pd.read_csv(v / f"{split}.csv").p_fall.to_numpy(np.float32)
    return (e[:, 3584:] if feat == "hidden" else e), pf.astype(np.float32)


def load(name, split, run, T, feat=None):
    if name == "FallVision":
        z = dict(np.load(ROOT / f"data/processed/run{run}_T{T}/{split}.npz"))
        z["y_event"] = z["y"]  # no segment labels -> video label (weak)
    else:
        z = dict(np.load(ROOT / f"data/processed/ext_{name}/run{run}_T{T}/{split}.npz"))
    if feat:
        z["emb"], z["pfall"] = load_vlm(name, split, T, feat)
    return z


def make_model(mode, T, cfg, n_out=1):
    if mode == "stgcn":
        return STGCN(num_outputs=n_out, dropout=cfg["dropout"])
    return ZeroShot() if mode == "zeroshot" else build_model(mode, cfg, T, n_out)


@torch.no_grad()
def predict(model, d, device, X=None, emb=None, pfall=None, bs=1024, act=torch.sigmoid):
    X = d["X"] if X is None else X
    emb = d.get("emb") if emb is None else emb
    pfall = d.get("pfall") if pfall is None else pfall
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        if isinstance(model, STGCN) or isinstance(getattr(model, "inner", None), STGCN):
            o = model(x)
        else:
            o = model(x, torch.from_numpy(np.ascontiguousarray(emb[i:i + bs])).to(device).float(),
                      torch.from_numpy(pfall[i:i + bs]).to(device))
        out.append(act(o))
    return torch.cat(out).cpu().numpy()


def clip_auroc(d, p):
    y = d["y_seg"] if "y_seg" in d else d["y"]
    return binary_metrics(y, p)["auroc"]


def video_auroc(d, p):
    _, vy, vp = video_level(d["video_id"], d["y"], p)
    return binary_metrics(vy, vp)["auroc"]


def evaluate_all(model, run, T, device, thr, feat=None):
    """static = motion removed from all inputs (skeleton last frame repeated + VLM features from 4 copies of last frame).
    skeleton + VLM models also compute static_skel (skeleton frozen only, original VLM)."""
    out = {}
    for name, split in EVAL_SETS:
        d = load(name, split, run, T, feat)
        p = predict(model, d, device)
        static = d["X"][:, :, -1:].repeat(T, axis=2)  # remove motion
        r = {"clip_auroc": clip_auroc(d, p), "video_auroc": video_auroc(d, p)}
        last = load_vlm(name, split, T, feat, "last") if feat else (None, None)
        if last is None:
            r["static_clip_auroc"] = r["static_video_auroc"] = float("nan")
        else:
            ps = predict(model, d, device, X=static, emb=last[0], pfall=last[1])
            r.update(static_clip_auroc=clip_auroc(d, ps), static_video_auroc=video_auroc(d, ps))
        if feat and model.use_skel:
            pk = predict(model, d, device, X=static)
            r.update(static_skel_clip_auroc=clip_auroc(d, pk), static_skel_video_auroc=video_auroc(d, pk))
        if name == "OF-Syn":
            elev = ofsyn_attr(d["video_id"], "camera_elevation")
            r["by_elevation"] = {e: {"clip_auroc": clip_auroc({k: v[elev == e] for k, v in d.items()
                                                               if isinstance(v, np.ndarray) and len(v) == len(elev)},
                                                              p[elev == e]),
                                     "video_auroc": video_auroc({k: v[elev == e] for k, v in d.items()
                                                                 if isinstance(v, np.ndarray) and len(v) == len(elev)},
                                                                p[elev == e])}
                                 for e in ("top", "high", "eye", "low")}
        if "action" in d:  # clip AUROC under alarm definition B (1 if fall or fallen)
            a = d["action"] >= 0
            r["alarm_clip_auroc"] = binary_metrics(np.isin(d["action"][a], [FALL, FALLEN]).astype(int), p[a])["auroc"]
            r["fall_rate_by_action"] = {ACTIONS[a]: float((p[d["action"] == a] >= thr).mean())
                                        for a in np.unique(d["action"]) if a >= 0}
        out[f"{name}/{split}"] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="stgcn")
    ap.add_argument("--config", default=None, help="default: stgcn -> configs/stgcn.yaml, otherwise configs/fusion.yaml")
    ap.add_argument("--train", nargs="+", default=["OF-Syn"])
    ap.add_argument("--run", choices=["A", "B"], default="A")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", choices=["event", "video", "multiclass"], default="event",
                    help="video = train / select external datasets on video label y too (control: event-label effect vs data volume)")
    ap.add_argument("--filter", default=None, help="OF-Syn train / val video filter, e.g. camera_elevation=high,top")
    ap.add_argument("--eval-only", default=None, help="stgcn checkpoint folder to evaluate only (uses best.pt and run from metrics.json)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config or ROOT / f"configs/{'stgcn' if args.mode == 'stgcn' else 'fusion'}.yaml"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    dev = args.device
    feat = feat_of(args.mode)

    va = apply_filter(load("OF-Syn", "val", args.run, args.T, feat), args.filter)
    ykey = {"event": "y_event", "video": "y", "multiclass": "action"}[args.label]
    multi = args.label == "multiclass"
    assert not (multi and "FallVision" in args.train), "FallVision has no action labels"
    keep = va[ykey] >= 0
    if args.eval_only:
        src = Path(args.eval_only).resolve()
        run = json.loads((src / "metrics.json").read_text())["run"]
        model = STGCN(dropout=cfg["dropout"])
        model.load_state_dict(torch.load(src / "best.pt", map_location="cpu", weights_only=True))
        model.to(dev)
        va = load("OF-Syn", "val", run, args.T)
        thr = threshold_for_recall(va["y_event"][keep], predict(model, va, dev)[keep], 0.9)
        res = {"source": str(src.relative_to(ROOT)), "train": ["FallVision"], "run": run, "val_thr_recall90": thr,
               "eval": evaluate_all(model, run, args.T, dev, thr)}
        out = ROOT / "results/event" / f"baseline_{src.name}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
        print(json.dumps({k: {m: round(v[m], 3) for m in v if isinstance(v[m], float)}
                          for k, v in res["eval"].items()}, indent=1))
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parts, masks, toks = [], [], []
    for name in args.train:
        d = load(name, "train", args.run, args.T, feat)
        if name == "OF-Syn":
            d = apply_filter(d, args.filter)
        m = d[ykey] >= 0
        masks.append(m)
        if feat == "tokens":
            toks.append(d["emb"])
            parts.append((d["X"][m], d[ykey][m], None, d["pfall"][m]))
        else:
            parts.append((d["X"][m], d[ykey][m]) + ((d["emb"][m], d["pfall"][m]) if feat else ()))
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    X = torch.from_numpy(cat(0)).to(dev)
    y = torch.from_numpy(cat(1)).to(dev)
    y = y.long() if multi else y.float()
    if feat == "tokens":  # tokens don't fit in RAM / GPU -> read from memmap per batch by (dataset index, row)
        src = np.concatenate([np.full(len(p[1]), j) for j, p in enumerate(parts)])
        src_row = np.concatenate([np.flatnonzero(m_) for m_ in masks])

        def E_get(idx):
            idx = idx.cpu().numpy()
            out = np.empty((len(idx), *toks[0].shape[1:]), np.float16)
            for j, t in enumerate(toks):
                sel = src[idx] == j
                if sel.any():
                    out[sel] = t[src_row[idx[sel]]]
            return torch.from_numpy(out).to(dev, non_blocking=True)
    else:
        E = torch.from_numpy(cat(2)).float().to(dev) if feat else None
        E_get = (lambda idx: E[idx]) if feat else None  # noqa: E731
    P = torch.from_numpy(cat(3)).to(dev) if feat else None
    vd = {k: v[keep] for k, v in va.items() if isinstance(v, np.ndarray) and len(v) == len(keep)}
    vy = vd[ykey]
    vmask = vd["y_event"] >= 0  # model selection: multiclass also uses event clip AUROC of p(fall) (same criterion as binary)

    tag = "+".join(args.train) + (f"[{args.filter.replace('=', '-').replace(',', '+')}]" if args.filter else "") + \
        {"event": "", "video": "-videolabel", "multiclass": "-multiclass"}[args.label]
    out = ROOT / "results/event" / ("" if args.mode == "stgcn" else args.mode) / f"{tag}_run{args.run}_T{args.T}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(args.mode, args.T, cfg, len(ACTIONS) if multi else 1).to(dev)
    score = AlarmScore(model, SCORES["fall"]) if multi else model  # binary score for eval / model selection
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(y) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    if multi:  # class weight = freq^-0.5 (mean 1), 0 for classes absent from training
        cnt = torch.bincount(y, minlength=len(ACTIONS)).float()
        cls_w = torch.where(cnt > 0, cnt.clamp(min=1) ** -0.5, torch.zeros_like(cnt))
        cls_w = cls_w / cls_w[cnt > 0].mean()
        pos_weight = torch.tensor(1.0)
        print(f"train {args.mode} {tag} run{args.run} seed{args.seed} | {len(y)} clips | class counts "
              + ", ".join(f"{ACTIONS[i]} {int(c)}" for i, c in enumerate(cnt) if c > 0))
    else:
        pos_weight = (y == 0).sum() / (y == 1).sum()
        print(f"train {args.mode} {tag} run{args.run} seed{args.seed} | {len(y)} clips (Fall {int(y.sum())}) "
              f"| pos_weight {pos_weight:.2f}")
    writer = SummaryWriter(out / "tb")
    best, log, t0 = -1, [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(y), device=dev)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x = X[idx]
            emb = E_get(idx) if feat else None
            if feat == "tokens":  # flip skeleton and patch grid together
                x, emb = hflip(x, emb, args.run, cfg["aug"]["flip_p"])
                x = augment(x.clone(), args.run, {**cfg["aug"], "flip_p": 0.0})
            elif args.mode == "stgcn" or model.use_skel:
                x = augment(x.clone(), args.run, cfg["aug"])
            pred = model(x) if args.mode == "stgcn" else model(x, emb, P[idx])
            logit, aux = pred if isinstance(pred, tuple) else (pred, None)
            if multi:
                loss = F.cross_entropy(logit, y[idx], weight=cls_w)
            else:
                loss = F.binary_cross_entropy_with_logits(logit, y[idx], pos_weight=pos_weight)
            if aux is not None:  # cross-attention skeleton aux head
                loss = loss + cfg["xattn_aux_weight"] * F.binary_cross_entropy_with_logits(aux, y[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if multi:
            pr = predict(model, vd, dev, act=lambda o: o.softmax(-1))
            vp, vl = pr[:, FALL], float(-np.log(pr[np.arange(len(vy)), vy].clip(1e-8)).mean())
            vb = vd["y_event"][vmask]
            auc, f1 = binary_metrics(vb, vp[vmask])["auroc"], float((pr.argmax(1) == vy).mean())  # f1 slot = 16-class accuracy
        else:
            vp = predict(model, vd, dev)
            auc, f1 = binary_metrics(vy, vp)["auroc"], binary_metrics(vy, vp)["f1"]
            vl = weighted_bce(vp, torch.from_numpy(vy), pos_weight)
        row = {"epoch": ep, "loss": tot / len(perm), "val_loss": vl,
               "val_clip_f1": f1, "val_video_f1": float("nan"), "val_video_auroc": auc,
               "lr": sched.get_last_lr()[0]}
        tb_log(writer, row)
        log.append(row)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best.pt")
            row["best"] = 1
        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            print(f"ep {ep:3d} loss {row['loss']:.4f} val_loss {row['val_loss']:.4f} OF-Syn val clip AUROC {auc:.3f}")
    writer.close()
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(log)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True))
    vb = vd["y_event"][vmask]
    thr = threshold_for_recall(vb, predict(score, vd, dev)[vmask], 0.9)
    res = {"mode": args.mode, "train": args.train, "label": args.label, "filter": args.filter, "run": args.run, "T": args.T,
           "seed": args.seed, "val_thr_recall90": thr, "best_val_clip_auroc": best,
           "train_time_s": round(time.time() - t0, 1), "eval": evaluate_all(score, args.run, args.T, dev, thr, feat)}
    if multi:  # eval alarm-definition-B score + per-action prediction distribution (row = true action, col = predicted action fraction)
        sb = AlarmScore(model, SCORES["fall+fallen"])
        res["eval_fall+fallen"] = evaluate_all(sb, args.run, args.T, dev,
                                               threshold_for_recall(vb, predict(sb, vd, dev)[vmask], 0.9), feat)
        res["confusion"] = {}
        for name in ("OF-Syn", "GMDCSA24", "CAUCAFall"):
            d = load(name, "test", args.run, args.T, feat)
            pred = predict(model, d, dev, act=lambda o: o.softmax(-1)).argmax(1)
            res["confusion"][name] = {ACTIONS[a]: {ACTIONS[b]: float((pred[d["action"] == a] == b).mean())
                                                   for b in np.unique(pred[d["action"] == a])}
                                      for a in np.unique(d["action"]) if a >= 0}
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    print(json.dumps({k: {m: round(v[m], 3) for m in v if isinstance(v[m], float)}
                      for k, v in res["eval"].items()}, indent=1))


if __name__ == "__main__":
    main()
