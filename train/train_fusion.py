"""Skeleton + VLM text fusion training. Train / eval protocol same as train_stgcn.py.

  python train/train_fusion.py --mode skel_text --run A --T 16 --seed 0
Input: data/processed/run{run}_T{T}/{split}.npz + data/processed/vlm_T{T}/{split}.csv, {split}_emb.npy or {split}_hidden.npy
Output:results/fusion/{mode}_run{run}_T{T}_s{seed}/{best.pt, log.csv, metrics.json, test_pred.npz}
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
from eval.metrics import threshold_for_recall  # noqa: E402
from models.fusion import MODES, build_model  # noqa: E402
from models.stgcn import FLIP_INDEX  # noqa: E402
from train.train_stgcn import LOG_FIELDS, augment, evaluate, load_split, tb_log, weighted_bce  # noqa: E402

VLM_FILE = {"text": "emb", "hidden": "hidden", "tokmean": "tokmean", "tokens": "tokens"}


def load_fusion_split(data_dir, vlm_dir, split, device, vlm_feat="text"):
    d = load_split(data_dir, split, device)
    cap = pd.read_csv(vlm_dir / f"{split}.csv")
    assert (cap.video_id.to_numpy() == d["video_id"]).all() and (cap.clip_idx.to_numpy() == np.load(
        data_dir / f"{split}.npz")["clip_idx"]).all(), f"{split}: VLM row order differs from npz"
    e = np.load(vlm_dir / f"{split}_{VLM_FILE[vlm_feat]}.npy")
    d["emb"] = torch.from_numpy(e if vlm_feat == "tokens" else e.astype(np.float32)).to(device)  # tokens kept in fp16 (~8GB)
    d["pfall"] = torch.from_numpy(cap.p_fall.to_numpy(np.float32)).to(device)
    return d


def hflip(x, tok, run, p, grid=6, n_time=2):
    """Horizontally flip skeleton and VLM patch grid together per sample (keeps cross-attention spatial alignment)."""
    flip = torch.rand(len(x), device=x.device) < p
    if not flip.any():
        return x, tok
    x, tok = x.clone(), tok.clone()
    c = 0.5 if run == "A" else 0.0
    xf = x[flip][..., FLIP_INDEX]
    xf[:, 0] = (2 * c - xf[:, 0]) * xf[:, 2]
    x[flip] = xf
    n, d = int(flip.sum()), tok.shape[-1]
    tok[flip, :-1] = tok[flip, :-1].view(n, n_time, grid, grid, d).flip(3).reshape(n, -1, d)
    return x, tok


@torch.no_grad()
def predict(model, d, bs=1024):
    model.eval()
    out = [torch.sigmoid(model(d["X"][i:i + bs], d["emb"][i:i + bs], d["pfall"][i:i + bs]))
           for i in range(0, len(d["y"]), bs)]
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/fusion.yaml"))
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--run", choices=["A", "B"], default="A")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = ROOT / cfg["data_root"] / f"run{args.run}_T{args.T}"
    vlm_dir = ROOT / cfg["data_root"] / f"vlm_T{args.T}"
    out = ROOT / cfg["out_root"] / f"{args.mode}_run{args.run}_T{args.T}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(args.mode, cfg, args.T).to(device)
    tr, va, te = (load_fusion_split(data_dir, vlm_dir, s, device, model.vlm_feat) for s in ("train", "val", "test"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(tr["y"]) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    pos_weight = (tr["y"] == 0).sum() / (tr["y"] == 1).sum()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.mode} run{args.run} T{args.T} seed{args.seed} | train {len(tr['y'])} clips "
          f"| params {n_params/1e6:.2f}M | {device}")

    writer = SummaryWriter(out / "tb")  # tensorboard --logdir results
    best_f1, log = -1, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(tr["y"]), device=device)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x, emb = tr["X"][idx], tr["emb"][idx]
            if model.vlm_feat == "tokens":  # flip handled separately, together with tokens
                x, emb = hflip(x, emb, args.run, cfg["aug"]["flip_p"])
                x = augment(x.clone(), args.run, {**cfg["aug"], "flip_p": 0.0})
            elif model.use_skel:
                x = augment(x.clone(), args.run, cfg["aug"])
            pred = model(x, emb, tr["pfall"][idx])
            logit, aux = pred if isinstance(pred, tuple) else (pred, None)
            loss = F.binary_cross_entropy_with_logits(logit, tr["y"][idx], pos_weight=pos_weight)
            if aux is not None:  # skeleton aux head (cross-attention)
                loss = loss + cfg["xattn_aux_weight"] * F.binary_cross_entropy_with_logits(
                    aux, tr["y"][idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)

        va_prob = predict(model, va)
        vm, _ = evaluate(va, va_prob, 0.5)
        row = {"epoch": ep, "loss": tot / len(perm), "val_loss": weighted_bce(va_prob, va["y"], pos_weight),
               "val_clip_f1": vm["clip"]["f1"], "val_video_f1": vm["video"]["f1"],
               "val_video_auroc": vm["video"]["auroc"], "lr": sched.get_last_lr()[0]}
        tb_log(writer, row)
        log.append(row)
        if vm["video"]["f1"] > best_f1:
            best_f1 = vm["video"]["f1"]
            torch.save(model.state_dict(), out / "best.pt")
            row["best"] = 1
        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            print(f"ep {ep:3d} loss {row['loss']:.4f} val clip F1 {row['val_clip_f1']:.3f} "
                  f"video F1 {row['val_video_f1']:.3f} AUROC {row['val_video_auroc']:.3f}")

    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(log)

    writer.close()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    _, (vy, vprob) = evaluate(va, predict(model, va), 0.5)
    thr90 = threshold_for_recall(vy, vprob, 0.9)
    te_prob = predict(model, te)
    res = {"mode": args.mode, "run": args.run, "T": args.T, "seed": args.seed, "params": n_params,
           "best_epoch": int(max(range(len(log)), key=lambda i: log[i]["val_video_f1"])),
           "val_thr_recall90": thr90, "train_time_s": round(time.time() - t0, 1),
           "test@0.5": evaluate(te, te_prob, 0.5)[0], "test@val_recall90": evaluate(te, te_prob, thr90)[0]}
    np.savez(out / "test_pred.npz", prob=te_prob, video_id=te["video_id"], y=te["y"].cpu().numpy())
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    t = res["test@0.5"]["video"]
    print(f"TEST video @0.5: F1 {t['f1']:.3f} recall {t['recall']:.3f} precision {t['precision']:.3f} "
          f"AUROC {t['auroc']:.3f}")


if __name__ == "__main__":
    main()
