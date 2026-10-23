"""ST-GCN Fall / No Fall training.

  python train/train_stgcn.py --run B --T 16 --seed 0
Defaults from configs/stgcn.yaml; CLI args take precedence. Data: data/processed/run{run}_T{T}/{split}.npz.
Output: results/stgcn/run{run}_T{T}_s{seed}/{best.pt (state_dict), log.csv, metrics.json}
Model selection by val video-level F1. Test reported at two thresholds: (a) 0.5, (b) value giving val video-level recall>=0.9.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, by_group, threshold_for_recall, video_level  # noqa: E402
from models.stgcn import FLIP_INDEX, STGCN  # noqa: E402


def load_split(d, split, device):
    z = np.load(d / f"{split}.npz")
    return {"X": torch.from_numpy(z["X"]).to(device), "y": torch.from_numpy(z["y"]).float().to(device),
            "video_id": z["video_id"], "scenario": z["scenario"]}


def augment(x, run, cfg):
    """x (N, 3, T, V). Run A coords centered at 0.5 (image center), Run B at 0 (hip center)."""
    n = x.shape[0]
    xy, mask = x[:, :2], x[:, 2:3]
    c = 0.5 if run == "A" else 0.0
    xy = xy - c

    flip = torch.rand(n, device=x.device) < cfg["flip_p"]
    xy[flip] = xy[flip][..., FLIP_INDEX] * torch.tensor([-1.0, 1.0], device=x.device).view(1, 2, 1, 1)
    mask[flip] = mask[flip][..., FLIP_INDEX]

    s = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * cfg["scale"]
    th = (torch.rand(n, device=x.device) * 2 - 1) * np.deg2rad(cfg["rotate_deg"])
    cos, sin = torch.cos(th).view(n, 1, 1), torch.sin(th).view(n, 1, 1)
    rx = cos * xy[:, 0] - sin * xy[:, 1]
    ry = sin * xy[:, 0] + cos * xy[:, 1]
    xy = torch.stack([rx, ry], 1) * s + c

    drop = (torch.rand(n, 1, 1, x.shape[3], device=x.device) < cfg["joint_drop_p"]).float()
    mask = mask * (1 - drop)
    xy = xy * mask  # invalid joint coords always 0 (same convention as normalize.py)
    return torch.cat([xy, mask], 1)


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    return torch.cat([torch.sigmoid(model(X[i:i + bs])) for i in range(0, len(X), bs)]).cpu().numpy()


def weighted_bce(prob, y, pos_weight):
    """Prob -> BCE with the same pos_weight as train loss (to compare val loss with train loss)."""
    p = torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6)
    y = y.float().cpu()
    return float(-(pos_weight.cpu() * y * p.log() + (1 - y) * (1 - p).log()).mean())


LOG_FIELDS = ["epoch", "loss", "val_loss", "val_clip_f1", "val_video_f1", "val_video_auroc", "lr", "best"]


def tb_log(writer, row):
    for k, tag in (("loss", "loss/train"), ("val_loss", "loss/val"), ("val_clip_f1", "val/clip_f1"),
                   ("val_video_f1", "val/video_f1"), ("val_video_auroc", "val/video_auroc"), ("lr", "lr")):
        writer.add_scalar(tag, row[k], row["epoch"])


def evaluate(data, prob, thr):
    y = data["y"].cpu().numpy().astype(int)
    vids, vy, vprob = video_level(data["video_id"], y, prob)
    vscen = np.array([v.split("/")[1].lower() for v in vids])
    return {"clip": binary_metrics(y, prob, thr), "video": binary_metrics(vy, vprob, thr),
            "video_by_scenario": by_group(vscen, vy, vprob, thr)}, (vy, vprob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/stgcn.yaml"))
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = ROOT / cfg["data_root"] / f"run{args.run}_T{args.T}"
    out = ROOT / cfg["out_root"] / f"run{args.run}_T{args.T}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = (load_split(data_dir, s, device) for s in ("train", "val", "test"))

    model = STGCN(dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(tr["y"]) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    pos_weight = (tr["y"] == 0).sum() / (tr["y"] == 1).sum()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run{args.run} T{args.T} seed{args.seed} | train {len(tr['y'])} clips | params {n_params/1e6:.2f}M "
          f"| pos_weight {pos_weight:.2f} | {device}")

    writer = SummaryWriter(out / "tb")  # tensorboard --logdir results
    best_f1, log = -1, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(tr["y"]), device=device)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x = augment(tr["X"][idx].clone(), args.run, cfg["aug"])
            loss = F.binary_cross_entropy_with_logits(model(x), tr["y"][idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)

        va_prob = predict(model, va["X"])
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
    _, (vy, vprob) = evaluate(va, predict(model, va["X"]), 0.5)
    thr90 = threshold_for_recall(vy, vprob, 0.9)
    te_prob = predict(model, te["X"])
    res = {"run": args.run, "T": args.T, "seed": args.seed, "params": n_params,
           "best_epoch": int(max(range(len(log)), key=lambda i: log[i]["val_video_f1"])),
           "val_thr_recall90": thr90, "train_time_s": round(time.time() - t0, 1),
           "test@0.5": evaluate(te, te_prob, 0.5)[0], "test@val_recall90": evaluate(te, te_prob, thr90)[0]}
    np.savez(out / "test_pred.npz", prob=te_prob, video_id=te["video_id"], y=te["y"].cpu().numpy())
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    t = res["test@0.5"]["video"]
    print(f"TEST video @0.5: F1 {t['f1']:.3f} recall {t['recall']:.3f} precision {t['precision']:.3f} "
          f"AUROC {t['auroc']:.3f} | @val-recall90 thr {thr90:.3f}: "
          f"recall {res['test@val_recall90']['video']['recall']:.3f} F1 {res['test@val_recall90']['video']['f1']:.3f}")


if __name__ == "__main__":
    main()
