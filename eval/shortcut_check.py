"""Check whether FallVision is solvable from 'static pose' alone (shortcut).

  python eval/shortcut_check.py
1. Static-pose baselines: joints of the clip's last frame only (Run A coords) -> MLP, and 2 hand-crafted features
   (torso angle, hip height) -> logistic regression. Trained on FallVision train, test video AUROC.
2. Motion usage: feed trained ST-GCN / fusion (a) frame-shuffled, (b) last-frame-repeated clips and compare AUROC.
3. Data bias: ratio of 'body horizontal + low in frame' frames per label x scenario.
Output: results/shortcut_check.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, video_level  # noqa: E402
from models.fusion import build_model  # noqa: E402
from models.stgcn import STGCN  # noqa: E402

DEV = "cuda:1"
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def load(split, run="A"):
    return dict(np.load(ROOT / f"data/processed/run{run}_T16/{split}.npz"))


def video_auroc(d, prob):
    _, vy, vp = video_level(d["video_id"], d["y"], prob)
    return binary_metrics(vy, vp)["auroc"]


def pose_feats(X):
    """X (N,3,T,V) Run A -> per frame (torso angle [0=vertical, 1=horizontal], hip y [0=top, 1=bottom], valid)."""
    xy, m = X[:, :2], X[:, 2]
    sh = (xy[..., L_SH] * m[:, None, :, L_SH] + xy[..., R_SH] * m[:, None, :, R_SH]) / \
        np.clip(m[:, None, :, L_SH] + m[:, None, :, R_SH], 1, None)
    hip = (xy[..., L_HIP] * m[:, None, :, L_HIP] + xy[..., R_HIP] * m[:, None, :, R_HIP]) / \
        np.clip(m[:, None, :, L_HIP] + m[:, None, :, R_HIP], 1, None)
    v = (m[:, :, [L_SH, R_SH]].max(-1) > 0) & (m[:, :, [L_HIP, R_HIP]].max(-1) > 0)
    d = sh - hip
    ang = np.abs(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]) + 1e-6)) / (np.pi / 2)  # 0 vertical, 1 horizontal
    return ang, hip[:, 1], v


def static_baselines(tr, te):
    out = {}
    # (a) last-frame joints (51-dim) -> MLP
    f = lambda d: torch.from_numpy(d["X"][:, :, -1].reshape(len(d["X"]), -1)).float().to(DEV)
    Xtr, Xte, ytr = f(tr), f(te), torch.from_numpy(tr["y"]).float().to(DEV)
    aucs = []
    for seed in range(3):
        torch.manual_seed(seed)
        mlp = nn.Sequential(nn.Linear(51, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)).to(DEV)
        opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(60):
            for i in torch.randperm(len(Xtr), device=DEV).split(256):
                loss = nn.functional.binary_cross_entropy_with_logits(mlp(Xtr[i]).squeeze(-1), ytr[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            aucs.append(video_auroc(te, torch.sigmoid(mlp(Xte).squeeze(-1)).cpu().numpy()))
    out["last-frame joints -> MLP"] = aucs
    # (b) torso angle + hip height (last frame), 2 numbers -> logistic regression
    def two(d):
        a, h, v = pose_feats(d["X"])
        return np.stack([np.where(v[:, -1], a[:, -1], 0), np.where(v[:, -1], h[:, -1], 0)], 1)
    lr = LogisticRegression(max_iter=1000).fit(two(tr), tr["y"])
    out["torso angle + hip height (2 numbers)"] = [video_auroc(te, lr.predict_proba(two(te))[:, 1])]
    return out


@torch.no_grad()
def perturb_tests(te, cfg):
    X = torch.from_numpy(te["X"]).to(DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    shuffled = X[:, :, torch.randperm(X.shape[2], generator=g)]
    last = X[:, :, -1:].expand_as(X).contiguous()
    tok = torch.from_numpy(np.load(ROOT / "data/processed/vlm_T16/test_tokmean.npy").astype(np.float32)).to(DEV)
    zeros = torch.zeros(len(X), device=DEV)
    res = {}
    for label, pattern, mode in (("ST-GCN Run A", "stgcn/runA_T16_s*", None),
                                 ("ST-GCN + VLM tokens", "fusion/skel_tokmean_runA_T16_s*", "skel_tokmean")):
        rows = {"original": [], "shuffled": [], "last frame repeated": []}
        for run_dir in sorted((ROOT / "results").glob(pattern)):
            model = STGCN(dropout=0.1) if mode is None else build_model(mode, cfg, 16)
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
            model.to(DEV).eval()
            for k, x in (("original", X), ("shuffled", shuffled), ("last frame repeated", last)):
                p = torch.cat([torch.sigmoid(model(x[i:i + 512]) if mode is None else
                                             model(x[i:i + 512], tok[i:i + 512], zeros[i:i + 512]))
                               for i in range(0, len(x), 512)]).cpu().numpy()
                rows[k].append(video_auroc(te, p))
        res[label] = rows
    return res


def bias_table():
    rows = []
    for split in ("train", "val", "test"):
        d = load(split)
        a, h, v = pose_feats(d["X"])
        low = (a > 0.6) & (h > 0.6) & v  # body near horizontal and low in frame
        rows.append(pd.DataFrame({"label": np.where(d["y"] == 1, "Fall", "No Fall"), "scenario": d["scenario"],
                                  "video": d["video_id"], "low_frac": low.mean(1), "any_low": low.any(1)}))
    df = pd.concat(rows)
    v = df.groupby(["label", "scenario", "video"]).agg(low_frac=("low_frac", "mean"), any_low=("any_low", "max"))
    return v.groupby(["label", "scenario"]).agg(videos=("any_low", "size"), low_frame_ratio=("low_frac", "mean"),
                                                videos_with_low_pose=("any_low", "mean")).round(3)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/fusion.yaml"))
    tr, te = load("train"), load("test")
    md = ["# Shortcut check (FallVision test, 500 videos, video-level AUROC)", ""]
    sb = static_baselines(tr, te)
    md += ["## 1. Static-pose baselines (no motion information)", "", "| Input | AUROC |", "|---|---|"]
    md += [f"| {k} | {np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "") + " |"
           for k, v in sb.items()]
    md += ["| (reference) ST-GCN Run A, 16 frames | 0.986 |", ""]
    pt = perturb_tests(te, cfg)
    md += ["## 2. Trained models on motion-removed input", "", "| Model | Original | Frames shuffled | Last frame repeated |", "|---|---|---|---|"]
    for m, r in pt.items():
        md.append(f"| {m} | " + " | ".join(f"{np.mean(r[k]):.3f}" for k in ("original", "shuffled", "last frame repeated")) + " |")
    md += ["", "## 3. Rate of 'body horizontal + low in frame' poses (train+val+test)", "",
           "low_frame_ratio = fraction of such frames, videos_with_low_pose = fraction of videos with at least one such clip", "",
           bias_table().to_markdown(), ""]
    (ROOT / "results/shortcut_check.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
