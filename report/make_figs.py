"""Report figure 2: FallVision test video-level confusion matrix (ST-GCN Run A seed 0, VLM zero-shot), threshold 0.5.

  python report/make_figs.py   -> report/figs/baseline.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.titlesize": 8, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.5})


def video_cm(video_id, y, p, thr=0.5):
    df = pd.DataFrame({"v": video_id, "y": y, "p": p}).groupby("v").agg(y=("y", "first"), p=("p", "max"))
    pred = (df.p >= thr).astype(int)
    return np.array([[((df.y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])


fig, ax = plt.subplots(1, 2, figsize=(3.6, 1.8))

te = np.load(ROOT / "data/processed/runA_T16/test.npz")
sg = np.load(ROOT / "results/stgcn/runA_T16_s0/test_pred.npz")
zs = pd.read_csv(ROOT / "data/processed/vlm_T16/test.csv")
for a, (title, cm) in zip(ax, [("(a) ST-GCN", video_cm(sg["video_id"], sg["y"], sg["prob"])),
                                   ("(b) Zero-shot VLM", video_cm(zs.video_id, te["y"], zs.p_fall))]):
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
    a.grid(False)
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.6 else INK)
    a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"],
          xlabel="predicted", title=title)
    a.tick_params(length=0)
ax[0].set_ylabel("true")
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(ROOT / "report/figs/baseline.pdf")
fig.savefig(ROOT / "report/figs/baseline.png", dpi=300)
print("saved")
