"""Report figure 1: pipeline (matplotlib box diagram). python report/make_pipeline.py -> report/figs/pipeline.pdf"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, SKEL, VLM, FUSE = "#0b0b0b", "#52514e", "#dbe9f8", "#fde4d9", "#e9e6f9"
fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.set(xlim=(0, 100), ylim=(0, 30))
ax.axis("off")


def box(x, y, w, h, text, fc="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.8))


box(0.5, 11, 11, 8, "RGB clip\n16 frames\n10 fps, 1.6 s")
# skeleton branch
box(16, 20, 14, 8, "YOLO11x-pose\n+ ByteTrack\n(main person)", SKEL)
box(34, 20, 14, 8, "skeleton\n3 x 16 x 17\n(x, y, mask)", SKEL)
box(52, 20, 13, 8, "ST-GCN\n(6 blocks)\n256-d", SKEL, bold=True)
# VLM branch
box(16, 2, 14, 8, "4 frames\n(2.5 fps)", VLM)
box(34, 2, 14, 8, "Qwen2.5-VL-7B\n(frozen)\nYes/No prompt", VLM, bold=True)
box(52, 2, 13, 8, "p_fall, hidden,\npatch tokens\n(73 x 3584)", VLM)
# fusion + head
box(70, 11, 13, 8, "fusion\nconcat MLP or\ncross-attention", FUSE, bold=True)
box(87, 11, 12.5, 8, "clip score\nFall vs No Fall\nor 16 actions", FUSE)
arrow(11.8, 17, 15.7, 23)
arrow(11.8, 13, 15.7, 7)
for x0, x1, y in [(30.3, 33.7, 24), (48.3, 51.7, 24), (30.3, 33.7, 6), (48.3, 51.7, 6), (83.3, 86.7, 15)]:
    arrow(x0, y, x1, y)
arrow(65.3, 24, 69.7, 18)
arrow(65.3, 6, 69.7, 12)
ax.text(93.2, 8.6, "video score = max over clips", ha="center", va="top", fontsize=6, color=MUTED)
fig.tight_layout(pad=0.1)
fig.savefig(ROOT / "report/figs/pipeline.pdf")
fig.savefig(ROOT / "report/figs/pipeline.png", dpi=200)
print("saved")
