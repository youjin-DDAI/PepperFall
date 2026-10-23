"""results/{stgcn,fusion}/*/log.csv -> TensorBoard event files (tb/ in each run folder).

  python eval/logs_to_tensorboard.py
  tensorboard --logdir results --port 6006
New runs are logged to tb/ directly by train_stgcn.py / train_fusion.py; this script converts older runs.
Runs that already have event files are skipped.
"""
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
TAGS = {"loss": "loss/train", "val_loss": "loss/val", "val_clip_f1": "val/clip_f1",
        "val_video_f1": "val/video_f1", "val_video_auroc": "val/video_auroc", "lr": "lr"}


def main():
    n = 0
    for log in sorted(ROOT.glob("results/*/*/log.csv")):
        out = log.parent / "tb"
        if out.exists() and any(out.glob("events.*")):
            continue
        w = SummaryWriter(out)
        for _, r in pd.read_csv(log).iterrows():
            for col, tag in TAGS.items():
                if col in r and pd.notna(r[col]):
                    w.add_scalar(tag, r[col], int(r.epoch))
        w.close()
        n += 1
    print(f"converted {n} runs (results/*/*/tb)")


if __name__ == "__main__":
    main()
