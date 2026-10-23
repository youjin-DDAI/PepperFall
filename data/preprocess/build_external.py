"""External datasets (cross-dataset eval of FallVision-trained models) keypoint CSV -> clip tensors + OmniFall labels.

  python data/preprocess/build_external.py --name GMDCSA24
Clips built as in build_clips.py (main track, 10fps, window T / stride T//2, Run A / B).
Labels from OmniFall segment labels (data/raw/omnifall_labels/{file}.csv, label 1 = fall):
  y       : video-level (1 if any fall segment). Same definition as FallVision training labels
  y_seg   : clip-level (1 if clip span overlaps a fall segment at all). For eval
  y_event : clip-level training label. 1 if overlap with fall >= EVENT_MIN_OVERLAP s, 0 if none, -1 in between (excluded from training)
  action  : OmniFall label id with most overlap with the clip (-1 if none)
split: OF-Syn uses the official random split (train / val / test), all others are test.
Output: data/processed/ext_{name}/run{A,B}_T{T}/{split}.npz (X, y, y_seg, y_event, action, video_id, clip_idx, scenario,
      subject), data/processed/ext_{name}/{split}.txt (video list, input to vlm_tokens.py)
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.preprocess.build_clips import TARGET_FPS, load_video, windows  # noqa: E402
from data.preprocess.normalize import preprocess_window  # noqa: E402

FALL = 1  # OmniFall label id
EVENT_MIN_OVERLAP = 0.3  # seconds


def omnifall_path_gmdcsa24(video_path):
    """'Subject 1/ADL/01.mp4' -> 'Subject_1/ADL/01'"""
    return str(Path(video_path).with_suffix("")).replace("Subject ", "Subject_")


def omnifall_path_by_stem(video_path, label_paths):
    """CAUCAFall: 'Subject.1/Hop/HopS1.avi' -> 'adl/HopS1' (OmniFall uses category folder / file name; file names are unique per video)"""
    by_stem = {p.split("/")[-1]: p for p in label_paths}
    return by_stem.get(Path(video_path).stem, Path(video_path).stem)


LABEL_DIR = ROOT / "data/raw/omnifall_labels"
DATASETS = {
    "GMDCSA24": {"labels": LABEL_DIR / "GMDCSA24.csv", "to_omnifall": lambda v, _: omnifall_path_gmdcsa24(v)},
    "CAUCAFall": {"labels": LABEL_DIR / "caucafall.csv", "to_omnifall": omnifall_path_by_stem},
    "OF-Syn": {"labels": ROOT / "data/raw/OF-Syn/meta/labels/of-syn.csv",
               "to_omnifall": lambda v, _: str(Path(v).with_suffix("")),
               "splits": ROOT / "data/raw/OF-Syn/meta/splits/syn/random"},
}


def _load(args):
    row, kp_root = args
    return row["video_path"], load_video(kp_root / row["csv_path"], int(row["n_frames"]), float(row["fps"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DATASETS))
    ap.add_argument("--T", type=int, nargs="+", default=[16])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    spec = DATASETS[args.name]
    kp_root = ROOT / "data/keypoints_ext" / args.name
    out = ROOT / "data/processed" / f"ext_{args.name}"

    meta = pd.read_csv(kp_root / "meta.csv")
    lab = pd.read_csv(spec["labels"])
    meta["of_path"] = meta.video_path.map(lambda v: spec["to_omnifall"](v, lab.path.unique()))
    missing = set(meta.of_path) - set(lab.path)
    if missing:
        raise SystemExit(f"{len(missing)} videos without OmniFall labels, e.g. {sorted(missing)[:3]}")
    falls = lab[lab.label == FALL].groupby("path")[["start", "end"]].apply(lambda d: d.to_numpy())
    subject = lab.groupby("path").subject.first()
    meta["label"] = meta.of_path.isin(falls.index).astype(int)
    meta["subject"] = meta.of_path.map(subject)

    if "splits" in spec:
        split_of = {p: s for s in ("train", "val", "test") for p in pd.read_csv(spec["splits"] / f"{s}.csv").path}
        meta["split"] = meta.of_path.map(split_of)
    else:
        meta["split"] = "test"
    out.mkdir(parents=True, exist_ok=True)
    for s, g in meta.groupby("split"):
        (out / f"{s}.txt").write_text("\n".join(g.video_path) + "\n")
    meta.to_csv(out / "meta.csv", index=False)
    segs = {p: g[["start", "end", "label"]].to_numpy() for p, g in lab.groupby("path")}

    with ProcessPoolExecutor(args.workers) as ex:
        videos = dict(ex.map(_load, [(r, kp_root) for _, r in meta.iterrows()], chunksize=8))

    for T in args.T:
        for run, split in ((r, s) for r in ("A", "B") for s in sorted(meta.split.unique())):
            X, y, yseg, yev, act, vid, cidx, subj = [], [], [], [], [], [], [], []
            for _, r in meta[meta.split == split].iterrows():
                v = videos[r.video_path]
                n = len(v["conf"])
                seg = segs[r.of_path]
                for k, w in enumerate(windows(n, T)):
                    t0, t1 = w[0] / TARGET_FPS, (min(w[-1], n - 1) + 1) / TARGET_FPS
                    ov = np.clip(np.minimum(seg[:, 1], t1) - np.maximum(seg[:, 0], t0), 0, None)
                    fall_ov = ov[seg[:, 2] == FALL].sum()
                    X.append(preprocess_window(run, v["xy"][w], v["xyn"][w], v["conf"][w], v["box"][w]))
                    y.append(r.label)
                    yseg.append(int(fall_ov > 0))
                    yev.append(1 if fall_ov >= EVENT_MIN_OVERLAP else (0 if fall_ov == 0 else -1))
                    act.append(int(seg[ov.argmax(), 2]) if ov.max() > 0 else -1)
                    vid.append(r.video_path)
                    cidx.append(k)
                    subj.append(r.subject)
            d = out / f"run{run}_T{T}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{split}.npz", X=np.stack(X), y=np.array(y, np.int64),
                                y_seg=np.array(yseg, np.int64), y_event=np.array(yev, np.int64),
                                action=np.array(act, np.int64), video_id=np.array(vid), clip_idx=np.array(cidx),
                                scenario=np.array([args.name.lower()] * len(y)), subject=np.array(subj))
            yy, ye = np.array(y), np.array(yev)
            print(f"{args.name} run{run} T{T} {split}: {(meta.split == split).sum()} videos, {len(yy)} clips "
                  f"(video-label Fall {yy.sum()}, event 1/0/ignore {(ye == 1).sum()}/{(ye == 0).sum()}/{(ye == -1).sum()})")
    sec = meta.n_frames / meta.fps
    print(f"mean length {sec.mean():.1f}s (min {sec.min():.1f}, max {sec.max():.1f}), {meta.subject.nunique()} subjects")


if __name__ == "__main__":
    main()
