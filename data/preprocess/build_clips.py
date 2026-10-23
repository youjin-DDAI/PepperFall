"""keypoint CSV -> ST-GCN clip tensors + split.

1. Dedup: use only keep=1 videos from data/raw/duplicates.csv
2. Main track: track_id with most frames (tie: largest mean box area). Other frames treated as detection failures
3. Source fps -> 10 fps nearest resample
4. Windows: length T, stride T//2. Videos shorter than T repeat the last frame (edge-pad). Every window gets the video label
5. Run A(raw xn,yn) / Run B(hip-center + torso scale) -> (N, 3, T, 17)  [normalize.py]
6. Split: by session_group, stratified by label x scenario, StratifiedGroupKFold(10, seed 0) -> fold0=test, fold1=val, rest train.
   Reuses data/processed/splits/{train,val,test}.txt if present (--resplit to regenerate)
Output: data/processed/run{A,B}_T{T}/{train,val,test}.npz (X, y, video_id, clip_idx, scenario), stats.md

Run: python data/preprocess/build_clips.py --T 16 32
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.preprocess.extract_pose import COCO17  # noqa: E402
from data.preprocess.normalize import preprocess_window  # noqa: E402

TARGET_FPS = 10
SPLITS = ("train", "val", "test")


def select_main_track(df):
    tracked = df[df.track_id >= 0]
    if tracked.empty:
        return None
    g = tracked.groupby("track_id")
    stats = pd.DataFrame({
        "n": g.frame.nunique(),
        "area": g.apply(lambda d: ((d.x2 - d.x1) * (d.y2 - d.y1)).mean(), include_groups=False),
    })
    return int(stats.sort_values(["n", "area"], ascending=False).index[0])


def load_video(csv_path, n_frames, fps):
    """-> dict of 10fps arrays: xy (F,17,2) pixels, xyn (F,17,2), conf (F,17), box (F,4), detected (F,)"""
    df = pd.read_csv(csv_path)
    tid = select_main_track(df)
    xy = np.zeros((n_frames, 17, 2), np.float32)
    xyn = np.zeros((n_frames, 17, 2), np.float32)
    conf = np.zeros((n_frames, 17), np.float32)
    box = np.full((n_frames, 4), np.nan, np.float32)
    if tid is not None:
        d = df[df.track_id == tid]
        f = d.frame.to_numpy()
        j = d.keypoint.map({k: i for i, k in enumerate(COCO17)}).to_numpy()
        xy[f, j] = d[["x", "y"]].to_numpy()
        xyn[f, j] = d[["xn", "yn"]].to_numpy()
        conf[f, j] = d.conf.to_numpy()
        first = d.drop_duplicates("frame")
        box[first.frame.to_numpy()] = first[["x1", "y1", "x2", "y2"]].to_numpy()

    n_out = max(1, int(np.floor(n_frames / fps * TARGET_FPS)))
    src = np.minimum(np.round(np.arange(n_out) / TARGET_FPS * fps).astype(int), n_frames - 1)
    return {"xy": xy[src], "xyn": xyn[src], "conf": conf[src], "box": box[src],
            "detected": np.isfinite(box[src, 0]), "main_track": -1 if tid is None else tid}


def windows(n, T):
    """Window start indices and per-window frame indices (edge-pad if short)."""
    if n <= T:
        return [np.minimum(np.arange(T), n - 1)]
    starts = list(range(0, n - T + 1, T // 2))
    if starts[-1] != n - T:  # align last window to the end so the tail isn't dropped
        starts.append(n - T)
    return [np.arange(s, s + T) for s in starts]


def _load(args):
    row, kp_root = args
    return row["video_path"], load_video(kp_root / row["csv_path"], int(row["n_frames"]), float(row["fps"]))


def make_splits(meta, split_dir, resplit):
    from sklearn.model_selection import StratifiedGroupKFold
    split_dir.mkdir(parents=True, exist_ok=True)
    files = {s: split_dir / f"{s}.txt" for s in SPLITS}
    if not resplit and all(f.exists() for f in files.values()):
        assign = {}
        for s, f in files.items():
            assign.update({p: s for p in f.read_text().splitlines() if p})
        missing = set(meta.video_path) - set(assign)
        if missing:
            raise SystemExit(f"{len(missing)} videos missing from split files. --resplit needed")
        return meta.video_path.map(assign)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    folds = np.empty(len(meta), int)
    strata = meta.label.astype(str) + "_" + meta.scenario  # balance label x scenario ratio across splits
    for k, (_, idx) in enumerate(sgkf.split(meta, strata, groups=meta.session_group)):
        folds[idx] = k
    split = pd.Series(np.where(folds == 0, "test", np.where(folds == 1, "val", "train")), index=meta.index)
    for s in SPLITS:
        files[s].write_text("\n".join(sorted(meta.video_path[split == s])) + "\n")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-root", default="data/keypoints_yolo")
    ap.add_argument("--dups", default="data/raw/duplicates.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--T", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--runs", nargs="+", default=["A", "B"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resplit", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="skip videos whose extraction isn't done (debug)")
    args = ap.parse_args()
    kp_root, out = Path(args.kp_root), Path(args.out)

    meta = pd.read_csv(kp_root / "meta.csv")
    dups = pd.read_csv(args.dups)
    kept = dups[dups.keep == 1][["video_path", "session_group"]]
    n_missing = len(set(kept.video_path) - set(meta.video_path))
    if n_missing and not args.allow_missing:
        raise SystemExit(f"{n_missing} videos without keypoints. Run after extraction finishes, or use --allow-missing")
    meta = meta.merge(kept, on="video_path").reset_index(drop=True)

    meta["split"] = make_splits(meta, out / "splits", args.resplit)

    with ProcessPoolExecutor(args.workers) as ex:
        videos = dict(ex.map(_load, [(r, kp_root) for _, r in meta.iterrows()], chunksize=32))

    stats_lines = []
    for T in args.T:
        for run in args.runs:
            d = out / f"run{run}_T{T}"
            d.mkdir(parents=True, exist_ok=True)
            for s in SPLITS:
                X, y, vid, cidx, scen = [], [], [], [], []
                for _, r in meta[meta.split == s].iterrows():
                    v = videos[r.video_path]
                    for k, w in enumerate(windows(len(v["conf"]), T)):
                        X.append(preprocess_window(run, v["xy"][w], v["xyn"][w], v["conf"][w], v["box"][w]))
                        y.append(r.label)
                        vid.append(r.video_path)
                        cidx.append(k)
                        scen.append(r.scenario)
                np.savez_compressed(d / f"{s}.npz", X=np.stack(X), y=np.array(y, np.int64),
                                    video_id=np.array(vid), clip_idx=np.array(cidx), scenario=np.array(scen))
                if run == args.runs[0]:
                    yy = np.array(y)
                    stats_lines.append(f"| {T} | {s} | {len(yy)} | {(yy == 1).sum()} | {(yy == 0).sum()} |")
            print(f"saved {d}")

    write_stats(out / "stats.md", meta, videos, stats_lines)


def write_stats(path, meta, videos, clip_lines):
    n10 = meta.video_path.map(lambda p: len(videos[p]["conf"]))
    det = meta.video_path.map(lambda p: videos[p]["detected"].mean())
    no_track = meta.video_path.map(lambda p: videos[p]["main_track"] < 0).sum()
    lab = meta.label.map({1: "Fall", 0: "No Fall"})
    L = ["# FallVision preprocessing stats", "",
         f"- Videos (after dedup): {len(meta)}",
         f"- Session groups: {meta.session_group.nunique()}",
         f"- Mean length: {(meta.n_frames / meta.fps).mean():.2f} s, median 10fps frames {int(n10.median())}",
         f"- Videos shorter than T=16 at 10fps: {(n10 < 16).sum()}, shorter than T=32: {(n10 < 32).sum()}",
         f"- Main-track missed-frame ratio (10fps): {1 - det.mean():.3%}",
         f"- No main track (no detections in any frame): {no_track}", "",
         "## Videos (split x label x scenario)", "",
         pd.crosstab([meta.split, lab], meta.scenario, margins=True).to_markdown(), "",
         "## Clips", "", "| T | split | clips | Fall | No Fall |", "|---|---|---|---|---|", *clip_lines, ""]
    path.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
