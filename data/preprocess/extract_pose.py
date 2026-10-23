"""FallVision Raw Video -> YOLO11x-pose + ByteTrack keypoint CSV.

Two outputs per video (data/keypoints_yolo/<source relative path>.{csv,json}):
  .csv  long format, frames x persons x 17 rows
        frame, track_id, box_conf, x1, y1, x2, y2, keypoint, x, y, xn, yn, conf
        - track_id=-1 : detection the tracker gave no id (has coords) or missed frame (coords NaN, conf=0)
        - every decoded frame has at least 17 rows (no missing frames)
  .json video meta (video_path, n_frames, fps, width, height, label, scenario, ...)
Skip if both exist -> safe to rerun / resume. Writes go to a tmp file, then rename.
Finally gathers all .json into <out>/meta.csv (can run alone with --merge-only).

Multi-GPU: launch N copies of the same command with --shard k --num-shards N --device k.
  for k in 0 1 2; do python data/preprocess/extract_pose.py --shard $k --num-shards 3 --device $k & done
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
HEADER = ["frame", "track_id", "box_conf", "x1", "y1", "x2", "y2",
          "keypoint", "x", "y", "xn", "yn", "conf"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SCENARIOS = {"bed", "chair", "stand"}


def find_videos(root):
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def parse_label(rel):
    """Parse label(1=Fall, 0=No Fall), scenario, archive(rar name) from path."""
    parts = rel.parts
    label = {"fall": 1, "no fall": 0}.get(parts[0].lower(), -1)
    scenario = next((p.lower() for p in parts if p.lower() in SCENARIOS), "unknown")
    archive = parts[-2] if len(parts) >= 2 else ""
    return label, scenario, archive


def frame_rows(fi, result):
    """One frame's result -> list of CSV rows."""
    boxes, kpts = result.boxes, result.keypoints
    if boxes is None or len(boxes) == 0 or kpts is None:
        return [[fi, -1, 0.0, "", "", "", "", k, "", "", "", "", 0.0] for k in COCO17], 0

    xyxy = boxes.xyxy.cpu().numpy()
    bconf = boxes.conf.cpu().numpy()
    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.full(len(boxes), -1)
    xy = kpts.xy.cpu().numpy()
    xyn = kpts.xyn.cpu().numpy()
    kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2])

    rows = []
    for i in range(len(boxes)):
        b = [round(float(v), 2) for v in xyxy[i]]
        for j, name in enumerate(COCO17):
            rows.append([fi, int(ids[i]), round(float(bconf[i]), 4), *b, name,
                         round(float(xy[i, j, 0]), 2), round(float(xy[i, j, 1]), 2),
                         round(float(xyn[i, j, 0]), 5), round(float(xyn[i, j, 1]), 5),
                         round(float(kconf[i, j]), 4)])
    return rows, len(boxes)


def process_video(model, video, rel, out_csv, out_json, args):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # persist=False: new tracker per track() call -> no id state leaks across videos
    results = model.track(source=str(video), stream=True, persist=False,
                          tracker=args.tracker, conf=args.conf, imgsz=args.imgsz,
                          device=args.device, quantize=16 if args.half else None, verbose=False)

    tmp_csv = out_csv.with_suffix(".csv.tmp")
    n_frames, n_detected, track_ids, shape = 0, 0, set(), (0, 0)
    with open(tmp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for fi, r in enumerate(results):
            rows, n_det = frame_rows(fi, r)
            w.writerows(rows)
            n_frames += 1
            n_detected += n_det > 0
            track_ids.update(row[1] for row in rows if row[1] >= 0)
            shape = r.orig_shape
    os.replace(tmp_csv, out_csv)

    label, scenario, archive = parse_label(rel)
    meta = {
        "video_path": str(rel),
        "csv_path": str(out_csv.relative_to(args.out)),
        "n_frames": n_frames,                  # frames actually decoded
        "n_frames_header": n_frames_header,    # frame count claimed by container
        "fps": round(fps, 3),
        "width": int(shape[1]),                # actual frame size after rotation meta
        "height": int(shape[0]),
        "label": label,
        "scenario": scenario,
        "archive": archive,
        "n_frames_detected": int(n_detected),
        "n_tracks": len(track_ids),
    }
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(meta))
    os.replace(tmp_json, out_json)


def merge_meta(out_root):
    out_root = Path(out_root)
    metas = [json.loads(p.read_text()) for p in sorted(out_root.rglob("*.json"))]
    if not metas:
        return 0
    tmp = out_root / "meta.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    os.replace(tmp, out_root / "meta.csv")
    return len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FallVision")
    ap.add_argument("--out", default="data/keypoints_yolo")
    ap.add_argument("--weights", default="weights/yolo11x-pose.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sub-shard", type=int, default=0, help="for re-splitting a slow shard: splits the shard's list again")
    ap.add_argument("--num-sub-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit number of videos")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    args.out = Path(args.out)

    if args.merge_only:
        print(f"meta.csv: {merge_meta(args.out)} videos")
        return

    videos = find_videos(args.root)[args.shard::args.num_shards][args.sub_shard::args.num_sub_shards]
    if args.limit:
        videos = videos[:args.limit]

    from ultralytics import YOLO
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    n_skip, failed = 0, []
    for video in tqdm(videos, desc=f"shard {args.shard}/{args.num_shards}"):
        rel = video.relative_to(args.root)
        out_csv = (args.out / rel).with_suffix(".csv")
        out_json = (args.out / rel).with_suffix(".json")
        if out_csv.exists() and out_json.exists():
            n_skip += 1
            continue
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            process_video(model, video, rel, out_csv, out_json, args)
        except Exception as e:  # don't let one corrupt video kill the whole shard
            failed.append((str(rel), repr(e)))
            tqdm.write(f"FAIL {rel}: {e!r}")

    if failed:
        with open(args.out / f"failed_shard{args.shard}_{args.sub_shard}.txt", "w") as f:
            f.writelines(f"{p}\t{e}\n" for p, e in failed)
    print(f"done: {len(videos)} videos, skipped {n_skip}, failed {len(failed)}")
    print(f"meta.csv: {merge_meta(args.out)} videos")


if __name__ == "__main__":
    main()
