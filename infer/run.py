"""Run the fall detector on a video file, the laptop camera, or Pepper's camera.

  python infer/run.py --ckpt CKPT_DIR --source video.mp4
  python infer/run.py --ckpt CKPT_DIR --source camera --show
  python infer/run.py --ckpt CKPT_DIR --source pepper:192.168.1.20 --show     # needs infer/pepper_bridge.py on Pepper
CKPT_DIR = a trained run folder (best.pt + metrics.json), e.g. results/event/skel_tokmean_q25-3b/OF-Syn-multiclass_runA_T16_s0.
--vlm-model must be the VLM the checkpoint was trained with (7B: no tag, q25-3b: Qwen2.5-VL-3B, q3-2b: Qwen3-VL-2B).
Alarm: score >= --thr in --k consecutive windows (0.8 s apart); then --cooldown seconds before the next alarm.
Score = p(fall) + p(fallen) for multiclass checkpoints (alarm also on the person already down), p(fall) for binary.
Output: one line per window on stdout, and --log CSV if given.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer.detector import FallDetector  # noqa: E402
from infer.sources import Pepper, open_source  # noqa: E402

VLM_OF_TAG = {"": "Qwen/Qwen2.5-VL-7B-Instruct", "q25-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
              "q3-2b": "Qwen/Qwen3-VL-2B-Instruct"}


def default_vlm(ckpt):
    mode_dir = Path(ckpt).resolve().parent.name  # e.g. skel_tokmean_q25-3b
    return VLM_OF_TAG[next((t for t in VLM_OF_TAG if t and mode_dir.endswith("_" + t)), "")]


def draw(frame, res, alarm):
    if res:
        txt = f"alarm {res['p_alarm']:.2f}" + (f" | {res['action']}" if "action" in res else "")
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if alarm else (0, 200, 0), 2)
    cv2.imshow("fall detector", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="video path | camera[:INDEX] | pepper:HOST[:PORT]")
    ap.add_argument("--pose", default="yolo11x-pose.pt", help=".pt (runs on --device) or CoreML .mlpackage")
    ap.add_argument("--vlm-model", default=None, help="default: inferred from the checkpoint folder name")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--say", default="Are you okay? I think you fell. Do you need help?")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    det = FallDetector(args.ckpt, args.pose, args.vlm_model or default_vlm(args.ckpt), args.device)
    src = open_source(args.source)
    log = csv.writer(open(args.log, "w", newline="")) if args.log else None
    if log:
        log.writerow(["frame", "time", "p_fall", "p_alarm", "action", "vlm_p_fall", "alarm", "latency_s"])
    streak, last_alarm, res = 0, -1e9, None
    while True:
        frame = src.read()
        if frame is None:
            break
        t0 = time.time()
        out = det.step(frame)
        alarm = False
        if out:
            res = out
            streak = streak + 1 if out["p_alarm"] >= args.thr else 0
            alarm = streak >= args.k and t0 - last_alarm > args.cooldown
            if alarm:
                last_alarm = t0
                if isinstance(src, Pepper):
                    src.send({"say": args.say})
            lat = time.time() - t0
            print(f"[{det.n:6d}] alarm={out['p_alarm']:.3f} fall={out['p_fall']:.3f} {out.get('action', '')} "
                  f"{'ALARM' if alarm else ''} ({lat:.2f}s)", flush=True)
            if log:
                log.writerow([det.n, f"{t0:.2f}", out["p_fall"], out["p_alarm"], out.get("action", ""),
                              out["vlm_p_fall"], int(alarm), f"{lat:.3f}"])
        if args.show and draw(frame, res, alarm or streak >= args.k):
            break
    src.close()


if __name__ == "__main__":
    main()
