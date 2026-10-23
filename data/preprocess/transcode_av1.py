"""OF-Syn AV1 videos -> H.264 640x360 mp4 (OpenCV can't read AV1). Frame count / fps unchanged.

  python data/preprocess/transcode_av1.py --src data/raw/OF-Syn/videos --dst data/raw/OF-Syn/videos_h264
PyAV (libdav1d decode, libx264 encode, crf 20). Skips already converted files. Writes to .tmp, then rename.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av


def convert(args):
    src, dst, width = args
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    with av.open(str(src)) as inp, av.open(str(tmp), "w") as out:
        vin = inp.streams.video[0]
        h = int(round(vin.height * width / vin.width / 2)) * 2
        vout = out.add_stream("libx264", rate=vin.average_rate)
        vout.width, vout.height, vout.pix_fmt = width, h, "yuv420p"
        vout.options = {"crf": "20", "preset": "veryfast"}
        n = 0
        for frame in inp.decode(video=0):
            for pkt in vout.encode(frame.reformat(width=width, height=h, format="yuv420p")):
                out.mux(pkt)
            n += 1
        for pkt in vout.encode():
            out.mux(pkt)
    os.replace(tmp, dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/OF-Syn/videos")
    ap.add_argument("--dst", default="data/raw/OF-Syn/videos_h264")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    src = Path(args.src)
    jobs = [(p, Path(args.dst) / p.relative_to(src), args.width) for p in sorted(src.rglob("*.mp4"))]
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(convert, jobs, chunksize=16)):
            done += 1
            if i % 1000 == 0:
                print(f"[{i}/{len(jobs)}] {r}", flush=True)
    print(f"done {done}/{len(jobs)}")


if __name__ == "__main__":
    main()
