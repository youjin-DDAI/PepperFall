"""Frames from the same span as ST-GCN clips -> Qwen2.5-VL last-layer hidden state (frozen). No text generation / MiniLM.

  python data/preprocess/vlm_hidden.py --T 16 --shard 0 --num-shards 3 --device cuda:0
  python data/preprocess/vlm_hidden.py --T 16 --merge-only
One forward pass with the Yes/No question (PROMPT from eval/vlm_zero_shot.py); last-layer hidden of the last token before the answer (3584-d).
The exact vector that produces the p_fall logit. Frame sampling same as vlm_captions.py.
Output: data/processed/vlm_T{T}/{split}_hidden_shard{k}.npz -> {split}_hidden.npy (N, 3584) float16, row order same as npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.preprocess.build_clips import TARGET_FPS  # noqa: E402
from data.preprocess.vlm_captions import N_FRAMES, VLM, clip_frames  # noqa: E402
from eval.vlm_zero_shot import PROMPT  # noqa: E402


def hidden(vlm, clips, fps):
    torch = vlm.torch
    with torch.no_grad():
        inp = vlm._inputs(clips, PROMPT, fps)
        h = vlm.model(**inp, output_hidden_states=True).hidden_states[-1][:, -1]  # left padding -> -1 is the last token
    return h.float().cpu().numpy().astype(np.float16)


def merge(out_dir, T, split):
    z = np.load(ROOT / f"data/processed/runA_T{T}/{split}.npz")
    key = {(v, int(c)): i for i, (v, c) in enumerate(zip(z["video_id"], z["clip_idx"]))}
    H = np.full((len(key), 0), 0, np.float16)
    filled = np.zeros(len(key), bool)
    for p in sorted(out_dir.glob(f"{split}_hidden_shard*.npz")):
        s = np.load(p)
        if H.shape[1] == 0:
            H = np.zeros((len(key), s["h"].shape[1]), np.float16)
        idx = [key[(v, int(c))] for v, c in zip(s["video_id"], s["clip_idx"])]
        H[idx], filled[idx] = s["h"], True
    if not filled.all():
        raise SystemExit(f"{split}: {(~filled).sum()} clips without hidden")
    np.save(out_dir / f"{split}_hidden.npy", H)
    print(f"{split}: hidden {H.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--root", default=str(ROOT / "data/raw/FallVision"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / f"data/processed/vlm_T{args.T}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        for s in args.split:
            merge(out_dir, args.T, s)
        return

    meta = pd.read_csv(ROOT / "data/keypoints_yolo/meta.csv").set_index("video_path")
    clip_fps = N_FRAMES / (args.T / TARGET_FPS)
    vlm = VLM(args.model, args.device)
    for split in args.split:
        out = out_dir / f"{split}_hidden_shard{args.shard}.npz"
        if out.exists():
            print(f"{out.name} exists, skip")
            continue
        vids = [l.strip() for l in open(ROOT / f"data/processed/splits/{split}.txt") if l.strip()]
        vids = vids[args.shard::args.num_shards]
        ids, cidx, hs, buf = [], [], [], []
        for i, v in enumerate(vids):
            m = meta.loc[v]
            buf += [(v, k, c) for k, c in enumerate(clip_frames(Path(args.root) / v, int(m.n_frames),
                                                               float(m.fps), args.T))]
            while len(buf) >= args.batch or (i == len(vids) - 1 and buf):
                b, buf = buf[:args.batch], buf[args.batch:]
                hs.append(hidden(vlm, [r[2] for r in b], clip_fps))
                ids += [r[0] for r in b]
                cidx += [r[1] for r in b]
            if i % 200 == 0:
                print(f"[{split} {i}/{len(vids)}]", flush=True)
        np.savez(out, h=np.concatenate(hs), video_id=np.array(ids), clip_idx=np.array(cidx))
        print(f"saved {out.name} {len(ids)} clips", flush=True)


if __name__ == "__main__":
    main()
