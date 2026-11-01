"""RGB frames from the same span as ST-GCN clips -> Qwen2.5-VL description + p_fall (frozen, no training).

  python data/preprocess/vlm_captions.py --T 16 --split train --shard 0 --num-shards 3 --device cuda:0
  python data/preprocess/vlm_captions.py --T 16 --merge-only      # merge shards + MiniLM embeddings
Clip spans same as build_clips.py (10fps resample -> windows(n, T)). 4 frames sampled uniformly per span.
- caption: describes posture/motion/surroundings (bed, floor, chair), not a verdict. Input to fusion text branch
- p_fall : first-token logit of a Yes/No question (same as eval/vlm_zero_shot.py), for ablation
Output: data/processed/vlm_T{T}/{split}_shard{k}.csv -> {split}.csv (video_id, clip_idx, caption, p_fall)
      {split}_emb.npy (N, 384), row order same as run{A,B}_T{T}/{split}.npz
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.preprocess.build_clips import TARGET_FPS, windows  # noqa: E402
from eval.vlm_zero_shot import PROMPT as YESNO_PROMPT  # noqa: E402

CAPTION_PROMPT = ("Describe the person's body posture and movement in these frames, and what they are "
                  "on or near (for example floor, bed, chair). Answer in two short sentences.")
N_FRAMES = 4
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def read_frames(path, keep, short_side=360):
    """Decode and keep only needed frame indices (videos with broken fps headers have 100k+ frames; loading all OOMs)."""
    keep, last = set(int(i) for i in keep), max(keep)
    cap = cv2.VideoCapture(str(path))
    frames, i = {}, 0
    ok = cap.grab()
    while ok and i <= last:
        if i in keep:
            f = cap.retrieve()[1]
            h, w = f.shape[:2]
            s = short_side / min(h, w)
            f = cv2.resize(f, (int(w * s) // 28 * 28, int(h * s) // 28 * 28), interpolation=cv2.INTER_AREA)
            frames[i] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
        ok = cap.grab()
    cap.release()
    return frames


def clip_frames(path, n_src, fps, T):
    """10fps indices of build_clips.load_video -> source frames. N_FRAMES per clip."""
    n10 = max(1, int(np.floor(n_src / fps * TARGET_FPS)))
    src = np.minimum(np.round(np.arange(n10) / TARGET_FPS * fps).astype(int), n_src - 1)
    picks = [src[w[np.linspace(0, T - 1, N_FRAMES).round().astype(int)]] for w in windows(n10, T)]
    frames = read_frames(path, np.concatenate(picks))
    last = max(frames)  # fall back to last frame if fewer frames than header claims
    return [[frames.get(i, frames[last]) for i in pick] for pick in picks]


class VLM:
    def __init__(self, name, device):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
        self.torch, self.qwen3 = torch, "Qwen3-VL" in name
        cls = Qwen3VLForConditionalGeneration if self.qwen3 else Qwen2_5_VLForConditionalGeneration
        self.model = cls.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.proc.tokenizer.padding_side = "left"
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]

    def _inputs(self, clips, prompt, fps):
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kw = {"fps": fps}
        if self.qwen3:  # frames are already sampled; pass their timing explicitly
            kw = {"video_metadata": [{"fps": fps, "total_num_frames": len(c), "frames_indices": list(range(len(c)))}
                                     for c in clips], "do_sample_frames": False}
        return self.proc(text=[text] * len(clips), videos=clips, padding=True,
                         return_tensors="pt", **kw).to(self.model.device)

    def run(self, clips, fps, max_new_tokens=60):
        torch = self.torch
        with torch.no_grad():
            inp = self._inputs(clips, YESNO_PROMPT, fps)
            logits = self.model(**inp).logits[:, -1].float()
            p = torch.sigmoid(torch.logsumexp(logits[:, self.yes], 1) - torch.logsumexp(logits[:, self.no], 1))
            inp = self._inputs(clips, CAPTION_PROMPT, fps)
            gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        caps = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return [c.strip().replace("\n", " ") for c in caps], p.cpu().numpy()


def merge(out_dir, T, split):
    from sentence_transformers import SentenceTransformer
    z = np.load(ROOT / f"data/processed/runA_T{T}/{split}.npz")
    order = pd.DataFrame({"video_id": z["video_id"], "clip_idx": z["clip_idx"]})
    parts = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"{split}_shard*.csv"))])
    df = order.merge(parts.drop_duplicates(["video_id", "clip_idx"]), on=["video_id", "clip_idx"], how="left")
    missing = df.caption.isna().sum()
    if missing:
        raise SystemExit(f"{split}: {missing} clips without caption")
    df.to_csv(out_dir / f"{split}.csv", index=False)
    enc = SentenceTransformer(TEXT_ENCODER, device="cuda")
    emb = enc.encode(df.caption.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    np.save(out_dir / f"{split}_emb.npy", emb.astype(np.float32))
    print(f"{split}: {len(df)} clips, emb {emb.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--root", default=str(ROOT / "data/raw/FallVision"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sub-shard", type=int, default=0)
    ap.add_argument("--num-sub-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="limit number of videos (for timing)")
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
        vids = [l.strip() for l in open(ROOT / f"data/processed/splits/{split}.txt") if l.strip()]
        vids = vids[args.shard::args.num_shards][:args.limit]
        sub = f"_{args.sub_shard}" if args.num_sub_shards > 1 else ""
        out = out_dir / f"{split}_shard{args.shard}{sub}.csv"
        done = set().union(*[set(pd.read_csv(f).video_id) for f in out_dir.glob(f"{split}_shard{args.shard}*.csv")])
        todo = [v for v in vids[args.sub_shard::args.num_sub_shards] if v not in done]  # split the full list so concurrent runs miss nothing
        print(f"{split} shard {args.shard}/{args.num_shards}: {len(todo)} videos todo ({len(done)} done)", flush=True)

        buf = []  # (video_id, clip_idx, frames); written to csv per video so restarts skip per video
        def flush(rows):
            caps, p = vlm.run([r[2] for r in rows], clip_fps)
            return [{"video_id": r[0], "clip_idx": r[1], "caption": c, "p_fall": float(q)}
                    for r, c, q in zip(rows, caps, p)]

        for i, v in enumerate(todo):
            m = meta.loc[v]
            clips = clip_frames(Path(args.root) / v, int(m.n_frames), float(m.fps), args.T)
            buf += [(v, k, c) for k, c in enumerate(clips)]
            if len(buf) >= args.batch or i == len(todo) - 1:
                rows = []
                for j in range(0, len(buf), args.batch):
                    rows += flush(buf[j:j + args.batch])
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                buf = []
            if i % 100 == 0:
                print(f"[{split} {i}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
