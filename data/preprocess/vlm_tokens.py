"""Frames from the same span as ST-GCN clips -> Qwen2.5-VL last-layer video patch tokens (frozen). Input to cross-attention fusion.

  python data/preprocess/vlm_tokens.py --T 16 --shard 0 --num-shards 3 --device cuda:0
  python data/preprocess/vlm_tokens.py --T 16 --merge-only
  python data/preprocess/vlm_tokens.py --ext GMDCSA24 --split test   # external dataset (build_external.py output)
  python data/preprocess/vlm_tokens.py --split test --frames last      # motion-removal check (output vlm_T16_last/)
  python data/preprocess/vlm_tokens.py --ext OF-Syn --mean-only        # save only tokmean / pfall, not full tokens (saves disk)
One forward pass with the Yes/No question (PROMPT from eval/vlm_zero_shot.py). 4 frames -> Qwen pairs them into 2 time steps,
(12 x 22) grid after 2x2 merge (22 x 12 for portrait). Avg-pool each time step to GRID x GRID -> 72 tokens,
then append the token before the answer (same vector as vlm_hidden.py) as a global token -> 73 tokens.
Position of token i (< 72): time step i // 36, grid row (i % 36) // GRID, col i % GRID. Grid centers = normalized image coords.
Output: data/processed/vlm_T{T}/{split}_tokens.npy (N, 73, 3584) float16,
      {split}_tokmean.npy (N, 7168) = [mean of 72 tokens, global]  (for concat-MLP control using the same info)
      {split}_pfall.npy (N,) zero-shot p(Fall) from the global token's Yes/No logit (only for extractions after this was added)
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

GRID = 6
N_TIME = N_FRAMES // 2  # Qwen2.5-VL temporal patch = 2 frames
N_TOK = N_TIME * GRID * GRID + 1


def tokens(vlm, clips, fps):
    torch = vlm.torch
    F = torch.nn.functional
    with torch.no_grad():
        inp = vlm._inputs(clips, PROMPT, fps)
        h = vlm.model(**inp, output_hidden_states=True).hidden_states[-1]
        vid = inp["input_ids"] == vlm.model.config.video_token_id
        merge = vlm.proc.video_processor.merge_size
        out = []
        for b, (t, gh, gw) in enumerate(inp["video_grid_thw"].tolist()):
            assert t == N_TIME, t
            v = h[b, vid[b]].view(t, gh // merge, gw // merge, -1).permute(0, 3, 1, 2).float()
            v = F.adaptive_avg_pool2d(v, GRID).permute(0, 2, 3, 1).reshape(t * GRID * GRID, -1)
            out.append(torch.cat([v, h[b, -1:].float()]))  # left padding -> -1 is the token before the answer
        logits = vlm.model.lm_head(h[:, -1]).float()
        p = torch.sigmoid(torch.logsumexp(logits[:, vlm.yes], 1) - torch.logsumexp(logits[:, vlm.no], 1))
    return torch.stack(out).cpu().numpy().astype(np.float16), p.cpu().numpy()


def to_mean(tok):
    return np.concatenate([tok[:, :-1].astype(np.float32).mean(1), tok[:, -1].astype(np.float32)], 1).astype(np.float16)


def merge(out_dir, clips_dir, split):
    z = np.load(clips_dir / f"{split}.npz")
    key = {(v, int(c)): i for i, (v, c) in enumerate(zip(z["video_id"], z["clip_idx"]))}
    parts = sorted(out_dir.glob(f"{split}_tokens_shard*.npz"))
    if parts and "tokmean" in np.load(parts[0]).files:  # extracted with --mean-only
        mean = np.zeros((len(key), np.load(parts[0])["tokmean"].shape[1]), np.float16)
        pfall = np.full(len(key), np.nan, np.float32)
        filled = np.zeros(len(key), bool)
        for p in parts:
            s = np.load(p)
            idx = np.array([key[(v, int(c))] for v, c in zip(s["video_id"], s["clip_idx"])])
            mean[idx], pfall[idx], filled[idx] = s["tokmean"], s["pfall"], True
        if not filled.all():
            raise SystemExit(f"{split}: {(~filled).sum()} clips without tokens")
        np.save(out_dir / f"{split}_tokmean.npy", mean)
        np.save(out_dir / f"{split}_pfall.npy", pfall)
        for p in parts:
            p.unlink()
        print(f"{split}: tokmean {mean.shape} (mean-only)")
        return
    dim = np.load(parts[0], mmap_mode="r")["tok"].shape[-1]
    tok = np.lib.format.open_memmap(out_dir / f"{split}_tokens.npy", "w+", np.float16, (len(key), N_TOK, dim))
    filled = np.zeros(len(key), bool)
    pfall = np.full(len(key), np.nan, np.float32)
    for p in parts:
        s = np.load(p)
        idx = np.array([key[(v, int(c))] for v, c in zip(s["video_id"], s["clip_idx"])])
        tok[idx], filled[idx] = s["tok"], True
        if "pfall" in s:
            pfall[idx] = s["pfall"]
        del s
        tok.flush()
        p.unlink()  # save disk: delete each shard once copied (tokens file fills sparsely)
    if not filled.all():
        raise SystemExit(f"{split}: {(~filled).sum()} clips without tokens")
    tok.flush()
    mean = to_mean(tok)
    np.save(out_dir / f"{split}_tokmean.npy", mean)
    if np.isfinite(pfall).all():
        np.save(out_dir / f"{split}_pfall.npy", pfall)
    print(f"{split}: tokens {tok.shape}, tokmean {mean.shape}")


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
    ap.add_argument("--frames", default="normal", choices=["normal", "last", "reverse"],
                    help="frame perturbation for shortcut checks: last = repeat last frame, reverse = reverse order")
    ap.add_argument("--mean-only", action="store_true", help="save only tokmean / pfall (no tokens for cross-attention)")
    ap.add_argument("--tag", default="", help="feature folder suffix for a non-default VLM, e.g. q25-3b -> vlm_T16_q25-3b")
    ap.add_argument("--ext", default=None, help="external dataset name (data/processed/ext_{ext}). Auto-merge after extraction")
    args = ap.parse_args()

    base = ROOT / "data/processed" / (f"ext_{args.ext}" if args.ext else "")
    out_dir = base / (f"vlm_T{args.T}" + ("" if args.frames == "normal" else f"_{args.frames}") +
                      (f"_{args.tag}" if args.tag else ""))
    clips_dir = base / f"runA_T{args.T}"
    split_dir = base if args.ext else ROOT / "data/processed/splits"
    if args.ext:
        h264 = ROOT / "data/raw" / args.ext / "videos_h264"  # OF-Syn: folder of AV1 originals transcoded to H.264
        args.root = str(h264 if h264.exists() else ROOT / "data/raw" / args.ext / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        for s in args.split:
            merge(out_dir, clips_dir, s)
        return

    meta = pd.read_csv(base / "meta.csv" if args.ext else ROOT / "data/keypoints_yolo/meta.csv").set_index("video_path")
    clip_fps = N_FRAMES / (args.T / TARGET_FPS)
    vlm = VLM(args.model, args.device)
    for split in args.split:
        out = out_dir / f"{split}_tokens_shard{args.shard}.npz"
        if out.exists():
            print(f"{out.name} exists, skip")
            continue
        vids = [l.strip() for l in open(split_dir / f"{split}.txt") if l.strip()]
        vids = vids[args.shard::args.num_shards]
        ids, cidx, toks, pf, buf = [], [], [], [], []
        for i, v in enumerate(vids):
            m = meta.loc[v]
            clips = clip_frames(Path(args.root) / v, int(m.n_frames), float(m.fps), args.T)
            if args.frames == "last":
                clips = [[c[-1]] * len(c) for c in clips]
            elif args.frames == "reverse":
                clips = [c[::-1] for c in clips]
            buf += [(v, k, c) for k, c in enumerate(clips)]
            while len(buf) >= args.batch or (i == len(vids) - 1 and buf):
                b, buf = buf[:args.batch], buf[args.batch:]
                t, p = tokens(vlm, [r[2] for r in b], clip_fps)
                toks.append(to_mean(t) if args.mean_only else t)
                pf.append(p)
                ids += [r[0] for r in b]
                cidx += [r[1] for r in b]
            if i % 200 == 0:
                print(f"[{split} {i}/{len(vids)}]", flush=True)
        np.savez(out, **{"tokmean" if args.mean_only else "tok": np.concatenate(toks)}, pfall=np.concatenate(pf), video_id=np.array(ids),
                 clip_idx=np.array(cidx))
        print(f"saved {out.name} {len(ids)} clips", flush=True)
    if (args.ext or args.frames != "normal") and args.num_shards == 1:
        for s in args.split:
            merge(out_dir, clips_dir, s)


if __name__ == "__main__":
    main()
