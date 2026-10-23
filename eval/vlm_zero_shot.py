"""Qwen2.5-VL zero-shot Fall / No Fall baseline (video-only, no training).

  python eval/vlm_zero_shot.py --split test --shard 0 --num-shards 3 --device cuda:0
  python eval/vlm_zero_shot.py --split test --merge-only
Run in conda env `fall` (torch 2.6, transformers 5.x).
Samples the whole video at 2 fps as one input; p(Fall) from the "Yes"/"No" logits of the first answer token.
Output: results/vlm_zero_shot/{model}/{split}_pred[_shard{k}].csv, {split}_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, by_group, threshold_for_recall  # noqa: E402

PROMPT = ("This video was recorded by a fixed indoor camera. "
          "Does any person fall down in this video? Answer with only Yes or No.")


def sample_frames(path, fps=2.0, max_frames=32, short_side=360):
    """Uniformly sample the whole video at fps. Qwen2.5-VL groups frames in pairs, so keep an even count."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    ok, f = cap.read()
    while ok:
        frames.append(f)
        ok, f = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames: {path}")
    n = max(2, min(max_frames, int(round(len(frames) / src_fps * fps))))
    n += n % 2
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idx:
        img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = short_side / min(h, w)
        out.append(Image.fromarray(cv2.resize(img, (int(w * s) // 28 * 28, int(h * s) // 28 * 28),
                                              interpolation=cv2.INTER_AREA)))
    return out


def load_model(name, device):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
    proc = AutoProcessor.from_pretrained(name)
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
    return model, proc, yes, no


def score(model, proc, yes, no, frames, fps):
    import torch
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], videos=[frames], fps=fps, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1].float()
    ly, ln = torch.logsumexp(logits[yes], 0), torch.logsumexp(logits[no], 0)
    p = torch.sigmoid(ly - ln).item()
    top = proc.tokenizer.decode(logits.argmax().item()).strip()
    return p, top, len(frames)


def metrics(df, thr):
    y, p = df.label.to_numpy(), df.p_fall.to_numpy()
    return {"video": binary_metrics(y, p, thr), "video_by_scenario": by_group(df.scenario.to_numpy(), y, p, thr)}


def merge(out_dir, split):
    parts = sorted(out_dir.glob(f"{split}_pred_shard*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates("video_path")
    df.to_csv(out_dir / f"{split}_pred.csv", index=False)
    res = {"split": split, "n_videos": len(df), "answer_not_yes_no": int((~df.top_token.isin(["Yes", "No"])).sum()),
           "@0.5": metrics(df, 0.5)}
    val = out_dir / "val_pred.csv"
    if split == "test" and val.exists():
        v = pd.read_csv(val)
        thr = threshold_for_recall(v.label.to_numpy(), v.p_fall.to_numpy(), 0.9)
        res["val_thr_recall90"] = thr
        res["@val_recall90"] = metrics(df, thr)
    json.dump(res, open(out_dir / f"{split}_metrics.json", "w"), indent=2, default=float)
    m = res["@0.5"]["video"]
    print(f"{split} n={len(df)} @0.5: F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
          f"AUROC {m['auroc']:.3f}")
    for s, sm in res["@0.5"]["video_by_scenario"].items():
        print(f"  {s:6s} F1 {sm['f1']:.3f} P {sm['precision']:.3f} R {sm['recall']:.3f} AUROC {sm['auroc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--root", default=str(ROOT / "data/raw/FallVision"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "results/vlm_zero_shot" / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        return merge(out_dir, args.split)

    meta = pd.read_csv(ROOT / "data/keypoints_yolo/meta.csv").set_index("video_path")
    vids = [l.strip() for l in open(ROOT / f"data/processed/splits/{args.split}.txt") if l.strip()]
    vids = vids[args.shard::args.num_shards][:args.limit]
    out = out_dir / f"{args.split}_pred_shard{args.shard}.csv"
    done = set(pd.read_csv(out).video_path) if out.exists() else set()
    todo = [v for v in vids if v not in done]
    print(f"{args.split} shard {args.shard}/{args.num_shards}: {len(todo)} todo ({len(done)} done)")

    model, proc, yes, no = load_model(args.model, args.device)
    for i, v in enumerate(todo):
        p, top, nf = score(model, proc, yes, no, sample_frames(Path(args.root) / v, args.fps), args.fps)
        row = pd.DataFrame([{"video_path": v, "label": int(meta.loc[v, "label"]),
                             "scenario": meta.loc[v, "scenario"], "p_fall": p, "top_token": top, "n_frames": nf}])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)
        if i % 50 == 0:
            print(f"[{i}/{len(todo)}] {v} label {row.label[0]} p_fall {p:.3f} ({top})", flush=True)


if __name__ == "__main__":
    main()
