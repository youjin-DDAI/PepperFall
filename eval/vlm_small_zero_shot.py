"""Zero-shot clip p(Fall) of smaller VLMs on the test sets, same clips / frames / prompt as vlm_tokens.py (7B).

  python eval/vlm_small_zero_shot.py --model Qwen/Qwen2.5-VL-3B-Instruct --device cuda:2
  python eval/vlm_small_zero_shot.py --summary
Clips: 4 frames per 1.6 s window (vlm_captions.clip_frames), short side 360. p(Fall) from the Yes/No logits of the first answer token.
Qwen models get the 4 frames as one video; other models as 4 images.
Output: results/vlm_small/{model}/{dataset}_pfall.npy (row order = runA_T16/test.npz), results/vlm_small/summary.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.preprocess.build_clips import TARGET_FPS  # noqa: E402
from data.preprocess.vlm_captions import N_FRAMES, clip_frames  # noqa: E402
from eval.metrics import binary_metrics  # noqa: E402
from eval.vlm_zero_shot import PROMPT  # noqa: E402

OUT = ROOT / "results/vlm_small"
DATASETS = {
    "FallVision": (ROOT / "data/processed", ROOT / "data/keypoints_yolo/meta.csv", ROOT / "data/raw/FallVision"),
    "GMDCSA24": (ROOT / "data/processed/ext_GMDCSA24", ROOT / "data/processed/ext_GMDCSA24/meta.csv",
                 ROOT / "data/raw/GMDCSA24/videos"),
    "CAUCAFall": (ROOT / "data/processed/ext_CAUCAFall", ROOT / "data/processed/ext_CAUCAFall/meta.csv",
                  ROOT / "data/raw/CAUCAFall/videos"),
}


class Scorer:
    def __init__(self, name, device):
        import torch
        import transformers as tf
        self.torch, self.name = torch, name
        cls = (tf.Qwen2_5_VLForConditionalGeneration if "Qwen2.5-VL" in name
               else tf.Qwen3VLForConditionalGeneration if "Qwen3-VL" in name else tf.AutoModelForImageTextToText)
        self.model = cls.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                         device_map=device).eval()
        self.proc = tf.AutoProcessor.from_pretrained(name)
        self.proc.tokenizer.padding_side = "left"
        self.video = "Qwen" in name
        if "SmolVLM" in name:  # transformers 5.20 bug: merger chunks by vision patch_size (14) instead of tokens per image
            inner, merge, seq_len = self.model.model, type(self.model.model).inputs_merger, self.proc.image_seq_len

            def fixed(input_ids, inputs_embeds, image_hidden_states):
                cfg = inner.config.vision_config
                old, cfg.patch_size = cfg.patch_size, seq_len
                try:
                    return merge(inner, input_ids, inputs_embeds, image_hidden_states)
                finally:
                    cfg.patch_size = old
            inner.inputs_merger = fixed
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")})

    def __call__(self, clips, fps):
        torch = self.torch
        if self.video:
            msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            kw = {"videos": clips, "fps": fps}
            if "Qwen3-VL" in self.name:
                kw = {"videos": clips, "video_metadata": [{"fps": fps, "total_num_frames": len(c), "frames_indices": list(range(len(c)))} for c in clips],
                      "do_sample_frames": False}
            inp = self.proc(text=[text] * len(clips), padding=True, return_tensors="pt", **kw)
        else:
            if len(clips) > 1:  # padded batches break the image-token merge; score one clip at a time
                return np.concatenate([self([c], fps) for c in clips])
            msgs = [{"role": "user", "content": [{"type": "image", "image": f} for f in clips[0]] + [{"type": "text", "text": PROMPT}]}]
            inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt", processor_kwargs={"do_image_splitting": False})
            inp["pixel_values"] = inp["pixel_values"].to(self.model.dtype)
        inp = inp.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inp).logits[:, -1].float()
        p = torch.sigmoid(torch.logsumexp(logits[:, self.yes], 1) - torch.logsumexp(logits[:, self.no], 1))
        return p.cpu().numpy()


def run(args):
    scorer = Scorer(args.model, args.device)
    out_dir = OUT / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_fps = N_FRAMES / (16 / TARGET_FPS)
    for ds in args.datasets:
        base, meta_path, root = DATASETS[ds]
        out = out_dir / f"{ds}_pfall.npy"
        if out.exists():
            print(f"{out} exists, skip")
            continue
        z = np.load(base / "runA_T16/test.npz")
        meta = pd.read_csv(meta_path).set_index("video_path")
        key = {(v, int(c)): i for i, (v, c) in enumerate(zip(z["video_id"], z["clip_idx"]))}
        pfall = np.full(len(key), np.nan, np.float32)
        vids = list(dict.fromkeys(z["video_id"]))
        buf = []
        for i, v in enumerate(vids):
            m = meta.loc[v]
            buf += [(key[(v, k)], c) for k, c in enumerate(clip_frames(root / v, int(m.n_frames), float(m.fps), 16))
                    if (v, k) in key]
            while len(buf) >= args.batch or (i == len(vids) - 1 and buf):
                b, buf = buf[:args.batch], buf[args.batch:]
                pfall[[r[0] for r in b]] = scorer([r[1] for r in b], clip_fps)
            if i % 50 == 0:
                print(f"[{ds} {i}/{len(vids)}]", flush=True)
        assert np.isfinite(pfall).all(), f"{ds}: {np.isnan(pfall).sum()} clips missing"
        np.save(out, pfall)
        print(f"saved {out}", flush=True)


def summary():
    ref = {"FallVision": None,
           "GMDCSA24": ROOT / "data/processed/ext_GMDCSA24/vlm_T16/test_pfall.npy",
           "CAUCAFall": ROOT / "data/processed/ext_CAUCAFall/vlm_T16/test_pfall.npy"}
    rows = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        r = {"model": d.name}
        for ds, (base, _, _) in DATASETS.items():
            f = d / f"{ds}_pfall.npy"
            if not f.exists():
                continue
            z, p = np.load(base / "runA_T16/test.npz"), np.load(f)
            vid = pd.DataFrame({"v": z["video_id"], "y": z["y"], "p": p}).groupby("v").agg(y=("y", "max"), p=("p", "max"))
            r[f"{ds} video"] = binary_metrics(vid.y.to_numpy(), vid.p.to_numpy(), 0.5)["auroc"]
            if "y_seg" in z.files:
                r[f"{ds} clip"] = binary_metrics(z["y_seg"], p, 0.5)["auroc"]
                if ref[ds] is not None and d.name == "Qwen2.5-VL-7B-Instruct":
                    r[f"{ds} corr w/ stored 7B"] = float(np.corrcoef(p, np.load(ref[ds]))[0, 1])
        rows.append(r)
    md = ["# Zero-shot VLM size comparison (clip p(Fall), AUROC)", "",
          "Same clips, 4 frames and prompt as the 7B features in the report. video = max over clips.", "",
          pd.DataFrame(rows).set_index("model").round(3).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    summary() if args.summary else run(args)


if __name__ == "__main__":
    main()
