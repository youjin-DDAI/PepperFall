"""Contact sheets of videos misclassified at video level (for error-type analysis).

  python eval/error_sheets.py --per-group 8
Input: results/error_clip_scores.csv (per-clip scores / predictions, FallVision test + external datasets)
Group = model x dataset x {FP: No Fall predicted as Fall, FN: missed Fall}, up to per-group random videos each.
One sheet: top row = 8 frames uniform over the whole video, bottom row = 4 frames of the model's top-scoring clip + score / action / VLM description.
Output: results/error_sheets/{model}/{dataset}_{FP|FN}_{k}.jpg, results/error_sheets/index.csv
"""
import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.preprocess.vlm_captions import clip_frames, read_frames  # noqa: E402

MODELS = {"VLM zero-shot": "zeroshot", "VLM hidden MLP": "hidden", "ST-GCN + VLM tokens": "fusion"}
H = 200


def video_info(dataset, vid):
    if dataset == "FallVision":
        meta = video_info.fv
        return ROOT / "data/raw/FallVision" / vid, meta.loc[vid]
    meta = video_info.ext.setdefault(dataset, pd.read_csv(ROOT / f"data/processed/ext_{dataset}/meta.csv")
                                     .set_index("video_path"))
    return ROOT / "data/raw" / dataset / "videos" / vid, meta.loc[vid]


video_info.fv = pd.read_csv(ROOT / "data/keypoints_yolo/meta.csv").set_index("video_path")
video_info.ext = {}


def strip(frames):
    ims = [f.resize((int(f.width * H / f.height), H)) for f in frames]
    out = Image.new("RGB", (sum(i.width for i in ims), H))
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width
    return out


def sheet(dataset, vid, clip_idx, text):
    path, m = video_info(dataset, vid)
    n = int(m.n_frames)
    overview = read_frames(path, np.linspace(0, n - 1, 8).round().astype(int))
    top, bottom = strip([overview[k] for k in sorted(overview)]), strip(clip_frames(path, n, float(m.fps), 16)[clip_idx])
    lines = textwrap.wrap(text, 150)
    w = max(top.width, bottom.width)
    canvas = Image.new("RGB", (w, 2 * H + 18 * len(lines) + 30), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, H + 6))
    d = ImageDraw.Draw(canvas)
    for i, l in enumerate(lines):
        d.text((6, 2 * H + 12 + 18 * i), l, fill="black")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=8)
    args = ap.parse_args()
    c = pd.read_csv(ROOT / "results/error_clip_scores.csv")
    caps = pd.read_csv(ROOT / "data/processed/vlm_T16/test.csv").set_index(["video_id", "clip_idx"]).caption
    actions = pd.read_csv(ROOT / "results/action_clip_scores.csv").set_index(["video_id", "clip_idx"]).action
    out_root = ROOT / "results/error_sheets"
    index = []
    rng = np.random.default_rng(0)
    for model, short in MODELS.items():
        v = c.groupby(["dataset", "video_id"]).agg(y=("y", "first"), pred=(f"{model}_pred", "max")).reset_index()
        v["pred"] = v.pred >= 0.5
        v = v[v.pred != v.y.astype(bool)]
        v["kind"] = np.where(v.y == 1, "FN", "FP")
        for (ds, kind), g in v.groupby(["dataset", "kind"]):
            g = g.iloc[rng.permutation(len(g))[:args.per_group]]
            for k, r in enumerate(g.itertuples()):
                clips = c[(c.dataset == ds) & (c.video_id == r.video_id)]
                best = clips.loc[clips[model].idxmax()]
                ci = int(best.clip_idx)
                key = (r.video_id, ci)
                info = (f"[{model}] {ds} {kind}  true={'Fall' if r.y else 'No Fall'}  video={r.video_id}  "
                        f"top clip {ci}/{len(clips) - 1} score={best[model]:.2f}")
                if key in actions.index:
                    info += f"  action={actions.loc[key]}"
                if ds == "FallVision" and key in caps.index:
                    info += f"  | VLM caption: {caps.loc[key]}"
                f = out_root / short / f"{ds}_{kind}_{k}.jpg"
                f.parent.mkdir(parents=True, exist_ok=True)
                sheet(ds, r.video_id, ci, info).save(f, quality=85)
                index.append({"file": str(f.relative_to(ROOT)), "model": model, "dataset": ds, "kind": kind,
                              "video_id": r.video_id, "top_clip": ci, "score": best[model]})
    pd.DataFrame(index).to_csv(out_root / "index.csv", index=False)
    print(f"{len(index)} sheets -> {out_root}")


if __name__ == "__main__":
    main()
