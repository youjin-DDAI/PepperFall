"""All models x all datasets results table -> results/all_results.md (mean ± std over seeds).

  python eval/summarize_all.py
FallVision: results/{stgcn,fusion}/*/metrics.json (test) + VLM zero-shot (video max of clip p_fall, data/processed/vlm_T16/*.csv)
External: results/external_all_seeds.csv (eval/eval_external.py)
FallVision clip labels are weak labels copied from the video label; external clip labels come from OmniFall fall segments.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, threshold_for_recall, video_level  # noqa: E402

MODELS = [  # (table name, FallVision results glob, external eval name or None)
    ("VLM zero-shot (Yes/No, clip max)", None, "VLM zero-shot (Yes/No, clip max)"),
    ("ST-GCN Run A, T=16", "stgcn/runA_T16_s*", "ST-GCN Run A (raw coords)"),
    ("ST-GCN Run B, T=16", "stgcn/runB_T16_s*", "ST-GCN Run B (normalized)"),
    ("ST-GCN Run A, T=32", "stgcn/runA_T32_s*", None),
    ("ST-GCN Run B, T=32", "stgcn/runB_T32_s*", None),
    ("VLM text only (MiniLM)", "fusion/text_runA_T16_s*", None),
    ("VLM hidden only", "fusion/hidden_runA_T16_s*", "VLM hidden only (MLP)"),
    ("VLM tokens only (mean)", "fusion/tokmean_runA_T16_s*", "VLM tokens only (mean, MLP)"),
    ("ST-GCN + VLM text", "fusion/skel_text_runA_T16_s*", None),
    ("ST-GCN + VLM text + p_fall", "fusion/skel_text_pfall_runA_T16_s*", None),
    ("ST-GCN + VLM hidden (concat)", "fusion/skel_hidden_runA_T16_s*", "ST-GCN + VLM hidden"),
    ("ST-GCN + VLM tokens (concat)", "fusion/skel_tokmean_runA_T16_s*", "ST-GCN + VLM tokens (mean, concat)"),
    ("ST-GCN x VLM tokens (cross-attn)", "fusion/skel_xattn_runA_T16_s*", "ST-GCN x VLM tokens (cross-attn)"),
]
METRICS = ["video_auroc", "clip_auroc", "f1_05", "f1_val"]


def fallvision_zero_shot():
    d = {s: pd.read_csv(ROOT / f"data/processed/vlm_T16/{s}.csv") for s in ("val", "test")}
    y = {s: np.load(ROOT / f"data/processed/runA_T16/{s}.npz")["y"] for s in d}
    _, vy, vp = video_level(d["val"].video_id.to_numpy(), y["val"], d["val"].p_fall.to_numpy())
    thr = threshold_for_recall(vy, vp, 0.9)
    p = d["test"].p_fall.to_numpy()
    _, vy, vp = video_level(d["test"].video_id.to_numpy(), y["test"], p)
    return [{"video_auroc": binary_metrics(vy, vp)["auroc"], "clip_auroc": binary_metrics(y["test"], p)["auroc"],
             "f1_05": binary_metrics(vy, vp, 0.5)["f1"], "f1_val": binary_metrics(vy, vp, thr)["f1"]}]


def fallvision(pattern):
    out = []
    for p in sorted((ROOT / "results").glob(f"{pattern}/metrics.json")):
        m = json.loads(p.read_text())
        out.append({"video_auroc": m["test@0.5"]["video"]["auroc"], "clip_auroc": m["test@0.5"]["clip"]["auroc"],
                    "f1_05": m["test@0.5"]["video"]["f1"], "f1_val": m["test@val_recall90"]["video"]["f1"]})
    return out


def external(ext, name, model):
    s = ext[(ext.dataset == name) & (ext.model == model)]
    out = []
    for seed, g in s.groupby("seed"):
        a, b = g[g.thr == "0.5"].iloc[0], g[g.thr == "val_recall90"].iloc[0]
        out.append({"video_auroc": a.video_auroc, "clip_auroc": a.clip_seg_auroc, "f1_05": a.video_f1,
                    "f1_val": b.video_f1})
    return out


def cell(runs, k):
    if not runs:
        return "—"
    v = np.array([r[k] for r in runs])
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    ext = pd.read_csv(ROOT / "results/external_all_seeds.csv")
    datasets = ["FallVision", "GMDCSA24", "CAUCAFall"]
    res = {}
    for label, fv_glob, ext_name in MODELS:
        res[label] = {"FallVision": fallvision_zero_shot() if fv_glob is None else fallvision(fv_glob)}
        for ds in datasets[1:]:
            res[label][ds] = external(ext, ds, ext_name) if ext_name else []

    titles = {"video_auroc": "Video-level AUROC", "clip_auroc": "Clip-level AUROC (FallVision: weak label, external: fall segments)",
              "f1_05": "Video F1, threshold 0.5", "f1_val": "Video F1, threshold = recall 0.9 on FallVision val"}
    md = ["# All results (trained on FallVision; external datasets evaluated without retraining / recalibration)", "",
          "Mean ± std over 3 seeds. VLM zero-shot has no training (no seeds). — = not evaluated.",
          "FallVision test 500 videos / GMDCSA24 160 videos (4 subjects) / CAUCAFall 100 videos (10 subjects).", ""]
    for k in METRICS:
        t = pd.DataFrame({ds: [cell(res[m[0]][ds], k) for m in MODELS] for ds in datasets},
                         index=[m[0] for m in MODELS])
        md += [f"## {titles[k]}", "", t.to_markdown(), ""]
    (ROOT / "results/all_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
