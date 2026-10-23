"""Cross-dataset eval: apply FallVision-trained checkpoints to external datasets without retraining / recalibration.

  python eval/eval_external.py --datasets GMDCSA24 CAUCAFall
Input: data/processed/ext_{name}/run{A,B}_T16/test.npz, vlm_T16/test_{tokens,tokmean,pfall}.npy (build_external.py, vlm_tokens.py)
Thresholds: 0.5 and the value chosen on FallVision val (val_thr_recall90 in metrics.json). Nothing is tuned on external data.
Metrics: video level (clip prob max, label = has a fall segment) + clip level (overlaps an OmniFall fall segment, y_seg)
Output: results/external_summary.md, results/external_all_seeds.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import binary_metrics, threshold_for_recall, video_level  # noqa: E402
from models.fusion import build_model  # noqa: E402
from models.stgcn import STGCN  # noqa: E402

T = 16
MODELS = [  # (table name, results folder glob, mode)  mode None = ST-GCN only
    ("ST-GCN Run A (raw coords)", "stgcn/runA_T16_s*", None),
    ("ST-GCN Run B (normalized)", "stgcn/runB_T16_s*", None),
    ("VLM hidden only (MLP)", "fusion/hidden_runA_T16_s*", "hidden"),
    ("VLM tokens only (mean, MLP)", "fusion/tokmean_runA_T16_s*", "tokmean"),
    ("ST-GCN + VLM hidden", "fusion/skel_hidden_runA_T16_s*", "skel_hidden"),
    ("ST-GCN + VLM tokens (mean, concat)", "fusion/skel_tokmean_runA_T16_s*", "skel_tokmean"),
    ("ST-GCN x VLM tokens (cross-attn)", "fusion/skel_xattn_runA_T16_s*", "skel_xattn"),
]


def zero_shot_threshold():
    """Pick the VLM zero-shot (video max of clip p_fall) threshold on FallVision val with the same rule."""
    v = pd.read_csv(ROOT / "data/processed/vlm_T16/val.csv")
    vids, vy, vp = video_level(v.video_id.to_numpy(), v.video_id.str.startswith("Fall/").astype(int).to_numpy(),
                               v.p_fall.to_numpy())
    return threshold_for_recall(vy, vp, 0.9)


def score(d, prob, thr):
    _, vy, vp = video_level(d["video_id"], d["y"], prob)
    v, c = binary_metrics(vy, vp, thr), binary_metrics(d["y_seg"], prob, thr)
    return {"video_f1": v["f1"], "video_precision": v["precision"], "video_recall": v["recall"],
            "video_auroc": v["auroc"], "clip_seg_f1": c["f1"], "clip_seg_auroc": c["auroc"]}


@torch.no_grad()
def predict(model, X, emb, bs=256):
    model.eval()
    zeros = torch.zeros(len(X), device=X.device)
    out = [torch.sigmoid(model(X[i:i + bs], emb[i:i + bs], zeros[i:i + bs]) if emb is not None
                         else model(X[i:i + bs])) for i in range(0, len(X), bs)]
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["GMDCSA24", "CAUCAFall"])
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/fusion.yaml"))
    zs_thr = zero_shot_threshold()

    rows = []
    for name in args.datasets:
        base = ROOT / "data/processed" / f"ext_{name}"
        d = {r: dict(np.load(base / f"run{r}_T{T}/test.npz")) for r in ("A", "B")}
        X = {r: torch.from_numpy(d[r]["X"]).to(args.device) for r in d}
        tok = np.load(base / f"vlm_T{T}/test_tokens.npy")
        emb = {"tokens": torch.from_numpy(tok).to(args.device),
               "hidden": torch.from_numpy(tok[:, -1].astype(np.float32)).to(args.device),
               "tokmean": torch.from_numpy(np.load(base / f"vlm_T{T}/test_tokmean.npy").astype(np.float32)).to(args.device)}
        ref = d["A"]
        pf = np.load(base / f"vlm_T{T}/test_pfall.npy")
        for thr_name, thr in (("0.5", 0.5), ("val_recall90", zs_thr)):
            rows.append({"dataset": name, "model": "VLM zero-shot (Yes/No, clip max)", "seed": 0, "thr": thr_name,
                         **score(ref, pf, thr)})

        for label, pattern, mode in MODELS:
            for run_dir in sorted((ROOT / "results").glob(pattern)):
                m = json.loads((run_dir / "metrics.json").read_text())
                run = m["run"]
                if mode is None:
                    model, e = STGCN(dropout=0.1), None
                else:
                    model = build_model(mode, cfg, T)
                    e = emb[model.vlm_feat]
                model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
                prob = predict(model.to(args.device), X[run], e)
                fv = m["test@0.5"]["video"]["f1"]
                for thr_name, thr in (("0.5", 0.5), ("val_recall90", m["val_thr_recall90"])):
                    rows.append({"dataset": name, "model": label, "seed": m["seed"], "thr": thr_name,
                                 "fallvision_video_f1": fv if thr_name == "0.5" else m["test@val_recall90"]["video"]["f1"],
                                 **score(d[run], prob, thr)})
        print(f"{name}: done")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/external_all_seeds.csv", index=False)
    cols = ["fallvision_video_f1", "video_f1", "video_precision", "video_recall", "video_auroc", "clip_seg_f1",
            "clip_seg_auroc"]
    order = ["VLM zero-shot (Yes/No, clip max)"] + [m[0] for m in MODELS]
    md = ["# Cross-dataset eval (trained on FallVision -> external datasets, no retraining / recalibration)", "",
          "fallvision_video_f1 = FallVision test video F1 of the same checkpoint (for comparison). clip_seg = clip level vs OmniFall fall segments.",
          "Mean ± std over seeds (VLM zero-shot has no seeds).", ""]
    for thr_name in ("0.5", "val_recall90"):
        for name in args.datasets:
            sub = df[(df.dataset == name) & (df.thr == thr_name)]
            g = sub.groupby("model")[cols]
            mean, std = g.mean(), g.std().fillna(0)
            t = pd.DataFrame({c: [f"{a:.3f} ± {b:.3f}" if np.isfinite(a) else "—" for a, b in zip(mean[c], std[c])]
                              for c in cols}, index=mean.index).reindex([o for o in order if o in mean.index])
            md += [f"## {name} (threshold {thr_name})", "", t.to_markdown(), ""]
    (ROOT / "results/external_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
