"""VLM zero-shot / ST-GCN / fusion comparison -> results/fusion_summary.md (test, mean ± std over seeds).

  python eval/summarize_fusion.py --run A --T 16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = [("video", "f1"), ("video", "precision"), ("video", "recall"), ("video", "auroc"),
        ("video/bed", "f1"), ("video/chair", "f1"), ("video/stand", "f1"), ("clip", "f1")]
HEADER = ["F1", "Precision", "Recall", "AUROC", "bed F1", "chair F1", "stand F1", "clip F1"]


def flat(m):
    """test@thr block of metrics.json -> {(level, metric): value}"""
    out = {("video", k): v for k, v in m["video"].items()}
    out.update({("clip", k): v for k, v in m.get("clip", {}).items()})
    for s, sm in m["video_by_scenario"].items():
        out.update({(f"video/{s}", k): v for k, v in sm.items()})
    return out


def row(name, runs):
    cells = []
    for c in COLS:
        vals = [r[c] for r in runs if c in r]
        if not vals:
            cells.append("—")
        elif len(vals) == 1:
            cells.append(f"{vals[0]:.3f}")
        else:
            cells.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")
    return [name, len(runs)] + cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="A")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--vlm", default="Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    zs = json.loads((ROOT / f"results/vlm_zero_shot/{args.vlm}/test_metrics.json").read_text())
    md = [f"# Fusion comparison (test 500 videos, Run {args.run}, T={args.T})", "",
          "VLM zero-shot takes the whole video once (no seeds, no clip level). Others: mean ± std over seeds.", ""]
    for thr_key, zs_key, title in (("test@0.5", "@0.5", "threshold 0.5"),
                                   ("test@val_recall90", "@val_recall90", "threshold = val video recall 0.9")):
        rows = [row("VLM zero-shot (Yes/No)", [flat(zs[zs_key])])]
        st = sorted((ROOT / "results/stgcn").glob(f"run{args.run}_T{args.T}_s*/metrics.json"))
        rows.append(row("ST-GCN (skeleton only)", [flat(json.loads(p.read_text())[thr_key]) for p in st]))
        for mode, name in (("text", "VLM text only (MiniLM + MLP)"), ("skel_text", "ST-GCN + VLM text"),
                           ("skel_text_pfall", "ST-GCN + VLM text + p_fall"),
                           ("hidden", "VLM hidden only (MLP)"), ("skel_hidden", "ST-GCN + VLM hidden"),
                           ("tokmean", "VLM tokens only (mean, MLP)"),
                           ("skel_tokmean", "ST-GCN + VLM tokens (mean, concat)"),
                           ("skel_xattn", "ST-GCN x VLM tokens (cross-attn)")):
            ps = sorted((ROOT / "results/fusion").glob(f"{mode}_run{args.run}_T{args.T}_s*/metrics.json"))
            if ps:
                rows.append(row(name, [flat(json.loads(p.read_text())[thr_key]) for p in ps]))
        df = pd.DataFrame(rows, columns=["model", "seeds"] + HEADER)
        md += [f"## {title}", "", df.to_markdown(index=False), ""]
    out = ROOT / "results/fusion_summary.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
