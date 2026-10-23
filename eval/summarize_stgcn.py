"""results/stgcn/*/metrics.json -> results/stgcn_summary.md (mean ± std over seeds)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["acc", "precision", "recall", "f1", "auroc"]


def main():
    rows = []
    for p in sorted((ROOT / "results/stgcn").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        for thr_key in ("test@0.5", "test@val_recall90"):
            for level in ("clip", "video"):
                r = {"run": m["run"], "T": m["T"], "seed": m["seed"], "thr": thr_key.split("@")[1], "level": level}
                r.update({k: m[thr_key][level][k] for k in METRICS})
                rows.append(r)
            for scen, sm in m[thr_key]["video_by_scenario"].items():
                r = {"run": m["run"], "T": m["T"], "seed": m["seed"], "thr": thr_key.split("@")[1],
                     "level": f"video/{scen}"}
                r.update({k: sm[k] for k in METRICS})
                rows.append(r)
    df = pd.DataFrame(rows)
    g = df.groupby(["thr", "level", "T", "run"])[METRICS]
    mean, std, n = g.mean(), g.std().fillna(0), g.size()
    table = mean.copy().astype(object)
    for c in METRICS:
        table[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[c], std[c])]
    table["seeds"] = n
    md = ["# ST-GCN results (test set, mean ± std over seeds)", "",
          "thr=0.5: fixed threshold / thr=val_recall90: threshold chosen for video-level val recall>=0.9", "",
          table.reset_index().to_markdown(index=False), ""]
    out = ROOT / "results/stgcn_summary.md"
    out.write_text("\n".join(md))
    df.to_csv(ROOT / "results/stgcn_all_seeds.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
