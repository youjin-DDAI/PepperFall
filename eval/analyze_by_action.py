"""Split external-dataset clips by OmniFall action label and compare the rate each model predicts Fall.

  python eval/analyze_by_action.py
Clip action = label of the OmniFall segment that overlaps the clip's time span the most. Threshold chosen on FallVision val.
A model that sees the event should be high only on "fall" and low on "lie_down / lying / fallen".
Output: results/action_analysis.md, results/action_clip_scores.csv (per-clip scores, for error analysis)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.preprocess.build_clips import TARGET_FPS, windows  # noqa: E402
from data.preprocess.build_external import DATASETS  # noqa: E402
from eval.eval_external import predict, zero_shot_threshold  # noqa: E402
from models.fusion import build_model  # noqa: E402
from models.stgcn import STGCN  # noqa: E402

T = 16
LABELS = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting", 5: "lie_down", 6: "lying",
          7: "stand_up", 8: "standing", 9: "other"}
MODELS = [("VLM zero-shot", None, None), ("ST-GCN Run A", "stgcn/runA_T16_s*", None),
          ("VLM hidden MLP", "fusion/hidden_runA_T16_s*", "hidden"),
          ("ST-GCN + VLM tokens", "fusion/skel_tokmean_runA_T16_s*", "skel_tokmean")]


def clip_actions(name):
    """Per clip (video_id, clip_idx, dominant action label)."""
    base = ROOT / "data/processed" / f"ext_{name}"
    meta = pd.read_csv(base / "meta.csv")
    lab = pd.read_csv(ROOT / "data/raw/omnifall_labels" / DATASETS[name]["labels"])
    rows = []
    for _, r in meta.iterrows():
        seg = lab[lab.path == r.of_path]
        n = max(1, int(np.floor(r.n_frames / r.fps * TARGET_FPS)))
        for k, w in enumerate(windows(n, T)):
            t0, t1 = w[0] / TARGET_FPS, (min(w[-1], n - 1) + 1) / TARGET_FPS
            ov = (np.minimum(seg.end, t1) - np.maximum(seg.start, t0)).clip(lower=0)
            rows.append({"video_id": r.video_path, "clip_idx": k,
                         "action": LABELS[int(seg.label.iloc[ov.argmax()])] if ov.max() > 0 else "none"})
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/fusion.yaml"))
    dev = "cuda:1"
    zs_thr = zero_shot_threshold()
    all_rows = []
    for name in ("GMDCSA24", "CAUCAFall"):
        base = ROOT / "data/processed" / f"ext_{name}"
        d = dict(np.load(base / f"runA_T{T}/test.npz"))
        X = torch.from_numpy(d["X"]).to(dev)
        tok = np.load(base / f"vlm_T{T}/test_tokens.npy")
        emb = {"hidden": torch.from_numpy(tok[:, -1].astype(np.float32)).to(dev),
               "tokmean": torch.from_numpy(np.load(base / f"vlm_T{T}/test_tokmean.npy").astype(np.float32)).to(dev)}
        act = clip_actions(name)
        assert (act.video_id.to_numpy() == d["video_id"]).all() and (act.clip_idx.to_numpy() == d["clip_idx"]).all()
        df = act.assign(dataset=name, y=d["y"], y_seg=d["y_seg"])
        df["VLM zero-shot"] = np.load(base / f"vlm_T{T}/test_pfall.npy")
        df["VLM zero-shot_pred"] = df["VLM zero-shot"] >= zs_thr
        for label, pattern, mode in MODELS[1:]:
            probs, preds = [], []
            for run_dir in sorted((ROOT / "results").glob(pattern)):
                m = json.loads((run_dir / "metrics.json").read_text())
                model = STGCN(dropout=0.1) if mode is None else build_model(mode, cfg, T)
                model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
                p = predict(model.to(dev), X, None if mode is None else emb[model.vlm_feat])
                probs.append(p)
                preds.append(p >= m["val_thr_recall90"])
            df[label] = np.mean(probs, 0)
            df[f"{label}_pred"] = np.mean(preds, 0)  # fraction of 3 seeds predicting Fall
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(ROOT / "results/action_clip_scores.csv", index=False)

    order = [a for a in ["walk", "standing", "sit_down", "sitting", "lie_down", "lying", "stand_up", "other",
                         "fall", "fallen"] if a in set(df.action)]
    md = ["# Fall rate per action (external dataset clips, FallVision val threshold)", "",
          "Value = fraction of that action's clips predicted Fall (trained models: mean of 3 seeds). Ideally high only for fall.", ""]
    for name in ("GMDCSA24", "CAUCAFall", None):
        sub = df if name is None else df[df.dataset == name]
        t = sub.groupby("action").agg(n=("y", "size"), **{m[0]: (f"{m[0]}_pred", "mean") for m in MODELS})
        t = t.reindex([a for a in order if a in t.index])
        md += [f"## {name or 'Both datasets combined'}", "", t.round(2).to_markdown(), ""]
    (ROOT / "results/action_analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
