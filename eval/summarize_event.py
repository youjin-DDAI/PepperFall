"""Event-label experiment comparison: re-evaluate every checkpoint with the same eval (train_event.evaluate_all).

  python eval/summarize_event.py
Targets: ST-GCN results/stgcn/run{A,B}_T16_s* (FallVision video labels, baseline) + results/event/*_T16_s* (event labels),
      VLM / fusion results/fusion/{mode}_runA_T16_s* (video labels) + results/event/{mode}/*_T16_s* (event labels),
      VLM zero-shot (no training).
Threshold (for per-action Fall rate) is OF-Syn val clip recall 0.9 (no filter), shared by all models.
Output: results/event_summary.md, results/event_all_seeds.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import threshold_for_recall  # noqa: E402
from train.train_event import ACTIONS, SCORES, AlarmScore, evaluate_all, feat_of, load, make_model, predict  # noqa: E402

DEV = "cuda:0"
HARD = ["sit_down", "lie_down", "stand_up", "kneel_down", "kneeling", "squat_down", "lying", "fallen"]


VLM_MODES = ("hidden", "tokmean", "skel_tokmean", "skel_xattn", "skel_xattn_sm")
NAME = {"stgcn": "ST-GCN", "hidden": "VLM hidden MLP", "tokmean": "VLM tokmean MLP",
        "skel_tokmean": "ST-GCN + VLM tokmean", "skel_xattn": "ST-GCN x VLM cross-attn",
        "skel_xattn_sm": "ST-GCN x VLM cross-attn (skeleton-main)", "zeroshot": "VLM zero-shot"}
FV = "FallVision (video label)"


def runs():
    """(mode, training data name, checkpoint folder or None)"""
    yield "zeroshot", "— (no training)", None
    for d in sorted((ROOT / "results/stgcn").glob("run[AB]_T16_s*")):
        yield "stgcn", FV, d
    for mode in VLM_MODES:
        for d in sorted((ROOT / "results/fusion").glob(f"{mode}_runA_T16_s*")):
            yield mode, FV, d
    for mode, sub in [("stgcn", "")] + [(m, m) for m in VLM_MODES]:
        for d in sorted((ROOT / "results/event" / sub).glob("*_T16_s*")):
            if not d.name.startswith("baseline_") and (d / "metrics.json").exists():
                yield mode, re.sub(r"_run[AB]_T16_s\d+$", "", d.name), d


def main():
    cfgs = {m: yaml.safe_load(open(ROOT / f"configs/{'stgcn' if m == 'stgcn' else 'fusion'}.yaml")) for m in NAME}
    rows = []
    for mode, train, d in runs():
        m = json.loads((d / "metrics.json").read_text()) if d else {"run": "A", "seed": 0}
        run, feat = m["run"], feat_of(mode)
        multi = m.get("label") == "multiclass"
        model = make_model(mode, 16, cfgs[mode], len(ACTIONS) if multi else 1)
        if d:
            model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=True))
        model.to(DEV)
        va = load("OF-Syn", "val", run, 16, feat)
        keep = va["y_event"] >= 0
        # multiclass models: one row per alarm score definition: p(fall), p(fall)+p(fallen)
        for sname, scorer in ([(f" [{k}]", AlarmScore(model, c)) for k, c in SCORES.items()] if multi else [("", model)]):
            thr = threshold_for_recall(va["y_event"][keep], predict(scorer, va, DEV)[keep], 0.9)
            ev = evaluate_all(scorer, run, 16, DEV, thr, feat)
            r = {"model": NAME[mode], "train": train + sname, "run": run if mode.startswith(("stgcn", "skel")) else "—",
                 "seed": m["seed"]}
            for ds, v in ev.items():
                n = ds.split("/")[0]
                r.update({f"{n} clip": v["clip_auroc"], f"{n} video": v["video_auroc"],
                          f"{n} clip static": v["static_clip_auroc"], f"{n} video static": v["static_video_auroc"],
                          f"{n} clip static skel": v.get("static_skel_clip_auroc", np.nan),
                          f"{n} alarm clip": v.get("alarm_clip_auroc", np.nan)})
                for e, ve in v.get("by_elevation", {}).items():
                    r[f"OF-Syn {e}"] = ve["clip_auroc"]
                if "fall_rate_by_action" in v and n in ("GMDCSA24", "CAUCAFall"):
                    for a, fr in v["fall_rate_by_action"].items():
                        r[f"{n} rate {a}"] = fr
            rows.append(r)
        print(mode, train, run, m["seed"], "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/event_all_seeds.csv", index=False)

    def table(cols, title):
        g = df.groupby(["model", "train", "run"], sort=False)[cols]
        mean, std = g.mean(), g.std().fillna(0)
        t = pd.DataFrame({c: [f"{a:.3f} ± {b:.3f}" if np.isfinite(a) else "—" for a, b in zip(mean[c], std[c])]
                          for c in cols}, index=mean.index)
        return [f"## {title}", "", t.to_markdown(), ""]

    sets = ["OF-Syn", "GMDCSA24", "CAUCAFall", "FallVision"]
    md = ["# Event-label vs video-label training (all models, T=16, mean ± std over seeds)", "",
          "clip = clip AUROC (overlaps a fall segment; video label for FallVision), video = video AUROC (clip max).",
          "static = motion removed: skeleton repeats the last frame 16 times, VLM features from 4 copies of the last frame.",
          "static skel = only skeleton frozen, VLM unchanged (does fusion use skeleton motion?).",
          "The more a model uses motion, the more it drops under static. run '—' = model without skeleton.", ""]
    md += table([f"{s} clip" for s in sets], "Clip AUROC")
    md += table([f"{s} video" for s in sets], "Video AUROC")
    md += table([f"{s} alarm clip" for s in sets[:3]],
                "Alarm definition B (fall or fallen clip = 1) clip AUROC — must also catch lying on the floor after a fall")
    md += table([c for s in sets for c in (f"{s} clip", f"{s} clip static")], "Clip AUROC before / after motion removal")
    md += table([c for s in sets for c in (f"{s} clip", f"{s} clip static skel")],
                "Clip AUROC with skeleton-only motion removal (VLM unchanged) — fusion only")
    md += table([f"OF-Syn {e}" for e in ("top", "high", "eye", "low")], "OF-Syn test clip AUROC by camera elevation")
    rate_cols = [c for c in df.columns if " rate " in c and c.split(" rate ")[1] in HARD + ["fall", "walk"]]
    md += table(sorted(rate_cols), "External datasets: Fall rate per action (threshold = OF-Syn val recall 0.9)")
    (ROOT / "results/event_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
