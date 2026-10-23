"""Fall / No Fall binary classification metrics. Clip level + video level (max clip prob within video)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auroc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "thr": float(thr),
    }


def video_level(video_id, y, prob):
    """Video label equals clip label. Video score = max clip prob."""
    vids, inv = np.unique(video_id, return_inverse=True)
    vprob = np.full(len(vids), -np.inf)
    np.maximum.at(vprob, inv, prob)
    vy = np.zeros(len(vids), int)
    vy[inv] = y
    return vids, vy, vprob


def threshold_for_recall(y, prob, target=0.9):
    """Highest threshold with recall >= target (preserves precision as much as possible)."""
    pos = np.sort(prob[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[min(k, len(pos) - 1)])


def by_group(groups, y, prob, thr):
    return {g: binary_metrics(y[groups == g], prob[groups == g], thr) for g in np.unique(groups)}
