"""Skeleton window -> ST-GCN input (3, T, 17). numpy only. Training (build_clips.py) and inference (demo) call the same function.

Input (one person, one window):
  xy   (T, 17, 2)  coords. Run A: 0-1 normalized (xn, yn), Run B: pixels
  conf (T, 17)     keypoint confidence. 0 on missed frames
  box  (T, 4)      xyxy pixels. NaN on missed frames
Output channels = (x, y, mask). Joints with conf < CONF_THR get coords 0, mask 0.
"""
import numpy as np

CONF_THR = 0.3
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _pair_center(xy, valid, a, b):
    """Mean of the valid joints among the two. NaN if both invalid. -> (T, 2)"""
    w = np.stack([valid[:, a], valid[:, b]], 1).astype(np.float32)  # (T, 2)
    pts = np.stack([xy[:, a], xy[:, b]], 1)                         # (T, 2, 2)
    n = w.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (pts * w[..., None]).sum(1) / n
    c[n[:, 0] == 0] = np.nan
    return c


def normalize_window(xy, conf, box):
    """Run B: hip center as origin, scaled by window-mean torso length.
    Frames with both hips invalid use box center. If torso is never measured, scale = half box height."""
    xy = xy.astype(np.float32)
    valid = conf >= CONF_THR

    hip = _pair_center(xy, valid, L_HIP, R_HIP)
    box_c = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    center = np.where(np.isnan(hip), box_c, hip)

    shoulder = _pair_center(xy, valid, L_SHOULDER, R_SHOULDER)
    torso = np.linalg.norm(shoulder - hip, axis=1)
    torso = torso[np.isfinite(torso) & (torso > 1e-6)]
    if len(torso):
        scale = float(torso.mean())
    else:
        h = (box[:, 3] - box[:, 1])
        h = h[np.isfinite(h) & (h > 1e-6)]
        scale = float(h.mean() / 2) if len(h) else 1.0

    out = (xy - center[:, None, :]) / scale
    return _to_tensor(out, valid)


def raw_window(xyn, conf):
    """Run A: 0-1 normalized image coords as is."""
    return _to_tensor(xyn.astype(np.float32), conf >= CONF_THR)


def _to_tensor(xy, valid):
    xy = np.where(valid[..., None] & np.isfinite(xy), xy, 0.0)
    mask = valid.astype(np.float32)
    return np.concatenate([xy, mask[..., None]], -1).transpose(2, 0, 1).astype(np.float32)  # (3, T, 17)


def preprocess_window(run, xy_px, xyn, conf, box):
    """run: 'A' | 'B'. Shared entry point for training/inference."""
    if run == "A":
        return raw_window(xyn, conf)
    if run == "B":
        return normalize_window(xy_px, conf, box)
    raise ValueError(run)
