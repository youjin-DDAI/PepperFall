"""Streaming fall detector for a laptop (Apple Silicon: device "mps") from a trained checkpoint folder.

Same preprocessing as training: 10 fps frames, 16-frame windows every 8 frames (1.6 s, stride 0.8 s),
YOLO11-pose, one followed person (nearest box frame to frame), Run A / B skeleton normalization (data/preprocess/normalize.py),
VLM tokmean + p(Fall) from 4 frames of the window (data/preprocess/vlm_tokens.py).
Checkpoint folder = results/event/{mode}[_{vlm tag}]/{run name}/ with best.pt and metrics.json.
"""
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.preprocess.normalize import preprocess_window  # noqa: E402
from models.fusion import build_model  # noqa: E402
from models.stgcn import STGCN  # noqa: E402

T, STRIDE, N_FRAMES = 16, 8, 4
VLM_PICK = np.linspace(0, T - 1, N_FRAMES).round().astype(int)  # same frames as vlm_captions.clip_frames
CLIP_FPS = N_FRAMES / (T / 10)
ACTIONS = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other",
           "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump"]
FALL, FALLEN = 1, 2


def vlm_image(bgr, short_side=360):
    """Same resize as vlm_captions.read_frames: short side 360, both sides multiples of 28."""
    h, w = bgr.shape[:2]
    s = short_side / min(h, w)
    f = cv2.resize(bgr, (int(w * s) // 28 * 28, int(h * s) // 28 * 28), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


class FallDetector:
    def __init__(self, ckpt, pose="yolo11x-pose.pt", vlm_model="Qwen/Qwen2.5-VL-3B-Instruct", device="mps"):
        ckpt = Path(ckpt)
        meta = json.loads((ckpt / "metrics.json").read_text())
        self.mode, self.run, self.multi = meta["mode"], meta["run"], meta.get("label") == "multiclass"
        self.dev = device
        sd = torch.load(ckpt / "best.pt", map_location="cpu", weights_only=True)
        n_out = len(ACTIONS) if self.multi else 1
        if self.mode == "stgcn":
            self.model = STGCN(num_outputs=n_out)
        else:
            cfg = yaml.safe_load(open(ROOT / "configs/fusion.yaml"))
            vlm_dim = sd["text_proj.0.weight"].shape[0]  # LayerNorm over the VLM feature
            self.model = build_model(self.mode, cfg, T, n_out, vlm_dim)
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.use_skel = self.mode == "stgcn" or getattr(self.model, "use_skel", False)
        self.feat = None if self.mode == "stgcn" else "hidden" if "hidden" in self.mode else "tokmean"

        from ultralytics import YOLO
        self.pose = YOLO(pose)
        self.vlm = None
        if self.feat:
            from data.preprocess.vlm_captions import VLM
            self.vlm = VLM(vlm_model, device)
            d = 2 * self.vlm.model.config.get_text_config().hidden_size
            want = sd["text_proj.0.weight"].shape[0]
            assert want in (d, d // 2), f"checkpoint expects VLM dim {want}, {vlm_model} gives {d}: wrong --vlm-model"
        self.buf = deque(maxlen=T)
        self.n = 0
        self.last_box = None

    def _detect(self, bgr):
        """All people in one frame: [(xy (17,2) px, xyn (17,2), conf (17,), box (4,))]. No tracker: at 10 fps
        ByteTrack drops or re-ids a person during the fast fall itself; _follow keeps the identity instead."""
        r = self.pose.predict(bgr, conf=0.25, imgsz=640, verbose=False,
                              device=self.dev if str(self.pose.ckpt_path or "").endswith(".pt") else None)[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return []
        kp, b = r.keypoints, r.boxes.xyxy.cpu().numpy()
        return [(kp.xy[i].cpu().numpy(), kp.xyn[i].cpu().numpy(), kp.conf[i].cpu().numpy(), b[i]) for i in range(len(b))]

    def _follow(self, people):
        """One person per frame: the detection nearest to the last chosen box (the largest person at start)."""
        if not people:
            return None
        if self.last_box is None:
            p = max(people, key=lambda d: (d[3][2] - d[3][0]) * (d[3][3] - d[3][1]))
        else:
            lc = (self.last_box[:2] + self.last_box[2:]) / 2
            p = min(people, key=lambda d: np.linalg.norm((d[3][:2] + d[3][2:]) / 2 - lc))
        self.last_box = p[3]
        return p

    def _window_skeleton(self):
        xy, xyn = np.zeros((T, 17, 2), np.float32), np.zeros((T, 17, 2), np.float32)
        conf, box = np.zeros((T, 17), np.float32), np.full((T, 4), np.nan, np.float32)
        for t, f in enumerate(self.buf):
            if f["person"] is not None:
                xy[t], xyn[t], conf[t], box[t] = f["person"]
        return preprocess_window(self.run, xy, xyn, conf, box)

    @torch.no_grad()
    def _score(self):
        X = self._window_skeleton()
        x = torch.from_numpy(X[None]).to(self.dev)
        if self.mode == "stgcn":
            logits, pfall = self.model(x), None
        else:
            from data.preprocess.vlm_tokens import to_mean, tokens
            tok, p = tokens(self.vlm, [[self.buf[i]["img"] for i in VLM_PICK]], CLIP_FPS)
            emb = to_mean(tok).astype(np.float32)
            if self.feat == "hidden":
                emb = emb[:, emb.shape[1] // 2:]
            pfall = float(p[0])
            logits = self.model(x, torch.from_numpy(emb).to(self.dev), torch.from_numpy(p.astype(np.float32)).to(self.dev))
        logits = logits.float().cpu()
        if self.multi:
            prob = logits.softmax(-1)[0].numpy()
            res = {"p_fall": float(prob[FALL]), "p_alarm": float(prob[FALL] + prob[FALLEN]),
                   "action": ACTIONS[int(prob.argmax())]}
        else:
            p = float(torch.sigmoid(logits.reshape(-1))[0])
            res = {"p_fall": p, "p_alarm": p}
        res["vlm_p_fall"] = pfall
        return res

    def step(self, bgr):
        """Feed one 10 fps frame. Returns a result dict every STRIDE frames once T frames are buffered, else None."""
        self.buf.append({"person": self._follow(self._detect(bgr)), "img": vlm_image(bgr) if self.vlm else None})
        self.n += 1
        if len(self.buf) < T or (self.n - T) % STRIDE:
            return None
        return self._score()
