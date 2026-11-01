"""Skeleton (ST-GCN) + VLM (MiniLM text embedding or Qwen hidden state) late-fusion MLP.

mode
  text            : text embedding only -> MLP
  skel_text       : ST-GCN feature (256) + text projection (128) -> MLP
  skel_text_pfall : above + VLM Yes/No logit (scalar)
  hidden          : Qwen last-layer hidden only (3584, token right before Yes/No answer) -> LayerNorm -> MLP
  skel_hidden     : ST-GCN feature + Qwen hidden projection (128) -> MLP. No text generation / MiniLM
  tokmean         : [mean of 72 Qwen patch tokens, global] (7168) only -> MLP
  skel_tokmean    : ST-GCN feature + [mean of 72 Qwen patch tokens, global] (7168) -> MLP. concat control with same info as skel_xattn
  skel_xattn      : CrossAttnFusion. ST-GCN (time x joint) tokens as query, Qwen patch tokens as key/value
  skel_xattn_sm   : skeleton-main variant. drops global (pre-answer) token + VLM attention dropout 0.5 -> skeleton path can decide alone
ST-GCN trained from scratch with the fusion head. VLM / MiniLM frozen (embeddings precomputed).
Modality dropout: in training, per sample with prob p, zero the entire skeleton or VLM input (never both).
"""
import torch
import torch.nn as nn

from models.stgcn import STGCN

MODES = ("text", "skel_text", "skel_text_pfall", "hidden", "skel_hidden", "tokmean", "skel_tokmean", "skel_xattn",
         "skel_xattn_sm")
VLM_DIM = {"text": 384, "hidden": 3584, "tokmean": 7168}  # MiniLM / Qwen2.5-VL-7B hidden / [token mean, global]
QWEN_DIM = 3584


class FusionNet(nn.Module):
    def __init__(self, mode, text_hidden=128, hidden=128, dropout=0.3, stgcn_dropout=0.1,
                 modality_drop=0.15, n_out=1, vlm_dim=None):
        super().__init__()
        assert mode in MODES, mode
        self.mode, self.modality_drop = mode, modality_drop
        self.use_skel, self.use_pfall = mode not in ("text", "hidden", "tokmean"), mode.endswith("pfall")
        self.vlm_feat = "hidden" if "hidden" in mode else "tokmean" if "tokmean" in mode else "text"
        self.stgcn = STGCN(dropout=stgcn_dropout) if self.use_skel else None
        dim = vlm_dim or VLM_DIM[self.vlm_feat]
        norm = [nn.LayerNorm(dim)] if self.vlm_feat != "text" else []  # normalize: Qwen hidden has a large value range
        self.text_proj = nn.Sequential(*norm, nn.Linear(dim, text_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        d = text_hidden + (self.stgcn.feat_dim if self.use_skel else 0) + int(self.use_pfall)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))  # n_out > 1: multiclass (OmniFall 16 actions)

    def forward(self, x, emb, pfall):
        """x (N,3,T,V) skeleton, emb (N,384) text embedding or (N,3584) Qwen hidden, pfall (N,) VLM p(Fall)."""
        vlm = [self.text_proj(emb)]
        if self.use_pfall:
            vlm.append(torch.logit(pfall.clamp(1e-4, 1 - 1e-4)).unsqueeze(1) / 4)
        vlm = torch.cat(vlm, 1)
        if not self.use_skel:
            return self.head(vlm).squeeze(-1)
        skel = self.stgcn.features(x)
        if self.training and self.modality_drop > 0:
            r = torch.rand(len(x), 1, device=x.device)
            skel = skel * (r >= self.modality_drop)
            vlm = vlm * ((r < self.modality_drop) | (r >= 2 * self.modality_drop))
        return self.head(torch.cat([skel, vlm], 1)).squeeze(-1)


class CrossAttnFusion(nn.Module):
    """ST-GCN attends to VLM patch tokens via cross-attention.

    query  : ST-GCN feature map (256, T', V) -> T'*V tokens + joint position (window-mean x, y, valid) + time embedding
    key/val: Qwen patch tokens (N_TIME * GRID^2) -> LayerNorm -> Linear(3584 -> dim) + grid-cell center position + time-bin embedding,
             final global token (token right before the answer) gets type embedding only
    Joint coords (Run A normalized image coords) share the grid-center coordinate frame, so joints can learn to attend to nearby patches (bed / floor).
    In training, per sample with prob vlm_drop, zero the entire attention output (works on skeleton alone) + skeleton aux head loss.
    """

    def __init__(self, n_time=2, grid=6, T=16, dim=256, heads=4, layers=2, dropout=0.1, stgcn_dropout=0.1,
                 vlm_drop=0.15, use_global=True):
        super().__init__()
        self.vlm_drop, self.n_time, self.grid, self.use_global = vlm_drop, n_time, grid, use_global
        self.use_skel, self.use_pfall, self.vlm_feat = True, False, "tokens"
        self.mode = "skel_xattn" if use_global else "skel_xattn_sm"
        self.stgcn = STGCN(dropout=stgcn_dropout)
        self.t_out = T // 4
        self.q_pos = nn.Sequential(nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.q_time = nn.Parameter(torch.zeros(self.t_out, 1, dim))
        self.kv_proj = nn.Sequential(nn.LayerNorm(QWEN_DIM), nn.Linear(QWEN_DIM, dim))
        self.k_pos = nn.Sequential(nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.k_time = nn.Parameter(torch.zeros(n_time, 1, dim))
        self.k_global = nn.Parameter(torch.zeros(1, dim))
        c = (torch.arange(grid) + 0.5) / grid
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("cell_xy", torch.stack([cx, cy], -1).view(1, grid * grid, 2))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                                   for _ in range(layers)])
        self.norm_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.norm_f = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                                nn.Linear(2 * dim, dim)) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.aux_head = nn.Linear(self.stgcn.feat_dim, 1)
        self.last_attn = None

    def queries(self, x):
        n, _, t, v = x.shape
        fm = self.stgcn.feature_map(x)  # (N, C, T', V)
        seg = t // fm.shape[2]
        m = x[:, 2:3].view(n, 1, fm.shape[2], seg, v)
        xy = (x[:, :2].view(n, 2, fm.shape[2], seg, v) * m).sum(3) / m.sum(3).clamp(min=1)
        pos = torch.cat([xy, m.amax(3)], 1).permute(0, 2, 3, 1)  # (N, T', V, 3)
        q = fm.permute(0, 2, 3, 1) + self.q_pos(pos) + self.q_time  # (N, T', V, C)
        return q.reshape(n, -1, q.shape[-1]), fm

    def keys(self, tok):
        n = tok.shape[0]
        kv = self.kv_proj(tok.float())
        g2 = self.grid * self.grid
        patch = kv[:, :-1].view(n, self.n_time, g2, -1) + self.k_pos(self.cell_xy).unsqueeze(1) + self.k_time
        if not self.use_global:  # skip the pre-answer token holding the VLM Yes/No decision; attend to scene patches only
            return patch.reshape(n, self.n_time * g2, -1)
        return torch.cat([patch.reshape(n, self.n_time * g2, -1), kv[:, -1:] + self.k_global], 1)

    def forward(self, x, tok, pfall=None, keep_attn=False):
        q, fm = self.queries(x)
        kv = self.keys(tok)
        keep = torch.ones(len(x), 1, 1, device=x.device)
        if self.training and self.vlm_drop > 0:
            keep = (torch.rand(len(x), 1, 1, device=x.device) >= self.vlm_drop).float()
        attn_maps = []
        for attn, nq, nf, ffn in zip(self.attn, self.norm_q, self.norm_f, self.ffn):
            a, w = attn(nq(q), kv, kv, need_weights=keep_attn)
            q = q + a * keep
            q = q + ffn(nf(q))
            attn_maps.append(w)
        if keep_attn:
            self.last_attn = attn_maps  # each (N, T'*V, n_kv), head mean
        logit = self.head(q.mean(1)).squeeze(-1)
        if self.training:
            return logit, self.aux_head(fm.mean(dim=(2, 3))).squeeze(-1)
        return logit


def build_model(mode, cfg, T, n_out=1, vlm_dim=None):
    assert n_out == 1 or not mode.startswith("skel_xattn"), "cross-attention supports binary output only"
    if mode == "skel_xattn":
        return CrossAttnFusion(T=T, stgcn_dropout=cfg["dropout"], **cfg["xattn"])
    if mode == "skel_xattn_sm":
        return CrossAttnFusion(T=T, stgcn_dropout=cfg["dropout"], **{**cfg["xattn"], "vlm_drop": 0.5}, use_global=False)
    return FusionNet(mode, stgcn_dropout=cfg["dropout"], n_out=n_out, vlm_dim=vlm_dim, **cfg["fusion"])
