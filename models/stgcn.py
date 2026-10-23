"""ST-GCN (Yan et al., AAAI 2018) for COCO-17 skeletons. Pure torch only (for porting to the demo notebook).

Input (N, C=3, T, V=17): (x, y, mask). Output: logit (N,), or (logit, feat (N, 256)) if forward(..., return_feat=True).
vs. original paper: 10 -> 6 blocks, COCO-17 graph, same spatial partition / edge importance.
"""
import numpy as np
import torch
import torch.nn as nn

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (0, 5), (0, 6),                            # head - shoulders
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def spatial_graph(num_node=17, edges=COCO_EDGES, center=0, max_hop=1):
    """Original paper's 'spatial' partition: root / centripetal / centrifugal -> A (K, V, V)."""
    adj = np.zeros((num_node, num_node))
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    hop = np.full((num_node, num_node), np.inf)
    reach = [np.linalg.matrix_power(adj + np.eye(num_node), d) for d in range(max_hop + 1)]
    for d in range(max_hop, -1, -1):
        hop[reach[d] > 0] = d
    hop_to_center = _hop_distance(adj, center)

    valid = (hop <= max_hop).astype(float)
    deg = valid.sum(0)
    norm = valid / deg[None, :]  # column-normalized (normalize_digraph)

    A = []
    for d in range(max_hop + 1):
        root, close, far = (np.zeros((num_node, num_node)) for _ in range(3))
        for i in range(num_node):
            for j in range(num_node):
                if hop[j, i] != d:
                    continue
                if hop_to_center[j] == hop_to_center[i]:
                    root[j, i] = norm[j, i]
                elif hop_to_center[j] > hop_to_center[i]:
                    close[j, i] = norm[j, i]
                else:
                    far[j, i] = norm[j, i]
        if d == 0:
            A.append(root)
        else:
            A.append(root + close)
            A.append(far)
    return np.stack(A).astype(np.float32)


def _hop_distance(adj, src):
    dist = np.full(len(adj), np.inf)
    dist[src], frontier = 0, [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.where(adj[u] > 0)[0]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, K, kt=9, stride=1, dropout=0.0, residual=True):
        super().__init__()
        self.K = K
        self.gcn = nn.Conv2d(in_ch, out_ch * K, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
            nn.BatchNorm2d(out_ch), nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.res = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, (stride, 1)), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.res(x)
        n, _, t, v = x.shape
        y = self.gcn(x).view(n, self.K, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, A)
        return self.relu(self.tcn(y) + res)


class STGCN(nn.Module):
    # (out_channels, temporal stride)
    DEFAULT_LAYERS = [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1)]

    def __init__(self, in_channels=3, num_outputs=1, layers=None, kt=9, dropout=0.0, edge_importance=True):
        super().__init__()
        A = torch.from_numpy(spatial_graph())
        self.register_buffer("A", A)
        K, V = A.shape[0], A.shape[1]
        layers = layers or self.DEFAULT_LAYERS

        self.data_bn = nn.BatchNorm1d(in_channels * V)
        blocks, c = [], in_channels
        for i, (out_c, s) in enumerate(layers):
            blocks.append(STGCNBlock(c, out_c, K, kt, s, 0.0 if i == 0 else dropout, residual=i > 0))
            c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones_like(A)) for _ in blocks] if edge_importance else [])
        self.feat_dim = c
        self.head = nn.Linear(c, num_outputs)

    def feature_map(self, x):
        """Feature before pooling (N, 256, T', V). T' = T / 4 (two stride-2 blocks)."""
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(n, v * c, t))
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for i, blk in enumerate(self.blocks):
            A = self.A * self.edge_importance[i] if len(self.edge_importance) else self.A
            x = blk(x, A)
        return x

    def features(self, x):
        return self.feature_map(x).mean(dim=(2, 3))  # global average pool over (T, V)

    def forward(self, x, return_feat=False):
        feat = self.features(x)
        out = self.head(feat)
        out = out.squeeze(-1) if out.shape[-1] == 1 else out
        return (out, feat) if return_feat else out
