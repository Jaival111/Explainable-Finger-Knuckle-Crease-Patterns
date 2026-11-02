"""Steps 3-4 - Similarity of Correspondence Graph Network, CGN (Sec. 3.3).

Given the top-c correspondences (sorted by matching score, Eq. 3-5):
  * Graph embedding: k-NN edges on the probe keypoints K^p'; the same edges are
    re-used for the gallery graph (isomorphic graphs, Eq. 6).
  * Positional encoding of the correspondence rank (Eq. 7): h = f + e.
  * Self-Graph: convolution along the node dimension (Eq. 10-12), shared
    (Siamese) between probe and gallery.
  * Cross-Node: arccos of the cosine similarity of each corresponding node pair
    on every layer (Eq. 13); the Tracker concatenates them (Eq. 14) into the
    node features of the tracked graph G^pg (same edges).
  * Graph similarity: l node-dimension convolutions on G^pg, sum readout per
    layer (Eq. 15), concatenation (Eq. 16) and MLP + sigmoid (Eq. 17).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(points, k):
    """points: (B, c, 2) -> neighbour indices (B, c, k), ordered by distance."""
    c = points.shape[1]
    assert c > k, f"need more correspondences ({c}) than neighbours ({k})"
    d = torch.cdist(points, points)
    eye = torch.eye(c, dtype=torch.bool, device=points.device)
    d = d.masked_fill(eye, float("inf"))
    return d.topk(k, dim=-1, largest=False).indices


def positional_encoding(c, d, device):
    """Sinusoidal encoding of the correspondence rank i' (Eq. 7) -> (c, d)."""
    pos = torch.arange(c, device=device, dtype=torch.float32)[:, None]
    two_alpha = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.pow(10000.0, two_alpha / d)
    pe = torch.zeros(c, d, device=device)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


class ConvNode(nn.Module):
    """Graph convolution along the node dimension (Eq. 10-12).

    A_i = [h_i^T | h_u1^T | ... | h_uk^T] in R^{din x (k+1)}
    h_i[d] = phi(bias_d + tr(W[d] A_i^T)),   W in R^{dout x din x (k+1)}
    """

    def __init__(self, din, dout, k, negative_slope=0.1):
        super().__init__()
        fan_in = din * (k + 1)
        w = torch.empty(dout, fan_in)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w.view(dout, din, k + 1))
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(dout).uniform_(-bound, bound))
        self.negative_slope = negative_slope

    def forward(self, h, nbr):
        """h: (B, c, din), nbr: (B, c, k) -> (B, c, dout)."""
        B, c, din = h.shape
        k = nbr.shape[-1]
        nb = torch.gather(h, 1, nbr.reshape(B, c * k, 1).expand(-1, -1, din)).view(B, c, k, din)
        A = torch.cat([h.unsqueeze(2), nb], dim=2)                       # (B, c, k+1, din)
        out = torch.einsum("bcjd,odj->bco", A, self.weight) + self.bias  # tr(W[d] A^T)
        return F.leaky_relu(out, self.negative_slope)


class CGN(nn.Module):
    def __init__(self, feature_dim=256, k=9, self_layers=3, sim_layers=3, sim_hidden=64,
                 head_hidden=128, use_pe=True, use_arccos=True):
        super().__init__()
        self.k = k
        self.use_pe = use_pe
        self.use_arccos = use_arccos
        # Descriptors are unit-norm while the raw sinusoidal code has norm sqrt(d/2);
        # a learnable scale (initialised to 1/sqrt(d)) keeps the two comparable.
        self.pe_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(feature_dim)))
        self.self_graph = nn.ModuleList([ConvNode(feature_dim, feature_dim, k) for _ in range(self_layers)])
        dims = [self_layers + 1] + [sim_hidden] * sim_layers
        self.sim_graph = nn.ModuleList([ConvNode(dims[i], dims[i + 1], k) for i in range(sim_layers)])
        self.head = nn.Sequential(
            nn.Linear(sim_hidden * sim_layers, head_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(head_hidden, 1))

    def cross_node(self, hp, hg):
        """One-to-one similarity between corresponding nodes (Eq. 13)."""
        cos = F.cosine_similarity(hp, hg, dim=-1, eps=1e-8)
        if self.use_arccos:
            return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        return cos

    def forward(self, desc_p, kpts_p, desc_g, return_details=False):
        """desc_p, desc_g: (B, c, D) features of corresponded keypoints sorted by
        correspondence score; kpts_p: (B, c, 2) probe locations.
        Returns logits (B,); s_G = sigmoid(logits)."""
        B, c, D = desc_p.shape
        nbr = knn_graph(kpts_p, self.k)                       # edges E^p (shared with E^g)

        hp, hg = desc_p, desc_g
        if self.use_pe:
            pe = positional_encoding(c, D, desc_p.device) * self.pe_scale
            hp, hg = hp + pe, hg + pe

        tracked = [self.cross_node(hp, hg)]                   # (0) h^pg
        for layer in self.self_graph:                         # Self-Graph
            hp, hg = layer(hp, nbr), layer(hg, nbr)
            tracked.append(self.cross_node(hp, hg))           # Cross-Node
        x = torch.stack(tracked, dim=-1)                      # Tracker, Eq. (14): (B, c, l+1)

        readouts = []
        for layer in self.sim_graph:                          # graph-level task on G^pg
            x = layer(x, nbr)
            readouts.append(x.sum(dim=1))                     # Eq. (15)
        h_graph = torch.cat(readouts, dim=-1)                 # Eq. (16)
        logits = self.head(h_graph).squeeze(-1)               # Eq. (17) before sigmoid

        if return_details:
            return logits, {"edges": nbr, "tracked": torch.stack(tracked, -1)}
        return logits
