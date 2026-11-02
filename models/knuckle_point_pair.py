"""Step 2 - KnucklePointPair (Sec. 3.2).

Self-/cross-attention GNN (SuperGlue-style) over the keypoints of the probe and
gallery images, followed by an optimal-transport partial assignment:

    D = F + MLP([K | S])                                        (Eq. 1)
    C = [[ (lD^p)(lD^g)^T , U1 ],                               (Eq. 2)
         [      U2        ,  u ]]

with learnable non-match scores U1 in R^{n x 1}, U2 in R^{1 x m} and u, solved
with t Sinkhorn iterations. Attribute names follow SuperGlue so its released
weights can optionally be used as initialisation.
"""
from copy import deepcopy

import torch
import torch.nn as nn


def MLP(channels, do_bn=True):
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < len(channels) - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, h, w):
    size = kpts.new_tensor([w, h])
    center = size / 2
    scaling = size.max() * 0.7
    return (kpts - center) / scaling


class KeypointEncoder(nn.Module):
    """MLP([K | S]) of Eq. (1)."""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        return self.encoder(torch.cat([kpts.transpose(1, 2), scores.unsqueeze(1)], dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in layer_names])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            src0, src1 = (desc1, desc0) if name == "cross" else (desc0, desc1)
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = desc0 + delta0, desc1 + delta1
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bins0, bins1, alpha, iters):
    """Differentiable partial assignment in log-space.

    scores: (B, n, m), bins0: (B, n, 1) = U1, bins1: (B, 1, m) = U2, alpha: (B, 1, 1) = u.
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = m * one, n * one
    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)
    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return Z - norm


class KnucklePointPair(nn.Module):
    def __init__(self, descriptor_dim=256, keypoint_encoder=(32, 64, 128, 256),
                 gnn_layers=("self", "cross") * 9, sinkhorn_iterations=100, max_keypoints=256):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        self.kenc = KeypointEncoder(descriptor_dim, list(keypoint_encoder))
        self.gnn = AttentionalGNN(descriptor_dim, list(gnn_layers))
        self.final_proj = nn.Conv1d(descriptor_dim, descriptor_dim, kernel_size=1, bias=True)
        # Learnable non-match scores of Eq. (2). Keypoints are sorted by detection
        # score, so U1[i] / U2[j] are tied to the keypoint rank.
        self.U1 = nn.Parameter(torch.ones(max_keypoints))
        self.U2 = nn.Parameter(torch.ones(max_keypoints))
        self.u = nn.Parameter(torch.tensor(1.0))

    def forward(self, data0, data1, image_size):
        """data*: dicts with keypoints (B,N,2), scores (B,N), descriptors (B,N,D).

        Returns log assignment matrix (B, n+1, m+1).
        """
        h, w = image_size
        desc0 = data0["descriptors"].transpose(1, 2)
        desc1 = data1["descriptors"].transpose(1, 2)
        kpts0 = normalize_keypoints(data0["keypoints"], h, w)
        kpts1 = normalize_keypoints(data1["keypoints"], h, w)

        desc0 = desc0 + self.kenc(kpts0, data0["scores"])          # Eq. (1)
        desc1 = desc1 + self.kenc(kpts1, data1["scores"])
        desc0, desc1 = self.gnn(desc0, desc1)                      # l attentional layers
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1) / self.descriptor_dim ** 0.5
        b, n, m = scores.shape
        bins0 = self.U1[:n].view(1, n, 1).expand(b, n, 1)
        bins1 = self.U2[:m].view(1, 1, m).expand(b, 1, m)
        alpha = self.u.view(1, 1, 1).expand(b, 1, 1)
        log_assign = log_optimal_transport(scores, bins0, bins1, alpha, self.sinkhorn_iterations)
        return {"log_assign": log_assign}

    def load_superglue_weights(self, path):
        """Initialise from a released SuperGlue checkpoint (bin_score -> U1, U2, u)."""
        state = torch.load(path, map_location="cpu")
        bin_score = state.pop("bin_score", None)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if bin_score is not None:
            with torch.no_grad():
                v = float(bin_score)
                self.U1.fill_(v)
                self.U2.fill_(v)
                self.u.fill_(v)
        return missing, unexpected


def mutual_matches(log_assign, threshold):
    """Mutual nearest-neighbour matches with confidence above `threshold`."""
    scores = log_assign[:, :-1, :-1]
    max0, max1 = scores.max(2), scores.max(1)
    indices0, indices1 = max0.indices, max1.indices
    arange0 = torch.arange(indices0.shape[1], device=indices0.device)[None]
    mutual0 = arange0 == indices1.gather(1, indices0)
    mscores0 = torch.where(mutual0, max0.values.exp(), scores.new_tensor(0))
    valid0 = mutual0 & (mscores0 > threshold)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    return indices0, mscores0


def matching_nll(log_assign, matches0, unmatched0, unmatched1):
    """Negative log-likelihood of the ground-truth partial assignment."""
    n = log_assign.shape[1] - 1
    m = log_assign.shape[2] - 1
    pos = matches0 >= 0
    lp_pos = log_assign[:, :n, :m].gather(2, matches0.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    lp_un0 = log_assign[:, :n, m]
    lp_un1 = log_assign[:, n, :m]
    total = (lp_pos * pos).sum() + (lp_un0 * unmatched0).sum() + (lp_un1 * unmatched1).sum()
    count = pos.sum() + unmatched0.sum() + unmatched1.sum()
    return -total / count.clamp(min=1)
