"""End-to-end matching pipeline (Fig. 2): detection -> correspondence -> graph -> score."""
import torch

import config
from common import gather_rows, images_to_tensor, load_checkpoint
from models import CGN, KnuckleCreasePoint, KnucklePointPair, mutual_matches


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_detector():
    return KnuckleCreasePoint(config.DESC_DIM, config.NMS_RADIUS, config.MAX_KEYPOINTS,
                              config.REMOVE_BORDERS)


def build_matcher():
    return KnucklePointPair(config.DESC_DIM, config.KENC_LAYERS, config.GNN_LAYERS,
                            config.SINKHORN_ITERS, config.MAX_KEYPOINTS)


def build_cgn():
    return CGN(config.DESC_DIM, config.KNN_K, config.CGN_SELF_LAYERS, config.CGN_SIM_LAYERS,
               config.CGN_SIM_HIDDEN, config.CGN_HEAD_HIDDEN, config.CGN_USE_PE, config.CGN_USE_ARCCOS)


def load_trained(device, detector=True, matcher=True, cgn=True):
    out = []
    if detector:
        m = build_detector().to(device)
        load_checkpoint(m, config.DETECTOR_CKPT, device)
        out.append(m.eval())
    if matcher:
        m = build_matcher().to(device)
        load_checkpoint(m, config.MATCHER_CKPT, device)
        out.append(m.eval())
    if cgn:
        m = build_cgn().to(device)
        load_checkpoint(m, config.CGN_CKPT, device)
        out.append(m.eval())
    return out


# ----------------------------------------------------------------------------
# Step 1: features
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(detector, images, device, batch_size=32, max_keypoints=None):
    """List of (H, W) images -> dict of stacked tensors on `device`."""
    out = {"keypoints": [], "scores": [], "descriptors": []}
    for s in range(0, len(images), batch_size):
        f = detector.extract(images_to_tensor(images[s:s + batch_size]).to(device), max_keypoints)
        for k in out:
            out[k].append(f[k])
    return {k: torch.cat(v) for k, v in out.items()}


def index_features(feats, idx):
    return {k: v[idx] for k, v in feats.items()}


def split_features(feats, b):
    return {k: v[:b] for k, v in feats.items()}, {k: v[b:] for k, v in feats.items()}


# ----------------------------------------------------------------------------
# Step 2: correspondences
# ----------------------------------------------------------------------------
def select_correspondences(M, c):
    """Top-c one-to-one correspondences from score matrix M (n, m), Eq. (3).

    Mutually matched pairs (row and column maxima) are taken first in decreasing
    score order; if fewer than c exist, the matched rows/columns are removed and
    the procedure repeats on the remainder. Output is sorted so that
    M[i1, j1] > M[i2, j2] > ... > M[ic, jc].
    """
    n, m = M.shape
    assert c <= min(n, m)
    M = M.clone()
    arange = torch.arange(n, device=M.device)
    rows, cols, vals = [], [], []
    remaining = c
    while remaining > 0:
        max0, arg0 = M.max(1)
        _, arg1 = M.max(0)
        mutual = (arg1[arg0] == arange) & (max0 >= 0)
        i = mutual.nonzero().squeeze(1)
        if i.numel() == 0:
            break
        j, v = arg0[i], max0[i]
        order = v.argsort(descending=True)[:remaining]
        i, j, v = i[order], j[order], v[order]
        rows.append(i)
        cols.append(j)
        vals.append(v)
        remaining -= i.numel()
        M[i, :] = -1.0
        M[:, j] = -1.0
    i, j, v = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    order = v.argsort(descending=True)
    return i[order], j[order], v[order]


@torch.no_grad()
def match_batch(matcher, f0, f1, num_corr, image_size, match_threshold=config.MATCH_THRESHOLD):
    """Runs KnucklePointPair on a batch of (probe, gallery) feature pairs."""
    log_assign = matcher(f0, f1, image_size)["log_assign"]
    P = log_assign[:, :-1, :-1].exp()
    idx0, idx1, corr = [], [], []
    for b in range(P.shape[0]):
        i, j, v = select_correspondences(P[b], num_corr)
        idx0.append(i)
        idx1.append(j)
        corr.append(v)
    matches0, mscores0 = mutual_matches(log_assign, match_threshold)
    return {
        "idx0": torch.stack(idx0), "idx1": torch.stack(idx1), "corr_scores": torch.stack(corr),
        "matches0": matches0, "mscores0": mscores0,
        "num_matches": (matches0 > -1).sum(1).float(),       # baseline score (match count)
    }


def graph_inputs(f0, f1, res):
    """Gathers F^p', K^p', F^g', K^g' (Eq. 4-5) for the selected correspondences."""
    fp = gather_rows(f0["descriptors"], res["idx0"])
    kp = gather_rows(f0["keypoints"], res["idx0"])
    fg = gather_rows(f1["descriptors"], res["idx1"])
    kg = gather_rows(f1["keypoints"], res["idx1"])
    return fp, kp, fg, kg


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------
@torch.no_grad()
def score_pairs(detector, matcher, cgn, feats, ii, jj, device, batch_size=64):
    """Scores ordered pairs (ii[k] probe, jj[k] gallery) given cached features."""
    image_size = (config.IMG_H, config.IMG_W)
    cgn_scores, n_matches, corr_sum = [], [], []
    for s in range(0, len(ii), batch_size):
        bi = torch.as_tensor(ii[s:s + batch_size], device=device)
        bj = torch.as_tensor(jj[s:s + batch_size], device=device)
        f0, f1 = index_features(feats, bi), index_features(feats, bj)
        res = match_batch(matcher, f0, f1, config.NUM_CORR, image_size)
        fp, kp, fg, _ = graph_inputs(f0, f1, res)
        cgn_scores.append(torch.sigmoid(cgn(fp, kp, fg)).cpu())
        n_matches.append(res["num_matches"].cpu())
        corr_sum.append(res["corr_scores"].sum(1).cpu())
    return (torch.cat(cgn_scores).numpy(), torch.cat(n_matches).numpy(), torch.cat(corr_sum).numpy())
