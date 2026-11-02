"""Stage 1: self-supervised training of KnuckleCreasePoint (Sec. 3.1).

Follows the SuperPoint protocol:
  1. Pseudo ground-truth crease keypoints are produced with homographic
     adaptation of a base detector (pretrained SuperPoint/MagicPoint weights if
     given via --pretrained, otherwise a classical crease-junction detector).
  2. The network is trained on (image, deformed image) pairs with the
     detector cross-entropy loss and the descriptor hinge loss.
  3. Labels are regenerated with the trained network and training repeats
     (--rounds).

Usage:
    python train_keypoint.py [--pretrained superpoint_v1.pth]
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import sato
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import get_device, images_to_tensor, save_checkpoint, seed_everything
from dataset import WarpPairDataset, list_samples, load_images, split_by_subject
from geometry import map_points, sample_homography
from pipeline import build_detector


# ----------------------------------------------------------------------------
# Base detectors for pseudo labels
# ----------------------------------------------------------------------------
def classical_crease_heatmap(batch):
    """Bootstrap detector: corner response on the dark-ridge (crease) map,
    i.e. crease junctions, endings and high-curvature crease points."""
    out = []
    for img in batch:
        ridge = sato(img, sigmas=(1, 2, 3), black_ridges=True).astype(np.float32)
        ridge /= ridge.max() + 1e-8
        harris = cv2.cornerHarris(ridge, blockSize=3, ksize=3, k=0.04)
        harris = np.clip(harris, 0, None)
        out.append(harris / (harris.max() + 1e-8))
    return np.stack(out)


def model_heatmap_fn(model, device):
    @torch.no_grad()
    def fn(batch):
        model.eval()
        logits, _ = model(images_to_tensor(list(batch)).to(device))
        return model.logits_to_heatmap(logits).cpu().numpy()
    return fn


def homographic_adaptation(img, heat_fn, rng, num_warps):
    """Aggregates detections over random homographies, mapped back to the image."""
    h, w = img.shape
    Hs, warped = [], [img]
    for _ in range(num_warps - 1):
        H = sample_homography(h, w, rng, **config.HOMOGRAPHY_ADAPT)
        Hs.append(H)
        warped.append(cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101))
    heats = heat_fn(np.stack(warped))
    acc = heats[0].copy()
    cnt = np.ones((h, w), np.float32)
    ones = np.ones((h, w), np.float32)
    for H, heat in zip(Hs, heats[1:]):
        Hinv = np.linalg.inv(H)
        acc += cv2.warpPerspective(heat, Hinv, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cnt += cv2.warpPerspective(ones, Hinv, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return acc / np.maximum(cnt, 1)


def heatmap_to_points(heat, radius=config.NMS_RADIUS, max_points=config.LABEL_MAX_POINTS,
                      rel_thresh=config.LABEL_REL_THRESH, border=config.REMOVE_BORDERS):
    dil = cv2.dilate(heat, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    mask = (heat >= dil) & (heat > rel_thresh * heat.max())
    if border > 0:
        mask[:border], mask[-border:], mask[:, :border], mask[:, -border:] = False, False, False, False
    ys, xs = np.nonzero(mask)
    order = np.argsort(-heat[ys, xs])[:max_points]
    return np.stack([xs[order], ys[order]], -1).astype(np.float32)


def generate_labels(images, heat_fn, rng, num_warps):
    return [heatmap_to_points(homographic_adaptation(img, heat_fn, rng, num_warps))
            for img in tqdm(images, desc="homographic adaptation")]


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def detector_loss(logits, kpmap, valid):
    """65-way cross-entropy per 8x8 cell (one random keypoint per cell, else dustbin)."""
    B, _, Hc, Wc = logits.shape
    lab = kpmap.view(B, Hc, 8, Wc, 8).permute(0, 2, 4, 1, 3).reshape(B, 64, Hc, Wc)
    lab = lab * 2 + torch.rand_like(lab) * 0.1           # random tie-break among keypoints
    lab = torch.cat([lab, torch.ones(B, 1, Hc, Wc, device=lab.device)], 1).argmax(1)
    valid_cells = valid.view(B, Hc, 8, Wc, 8).amin(dim=(2, 4))
    ce = F.cross_entropy(logits, lab, reduction="none")
    return (ce * valid_cells).sum() / valid_cells.sum().clamp(min=1)


def descriptor_loss(desc0, desc1, H, disp, valid1):
    """SuperPoint descriptor hinge loss between cells related by the deformation."""
    B, D, Hc, Wc = desc0.shape
    d0 = F.normalize(desc0, dim=1).flatten(2)
    d1 = F.normalize(desc1, dim=1).flatten(2)
    ys, xs = torch.meshgrid(torch.arange(Hc, device=desc0.device), torch.arange(Wc, device=desc0.device),
                            indexing="ij")
    centers = torch.stack([xs, ys], -1).reshape(-1, 2).float() * 8 + 3.5
    centers = centers[None].expand(B, -1, -1)
    warped = map_points(centers, H, disp)
    s = (torch.cdist(warped, centers) <= 7.5).float()      # (B, N0, N1)
    dot = torch.einsum("bdn,bdm->bnm", d0, d1)
    pos = config.DESC_LAMBDA_D * s * F.relu(config.DESC_POS_MARGIN - dot)
    neg = (1 - s) * F.relu(dot - config.DESC_NEG_MARGIN)
    v1 = valid1.view(B, Hc, 8, Wc, 8).amin(dim=(2, 4)).reshape(B, 1, -1)
    return ((pos + neg) * v1).sum() / (v1.sum() * s.shape[1]).clamp(min=1)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_round(model, images, labels, args, device, round_idx):
    ds = WarpPairDataset(images, labels, repeats=config.KP_REPEATS)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS,
                        drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        tot = {"det": 0.0, "desc": 0.0}
        pbar = tqdm(loader, desc=f"round {round_idx} epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits0, desc0 = model(batch["image0"])
            logits1, desc1 = model(batch["image1"])
            l_det = (detector_loss(logits0, batch["kpmap0"], batch["valid0"][:, 0])
                     + detector_loss(logits1, batch["kpmap1"], batch["valid1"][:, 0]))
            l_desc = descriptor_loss(desc0, desc1, batch["H"], batch["disp"], batch["valid1"][:, 0])
            loss = l_det + config.DESC_WEIGHT * l_desc
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["det"] += l_det.item()
            tot["desc"] += l_desc.item()
            pbar.set_postfix(det=f"{l_det.item():.3f}", desc=f"{l_desc.item():.3f}")
        n = len(loader)
        print(f"[round {round_idx}] epoch {epoch + 1}: det={tot['det'] / n:.4f} desc={tot['desc'] / n:.4f}")
        save_checkpoint(model, config.DETECTOR_CKPT, round=round_idx, epoch=epoch + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default=None,
                    help="optional SuperPoint weights (e.g. superpoint_v1.pth) for initialisation")
    ap.add_argument("--rounds", type=int, default=config.KP_ROUNDS)
    ap.add_argument("--epochs", type=int, default=config.KP_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.KP_BATCH)
    ap.add_argument("--lr", type=float, default=config.KP_LR)
    ap.add_argument("--ha-warps", type=int, default=config.HA_NUM_WARPS)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    train_samples, _ = split_by_subject(list_samples())
    images = load_images(train_samples)
    print(f"Training KnuckleCreasePoint on {len(images)} images ({device})")

    model = build_detector().to(device)
    if args.pretrained:
        model.load_pretrained(args.pretrained)
        print(f"Initialised from {args.pretrained}")

    rng = np.random.default_rng(config.SEED)
    for r in range(args.rounds):
        if r == 0 and not args.pretrained:
            heat_fn = classical_crease_heatmap
        else:
            heat_fn = model_heatmap_fn(model, device)
        labels = generate_labels(images, heat_fn, rng, args.ha_warps)
        print(f"[round {r}] mean pseudo keypoints per image: {np.mean([len(l) for l in labels]):.1f}")
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(config.OUTPUT_DIR / f"pseudo_labels_round{r}.npz",
                 **{s.path.name: l for s, l in zip(train_samples, labels)})
        train_round(model, images, labels, args, device, r)
    print(f"Saved detector to {config.DETECTOR_CKPT}")


if __name__ == "__main__":
    main()
