"""Stage 2: training KnucklePointPair (Sec. 3.2).

Ground-truth correspondences come from synthetic deformations (homography +
elastic crease deformation) of the training images: detected keypoints are
projected with the known mapping, mutual nearest neighbours closer than
GT_POS_THRESH are positives and keypoints farther than GT_NEG_THRESH from any
projection are assigned to the non-match (dustbin) row/column. The loss is the
negative log-likelihood of this partial assignment.

Usage:
    python train_matcher.py [--init-superglue superglue_outdoor.pth]
"""
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import get_device, load_checkpoint, save_checkpoint, seed_everything
from dataset import WarpPairDataset, list_samples, load_images, split_by_subject
from geometry import map_points, sample_field
from models import matching_nll, mutual_matches
from pipeline import build_detector, build_matcher, split_features


@torch.no_grad()
def ground_truth_matches(kp0, kp1, H, disp, valid1, pos_th=config.GT_POS_THRESH,
                         neg_th=config.GT_NEG_THRESH):
    B, N, _ = kp0.shape
    M = kp1.shape[1]
    proj = map_points(kp0, H, disp)
    vis0 = sample_field(valid1, proj)[..., 0] > 0.5
    vis1 = sample_field(valid1, kp1)[..., 0] > 0.5
    d = torch.cdist(proj, kp1)
    d = d.masked_fill(~vis0[:, :, None] | ~vis1[:, None, :], float("inf"))
    min0, arg0 = d.min(2)
    min1, arg1 = d.min(1)
    mutual = arg1.gather(1, arg0) == torch.arange(N, device=kp0.device)[None]
    pos = mutual & (min0 < pos_th)
    matches0 = torch.where(pos, arg0, torch.full_like(arg0, -1))
    matched1 = torch.zeros(B, M + 1, device=kp0.device)
    matched1.scatter_(1, torch.where(pos, arg0, torch.full_like(arg0, M)), 1.0)
    matched1 = matched1[:, :M].bool()
    unmatched0 = (~pos) & (min0 > neg_th)
    unmatched1 = (~matched1) & (min1 > neg_th)
    return matches0, unmatched0, unmatched1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-superglue", type=str, default=None,
                    help="optional SuperGlue weights used as initialisation")
    ap.add_argument("--epochs", type=int, default=config.MATCH_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.MATCH_BATCH)
    ap.add_argument("--lr", type=float, default=config.MATCH_LR)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    image_size = (config.IMG_H, config.IMG_W)

    detector = build_detector().to(device)
    load_checkpoint(detector, config.DETECTOR_CKPT, device)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    matcher = build_matcher().to(device)
    if args.init_superglue:
        missing, unexpected = matcher.load_superglue_weights(args.init_superglue)
        print(f"Initialised from {args.init_superglue} (missing={len(missing)}, unexpected={len(unexpected)})")

    train_samples, _ = split_by_subject(list_samples())
    images = load_images(train_samples)
    loader = DataLoader(WarpPairDataset(images, repeats=config.MATCH_REPEATS), batch_size=args.batch_size,
                        shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    opt = torch.optim.Adam(matcher.parameters(), lr=args.lr)
    print(f"Training KnucklePointPair on {len(images)} images ({device})")

    for epoch in range(args.epochs):
        matcher.train()
        tot_loss, tot_correct, tot_pred, tot_gt = 0.0, 0, 0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["image0"].shape[0]
            with torch.no_grad():
                f0, f1 = split_features(detector.extract(torch.cat([batch["image0"], batch["image1"]])), B)
            gt0, un0, un1 = ground_truth_matches(f0["keypoints"], f1["keypoints"], batch["H"],
                                                 batch["disp"], batch["valid1"])
            log_assign = matcher(f0, f1, image_size)["log_assign"]
            loss = matching_nll(log_assign, gt0, un0, un1)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred0, _ = mutual_matches(log_assign, config.MATCH_THRESHOLD)
                tot_correct += ((pred0 == gt0) & (gt0 >= 0)).sum().item()
                tot_pred += (pred0 >= 0).sum().item()
                tot_gt += (gt0 >= 0).sum().item()
            tot_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        n = len(loader)
        print(f"epoch {epoch + 1}: loss={tot_loss / n:.4f} "
              f"precision={tot_correct / max(tot_pred, 1):.3f} recall={tot_correct / max(tot_gt, 1):.3f}")
        save_checkpoint(matcher, config.MATCHER_CKPT, epoch=epoch + 1)
    print(f"Saved matcher to {config.MATCHER_CKPT}")


if __name__ == "__main__":
    main()
