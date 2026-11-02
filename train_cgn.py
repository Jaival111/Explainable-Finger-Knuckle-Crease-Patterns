"""Stage 3: training the correspondence graph network CGN (Sec. 3.3).

The frozen KnuckleCreasePoint + KnucklePointPair produce the top-c
correspondences of each (probe, gallery) pair; the CGN is trained with binary
cross-entropy on genuine / impostor labels. The best epoch is selected by EER
on held-out (subject-disjoint) validation subjects.

Usage:
    python train_cgn.py
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import get_device, load_checkpoint, save_checkpoint, seed_everything
from dataset import (PairDataset, all_ordered_pairs, list_samples, load_images, make_training_pairs,
                     split_by_subject, split_train_val)
from metrics import summarize
from pipeline import (build_cgn, build_detector, build_matcher, extract_features, graph_inputs,
                      match_batch, score_pairs, split_features)


def validation_pairs(subjects, rng, n_impostors):
    ii, jj, labels = all_ordered_pairs(subjects)
    gen = np.nonzero(labels == 1)[0]
    imp = np.nonzero(labels == 0)[0]
    imp = rng.choice(imp, size=min(n_impostors, len(imp)), replace=False)
    sel = np.r_[gen, imp]
    return ii[sel], jj[sel], labels[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.CGN_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.CGN_BATCH)
    ap.add_argument("--lr", type=float, default=config.CGN_LR)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    image_size = (config.IMG_H, config.IMG_W)

    detector, matcher = build_detector().to(device), build_matcher().to(device)
    load_checkpoint(detector, config.DETECTOR_CKPT, device)
    load_checkpoint(matcher, config.MATCHER_CKPT, device)
    for m in (detector, matcher):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    cgn = build_cgn().to(device)

    train_samples, _ = split_by_subject(list_samples())
    tr_samples, val_samples = split_train_val(train_samples)
    tr_images, val_images = load_images(tr_samples), load_images(val_samples)
    tr_subjects = [s.subject for s in tr_samples]
    rng = np.random.default_rng(config.SEED)

    if val_samples:
        val_feats = extract_features(detector, val_images, device)
        vi, vj, vl = validation_pairs([s.subject for s in val_samples], rng, config.CGN_VAL_IMPOSTORS)

    opt = torch.optim.Adam(cgn.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_eer = float("inf")
    print(f"Training CGN: {len(tr_samples)} train images, {len(val_samples)} validation images ({device})")

    for epoch in range(args.epochs):
        pairs = make_training_pairs(tr_subjects, rng, config.CGN_IMPOSTOR_RATIO)
        loader = DataLoader(PairDataset(tr_images, pairs, augment=not args.no_augment),
                            batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
        cgn.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            img0, img1 = batch["image0"].to(device), batch["image1"].to(device)
            label = batch["label"].to(device)
            B = img0.shape[0]
            with torch.no_grad():
                f0, f1 = split_features(detector.extract(torch.cat([img0, img1])), B)
                res = match_batch(matcher, f0, f1, config.NUM_CORR, image_size)
                fp, kp, fg, _ = graph_inputs(f0, f1, res)
            logits = cgn(fp, kp, fg)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * B
            tot_acc += ((logits > 0).float() == label).float().sum().item()
            n += B
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        msg = f"epoch {epoch + 1}: loss={tot_loss / n:.4f} acc={tot_acc / n:.3f}"

        if val_samples:
            cgn.eval()
            scores, _, _ = score_pairs(detector, matcher, cgn, val_feats, vi, vj, device)
            stats = summarize(scores[vl == 1], scores[vl == 0])
            msg += f" | val EER={stats['eer'] * 100:.2f}% AUC={stats['auc']:.4f}"
            if stats["eer"] < best_eer:
                best_eer = stats["eer"]
                save_checkpoint(cgn, config.CGN_CKPT, epoch=epoch + 1, val_eer=best_eer)
                msg += " (saved)"
        else:
            save_checkpoint(cgn, config.CGN_CKPT, epoch=epoch + 1)
        print(msg)
    print(f"Saved CGN to {config.CGN_CKPT}")


if __name__ == "__main__":
    main()
