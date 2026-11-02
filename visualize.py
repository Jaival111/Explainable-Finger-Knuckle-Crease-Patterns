"""Explainable match visualisation (Fig. 1 / Fig. 3 style).

Shows detected crease keypoints, the top-c corresponded keypoints, the k-NN
correspondence graphs on both images, the graph similarity score s_G and lists
the correspondences together with the tracked Cross-Node angles per layer.

Usage:
    python visualize.py --probe Knuckle_Images/001_1.bmp --gallery Knuckle_Images/001_2.bmp
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from common import get_device, images_to_tensor
from dataset import load_image
from pipeline import graph_inputs, load_trained, match_batch


def draw_graph(ax, pts, edges, color):
    for i, nb in enumerate(edges):
        for j in nb:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=color, lw=0.6, alpha=0.7)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=color)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", type=int, default=15, help="number of correspondences to print")
    args = ap.parse_args()

    device = get_device()
    detector, matcher, cgn = load_trained(device)
    img_p, img_g = load_image(args.probe), load_image(args.gallery)
    feats = detector.extract(images_to_tensor([img_p, img_g]).to(device))
    f0 = {k: v[:1] for k, v in feats.items()}
    f1 = {k: v[1:] for k, v in feats.items()}
    res = match_batch(matcher, f0, f1, config.NUM_CORR, (config.IMG_H, config.IMG_W))
    fp, kp, fg, kg = graph_inputs(f0, f1, res)
    logit, details = cgn(fp, kp, fg, return_details=True)
    s_g = torch.sigmoid(logit).item()

    kp_all_p, kp_all_g = f0["keypoints"][0].cpu().numpy(), f1["keypoints"][0].cpu().numpy()
    kp, kg = kp[0].cpu().numpy(), kg[0].cpu().numpy()
    edges = details["edges"][0].cpu().numpy()
    tracked = details["tracked"][0].cpu().numpy()
    corr = res["corr_scores"][0].cpu().numpy()
    w = img_p.shape[1]
    pair = np.hstack([img_p, img_g])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(pair, cmap="gray")
    axes[0].scatter(kp_all_p[:, 0], kp_all_p[:, 1], s=4, c="r")
    axes[0].scatter(kp_all_g[:, 0] + w, kp_all_g[:, 1], s=4, c="r")
    axes[0].set_title("Detected knuckle crease keypoints")
    axes[1].imshow(pair, cmap="gray")
    cmap = plt.get_cmap("viridis")
    for a, b, c in zip(kp, kg, corr):
        axes[1].plot([a[0], b[0] + w], [a[1], b[1]], color=cmap(float(c)), lw=0.7)
    axes[1].set_title(f"Top-{config.NUM_CORR} correspondences")
    axes[2].imshow(pair, cmap="gray")
    draw_graph(axes[2], kp, edges, "tab:blue")
    draw_graph(axes[2], kg + [w, 0], edges, "tab:orange")
    axes[2].set_title(f"k-NN graphs (k={config.KNN_K}),  s_G = {s_g:.4f}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out = Path(args.out) if args.out else config.OUTPUT_DIR / "vis" / f"{Path(args.probe).stem}_vs_{Path(args.gallery).stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Graph similarity score s_G = {s_g:.4f}  ({'match' if s_g >= 0.5 else 'non-match'} at threshold 0.5)")
    print(f"Mutual matches above {config.MATCH_THRESHOLD}: {int(res['num_matches'].item())}")
    layers = " ".join(f"{'ang' + str(l):>7}" for l in range(tracked.shape[1]))
    print(f"{'#':>3} {'probe (x,y)':>14} {'gallery (x,y)':>14} {'M_ij':>6} {layers}")
    for r in range(min(args.list, len(kp))):
        angles = " ".join(f"{a:7.3f}" for a in tracked[r])
        print(f"{r:>3} ({kp[r, 0]:5.1f},{kp[r, 1]:5.1f})  ({kg[r, 0]:5.1f},{kg[r, 1]:5.1f}) {corr[r]:6.3f} {angles}")
    print(f"Saved visualisation to {out}")


if __name__ == "__main__":
    main()
