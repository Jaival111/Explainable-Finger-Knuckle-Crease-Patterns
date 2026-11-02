"""All-to-all evaluation on the held-out (subject-disjoint) test subjects.

Every ordered (probe, gallery) pair i != j is matched. Reports EER, AUC and GAR
at FAR = 1e-5 ... 1e-1 for the CGN score s_G and for two correspondence-only
baselines (number of mutual matches, sum of the top-c correspondence scores).

Usage:
    python evaluate.py [--split test|all]
"""
import argparse
import json

import numpy as np

import config
from common import get_device, seed_everything
from dataset import all_ordered_pairs, list_samples, load_images, split_by_subject
from metrics import plot_rocs, summarize
from pipeline import extract_features, load_trained, score_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "all"], default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    detector, matcher, cgn = load_trained(device)

    samples = list_samples()
    if args.split == "test":
        _, samples = split_by_subject(samples)
    images = load_images(samples)
    subjects = [s.subject for s in samples]
    feats = extract_features(detector, images, device)

    ii, jj, labels = all_ordered_pairs(subjects)
    print(f"{len(samples)} images, {int(labels.sum())} genuine / {int((1 - labels).sum())} impostor pairs")
    cgn_s, n_matches, corr_sum = score_pairs(detector, matcher, cgn, feats, ii, jj, device, args.batch_size)

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"scores_{args.split}.npz", probe=ii, gallery=jj, label=labels,
             cgn=cgn_s, num_matches=n_matches, corr_sum=corr_sum,
             files=np.array([s.path.name for s in samples]))

    methods = {"CGN (ours)": cgn_s, "#mutual matches": n_matches, "sum top-c corr.": corr_sum}
    gen, imp = labels == 1, labels == 0
    results = {name: summarize(s[gen], s[imp]) for name, s in methods.items()}

    fars = [k for k in next(iter(results.values())) if k.startswith("gar@")]
    header = f"{'Method':<20}{'EER':>9}{'AUC':>9}" + "".join(f"{k.replace('gar@far=', 'GAR@'):>14}" for k in fars)
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<20}{r['eer'] * 100:>8.2f}%{r['auc']:>9.4f}"
              + "".join(f"{r[k] * 100:>13.2f}%" for k in fars))
    r0 = next(iter(results.values()))
    print(f"(FAR values below {r0['min_resolvable_far']:.1e} are not resolvable with "
          f"{r0['n_impostor']} impostor scores.)")

    with open(out_dir / f"results_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_rocs({n: (s[gen], s[imp]) for n, s in methods.items()}, out_dir / f"roc_{args.split}.png",
              title=f"All-to-all ROC ({args.split})")
    print(f"Saved scores, results and ROC plot to {out_dir}")


if __name__ == "__main__":
    main()
