# CGN: Explainable Finger-Crease (Knuckle) Matching

PyTorch implementation of *Zhou, Dong & Kumar, "Towards Explainable and Unprecedented
Accuracy in Matching Challenging Finger Crease Patterns", CVPR 2025*, adapted to the
`Knuckle_Images/` [dataset](https://www.kaggle.com/datasets/s3programmerlead/palmprint-and-finger-knuckle-pattern-dataset) (`<subject>_<sample>.bmp`, 5 samples per subject).

## Mapping paper → code

| Paper | File |
|---|---|
| 3.1 KnuckleCreasePoint (SuperPoint detector + descriptor, homographic adaptation) | `models/knuckle_crease_point.py`, `train_keypoint.py` |
| 3.2 KnucklePointPair (Eq. 1–2: keypoint encoder, self/cross attention, cost matrix with learnable non-match scores U1, U2, u, Sinkhorn) | `models/knuckle_point_pair.py`, `train_matcher.py` |
| Eq. 3–5 top-c correspondence selection | `pipeline.select_correspondences` |
| 3.3 Graph embedding (k-NN, isomorphic graphs), positional encoding (Eq. 7), ConvNode (Eq. 10–12), Cross-Node arccos + Tracker (Eq. 13–14), graph similarity (Eq. 15–17) | `models/cgn.py`, `train_cgn.py` |
| 4 All-to-all evaluation, ROC, GAR@FAR | `evaluate.py`, `metrics.py` |
| Fig. 1/3 explainable visualisation | `visualize.py` |
| 3.4 / 4.3 Uniqueness: joint mixture model (Eq. 18, EM + BIC), Poisson FRC (Eq. 19–22) | `uniqueness.py` |

All hyper-parameters are in `config.py`.

## Running

```bash
pip install -r requirements.txt
```

```bash
python train_keypoint.py
```

```bash
python train_matcher.py
```

```bash
python train_cgn.py
```

```bash
python evaluate.py
```

```bash
python visualize.py --probe Knuckle_Images/Knuckle_Images/041_1.bmp --gallery Knuckle_Images/Knuckle_Images/041_3.bmp
```

```bash
python uniqueness.py
```

Outputs (checkpoints, scores, ROC plots, visualisations, uniqueness tables) go to `outputs/`.

Optional initialisation: `train_keypoint.py --pretrained superpoint_v1.pth` (official
MagicLeap SuperPoint weights, which is what the paper starts from) and
`train_matcher.py --init-superglue superglue_outdoor.pth`. Without them the detector is
bootstrapped from a classical crease-junction detector.

## Results: this implementation vs. the paper

Numbers for this implementation come from `outputs/results_test.json`,
`outputs/roc_test.png` and `outputs/uniqueness.csv`. Paper numbers are copied from
Tables 2–5 of the paper.

> **Read these comparisons with care.** The paper and this implementation use different
> datasets and different protocols, at very different scales (see the next section). The
> tables show where the two sit side by side. They do not show which method is better.

### Test protocol at a glance

| | Paper, own dataset (Tab. 2) | Paper, Hand Dorsal [47] (Tab. 3) | Paper, FK v3.0 [11] (Tab. 4) | **This implementation** |
|---|---|---|---|---|
| Subjects tested | 351 | 712 | 104 | **20** (subject-disjoint) |
| Genuine scores | 31,590 | 14,240 | 3,744 | **392** |
| Impostor scores | 12,285,000 | 12,655,800 | 385,632 | **9,310** |
| Lowest FAR you can measure | ≈ 8×10⁻⁸ | ≈ 8×10⁻⁸ | ≈ 2.6×10⁻⁶ | **≈ 1.07×10⁻⁴** |
| Train / test relation | within-database | cross-database, no fine-tuning | cross-database, no fine-tuning | within-database, disjoint subjects |
| Protocol | all-to-all | all-to-all | all-to-all | all-to-all (ordered pairs, i ≠ j) |

### Verification accuracy (GAR at fixed FAR)

| Method / dataset | GAR @ FAR=10⁻⁵ | GAR @ FAR=10⁻⁴ | GAR @ FAR=10⁻³ | GAR @ FAR=10⁻² | GAR @ FAR=10⁻¹ | EER | AUC |
|---|---|---|---|---|---|---|---|
| Paper CGN, own dataset | 89.99% | 93.19% | – | 97.32% | – | ROC plot only | – |
| Paper CGN, Hand Dorsal [47] | 80.19% | 92.88% | – | 97.09% | – | ROC plot only | – |
| Paper CGN, FK v3.0 [11] | 55.41% | 66.35% | – | 80.95% | – | ROC plot only | – |
| **This implementation, CGN** | 90.82%\* | 90.82%\* | **96.43%** | **97.96%** | **98.21%** | **1.35%** | 0.9956 |
| This implementation, number of mutual matches (baseline) | 59.18%\* | 59.18%\* | 89.29% | 94.64% | 97.96% | 3.15% | 0.9958 |
| This implementation, sum of top-c match scores (baseline) | 82.65%\* | 82.65%\* | 88.52% | 94.90% | 97.96% | 3.62% | 0.9937 |

\* *Below the lowest measurable FAR (1.07×10⁻⁴), so these values are really the GAR at
zero false accepts among 9,310 impostors. They are not true 10⁻⁵ or 10⁻⁴ operating
points. To compare the two FAR columns with the paper, you would need far more impostor
pairs.*

For context, the paper's strongest non-graph baselines on its own dataset were:
EfficientNetv2-m, 67.22% / 81.78% / 98.06%, and RFNet-RSIL, 56.39% / 66.33% / 89.28%
(GAR at FAR = 10⁻⁵ / 10⁻⁴ / 10⁻²). On FK v3.0, every CNN baseline stayed below 3% GAR
at 10⁻⁴.

**What the accuracy numbers suggest**
- **CGN beats simple match counting.** On the same data, the graph-similarity score
  roughly halves the EER compared with scoring by correspondences alone (1.35% vs.
  3.15% and 3.62%). At FAR = 10⁻³ it gains 7–8 GAR points, and it is far stronger at
  zero false accepts (90.8% vs. 59.2% and 82.7%). This is consistent with the paper's
  claim (Sec. 3.3) that graph structure adds discriminative information beyond the
  number or similarity of correspondences.
- **At FAR = 10⁻²,** this implementation (97.96%) is in the same range as the paper on
  its own dataset (97.32%) and on Hand Dorsal (97.09%). However, this test has about
  2% of the paper's genuine pairs and under 0.1% of its impostor pairs, so the
  confidence intervals are wide.
- **The paper's hardest result is not reproduced.** It is cross-pose, cross-database
  matching on FK v3.0 (55.41% at 10⁻⁵). This repository has no cross-database test.

### Uniqueness (FRC) at the paper's main operating point, m = n = 69, w = 17, α = 0.05

| | Δk (px) | Δf | p(Tp, Tg) | λ_α | FRC_α |
|---|---|---|---|---|---|
| Paper (Tab. 5, Hand Dorsal [47]) | 6.750 | 0.726 | 2.980×10⁻⁴ | 6.613 | 1.860×10⁻⁴ |
| **This implementation (Knuckle_Images)** | 56.533 | 0.753 | 8.132×10⁻⁴ | 3.872 | 7.333×10⁻⁷ |

How FRC_α changes in this implementation when one parameter is varied:

| Varied parameter | Values | FRC_α |
|---|---|---|
| w (Δk = 56.533, Δf = 0.753) | 15 / 17 / 19 / 21 | 1.38×10⁻⁵ / 7.33×10⁻⁷ / 3.13×10⁻⁸ / 1.09×10⁻⁹ |
| Δk (w = 17) | 31.41 / 43.97 / 56.53 / 69.10 / 81.66 | 1.56×10⁻¹³ / 1.70×10⁻⁹ / 7.33×10⁻⁷ / 4.03×10⁻⁵ / 5.77×10⁻⁴ |
| Δf (w = 17) | 0.653 / 0.703 / 0.753 / 0.803 / 0.853 | 2.43×10⁻¹⁹ / 3.13×10⁻¹² / 7.33×10⁻⁷ / 3.46×10⁻³ / 3.48×10⁻¹ |

**What the uniqueness numbers suggest**
- **The trends match the paper.** FRC falls quickly as w increases, rises with looser
  tolerances, and is especially sensitive to Δf, just as in Table 5.
- **The absolute values are not comparable.**
  - Δk was estimated as 56.5 px instead of 6.75 px. It is measured from matcher
    correspondences between unaligned captures, not from annotated ground truth, so it
    includes the pose shift between images.
  - BIC chose a single mixture component (N* = 1). Every template therefore gets the
    same density, so p(Tp, Tg) is the same for every pair, and α-trimming has no
    effect (FRC_α = FRC).
  - The paper's λ_α (6.613) is not equal to m·n·p̄ (= 1.42), which suggests its
    per-template averaging differs from ours.
- **For a more meaningful estimate,** align the images before measuring Δk, or pass
  `--delta-k 6.75 --delta-f 0.726` to `uniqueness.py` to reuse the paper's tolerances.

## Differences between the paper's method and this implementation

### Data and protocol

| Aspect | Paper | This implementation |
|---|---|---|
| Training data | Own multi-pose 4K video dataset: 351 subjects, about 25 frames per video, 25,000 training images | `Knuckle_Images`: 50 subjects, 249 images (5 per subject; `050_5` is missing) |
| Test data | 351 subjects × 10 frames, plus cross-database tests on [47] and [11] | 20 held-out subjects (99 images), same database |
| Split | Train on own dataset; test within it and on other databases without fine-tuning | Seeded subject split: 30 train subjects (25 for CGN training, 5 for validation), 20 test subjects |
| Input | 152 × 200 knuckle ROI | Resized to 200 × 152 (H × W), grayscale, CLAHE contrast enhancement (not in the paper) |
| Baselines | ResNet-101, DenseNet-161, EfficientNetv2-m, ViT-B, FKNet, RFNet, STResNet; graph models GMN, SimGNN, MGMN, ERIC | Two scores from our own pipeline: number of mutual matches and the sum of the top-c match scores. The paper's baselines are not reimplemented. |

### Step 1: KnuckleCreasePoint (Sec. 3.1)

| Aspect | Paper | This implementation |
|---|---|---|
| Architecture | SuperPoint | SuperPoint (same layer layout and names, so the official weights load directly) |
| Initialisation | SuperPoint training: MagicPoint on synthetic shapes, then homographic adaptation (Supp. C.2) | Optional `--pretrained superpoint_v1.pth`. Otherwise bootstrapped from a classical crease-junction detector (Sato ridge filter + Harris). No synthetic-shapes stage. |
| Training labels | Homographic adaptation; knuckle keypoint ground truth available (Fig. 7) | 2 rounds of homographic adaptation (64 warps) on pseudo-labels; no manual ground truth |
| Augmentation | Not detailed in the main paper | Homography plus a smooth **elastic field** that imitates crease deformation, plus photometric changes |
| Keypoints per image | n (not fixed in the main text) | Exactly 256, sorted by score; NMS radius 3 |

### Step 2: KnucklePointPair (Sec. 3.2)

| Aspect | Paper | This implementation |
|---|---|---|
| Network | Self/cross attention GNN (as in SuperGlue), Eq. 1–2 | Same: keypoint-encoder MLP on [K \| S], 9 × (self, cross) layers, 4 heads |
| Non-match scores | Learnable U1 ∈ ℝⁿˣ¹, U2 ∈ ℝ¹ˣᵐ, u | Same. Implemented as per-rank vectors of length 256, which the fixed, sorted keypoint count makes possible. Optional SuperGlue initialisation copies `bin_score` into them. |
| Sinkhorn | t iterations | t = 100, in log space |
| Correspondence supervision | Ground-truth correspondences (Fig. 7, Supp. C) | **Synthetic only.** Known homography + elastic mapping; positive if a mutual nearest neighbour is within 3 px, dustbin if more than 5 px away. Never trained on real cross-image genuine pairs. |
| Top-c selection (Eq. 3) | Top-k with c by score in M | c = 64. Mutual-best pairs first, then repeated on the remaining rows and columns until c one-to-one pairs exist; sorted by M. |

### Steps 3–4: CGN (Sec. 3.3)

| Aspect | Paper | This implementation |
|---|---|---|
| Graph | k-NN on probe keypoints, reused for the gallery graph (Eq. 6) | Same, k = 9 (as in Fig. 3); neighbours ordered by distance so each slot of the ConvNode weight is consistent |
| Positional encoding (Eq. 7) | h = f + e(i′) | h = f + **s**·e(i′), where s is a learnable scale initialised to 1/√d so the code does not swamp the unit-norm descriptors |
| ConvNode (Eq. 10–12) | tr(W[d] Aᵀ) along the node dimension; activation φ not specified | Same operation; φ = LeakyReLU(0.1) |
| Cross-Node and Tracker (Eq. 13–14) | arccos of cosine similarity, l = 3 | Same, l = 3 |
| Graph similarity (Eq. 15–17) | l = 3 ConvNode layers, sum readout, MLP + sigmoid | Same. Hidden width 64; MLP 192→128→1 with dropout 0.2 |
| Training | Loss and optimiser not given in the main text | BCE on genuine and impostor pairs (1:1, resampled every epoch), Adam 1e-4 with cosine schedule, 60 epochs, mild warp and photometric augmentation; best epoch chosen by validation EER. Detector and matcher frozen. |

### Uniqueness (Sec. 3.4)

| Aspect | Paper | This implementation |
|---|---|---|
| Data | Hand Dorsal dataset [47] | All 249 `Knuckle_Images` images |
| Δk, Δf | From ground-truth correspondences, 97.5% quantile → 6.750 px, 0.726 | 97.5% quantile of distances between matcher correspondences on genuine pairs (confidence > 0.5) → 56.53 px, 0.753. The images are not aligned first. Both values can be overridden with flags. |
| Mixture model (Eq. 18) | Joint Gaussian mixture, 2-D location × 256-D descriptor, EM + BIC | Same, block-diagonal covariances, BIC over N = 1…8 (N* = 1 selected) |
| Template density f_T | Per-template mixture, following [32, 40] | Global components with **template-specific weights** (mean responsibilities of the template's keypoints) |
| p(Tp, Tg) (Eq. 20) | Integral over the tolerance region | Σ w_a^p w_b^g P(location match) P(descriptor match), with each component pair estimated by Monte Carlo (200k samples) |
| FRC (Eq. 19, 21, 22) | Poisson tail, α-trimmed mean over z(z−1)/2 identity pairs | Same formulas, averaged over all impostor template pairs |
