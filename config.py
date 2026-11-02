"""Global configuration for the CGN finger-knuckle matching framework.

Implementation of: Zhou, Dong, Kumar, "Towards Explainable and Unprecedented
Accuracy in Matching Challenging Finger Crease Patterns", CVPR 2025.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Knuckle_Images"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

DETECTOR_CKPT = CKPT_DIR / "knuckle_crease_point.pth"
MATCHER_CKPT = CKPT_DIR / "knuckle_point_pair.pth"
CGN_CKPT = CKPT_DIR / "cgn.pth"

SEED = 42
DEVICE = "cuda"            # falls back to cpu if cuda is unavailable
NUM_WORKERS = 0            # 0 is safest on Windows

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# The paper feeds the CGN pipeline with 152 x 200 knuckle images (Sec. 4.4).
# Both sides must be divisible by 8 (SuperPoint cell size).
IMG_H, IMG_W = 200, 152
USE_CLAHE = True           # local contrast enhancement of crease patterns

# Subject-disjoint protocol: the first N_TRAIN_SUBJECTS (after a seeded shuffle)
# are used for training, the rest for the all-to-all test protocol.
N_TRAIN_SUBJECTS = 30
# Of the training subjects, this many are held out to select the best CGN epoch.
N_VAL_SUBJECTS = 5

# ----------------------------------------------------------------------------
# Step 1: KnuckleCreasePoint (SuperPoint-style detector / descriptor, Sec. 3.1)
# ----------------------------------------------------------------------------
DESC_DIM = 256
NMS_RADIUS = 3
MAX_KEYPOINTS = 256        # n = m keypoints per image (fixed, sorted by score)
REMOVE_BORDERS = 4

KP_ROUNDS = 2              # rounds of homographic adaptation + retraining
KP_EPOCHS = 60
KP_BATCH = 8
KP_LR = 1e-4
KP_REPEATS = 4             # random warps per image per epoch
HA_NUM_WARPS = 64          # homographic adaptation warps for pseudo labels
LABEL_MAX_POINTS = 200
LABEL_REL_THRESH = 0.05    # keep NMS maxima above this fraction of the max
DESC_POS_MARGIN = 1.0
DESC_NEG_MARGIN = 0.2
DESC_LAMBDA_D = 250.0
DESC_WEIGHT = 1.0

# ----------------------------------------------------------------------------
# Step 2: KnucklePointPair (attentional GNN + Sinkhorn, Sec. 3.2)
# ----------------------------------------------------------------------------
GNN_LAYERS = ["self", "cross"] * 9
KENC_LAYERS = [32, 64, 128, 256]
SINKHORN_ITERS = 100       # t iterations
MATCH_THRESHOLD = 0.2
MATCH_EPOCHS = 100
MATCH_BATCH = 8
MATCH_LR = 1e-4
MATCH_REPEATS = 4
GT_POS_THRESH = 3.0        # px, reprojection distance for a positive match
GT_NEG_THRESH = 5.0        # px, beyond this a keypoint is assigned to the dustbin

# ----------------------------------------------------------------------------
# Step 3/4: Correspondence graph network (CGN, Sec. 3.3)
# ----------------------------------------------------------------------------
NUM_CORR = 64              # c, top correspondences forming the graph
KNN_K = 9                  # k-NN edges (Fig. 3 uses k = 9)
CGN_SELF_LAYERS = 3        # l Self-Graph layers
CGN_SIM_LAYERS = 3         # l graph-similarity layers
CGN_SIM_HIDDEN = 64
CGN_HEAD_HIDDEN = 128
CGN_USE_PE = True
CGN_USE_ARCCOS = True
CGN_EPOCHS = 60
CGN_BATCH = 16
CGN_LR = 1e-4
CGN_IMPOSTOR_RATIO = 1.0   # impostor pairs sampled per genuine pair each epoch
CGN_VAL_IMPOSTORS = 2000

# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
HOMOGRAPHY_TRAIN = dict(max_angle=15.0, scale=(0.85, 1.15), max_translation=0.08, perspective=0.08)
ELASTIC_TRAIN = dict(alpha=3.0, sigma=12.0)          # simulates local crease deformation
HOMOGRAPHY_ADAPT = dict(max_angle=10.0, scale=(0.9, 1.1), max_translation=0.05, perspective=0.05)
HOMOGRAPHY_CGN = dict(max_angle=5.0, scale=(0.95, 1.05), max_translation=0.03, perspective=0.03)
ELASTIC_CGN = dict(alpha=1.5, sigma=12.0)
PHOTOMETRIC = dict(brightness=0.15, contrast=(0.75, 1.3), gamma=(0.75, 1.35),
                   noise=0.02, blur_prob=0.3, shading=0.15)

# ----------------------------------------------------------------------------
# Evaluation / uniqueness (Sec. 3.4, 4.3)
# ----------------------------------------------------------------------------
FAR_POINTS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
UNI_TEMPLATE_SIZE = 69     # m = n keypoints per template
UNI_W = 17                 # minimum number of matches w
UNI_ALPHA = 0.05
UNI_MAX_COMPONENTS = 8     # BIC search range 1..N
UNI_MC_SAMPLES = 200_000   # Monte-Carlo samples per component pair
UNI_GENUINE_CONF = 0.5     # correspondence confidence used to estimate deltas
UNI_DELTA_QUANTILE = 0.975
