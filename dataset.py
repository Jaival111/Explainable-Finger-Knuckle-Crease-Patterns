"""Knuckle_Images dataset handling.

File names follow  <subject>_<sample>.bmp  (e.g. 001_3.bmp).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from geometry import map_points_np, photometric_augment, random_warp

_PATTERN = re.compile(r"^(\d+)_(\d+)\.(bmp|png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    path: Path
    subject: int
    index: int


def list_samples(data_dir=config.DATA_DIR):
    # Searched recursively, so images may also sit in a nested sub-folder
    # (e.g. Knuckle_Images/Knuckle_Images/001_1.bmp).
    samples = []
    for p in sorted(Path(data_dir).rglob("*")):
        m = _PATTERN.match(p.name)
        if p.is_file() and m:
            samples.append(Sample(p, int(m.group(1)), int(m.group(2))))
    if not samples:
        raise RuntimeError(f"No images matching <subject>_<sample>.bmp found in {data_dir}")
    return samples


def split_by_subject(samples, n_train=config.N_TRAIN_SUBJECTS, seed=config.SEED):
    subjects = sorted({s.subject for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    train_ids = set(subjects[:n_train])
    train = [s for s in samples if s.subject in train_ids]
    test = [s for s in samples if s.subject not in train_ids]
    return train, test


def split_train_val(train_samples, n_val=config.N_VAL_SUBJECTS, seed=config.SEED):
    subjects = sorted({s.subject for s in train_samples})
    rng = np.random.RandomState(seed + 1)
    rng.shuffle(subjects)
    val_ids = set(subjects[:n_val])
    return ([s for s in train_samples if s.subject not in val_ids],
            [s for s in train_samples if s.subject in val_ids])


def load_image(path, h=config.IMG_H, w=config.IMG_W, clahe=config.USE_CLAHE):
    """Grayscale knuckle image resized to (h, w), float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)
    return img.astype(np.float32) / 255.0


def load_images(samples):
    return [load_image(s.path) for s in samples]


def _rng():
    # Derived from torch's RNG so that each DataLoader worker gets its own stream.
    return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))


def points_to_map(pts, h, w):
    m = np.zeros((h, w), np.float32)
    if len(pts):
        xy = np.round(pts).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        m[xy[:, 1], xy[:, 0]] = 1.0
    return m


class WarpPairDataset(Dataset):
    """Self-supervised pairs (I, deform(I)) with a known dense point mapping.

    Used to train KnuckleCreasePoint (with pseudo keypoint labels) and
    KnucklePointPair (ground-truth correspondences from the known mapping).
    """

    def __init__(self, images, labels=None, repeats=1,
                 homography_cfg=config.HOMOGRAPHY_TRAIN, elastic_cfg=config.ELASTIC_TRAIN,
                 photometric_cfg=config.PHOTOMETRIC):
        self.images = images
        self.labels = labels
        self.repeats = repeats
        self.homography_cfg = homography_cfg
        self.elastic_cfg = elastic_cfg
        self.photometric_cfg = photometric_cfg

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, idx):
        rng = _rng()
        i = idx % len(self.images)
        img = self.images[i]
        h, w = img.shape
        warped, valid, H, disp = random_warp(img, rng, self.homography_cfg, self.elastic_cfg)
        item = {
            "image0": torch.from_numpy(photometric_augment(img, rng, **self.photometric_cfg))[None],
            "image1": torch.from_numpy(photometric_augment(warped, rng, **self.photometric_cfg))[None],
            "valid0": torch.ones(1, h, w),
            "valid1": torch.from_numpy(valid)[None],
            "H": torch.from_numpy(H),
            "disp": torch.from_numpy(disp),
        }
        if self.labels is not None:
            kp0 = self.labels[i]
            kp1 = map_points_np(kp0, H, disp)
            inside = ((kp1[:, 0] >= 0) & (kp1[:, 0] <= w - 1) & (kp1[:, 1] >= 0) & (kp1[:, 1] <= h - 1)) \
                if len(kp1) else np.zeros(0, bool)
            kp1 = kp1[inside]
            if len(kp1):
                xi = np.round(kp1).astype(int)
                kp1 = kp1[valid[xi[:, 1], xi[:, 0]] > 0.5]
            item["kpmap0"] = torch.from_numpy(points_to_map(kp0, h, w))
            item["kpmap1"] = torch.from_numpy(points_to_map(kp1, h, w))
        return item


def make_training_pairs(subjects, rng, impostor_ratio=1.0):
    """All ordered genuine pairs plus randomly sampled impostor pairs."""
    subjects = np.asarray(subjects)
    n = len(subjects)
    genuine = [(i, j, 1.0) for i in range(n) for j in range(n) if i != j and subjects[i] == subjects[j]]
    n_imp = int(round(len(genuine) * impostor_ratio))
    impostor = []
    while len(impostor) < n_imp:
        i, j = rng.integers(0, n, 2)
        if subjects[i] != subjects[j]:
            impostor.append((int(i), int(j), 0.0))
    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


def all_ordered_pairs(subjects):
    """All-to-all protocol: every ordered (probe, gallery) pair with i != j."""
    subjects = np.asarray(subjects)
    n = len(subjects)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    labels = (subjects[ii] == subjects[jj]).astype(np.float32)
    return ii, jj, labels


class PairDataset(Dataset):
    """(probe, gallery, label) image pairs for training the CGN."""

    def __init__(self, images, pairs, augment=True):
        self.images = images
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _prep(self, img, rng):
        if not self.augment:
            return img
        img = random_warp(img, rng, config.HOMOGRAPHY_CGN, config.ELASTIC_CGN)[0]
        return photometric_augment(img, rng, **config.PHOTOMETRIC)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        rng = _rng()
        return {
            "image0": torch.from_numpy(self._prep(self.images[i], rng))[None],
            "image1": torch.from_numpy(self._prep(self.images[j], rng))[None],
            "label": torch.tensor(label, dtype=torch.float32),
        }
