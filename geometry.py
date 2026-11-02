"""Geometric / photometric augmentation and point mapping utilities.

A random deformation is a homography H followed by a smooth elastic field d
(the elastic part mimics local knuckle-crease deformation under finger bending).
The warped image is  I'(r) = I_H(r + d(r)),  so a point p of the original image
maps to  q = H p  and then  r ~= q - d(q)  (first-order inverse of the field).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def sample_homography(h, w, rng, max_angle=15.0, scale=(0.85, 1.15),
                      max_translation=0.08, perspective=0.08):
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    center = np.float32([(w - 1) / 2.0, (h - 1) / 2.0])
    a = np.deg2rad(rng.uniform(-max_angle, max_angle))
    s = rng.uniform(*scale)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    dst = (src - center) @ rot.T * s + center
    dst += rng.uniform(-perspective, perspective, size=(4, 2)) * [w, h]
    dst += rng.uniform(-max_translation, max_translation, size=(1, 2)) * [w, h]
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)).astype(np.float32)


def sample_elastic_field(h, w, rng, alpha=3.0, sigma=12.0):
    """Smooth random displacement field (2, h, w) with max magnitude alpha px."""
    if alpha <= 0:
        return np.zeros((2, h, w), np.float32)
    field = rng.uniform(-1, 1, size=(2, h, w)).astype(np.float32)
    for c in range(2):
        field[c] = cv2.GaussianBlur(field[c], (0, 0), sigma)
    field /= np.abs(field).max() + 1e-8
    return (field * alpha).astype(np.float32)


def random_warp(img, rng, homography_cfg, elastic_cfg):
    """Returns warped image, valid mask, homography (3x3) and elastic field (2,h,w)."""
    h, w = img.shape
    H = sample_homography(h, w, rng, **homography_cfg)
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    valid = cv2.warpPerspective(np.ones_like(img), H, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    disp = sample_elastic_field(h, w, rng, **elastic_cfg)
    if np.any(disp):
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx, my = gx + disp[0], gy + disp[1]
        warped = cv2.remap(warped, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        valid = cv2.remap(valid, mx, my, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    return warped.astype(np.float32), valid.astype(np.float32), H, disp


def photometric_augment(img, rng, brightness=0.15, contrast=(0.75, 1.3), gamma=(0.75, 1.35),
                        noise=0.02, blur_prob=0.3, shading=0.15):
    h, w = img.shape
    out = img.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * rng.uniform(*contrast) + mean + rng.uniform(-brightness, brightness)
    if shading > 0:
        gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        a, b = rng.uniform(-shading, shading, 2)
        out = out + a * gx + b * gy
    out = np.clip(out, 0, 1) ** rng.uniform(*gamma)
    if rng.random() < blur_prob:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))
    out = out + rng.normal(0, rng.uniform(0, noise), out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# Point mapping (numpy)
# ----------------------------------------------------------------------------
def bilinear_sample_np(field, pts):
    """field: (C, h, w), pts: (N, 2) xy -> (N, C)."""
    _, h, w = field.shape
    x = np.clip(pts[:, 0], 0, w - 1)
    y = np.clip(pts[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    out = (field[:, y0, x0] * (1 - wx) * (1 - wy) + field[:, y0, x1] * wx * (1 - wy)
           + field[:, y1, x0] * (1 - wx) * wy + field[:, y1, x1] * wx * wy)
    return out.T


def warp_points_np(pts, H):
    if len(pts) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def map_points_np(pts, H, disp):
    q = warp_points_np(pts, H)
    if len(q) == 0:
        return q
    return q - bilinear_sample_np(disp, q)


# ----------------------------------------------------------------------------
# Point mapping (torch, batched)
# ----------------------------------------------------------------------------
def warp_points(pts, H):
    """pts: (B, N, 2), H: (B, 3, 3)."""
    ones = torch.ones_like(pts[..., :1])
    p = torch.cat([pts, ones], -1) @ H.transpose(1, 2)
    z = p[..., 2:]
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    return p[..., :2] / z


def sample_field(field, pts):
    """field: (B, C, h, w), pts: (B, N, 2) pixel xy -> (B, N, C)."""
    _, _, h, w = field.shape
    gx = pts[..., 0] / (w - 1) * 2 - 1
    gy = pts[..., 1] / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(1)
    out = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2).transpose(1, 2)


def map_points(pts, H, disp):
    q = warp_points(pts, H)
    return q - sample_field(disp, q)
