"""
data/augmentations.py — Hyperspectral two-view augmentation pipeline for Pix2Rep-v2.

Given a single patch tensor x ∈ (C, H, W):
  1. Independently apply *intensity* augmentations to obtain v and v'.
  2. Apply a random *spatial* affine transform φ to v' only (asymmetric).
  3. Return (v, v', theta_inv) where theta_inv is the inverse affine grid
     needed to match corresponding pixels between v and v'.

All operations are differentiable / GPU-friendly where relevant.
"""

import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────── intensity augmentations ──────────────────────

def band_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out a random fraction p of spectral bands."""
    C = x.shape[0]
    n_drop = max(1, int(C * p))
    drop_idx = torch.randperm(C)[:n_drop]
    x = x.clone()
    x[drop_idx] = 0.0
    return x


def brightness_contrast_jitter(x: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    """Per-band independent brightness and contrast jitter."""
    C = x.shape[0]
    # brightness: additive shift in [-brightness, +brightness]
    b = (torch.rand(C, 1, 1, device=x.device) * 2 - 1) * brightness
    # contrast: multiplicative factor in [1-contrast, 1+contrast]
    c = 1.0 + (torch.rand(C, 1, 1, device=x.device) * 2 - 1) * contrast
    return torch.clamp(x * c + b, 0.0, 1.0)


def spectral_reversal(x: torch.Tensor) -> torch.Tensor:
    """Key augmentation from Pix2Rep: intensity reversal x := 1 - x."""
    return 1.0 - x


def additive_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)


# ────────────────────────────── spatial transform ─────────────────────────────

def _rotation_matrix(angle_deg: float) -> torch.Tensor:
    """2×2 rotation matrix for angle_deg (CCW)."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def random_affine_theta(
    h: int,
    w: int,
    angles: Tuple[int, ...] = (0, 90, 180, 270),
    flip_prob: float = 0.5,
    zoom_range: Tuple[float, float] = (0.85, 1.0),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a random affine transformation matrix theta of shape (1, 2, 3),
    suitable for F.affine_grid.

    The transform combines:
      - One of the cardinal rotations (0/90/180/270°)
      - Optional horizontal flip
      - Random zoom (crop + rescale)

    Returns theta as a (1, 2, 3) tensor on *device*.
    """
    angle = random.choice(angles)
    R = _rotation_matrix(angle)                 # 2×2

    # Flip
    if random.random() < flip_prob:
        flip_mat = torch.tensor([[-1.0, 0], [0, 1.0]])
        R = flip_mat @ R

    # Zoom: scale factor > 1 means we see more of the image (zoom out)
    # We implement zoom as a scale in the normalised [-1,1] coordinate space.
    # zoom < 1 → zoom-in (crop); zoom > 1 → zoom-out (pad) — we use zoom-in only.
    zoom = random.uniform(*zoom_range)          # in [0.85, 1.0] → zoom-in crop

    # Full theta: [R | t] with t=0 (centred crop)
    theta = torch.zeros(2, 3)
    theta[:2, :2] = R / zoom                   # scale by 1/zoom in grid coords
    # theta[:, 2] stays 0 (no translation)

    return theta.unsqueeze(0).to(device)        # (1, 2, 3)


def apply_spatial_transform(
    x: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Apply affine transform theta (1, 2, 3) to x (C, H, W).
    Returns warped tensor of same shape, with border values set to 0.
    """
    C, H, W = x.shape
    x4d = x.unsqueeze(0)                        # (1, C, H, W)
    grid = F.affine_grid(theta, x4d.shape, align_corners=False)  # (1, H, W, 2)
    warped = F.grid_sample(x4d, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.squeeze(0)                    # (C, H, W)


def invert_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a (1, 2, 3) affine matrix.

    theta = [A | t] where A is (2×2) and t is (2×1).
    Inverse: [A^{-1} | -A^{-1} t]
    """
    A = theta[0, :, :2]                         # (2, 2)
    t = theta[0, :, 2:]                         # (2, 1)
    A_inv = torch.inverse(A)
    t_inv = -A_inv @ t
    theta_inv = torch.cat([A_inv, t_inv], dim=1).unsqueeze(0)  # (1, 2, 3)
    return theta_inv.to(theta.device)


# ────────────────────────────── two-view pipeline ────────────────────────────

class Pix2RepAugmentation:
    """
    Generates a pair of augmented views (v, v') from a single HSI patch x.

    v  — intensity augmented only
    v' — intensity augmented AND spatially transformed

    Usage::
        aug = Pix2RepAugmentation(cfg)
        v, v_prime, theta_inv = aug(patch)   # patch: (C, H, W) float32 tensor

    theta_inv (1, 2, 3) can be used to map pixel coordinates in v' back to v.
    """

    def __init__(self, cfg):
        self.band_dropout_p = cfg.band_dropout_p
        self.brightness = cfg.brightness_range
        self.contrast = cfg.contrast_range
        self.noise_std = cfg.noise_std

    def _intensity_aug(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all intensity augmentations in random order."""
        ops = [
            lambda t: brightness_contrast_jitter(t, self.brightness, self.contrast),
            lambda t: band_dropout(t, self.band_dropout_p),
            lambda t: spectral_reversal(t) if random.random() < 0.5 else t,
            lambda t: additive_gaussian_noise(t, self.noise_std),
        ]
        random.shuffle(ops)
        for op in ops:
            x = op(x)
        return x

    def __call__(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (C, H, W) float32 patch in [0, 1].

        Returns:
            v:         (C, H, W) intensity-augmented view
            v_prime:   (C, H, W) intensity- AND spatially-augmented view
            theta_inv: (1, 2, 3) inverse spatial transform (maps v'→v coordinates)
        """
        device = x.device
        C, H, W = x.shape

        # Independent intensity augmentations
        v = self._intensity_aug(x.clone())
        v_prime_intensity = self._intensity_aug(x.clone())

        # Spatial transform applied to v' only
        theta = random_affine_theta(H, W, device=device)
        v_prime = apply_spatial_transform(v_prime_intensity, theta)
        theta_inv = invert_theta(theta)

        return v, v_prime, theta_inv
