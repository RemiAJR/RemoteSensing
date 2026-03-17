"""
tests/test_pipeline.py — Offline unit tests for the Pix2Rep-v2 pipeline.
No GPU and no real data files required.

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from config import Config
from data.augmentations import (
    Pix2RepAugmentation,
    spectral_reversal,
    band_dropout,
    random_affine_theta,
    apply_spatial_transform,
)
from models.unet import HyperspectralUNet
from models.projection_head import DenseProjectionHead
from losses.barlow_twins_pixel import PixelBarlowTwinsLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(
        n_bands=16,            # small for tests
        spectral_reduced=8,
        embed_dim=32,
        proj_dim=16,
        proj_hidden_dim=24,
        n_pixels_M=50,
        patch_size=32,
        epochs=1,
        batch_size=2,
    )


@pytest.fixture
def small_patch(cfg):
    return torch.rand(cfg.n_bands, cfg.patch_size, cfg.patch_size)


# ── Augmentations ────────────────────────────────────────────────────────────

def test_spectral_reversal(small_patch):
    rev = spectral_reversal(small_patch)
    assert torch.allclose(rev + small_patch, torch.ones_like(small_patch))


def test_band_dropout_zeroes_bands(small_patch, cfg):
    dropped = band_dropout(small_patch, p=0.5)
    n_zero = (dropped.abs().sum(dim=(-2, -1)) == 0).sum().item()
    assert n_zero >= 1


def test_band_dropout_noop_when_zero_probability(small_patch):
    dropped = band_dropout(small_patch, p=0.0)
    assert torch.allclose(dropped, small_patch)


def test_two_view_shapes(cfg, small_patch):
    aug = Pix2RepAugmentation(cfg)
    v, vp, theta_vprime_to_v = aug(small_patch)
    assert v.shape == small_patch.shape
    assert vp.shape == small_patch.shape
    assert theta_vprime_to_v.shape == (1, 2, 3)


def test_affine_zoom_in_avoids_padding():
    x = torch.ones(1, 64, 64)
    theta = random_affine_theta(
        64, 64, angles=(0,), flip_prob=0.0, zoom_range=(0.85, 0.85)
    )
    y = apply_spatial_transform(x, theta)
    assert float(y.min()) > 0.99


# ── U-Net ─────────────────────────────────────────────────────────────────────

def test_unet_output_shape(cfg):
    model = HyperspectralUNet(
        n_bands=cfg.n_bands,
        spectral_ch=cfg.spectral_reduced,
        embed_dim=cfg.embed_dim,
    )
    x = torch.rand(2, cfg.n_bands, cfg.patch_size, cfg.patch_size)
    out = model(x)
    assert out.shape == (2, cfg.embed_dim, cfg.patch_size, cfg.patch_size)


def test_unet_no_spectral_reduction(cfg):
    """Verify U-Net works when spectral reduction is disabled."""
    model = HyperspectralUNet(n_bands=cfg.n_bands, spectral_ch=0, embed_dim=cfg.embed_dim)
    x = torch.rand(1, cfg.n_bands, cfg.patch_size, cfg.patch_size)
    out = model(x)
    assert out.shape == (1, cfg.embed_dim, cfg.patch_size, cfg.patch_size)


# ── Projection head ───────────────────────────────────────────────────────────

def test_proj_head_shape(cfg):
    head = DenseProjectionHead(cfg.embed_dim, cfg.proj_hidden_dim, cfg.proj_dim)
    x = torch.rand(2, cfg.embed_dim, cfg.patch_size, cfg.patch_size)
    out = head(x)
    assert out.shape == (2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)


# ── Loss ──────────────────────────────────────────────────────────────────────

def test_loss_identical_views(cfg):
    """Identical views → loss should be near zero (on-diagonal ≈ 1, off-diagonal ≈ 0)."""
    loss_fn = PixelBarlowTwinsLoss(proj_dim=cfg.proj_dim, n_pixels=cfg.n_pixels_M, lambda_=5e-3)
    z = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)
    loss = loss_fn(z, z.clone())
    assert loss.item() < 0.5, f"Expected near-zero loss for identical views, got {loss.item()}"


def test_loss_random_views_larger(cfg):
    """Independent random views → loss should be non-trivial."""
    loss_fn = PixelBarlowTwinsLoss(proj_dim=cfg.proj_dim, n_pixels=cfg.n_pixels_M, lambda_=5e-3)
    z1 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)
    z2 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)
    loss = loss_fn(z1, z2)
    assert loss.item() > 0.0


def test_loss_is_differentiable(cfg):
    """Loss must be differentiable to allow backprop."""
    loss_fn = PixelBarlowTwinsLoss(proj_dim=cfg.proj_dim, n_pixels=cfg.n_pixels_M, lambda_=5e-3)
    z1 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size, requires_grad=True)
    z2 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size, requires_grad=True)
    loss = loss_fn(z1, z2)
    loss.backward()
    assert z1.grad is not None
    assert z2.grad is not None


def test_loss_uses_theta_alignment(cfg):
    """Aligned transform metadata should reduce loss vs misaligned metadata."""
    torch.manual_seed(0)
    B, D, H, W = 2, cfg.proj_dim, cfg.patch_size, cfg.patch_size
    z = torch.randn(B, D, H, W)

    theta = random_affine_theta(
        H, W, angles=(90,), flip_prob=0.0, zoom_range=(1.0, 1.0)
    ).repeat(B, 1, 1)
    grid = F.affine_grid(theta, z.shape, align_corners=False)
    z_prime = F.grid_sample(
        z, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    loss_fn = PixelBarlowTwinsLoss(proj_dim=cfg.proj_dim, n_pixels=cfg.n_pixels_M, lambda_=5e-3)
    eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).repeat(B, 1, 1)
    loss_aligned = loss_fn(z, z_prime, theta_vprime_to_v=theta)
    loss_misaligned = loss_fn(z, z_prime, theta_vprime_to_v=eye)
    assert loss_aligned.item() < loss_misaligned.item()


def test_loss_raises_on_empty_valid_mask(cfg):
    loss_fn = PixelBarlowTwinsLoss(proj_dim=cfg.proj_dim, n_pixels=cfg.n_pixels_M, lambda_=5e-3)
    z1 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)
    z2 = torch.randn(2, cfg.proj_dim, cfg.patch_size, cfg.patch_size)
    empty = torch.zeros(2, cfg.patch_size, cfg.patch_size, dtype=torch.bool)
    with pytest.raises(ValueError, match="No valid pixels"):
        loss_fn(z1, z2, valid_mask=empty)


def test_config_rejects_too_few_samples_for_proj_dim():
    with pytest.raises(ValueError, match="batch_size \\* n_pixels_M"):
        Config(batch_size=1, n_pixels_M=8, proj_dim=16)


# ── End-to-end overfit test ───────────────────────────────────────────────────

def test_loss_decreases_on_small_batch(cfg):
    """10 gradient steps on a tiny batch → loss must decrease."""
    torch.manual_seed(0)
    backbone = HyperspectralUNet(n_bands=cfg.n_bands, spectral_ch=cfg.spectral_reduced, embed_dim=cfg.embed_dim)
    head     = DenseProjectionHead(cfg.embed_dim, cfg.proj_hidden_dim, cfg.proj_dim)
    loss_fn  = PixelBarlowTwinsLoss(cfg.proj_dim, cfg.n_pixels_M, cfg.lambda_barlow)
    aug      = Pix2RepAugmentation(cfg)

    # Use a fixed pair of views
    patch = torch.rand(cfg.n_bands, cfg.patch_size, cfg.patch_size)
    v, vp, theta_vprime_to_v = aug(patch)
    v  = v.unsqueeze(0)    # (1, C, H, W)
    vp = vp.unsqueeze(0)

    optimiser = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=1e-3
    )

    losses = []
    for _ in range(10):
        z  = head(backbone(v))
        zp = head(backbone(vp))
        loss = loss_fn(z, zp, theta_vprime_to_v=theta_vprime_to_v)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
