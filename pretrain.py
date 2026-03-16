"""
pretrain.py — Main SSL pretraining script for Pix2Rep-v2 on MUMUCD PRISMA data.

Usage
─────
  # Basic run (uses config.py defaults)
  python pretrain.py

  # Override any config field on the command line
  python pretrain.py --epochs 50 --batch_size 4 --data_root data/mumucd

The script:
  1. Loads MUMUCD PRISMA patches (from disk or HDF5 cache).
  2. For each batch, generates two augmented views (v, v') via Pix2RepAugmentation.
  3. Passes both through U-Net + projection head.
  4. Computes pixel-level Barlow Twins loss.
  5. Saves checkpoints every `save_every` epochs to `checkpoints/`.
  6. Logs loss to TensorBoard (optional) and stdout.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data.mumucd_dataset import MUMUCDPatchDataset
from data.augmentations import Pix2RepAugmentation
from models.unet import HyperspectralUNet
from models.projection_head import DenseProjectionHead
from losses.barlow_twins_pixel import PixelBarlowTwinsLoss


# ─────────────────────────────────── logging ──────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ────────────────────────────────── helpers ───────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    epoch: int,
    backbone: nn.Module,
    head: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    loss: float,
    cfg: Config,
):
    path = cfg.checkpoint_dir / f"pretrain_epoch{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dict":     head.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": cfg,
    }, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    backbone: nn.Module,
    head: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    optimiser.load_state_dict(ckpt["optimiser_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    log.info(f"Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ──────────────────────────────── collate fn ─────────────────────────────────

def ssl_collate_fn(batch, augmentation: Pix2RepAugmentation):
    """
    Given a list of raw patches (C, H, W), generate two augmented views per
    patch and stack into batched tensors.

    Returns:
        v:         (B, C, H, W)
        v_prime:   (B, C, H, W)
        theta_inv: (B, 2, 3)
    """
    vs, vps, thetas = [], [], []
    for patch in batch:
        v, vp, theta_inv = augmentation(patch)
        vs.append(v)
        vps.append(vp)
        thetas.append(theta_inv.squeeze(0))     # (2, 3)
    return (
        torch.stack(vs),                        # (B, C, H, W)
        torch.stack(vps),                       # (B, C, H, W)
        torch.stack(thetas),                    # (B, 2, 3)
    )


# ───────────────────────────────── training ───────────────────────────────────

def train(cfg: Config, resume_from: str | None = None):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    cache_path = cfg.cache_dir / "mumucd_patches.h5" if cfg.cache_dir else None
    dataset = MUMUCDPatchDataset(
        data_root=cfg.data_root,
        patch_size=cfg.patch_size,
        stride=cfg.patch_stride,
        cache_path=cache_path,
    )
    log.info(f"Dataset: {len(dataset)} patches from {cfg.data_root}")

    augmentation = Pix2RepAugmentation(cfg)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        collate_fn=lambda b: ssl_collate_fn(b, augmentation),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    backbone = HyperspectralUNet(
        n_bands=cfg.n_bands,
        spectral_ch=cfg.spectral_reduced,
        embed_dim=cfg.embed_dim,
    ).to(device)

    head = DenseProjectionHead(
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.proj_hidden_dim,
        proj_dim=cfg.proj_dim,
    ).to(device)

    total_params = sum(p.numel() for p in backbone.parameters()) + \
                   sum(p.numel() for p in head.parameters())
    log.info(f"Parameters: {total_params / 1e6:.1f} M")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    params = list(backbone.parameters()) + list(head.parameters())
    optimiser = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.t_max, eta_min=cfg.lr * 0.01,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = PixelBarlowTwinsLoss(
        proj_dim=cfg.proj_dim,
        n_pixels=cfg.n_pixels_M,
        lambda_=cfg.lambda_barlow,
    ).to(device)

    # ── Optional TensorBoard ──────────────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(cfg.log_dir))
        log.info(f"TensorBoard writer → {cfg.log_dir}")
    except ImportError:
        log.info("TensorBoard not available — logging to stdout only.")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if resume_from is not None:
        start_epoch = load_checkpoint(
            resume_from, backbone, head, optimiser, scheduler, device
        ) + 1

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        backbone.train()
        head.train()
        epoch_loss = 0.0

        for batch_idx, (v, v_prime, _theta_inv) in enumerate(loader, 1):
            v       = v.to(device, non_blocking=True)
            v_prime = v_prime.to(device, non_blocking=True)

            # Forward
            z       = head(backbone(v))           # (B, D, H, W)
            z_prime = head(backbone(v_prime))     # (B, D, H, W)

            loss = criterion(z, z_prime)

            # Backward
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

            if batch_idx % cfg.log_every == 0:
                avg = epoch_loss / batch_idx
                lr  = optimiser.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch:3d}/{cfg.epochs} | "
                    f"Batch {batch_idx:4d}/{len(loader)} | "
                    f"Loss {avg:.4f} | LR {lr:.2e}"
                )
                if writer:
                    global_step = (epoch - 1) * len(loader) + batch_idx
                    writer.add_scalar("loss/batch", avg, global_step)

        scheduler.step()

        avg_epoch_loss = epoch_loss / len(loader)
        log.info(f"─ Epoch {epoch:3d} complete | avg loss: {avg_epoch_loss:.4f}")
        if writer:
            writer.add_scalar("loss/epoch", avg_epoch_loss, epoch)
            writer.add_scalar("lr", optimiser.param_groups[0]["lr"], epoch)

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(epoch, backbone, head, optimiser, scheduler, avg_epoch_loss, cfg)

    if writer:
        writer.close()

    log.info("Pretraining complete.")
    return backbone, head


# ───────────────────────────────── CLI ───────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pix2Rep-v2 SSL pretraining on MUMUCD PRISMA hyperspectral data"
    )
    # Expose the most-common config fields as CLI flags
    cfg_defaults = Config()
    parser.add_argument("--data_root",       default=str(cfg_defaults.data_root))
    parser.add_argument("--cache_dir",       default=str(cfg_defaults.cache_dir))
    parser.add_argument("--patch_size",      type=int,   default=cfg_defaults.patch_size)
    parser.add_argument("--n_bands",         type=int,   default=cfg_defaults.n_bands)
    parser.add_argument("--spectral_reduced",type=int,   default=cfg_defaults.spectral_reduced)
    parser.add_argument("--embed_dim",       type=int,   default=cfg_defaults.embed_dim)
    parser.add_argument("--proj_dim",        type=int,   default=cfg_defaults.proj_dim)
    parser.add_argument("--batch_size",      type=int,   default=cfg_defaults.batch_size)
    parser.add_argument("--epochs",          type=int,   default=cfg_defaults.epochs)
    parser.add_argument("--lr",              type=float, default=cfg_defaults.lr)
    parser.add_argument("--weight_decay",    type=float, default=cfg_defaults.weight_decay)
    parser.add_argument("--n_pixels_M",      type=int,   default=cfg_defaults.n_pixels_M)
    parser.add_argument("--lambda_barlow",   type=float, default=cfg_defaults.lambda_barlow)
    parser.add_argument("--num_workers",     type=int,   default=cfg_defaults.num_workers)
    parser.add_argument("--seed",            type=int,   default=cfg_defaults.seed)
    parser.add_argument("--checkpoint_dir",  default=str(cfg_defaults.checkpoint_dir))
    parser.add_argument("--log_dir",         default=str(cfg_defaults.log_dir))
    parser.add_argument("--save_every",      type=int,   default=cfg_defaults.save_every)
    parser.add_argument("--resume_from",     default=None,
                        help="Path to a checkpoint file to resume training from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = Config(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        patch_size=args.patch_size,
        n_bands=args.n_bands,
        spectral_reduced=args.spectral_reduced,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_pixels_M=args.n_pixels_M,
        lambda_barlow=args.lambda_barlow,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_every=args.save_every,
    )

    train(cfg, resume_from=args.resume_from)
