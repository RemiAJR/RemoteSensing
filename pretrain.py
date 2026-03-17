"""
pretrain.py — Main SSL pretraining script for Pix2Rep-v2 on MUMUCD PRISMA data.

Usage
─────
  # Basic run (uses config.py defaults)
  python pretrain.py

  # Override any config field on the command line
  python pretrain.py --epochs 50 --batch_size 4 --data_root data/mumucd

  # VM-safe relaunch profile (disables cache, lowers RAM pressure)
  python pretrain.py --vm_safe --resume_latest

  # Explicit resume from a checkpoint
  python pretrain.py --resume_from checkpoints/pretrain_epoch0002.pt

  # Budget-aware run (caps steps per epoch)
  python pretrain.py --resume_from checkpoints/pretrain_epoch0002.pt --epochs 5 \
      --max_batches_per_epoch 150 --log_every 10

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
    *,
    batch_idx: int = 0,
    global_step: int = 0,
    path: Path | None = None,
    label: str = "Checkpoint",
):
    target_path = path or (cfg.checkpoint_dir / f"pretrain_epoch{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dict":     head.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": cfg,
    }, target_path)
    log.info(f"{label} saved → {target_path}")


def load_checkpoint(
    path: str | Path,
    backbone: nn.Module,
    head: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    optimiser.load_state_dict(ckpt["optimiser_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    batch_idx = int(ckpt.get("batch_idx", 0))
    global_step = int(ckpt.get("global_step", 0))
    if batch_idx > 0:
        log.info(
            f"Resumed from checkpoint: {path} "
            f"(epoch {epoch}, batch {batch_idx}, global_step {global_step})"
        )
    else:
        log.info(f"Resumed from checkpoint: {path} (epoch {epoch})")
    return epoch, batch_idx, global_step


def _find_latest_checkpoint(checkpoint_dir: Path, latest_checkpoint_name: str) -> Path | None:
    latest_ckpt = checkpoint_dir / latest_checkpoint_name
    if latest_ckpt.exists():
        return latest_ckpt

    epoch_ckpts: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("pretrain_epoch*.pt"):
        suffix = path.stem.replace("pretrain_epoch", "")
        if suffix.isdigit():
            epoch_ckpts.append((int(suffix), path))

    if not epoch_ckpts:
        return None
    return max(epoch_ckpts, key=lambda item: item[0])[1]


def _resolve_resume_path(
    resume_from: str | None,
    resume_latest: bool,
    checkpoint_dir: str | Path,
    latest_checkpoint_name: str = "pretrain_latest.pt",
) -> str | None:
    if resume_from:
        return str(resume_from)
    if not resume_latest:
        return None

    latest = _find_latest_checkpoint(Path(checkpoint_dir), latest_checkpoint_name)
    if latest is None:
        raise FileNotFoundError(
            f"--resume_latest requested but no checkpoint found in {checkpoint_dir}"
        )
    return str(latest)


# ──────────────────────────────── collate fn ─────────────────────────────────

def ssl_collate_fn(batch, augmentation: Pix2RepAugmentation):
    """
    Given a list of raw patches (C, H, W), generate two augmented views per
    patch and stack into batched tensors.

    Returns:
        v:         (B, C, H, W)
        v_prime:   (B, C, H, W)
        theta_vprime_to_v: (B, 2, 3)
    """
    vs, vps, thetas = [], [], []
    for patch in batch:
        v, vp, theta_vprime_to_v = augmentation(patch)
        vs.append(v)
        vps.append(vp)
        thetas.append(theta_vprime_to_v.squeeze(0))  # (2, 3)
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
    cache_path = cfg.cache_dir / "mumucd_patches.h5" if cfg.use_cache else None
    dataset = MUMUCDPatchDataset(
        data_root=cfg.data_root,
        patch_size=cfg.patch_size,
        stride=cfg.patch_stride,
        cache_path=cache_path,
    )
    log.info(f"Dataset: {len(dataset)} patches from {cfg.data_root}")

    augmentation = Pix2RepAugmentation(cfg)

    persistent_workers = cfg.persistent_workers and cfg.num_workers > 0
    prefetch_factor = cfg.prefetch_factor if cfg.num_workers > 0 else None
    log.info(
        "DataLoader config | batch_size=%s | workers=%s | prefetch_factor=%s | persistent_workers=%s",
        cfg.batch_size,
        cfg.num_workers,
        prefetch_factor,
        persistent_workers,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
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

    compile_available = torch.cuda.is_available() and hasattr(torch, "compile")

    total_params = sum(p.numel() for p in backbone.parameters()) + \
                   sum(p.numel() for p in head.parameters())
    log.info(f"Parameters: {total_params / 1e6:.1f} M")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    # Linear LR scaling rule: scale LR proportional to batch_size increase
    base_bs = 8
    scaled_lr = cfg.lr * (cfg.batch_size / base_bs)
    log.info(f"LR scaled: {cfg.lr:.2e} → {scaled_lr:.2e} (batch {base_bs}→{cfg.batch_size})")

    params = list(backbone.parameters()) + list(head.parameters())
    optimiser = torch.optim.Adam(params, lr=scaled_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.t_max, eta_min=scaled_lr * 0.01,
    )

    # ── AMP (mixed precision bfloat16 for H100) ──────────────────────────────
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    log.info(f"AMP: {use_amp} (dtype={amp_dtype})")

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
    resume_batch_idx = 0
    global_step = 0
    if resume_from is not None:
        resumed_epoch, resumed_batch_idx, resumed_global_step = load_checkpoint(
            resume_from, backbone, head, optimiser, scheduler, device
        )
        global_step = resumed_global_step
        if resumed_batch_idx > 0:
            start_epoch = max(resumed_epoch, 1)
            resume_batch_idx = resumed_batch_idx
        else:
            start_epoch = resumed_epoch + 1

    # Compile only for fresh runs; resumed checkpoints store eager state-dict keys.
    if compile_available and resume_from is None:
        log.info("Compiling backbone and head with torch.compile...")
        backbone = torch.compile(backbone)
        head = torch.compile(head)
        params = list(backbone.parameters()) + list(head.parameters())
    elif compile_available and resume_from is not None:
        log.info("Skipping torch.compile for resumed run to keep checkpoint compatibility.")

    # ── Training loop ─────────────────────────────────────────────────────────
    full_batches_per_epoch = len(loader)
    target_batches_per_epoch = (
        min(full_batches_per_epoch, cfg.max_batches_per_epoch)
        if cfg.max_batches_per_epoch > 0
        else full_batches_per_epoch
    )
    if cfg.max_batches_per_epoch > 0:
        log.info(
            "Epoch batch cap enabled: %s/%s batches per epoch.",
            target_batches_per_epoch,
            full_batches_per_epoch,
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        backbone.train()
        head.train()
        epoch_loss = 0.0
        processed_batches = 0
        batches_to_skip = resume_batch_idx if (epoch == start_epoch and resume_batch_idx > 0) else 0
        if batches_to_skip >= target_batches_per_epoch:
            log.warning(
                "Resume batch index %s exceeds current epoch cap %s. "
                "Restarting epoch %s from batch 1.",
                batches_to_skip,
                target_batches_per_epoch,
                epoch,
            )
            batches_to_skip = 0
        if batches_to_skip > 0:
            log.info(f"Skipping first {batches_to_skip} already-processed batches in epoch {epoch}.")

        for batch_idx, (v, v_prime, theta_vprime_to_v) in enumerate(loader, 1):
            if batches_to_skip and batch_idx <= batches_to_skip:
                continue

            v = v.to(device, non_blocking=True)
            v_prime = v_prime.to(device, non_blocking=True)
            theta_vprime_to_v = theta_vprime_to_v.to(device, non_blocking=True)

            # Forward with AMP
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                z = head(backbone(v))                 # (B, D, H, W)
                z_prime = head(backbone(v_prime))     # (B, D, H, W)
                loss = criterion(z, z_prime, theta_vprime_to_v=theta_vprime_to_v)

            # Backward
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            processed_batches += 1
            global_step += 1

            if batch_idx % cfg.log_every == 0:
                avg = epoch_loss / processed_batches
                lr  = optimiser.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch:3d}/{cfg.epochs} | "
                    f"Batch {processed_batches:4d}/{target_batches_per_epoch} "
                    f"(data idx {batch_idx:4d}/{full_batches_per_epoch}) | "
                    f"Loss {avg:.4f} | LR {lr:.2e}"
                )
                if writer:
                    writer.add_scalar("loss/batch", avg, global_step)

            if cfg.save_latest_every_batches > 0 and batch_idx % cfg.save_latest_every_batches == 0:
                running_avg = epoch_loss / processed_batches
                save_checkpoint(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    backbone=backbone,
                    head=head,
                    optimiser=optimiser,
                    scheduler=scheduler,
                    loss=running_avg,
                    cfg=cfg,
                    path=cfg.checkpoint_dir / cfg.latest_checkpoint_name,
                    label="Latest checkpoint",
                )

            if processed_batches >= target_batches_per_epoch:
                break

        scheduler.step()

        if processed_batches == 0:
            raise RuntimeError(
                f"No batches were processed in epoch {epoch}. "
                "Check resume checkpoint metadata and dataloader settings."
            )

        avg_epoch_loss = epoch_loss / processed_batches
        log.info(f"─ Epoch {epoch:3d} complete | avg loss: {avg_epoch_loss:.4f}")
        if writer:
            writer.add_scalar("loss/epoch", avg_epoch_loss, epoch)
            writer.add_scalar("lr", optimiser.param_groups[0]["lr"], epoch)

        # Save periodic named checkpoint and refresh latest checkpoint at epoch boundary.
        save_checkpoint(
            epoch=epoch,
            batch_idx=0,
            global_step=global_step,
            backbone=backbone,
            head=head,
            optimiser=optimiser,
            scheduler=scheduler,
            loss=avg_epoch_loss,
            cfg=cfg,
        )
        save_checkpoint(
            epoch=epoch,
            batch_idx=0,
            global_step=global_step,
            backbone=backbone,
            head=head,
            optimiser=optimiser,
            scheduler=scheduler,
            loss=avg_epoch_loss,
            cfg=cfg,
            path=cfg.checkpoint_dir / cfg.latest_checkpoint_name,
            label="Latest checkpoint",
        )
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            log.info(f"  (periodic checkpoint at epoch {epoch})")


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
    parser.add_argument("--prefetch_factor", type=int,   default=cfg_defaults.prefetch_factor)
    parser.add_argument("--persistent_workers", action="store_true",
                        default=cfg_defaults.persistent_workers,
                        help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--no_persistent_workers", action="store_true",
                        help="Disable persistent DataLoader workers.")
    parser.add_argument("--seed",            type=int,   default=cfg_defaults.seed)
    parser.add_argument("--checkpoint_dir",  default=str(cfg_defaults.checkpoint_dir))
    parser.add_argument("--log_dir",         default=str(cfg_defaults.log_dir))
    parser.add_argument("--save_every",      type=int,   default=cfg_defaults.save_every)
    parser.add_argument("--log_every",       type=int,   default=cfg_defaults.log_every)
    parser.add_argument("--max_batches_per_epoch", type=int,
                        default=cfg_defaults.max_batches_per_epoch,
                        help="Cap training batches per epoch (0 uses full dataset).")
    parser.add_argument("--save_latest_every_batches", type=int,
                        default=cfg_defaults.save_latest_every_batches,
                        help="Overwrite latest checkpoint every N batches (0 disables).")
    parser.add_argument("--latest_checkpoint_name", default=cfg_defaults.latest_checkpoint_name)
    parser.add_argument("--use_cache",       action="store_true",
                        default=cfg_defaults.use_cache,
                        help="Enable HDF5 patch cache.")
    parser.add_argument("--no_cache",        action="store_true",
                        help="Disable HDF5 patch cache.")
    parser.add_argument("--resume_from",     default=None,
                        help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--resume_latest",   action="store_true",
                        help="Resume from latest checkpoint in checkpoint_dir.")
    parser.add_argument("--vm_safe",         action="store_true",
                        help="Apply a memory-safe profile for constrained VMs.")
    return parser.parse_args()


def _resolve_use_cache(args: argparse.Namespace, cfg_defaults: Config) -> bool:
    if args.no_cache:
        return False
    return bool(args.use_cache or cfg_defaults.use_cache)


def _resolve_persistent_workers(args: argparse.Namespace, cfg_defaults: Config) -> bool:
    if args.no_persistent_workers:
        return False
    return bool(args.persistent_workers or cfg_defaults.persistent_workers)


def _apply_vm_safe_profile(args: argparse.Namespace):
    if not args.vm_safe:
        return

    args.no_cache = True
    args.use_cache = False
    args.batch_size = min(args.batch_size, 16)
    args.num_workers = min(args.num_workers, 2)
    args.prefetch_factor = min(args.prefetch_factor, 2)
    args.persistent_workers = False
    args.no_persistent_workers = True
    args.log_every = max(args.log_every, 10)
    if args.save_latest_every_batches <= 0:
        args.save_latest_every_batches = 250

    log.info(
        "Applied --vm_safe profile | no_cache=True | batch_size=%s | num_workers=%s | "
        "prefetch_factor=%s | persistent_workers=False | log_every=%s",
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.log_every,
    )


if __name__ == "__main__":
    args = _parse_args()
    cfg_defaults = Config()
    _apply_vm_safe_profile(args)
    use_cache = _resolve_use_cache(args, cfg_defaults)
    persistent_workers = _resolve_persistent_workers(args, cfg_defaults)

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
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        max_batches_per_epoch=args.max_batches_per_epoch,
        save_latest_every_batches=args.save_latest_every_batches,
        latest_checkpoint_name=args.latest_checkpoint_name,
        use_cache=use_cache,
    )

    resume_from = _resolve_resume_path(
        resume_from=args.resume_from,
        resume_latest=args.resume_latest,
        checkpoint_dir=cfg.checkpoint_dir,
        latest_checkpoint_name=cfg.latest_checkpoint_name,
    )
    train(cfg, resume_from=resume_from)
