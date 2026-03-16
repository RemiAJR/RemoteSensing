"""
config.py — Hyperparameters and paths for Pix2Rep-v2 SSL pretraining on MUMUCD PRISMA data.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: Path = Path("data/mumucd")          # folder with extracted PRISMA GeoTIFFs
    cache_dir: Path = Path("data/cache")           # HDF5 patch cache (optional)
    patch_size: int = 128                          # spatial patch size (px)
    patch_stride: int = 128                        # non-overlapping → stride == patch_size
    n_bands: int = 239                             # total PRISMA bands (66 VNIR + 173 SWIR)
    # If < n_bands, a 1×1 conv spectral-reduction layer is used in the encoder
    spectral_reduced: int = 64                     # set to 0 to skip reduction layer

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim: int = 1024                          # U-Net output channels
    proj_dim: int = 256                            # projection head output channels
    proj_hidden_dim: int = 512                     # projection head hidden channels

    # ── SSL loss (Barlow Twins) ────────────────────────────────────────────────
    n_pixels_M: int = 1000                         # sampled pixels per image per view
    lambda_barlow: float = 5e-3                    # off-diagonal regularisation weight

    # ── Optimisation ──────────────────────────────────────────────────────────
    batch_size: int = 8
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-4
    # cosine annealing restarts (set to epochs for a single cosine cycle)
    t_max: int = 200

    # ── Augmentation ──────────────────────────────────────────────────────────
    band_dropout_p: float = 0.10                   # fraction of bands zeroed per view
    brightness_range: float = 0.20                 # ±range for per-band brightness jitter
    contrast_range: float = 0.20                   # ±range for per-band contrast jitter
    noise_std: float = 0.02                        # additive Gaussian noise σ

    # ── Logging / checkpoints ─────────────────────────────────────────────────
    log_dir: Path = Path("runs/pretrain")
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 10                           # save checkpoint every N epochs
    log_every: int = 50                            # log loss every N batches

    # ── Hardware ──────────────────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
