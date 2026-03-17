"""
config.py — Hyperparameters and paths for Pix2Rep-v2 SSL pretraining on MUMUCD PRISMA data.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: Path = Path("data/mumucd")          # folder with extracted PRISMA scenes
    cache_dir: Path = Path("data/cache")           # HDF5 patch cache (optional)
    use_cache: bool = False                        # set False to disable HDF5 caching
    patch_size: int = 128                          # spatial patch size (px)
    patch_stride: int = 128                        # non-overlapping → stride == patch_size
    n_bands: int = 230                             # total PRISMA bands
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
    batch_size: int = 64
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-4
    # cosine annealing restarts (set to epochs for a single cosine cycle)
    t_max: int = 30

    # ── Augmentation ──────────────────────────────────────────────────────────
    band_dropout_p: float = 0.10                   # fraction of bands zeroed per view
    brightness_range: float = 0.20                 # ±range for per-band brightness jitter
    contrast_range: float = 0.20                   # ±range for per-band contrast jitter
    noise_std: float = 0.02                        # additive Gaussian noise σ
    salt_pepper_p: float = 0.01                    # probability of salt & pepper noise per pixel
    spectral_scale_range: float = 0.20             # ±range for global spectral scaling
    random_erasing_p: float = 0.5                  # probability of spatial random erasing
    random_erasing_scale: tuple = (0.02, 0.1)      # fraction of image area to randomly erase

    # ── Logging / checkpoints ─────────────────────────────────────────────────
    log_dir: Path = Path("runs/pretrain")
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 1                            # save checkpoint every N epochs
    log_every: int = 10                            # log loss every N batches

    # ── Hardware ──────────────────────────────────────────────────────────────
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be > 0, got {self.patch_stride}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.n_pixels_M <= 0:
            raise ValueError(f"n_pixels_M must be > 0, got {self.n_pixels_M}")
        if self.proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {self.proj_dim}")

        # Keep correlation estimation in a sane regime.
        n_samples = self.batch_size * self.n_pixels_M
        if n_samples < self.proj_dim:
            raise ValueError(
                f"batch_size * n_pixels_M must be >= proj_dim "
                f"(got {n_samples} < {self.proj_dim})"
            )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
