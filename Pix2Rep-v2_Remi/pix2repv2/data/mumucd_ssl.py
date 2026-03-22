import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from pix2repv2.utils import utils


def _rescale_intensity(patch: np.ndarray, Imin_range, Imax_range, percentiles):
    """Rescale intensity to random [Imin, Imax] using percentile clipping."""
    Imin = np.random.uniform(*Imin_range)
    Imax = np.random.uniform(*Imax_range)
    if Imin >= Imax:
        return patch
    lo = np.percentile(patch, percentiles[0])
    hi = np.percentile(patch, percentiles[1])
    if hi - lo < 1e-8:
        return patch
    out = (patch - lo) / (hi - lo)
    out = out * (Imax - Imin) + Imin
    return np.clip(out, Imin, Imax, out=out)


def _random_gamma(patch: np.ndarray, log_gamma: float, p: float):
    """Apply random gamma correction."""
    if np.random.random() > p:
        return patch
    gamma = np.exp(np.random.uniform(-log_gamma, log_gamma))
    mn = patch.min()
    mx = patch.max()
    if mx - mn < 1e-8:
        return patch
    normed = (patch - mn) / (mx - mn)
    return mn + (mx - mn) * np.power(normed, gamma)


def _augment_patch(patch_np: np.ndarray, cfg_transform) -> np.ndarray:
    """Apply augmentations in numpy (no torchio overhead)."""
    patch_np = _rescale_intensity(
        patch_np,
        Imin_range=(0.0, 0.3),
        Imax_range=(0.7, 1.0),
        percentiles=(1, 99),
    )
    patch_np = _random_gamma(
        patch_np,
        log_gamma=cfg_transform.log_gamma,
        p=cfg_transform.random_gamma_p,
    )
    return patch_np


def _load_all_cubes(paths: list[str]) -> list[np.ndarray]:
    """Load all city cubes into RAM at startup."""
    cubes = []
    t0 = time.time()
    for i, p in enumerate(paths):
        cube = np.load(p)
        cubes.append(cube)
        if (i + 1) % 10 == 0 or i == len(paths) - 1:
            elapsed = time.time() - t0
            logger.info(
                f"  Loaded {i+1}/{len(paths)} cubes "
                f"({sum(c.nbytes for c in cubes)/1e9:.1f} GB, {elapsed:.0f}s)"
            )
    return cubes


class MUMUCD_PatchSSL(Dataset):
    """Patch-based SSL dataset for MUMUCD PRISMA hyperspectral images.

    All city cubes are pre-loaded into RAM at init for instant patch access.
    """

    def __init__(
        self,
        cfg: dict,
        image_paths: list[str] | None = None,
        cubes: list[np.ndarray] | None = None,
        data_folder_path: str = "/workspace/RemoteSensing/data/mumucd_npy",
        patches_per_image: int = 100,
        patch_size: int = 128,
        apply_augmentations: bool = True,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.apply_augmentations = apply_augmentations
        self.cfg_transform = self.cfg.pretraining.pretraining_transform

        if image_paths is not None:
            self.image_paths = list(image_paths)
        else:
            self.image_paths = sorted(
                glob.glob(os.path.join(data_folder_path, "*.npy"))
            )
        assert len(self.image_paths) > 0, (
            f"No .npy files found in {data_folder_path}/. "
            "Run the NetCDF-to-npy conversion script first."
        )

        # Load all cubes into RAM (or reuse already-loaded cubes)
        if cubes is not None:
            self.cubes = cubes
        else:
            logger.info(f"Loading {len(self.image_paths)} cubes into RAM...")
            self.cubes = _load_all_cubes(self.image_paths)

        total_gb = sum(c.nbytes for c in self.cubes) / 1e9
        logger.info(
            f"MUMUCD PRISMA: {len(self.cubes)} cities in RAM ({total_gb:.1f} GB), "
            f"{patches_per_image} patches each -> {len(self)} samples/epoch"
        )

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def _sample_patch(self, city_idx: int) -> np.ndarray:
        cube = self.cubes[city_idx]
        img_h, img_w, _ = cube.shape
        ps = self.patch_size
        row = np.random.randint(0, img_h - ps + 1)
        col = np.random.randint(0, img_w - ps + 1)
        return cube[row : row + ps, col : col + ps, :].copy()

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        city_idx = idx // self.patches_per_image
        patch_np = self._sample_patch(city_idx)

        if self.apply_augmentations:
            view1_np = _augment_patch(patch_np.copy(), self.cfg_transform)
            view2_np = _augment_patch(patch_np.copy(), self.cfg_transform)
        else:
            view1_np = _rescale_intensity(patch_np.copy(), (0.0, 0.3), (0.7, 1.0), (1, 99))
            view2_np = _rescale_intensity(patch_np.copy(), (0.0, 0.3), (0.7, 1.0), (1, 99))

        view1 = torch.from_numpy(rearrange(view1_np, "H W C -> C H W 1").astype(np.float32))
        view2 = torch.from_numpy(rearrange(view2_np, "H W C -> C H W 1").astype(np.float32))

        affine_matrix = utils.generate_single_affine_spatial_transform(
            is_rotated=self.cfg.pretraining.is_rotated,
            is_cropped=self.cfg.pretraining.is_cropped,
            is_flipped=self.cfg.pretraining.is_flipped,
            is_translation=self.cfg.pretraining.is_translation,
            max_angle=self.cfg.pretraining.max_angle,
            max_crop=self.cfg.pretraining.max_crop,
        )
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0),
            [1, view1.shape[0], view1.shape[1], view1.shape[2]],
            align_corners=True,
        )

        return view1, view2, grid


def create_train_val_subsets(
    patch_dataset: MUMUCD_PatchSSL,
    cfg: dict = None,
) -> tuple[MUMUCD_PatchSSL, MUMUCD_PatchSSL]:
    """Split by city. Reuses already-loaded cubes to avoid double loading."""
    all_paths = list(patch_dataset.image_paths)
    all_cubes = list(patch_dataset.cubes)

    # Shuffle cities deterministically
    rng = np.random.RandomState(cfg.data.random_seed)
    indices = list(range(len(all_paths)))
    rng.shuffle(indices)

    n_val = max(1, int(round(cfg.data.val_ratio * len(all_paths))))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_paths = [all_paths[i] for i in train_idx]
    train_cubes = [all_cubes[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_cubes = [all_cubes[i] for i in val_idx]

    logger.info(f"Train: {len(train_paths)} cities, Val: {len(val_paths)} cities")

    train_ds = MUMUCD_PatchSSL(
        cfg=cfg,
        image_paths=train_paths,
        cubes=train_cubes,
        patches_per_image=patch_dataset.patches_per_image,
        patch_size=patch_dataset.patch_size,
        apply_augmentations=True,
    )
    val_ds = MUMUCD_PatchSSL(
        cfg=cfg,
        image_paths=val_paths,
        cubes=val_cubes,
        patches_per_image=patch_dataset.patches_per_image,
        patch_size=patch_dataset.patch_size,
        apply_augmentations=False,
    )
    return train_ds, val_ds


def build_patch_loader(
    patch_dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    num_workers = 8
    logger.info(f"Using num_workers = {num_workers}")

    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
