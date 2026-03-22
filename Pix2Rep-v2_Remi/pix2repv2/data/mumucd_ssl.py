import glob
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from netCDF4 import Dataset as NetCDFDataset
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split

from pix2repv2.utils import utils
from pix2repv2.utils.augmentations import RandomRescaleIntensity


@lru_cache(maxsize=70)
def _get_prisma_variable(path: str):
    ds = NetCDFDataset(path, mode="r")
    ds.set_auto_mask(False)
    return ds, ds.variables["sr"]


@lru_cache(maxsize=70)
def _load_prisma_shape(path: str) -> tuple[int, int, int]:
    _, sr = _get_prisma_variable(path)
    return tuple(sr.shape)


def _load_prisma_patch(path: str, row: int, col: int, patch_size: int) -> np.ndarray:
    _, sr = _get_prisma_variable(path)
    patch = sr[row : row + patch_size, col : col + patch_size, :]
    return np.asarray(patch, dtype=np.float32)


class MUMUCD_PatchSSL(Dataset):
    """Patch-based SSL dataset for MUMUCD PRISMA hyperspectral images.

    Fixed-size 128x128 patches are sampled on the fly at each __getitem__ call,
    which re-randomizes the patch distribution across epochs without introducing
    zoom-based scale augmentation.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = "/workspace/RemoteSensing/data/mumucd",
        patches_per_image: int = 100,
        patch_size: int = 128,
        apply_augmentations: bool = True,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.apply_augmentations = apply_augmentations
        self._cached_path = None
        self._cached_sr = None
        self.image_paths: list[str] = sorted(
            glob.glob(os.path.join(data_folder_path, "*", "*-before-prs.nc"))
        )
        assert len(self.image_paths) > 0, (
            f"No *-before-prs.nc files found in {data_folder_path}/*/. "
            "Make sure city subdirectories are extracted."
        )
        logger.info(
            f"MUMUCD PRISMA: found {len(self.image_paths)} cities, "
            f"{patches_per_image} patches each -> {len(self)} samples/epoch"
        )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )

        cfg_transform = self.cfg.pretraining.pretraining_transform
        self.pretraining_transform = tio.Compose(
            [
                RandomRescaleIntensity(
                    Imin_range=(0.0, 0.3),
                    Imax_range=(0.7, 1.0),
                    percentiles=(1, 99),
                ),
                tio.RandomGamma(
                    log_gamma=cfg_transform.log_gamma,
                    p=cfg_transform.random_gamma_p,
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def _load_city_cube(self, path: str) -> np.ndarray:
        if self._cached_path != path or self._cached_sr is None:
            _, sr = _get_prisma_variable(path)
            self._cached_sr = np.asarray(sr[:, :, :], dtype=np.float32)
            self._cached_path = path
        return self._cached_sr

    def _sample_patch(self, path: str) -> torch.Tensor:
        sr = self._load_city_cube(path)
        img_h, img_w, _ = sr.shape
        ps = self.patch_size
        row = np.random.randint(0, img_h - ps + 1)
        col = np.random.randint(0, img_w - ps + 1)
        patch_np = sr[row : row + ps, col : col + ps, :]

        return torch.tensor(
            rearrange(patch_np, "H W C -> C H W 1"),
            dtype=torch.float32,
        )

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        city_idx = idx // self.patches_per_image
        path = self.image_paths[city_idx]
        patch_tensor = self._sample_patch(path)

        if self.apply_augmentations:
            view1 = self.pretraining_transform(patch_tensor)
            view2 = self.pretraining_transform(patch_tensor)
        else:
            view1 = self.intensity_rescaler(patch_tensor)
            view2 = self.intensity_rescaler(patch_tensor)

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
            rearrange(view1, "C W H D -> D C W H").shape,
            align_corners=True,
        )

        return view1, view2, grid


class ValidationWrapper(torch.utils.data.Dataset):
    """Disable photometric augmentations for validation items only."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        ds = self.subset.dataset
        old = ds.apply_augmentations
        ds.apply_augmentations = False
        item = ds[self.subset.indices[idx]]
        ds.apply_augmentations = old
        return item


def create_train_val_subsets(
    patch_dataset: torch.utils.data.Dataset,
    cfg: dict = None,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    total_samples = len(patch_dataset)
    val_samples = max(1, int(round(cfg.data.val_ratio * total_samples)))
    train_samples = total_samples - val_samples

    generator = torch.Generator().manual_seed(cfg.data.random_seed)
    train_subset, val_subset = random_split(
        dataset=patch_dataset,
        lengths=[train_samples, val_samples],
        generator=generator,
    )
    val_subset = ValidationWrapper(val_subset)
    return train_subset, val_subset


def build_patch_loader(
    patch_dataset: torch.utils.data.Dataset
    | torch.utils.data.Subset
    | torch.utils.data.ConcatDataset,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    num_workers = 0
    logger.info(f"Using num_workers = {num_workers}")

    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
