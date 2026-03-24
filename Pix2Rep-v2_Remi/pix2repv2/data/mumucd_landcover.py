from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


DW_CLASS_NAMES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_PALETTE = np.array(
    [
        [0x41, 0x9B, 0xDF],
        [0x39, 0x7D, 0x49],
        [0x88, 0xB0, 0x53],
        [0x7A, 0x87, 0xC6],
        [0xE4, 0x96, 0x35],
        [0xDF, 0xC3, 0x5A],
        [0xC4, 0x28, 0x1B],
        [0xA5, 0x9B, 0x8F],
        [0xB3, 0x9F, 0xE1],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class MUMUCDLandCoverSample:
    city: str
    timepoint: str
    image_path: str
    label_path: str


@lru_cache(maxsize=32)
def _open_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def discover_city_names(data_folder_path: str) -> list[str]:
    return sorted(path.name for path in Path(data_folder_path).iterdir() if path.is_dir())


def build_landcover_samples(
    cities: list[str],
    timepoints: tuple[str, ...],
    data_folder_path: str = "/workspace/RemoteSensing/data/mumucd",
) -> list[MUMUCDLandCoverSample]:
    samples: list[MUMUCDLandCoverSample] = []
    for city in cities:
        city_dir = Path(data_folder_path) / city
        for timepoint in timepoints:
            image_path = city_dir / f"{city}-{timepoint}-prs.nc"
            label_path = city_dir / f"{city}-{timepoint}-dw.nc"
            if image_path.exists() and label_path.exists():
                samples.append(
                    MUMUCDLandCoverSample(
                        city=city,
                        timepoint=timepoint,
                        image_path=str(image_path),
                        label_path=str(label_path),
                    )
                )
    if not samples:
        raise FileNotFoundError("No supervised MUMUCD PRISMA/Dynamic-World samples found.")
    return samples


def compute_class_weights(
    samples: list[MUMUCDLandCoverSample],
    n_classes: int = 9,
) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for sample in samples:
        label_ds = _open_dataset(sample.label_path)
        labels = np.asarray(label_ds["lcc"].values, dtype=np.int64)
        labels = labels[(labels >= 0) & (labels < n_classes)]
        if labels.size == 0:
            continue
        bincount = np.bincount(labels.reshape(-1), minlength=n_classes)
        counts += bincount[:n_classes]

    total = counts.sum()
    if total <= 0:
        return torch.ones(n_classes, dtype=torch.float32)

    freqs = counts / total
    weights = 1.0 / np.sqrt(freqs + 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def _rescale_intensity(image: np.ndarray) -> np.ndarray:
    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


class MUMUCDLandCoverPatchDataset(Dataset):
    def __init__(
        self,
        samples: list[MUMUCDLandCoverSample],
        patch_size: int = 128,
        patches_per_image: int = 12,
        is_training: bool = True,
        seed: int = 42,
    ):
        self.samples = samples
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.is_training = is_training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _select_patch_origin(self, idx: int, image_h: int, image_w: int) -> tuple[int, int]:
        max_row = image_h - self.patch_size
        max_col = image_w - self.patch_size
        if self.is_training:
            row = np.random.randint(0, max_row + 1)
            col = np.random.randint(0, max_col + 1)
            return row, col

        rng = np.random.default_rng(self.seed + idx)
        row = int(rng.integers(0, max_row + 1))
        col = int(rng.integers(0, max_col + 1))
        return row, col

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx // self.patches_per_image]
        image_ds = _open_dataset(sample.image_path)
        label_ds = _open_dataset(sample.label_path)
        image_h = int(image_ds.sizes["nj"])
        image_w = int(image_ds.sizes["ni"])
        row, col = self._select_patch_origin(idx, image_h, image_w)

        image_patch = image_ds["sr"].isel(
            nj=slice(row, row + self.patch_size),
            ni=slice(col, col + self.patch_size),
        ).values
        label_patch = label_ds["lcc"].isel(
            nj=slice(row, row + self.patch_size),
            ni=slice(col, col + self.patch_size),
        ).values

        image_patch = _rescale_intensity(np.asarray(image_patch, dtype=np.float32))
        label_patch = np.clip(np.asarray(label_patch, dtype=np.int64), 0, 8)

        if self.is_training and np.random.random() < 0.5:
            image_patch = np.flip(image_patch, axis=0).copy()
            label_patch = np.flip(label_patch, axis=0).copy()
        if self.is_training and np.random.random() < 0.5:
            image_patch = np.flip(image_patch, axis=1).copy()
            label_patch = np.flip(label_patch, axis=1).copy()

        image_tensor = torch.from_numpy(
            rearrange(image_patch, "h w c -> c h w 1").astype(np.float32, copy=False)
        )
        label_tensor = torch.from_numpy(
            rearrange(label_patch, "h w -> 1 h w 1").astype(np.int64, copy=False)
        )
        return image_tensor, label_tensor


def build_landcover_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )


def load_center_crop(
    sample: MUMUCDLandCoverSample,
    crop_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    image_ds = _open_dataset(sample.image_path)
    label_ds = _open_dataset(sample.label_path)
    image_h = int(image_ds.sizes["nj"])
    image_w = int(image_ds.sizes["ni"])
    row = max(0, (image_h - crop_size) // 2)
    col = max(0, (image_w - crop_size) // 2)

    image_crop = image_ds["sr"].isel(
        nj=slice(row, row + crop_size),
        ni=slice(col, col + crop_size),
    ).values
    label_crop = label_ds["lcc"].isel(
        nj=slice(row, row + crop_size),
        ni=slice(col, col + crop_size),
    ).values

    image_crop = _rescale_intensity(np.asarray(image_crop, dtype=np.float32))
    label_crop = np.clip(np.asarray(label_crop, dtype=np.int64), 0, 8)
    return image_crop, label_crop


def filter_samples_by_metadata(
    samples: list[MUMUCDLandCoverSample],
    keep_keys: set[tuple[str, str]],
) -> list[MUMUCDLandCoverSample]:
    return [sample for sample in samples if (sample.city, sample.timepoint) in keep_keys]
