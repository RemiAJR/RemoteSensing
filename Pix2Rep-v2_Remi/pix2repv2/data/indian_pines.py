from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


INDIAN_PINES_CLASS_NAMES = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers",
]

INDIAN_PINES_PALETTE = np.array(
    [
        [0xE6, 0x19, 0x4B],
        [0x3C, 0xB4, 0x4B],
        [0xFF, 0xE1, 0x19],
        [0x00, 0x8A, 0xFF],
        [0xF5, 0x82, 0x31],
        [0x91, 0x1E, 0xB4],
        [0x46, 0xF0, 0xF0],
        [0xF0, 0x32, 0xE6],
        [0xD2, 0xF5, 0x3C],
        [0xFA, 0xBE, 0xBE],
        [0x00, 0x80, 0x80],
        [0xE6, 0xBE, 0xFF],
        [0xAA, 0x6E, 0x28],
        [0x80, 0x80, 0x00],
        [0x80, 0x80, 0x80],
        [0x00, 0x00, 0x80],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class IndianPinesSplit:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray


def _normalize_minmax(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    cube_min = float(cube.min())
    cube_max = float(cube.max())
    if cube_max - cube_min < 1e-8:
        return np.zeros_like(cube, dtype=np.float32)
    return (cube - cube_min) / (cube_max - cube_min)


def load_indian_pines(
    data_path: str,
    labels_path: str,
    data_key: str = "data",
    labels_key: str = "groundT",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cube = np.asarray(loadmat(data_path)[data_key], dtype=np.float32)
    labels = np.asarray(loadmat(labels_path)[labels_key], dtype=np.int64)
    if normalize:
        cube = _normalize_minmax(cube)
    return cube, labels


def compute_pca(
    cube: np.ndarray,
    num_components: int,
    whiten: bool = True,
) -> np.ndarray:
    if num_components <= 0 or num_components >= cube.shape[-1]:
        return cube.astype(np.float32, copy=False)

    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float64, copy=False)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:num_components]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    projected = centered @ eigvecs
    if whiten:
        projected = projected / np.sqrt(np.maximum(eigvals, 1e-8))
    return projected.reshape(cube.shape[0], cube.shape[1], num_components).astype(np.float32)


def pad_channels(cube: np.ndarray, target_channels: int) -> np.ndarray:
    if target_channels <= 0:
        return cube.astype(np.float32, copy=False)
    current_channels = int(cube.shape[-1])
    if current_channels == target_channels:
        return cube.astype(np.float32, copy=False)
    if current_channels > target_channels:
        raise ValueError(
            f"Cannot pad from {current_channels} to {target_channels}: target is smaller."
        )
    pad_width = target_channels - current_channels
    padding = np.zeros((*cube.shape[:2], pad_width), dtype=np.float32)
    return np.concatenate([cube.astype(np.float32, copy=False), padding], axis=-1)


def split_indices(
    labels_flat: np.ndarray,
    class_num: int,
    train_num: int,
    val_num: int,
    train_ratio: float,
    val_ratio: float,
    split_type: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_type not in {"number", "ratio"}:
        raise ValueError(f"Unknown split_type: {split_type}")

    rng = np.random.default_rng(seed)
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    test_index: list[np.ndarray] = []

    for class_idx in range(class_num):
        idx = np.where(labels_flat == class_idx + 1)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        if idx.size == 0:
            continue

        if split_type == "ratio":
            n_train = int(np.ceil(idx.size * train_ratio))
            n_val = int(np.ceil(idx.size * val_ratio))
        else:
            n_train = min(int(train_num), idx.size)
            n_val = min(int(val_num), max(idx.size - n_train, 0))

        train_index.append(idx[:n_train])
        val_index.append(idx[n_train : n_train + n_val])
        test_index.append(idx[n_train + n_val :])

    return (
        np.concatenate(train_index, axis=0),
        np.concatenate(val_index, axis=0),
        np.concatenate(test_index, axis=0),
    )


def build_split_masks(
    labels: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: np.ndarray,
) -> IndianPinesSplit:
    labels_flat = labels.reshape(-1)

    train_mask = np.zeros_like(labels_flat, dtype=np.int64)
    train_mask[train_index] = labels_flat[train_index]

    val_mask = np.zeros_like(labels_flat, dtype=np.int64)
    val_mask[val_index] = labels_flat[val_index]

    test_mask = np.zeros_like(labels_flat, dtype=np.int64)
    test_mask[test_index] = labels_flat[test_index]

    shape = labels.shape
    return IndianPinesSplit(
        train_mask=train_mask.reshape(shape),
        val_mask=val_mask.reshape(shape),
        test_mask=test_mask.reshape(shape),
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
    )


def _window_positions(
    image_shape: tuple[int, int],
    window_size: int,
    overlap_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    if window_size > height or window_size > width:
        raise ValueError(
            f"window_size={window_size} does not fit image_shape={image_shape}."
        )
    stride = window_size - overlap_size
    if stride <= 0:
        raise ValueError(
            f"overlap_size={overlap_size} must be smaller than window_size={window_size}."
        )

    y_end, x_end = np.subtract(image_shape, (window_size, window_size))
    x_positions = np.linspace(
        0,
        x_end,
        int(np.ceil(x_end / float(stride))) + 1,
        endpoint=True,
    ).astype(int)
    y_positions = np.linspace(
        0,
        y_end,
        int(np.ceil(y_end / float(stride))) + 1,
        endpoint=True,
    ).astype(int)
    return y_positions, x_positions


def build_sliding_windows(
    cube: np.ndarray,
    label_mask: np.ndarray,
    window_size: int,
    overlap_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_positions, x_positions = _window_positions(
        image_shape=(cube.shape[0], cube.shape[1]),
        window_size=window_size,
        overlap_size=overlap_size,
    )
    num_windows = len(y_positions) * len(x_positions)
    image_windows = np.zeros(
        (num_windows, window_size, window_size, cube.shape[-1]),
        dtype=np.float32,
    )
    label_windows = np.zeros((num_windows, window_size, window_size), dtype=np.int64)

    window_idx = 0
    for y in y_positions:
        for x in x_positions:
            image_windows[window_idx] = cube[y : y + window_size, x : x + window_size, :]
            label_windows[window_idx] = label_mask[y : y + window_size, x : x + window_size]
            window_idx += 1

    return image_windows, label_windows


def combine_patch_predictions(
    patch_predictions: np.ndarray,
    image_shape: tuple[int, int],
    window_size: int,
    overlap_size: int,
) -> np.ndarray:
    y_positions, x_positions = _window_positions(image_shape, window_size, overlap_size)
    combined = np.zeros(image_shape, dtype=np.int64)
    overlap = overlap_size // 2

    patch_idx = 0
    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            patch = patch_predictions[patch_idx]
            patch_idx += 1
            if y_idx == 0 and x_idx == 0:
                combined[y : y + window_size, x : x + window_size] = patch
            elif y_idx == 0 and x_idx > 0:
                combined[y : y + window_size, x + overlap : x + window_size] = patch[:, overlap:]
            elif y_idx > 0 and x_idx == 0:
                combined[y + overlap : y + window_size, x : x + window_size] = patch[overlap:, :]
            else:
                combined[y + overlap : y + window_size, x + overlap : x + window_size] = patch[
                    overlap:,
                    overlap:,
                ]
    return combined


def compute_class_weights(label_mask: np.ndarray, n_classes: int) -> torch.Tensor:
    labels = label_mask[label_mask > 0] - 1
    if labels.size == 0:
        return torch.ones(n_classes, dtype=torch.float32)

    counts = np.bincount(labels.reshape(-1), minlength=n_classes).astype(np.float64)
    freqs = counts / counts.sum()
    weights = 1.0 / np.sqrt(freqs + 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_confusion_matrix(
    prediction_map: np.ndarray,
    target_map: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    valid = target_map > 0
    if not np.any(valid):
        return np.zeros((n_classes, n_classes), dtype=np.int64)

    flat_targets = target_map[valid].reshape(-1) - 1
    flat_preds = prediction_map[valid].reshape(-1) - 1
    encoded = flat_targets * n_classes + flat_preds
    counts = np.bincount(encoded, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(np.int64)


def compute_metrics_from_confusion_matrix(confusion_matrix: np.ndarray) -> dict[str, object]:
    conf = confusion_matrix.astype(np.float64, copy=False)
    total = conf.sum()
    correct = np.trace(conf)
    oa = float(correct / total) if total > 0 else 0.0

    per_class_total = conf.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(conf),
        np.maximum(per_class_total, 1.0),
        out=np.zeros(conf.shape[0], dtype=np.float64),
        where=per_class_total > 0,
    )
    aa = float(per_class_acc.mean()) if per_class_acc.size > 0 else 0.0

    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    pe = float((row_sums * col_sums).sum() / (total * total)) if total > 0 else 0.0
    if total == 0 or abs(1.0 - pe) < 1e-12:
        kappa = 0.0
    else:
        kappa = float((oa - pe) / (1.0 - pe))

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_acc": {
            class_name: float(per_class_acc[idx])
            for idx, class_name in enumerate(INDIAN_PINES_CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    table = np.vstack([np.zeros((1, 3), dtype=np.uint8), INDIAN_PINES_PALETTE])
    return table[np.clip(label_map, 0, len(table) - 1)]


class IndianPinesPatchDataset(Dataset):
    def __init__(self, image_patches: np.ndarray, label_patches: np.ndarray):
        self.image_patches = np.asarray(image_patches, dtype=np.float32)
        self.label_patches = np.asarray(label_patches, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.image_patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_patch = torch.from_numpy(
            np.transpose(self.image_patches[index], (2, 0, 1))[..., None].astype(np.float32)
        )
        label_patch = torch.from_numpy(self.label_patches[index][None, ..., None].astype(np.int64))
        return image_patch, label_patch


def build_loader(
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
