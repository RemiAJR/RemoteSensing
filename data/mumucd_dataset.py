"""
data/mumucd_dataset.py — PRISMA scene loader, non-overlapping patch extraction, and
optional HDF5 caching for the MUMUCD dataset.

Expected layout after extraction:
    data/mumucd/
        <scene_id>/*prs*.nc     (MUMUCD NetCDF)
        OR
        <scene_id>/*prisma*.tif (GeoTIFF)
        ...

The dataset returns patch tensors of shape (C, patch_size, patch_size) in [0, 1].
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ─────────────────────────────── helpers ──────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    """Per-band min-max normalisation to [0, 1], robust to constant bands."""
    arr = arr.astype(np.float32)
    mn = arr.min(axis=(-2, -1), keepdims=True)
    mx = arr.max(axis=(-2, -1), keepdims=True)
    denom = np.where(mx - mn > 1e-8, mx - mn, 1.0)
    return (arr - mn) / denom


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert a 3D array to (C, H, W), inferring channel axis as the smallest axis.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    channel_axis = int(np.argmin(arr.shape))
    if channel_axis == 0:
        return arr
    if channel_axis == 1:
        return np.transpose(arr, (1, 0, 2))
    return np.transpose(arr, (2, 0, 1))


def _shape_to_chw(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert a 3D shape tuple to (C, H, W) using smallest-axis-as-channel."""
    channel_axis = int(np.argmin(shape))
    if channel_axis == 0:
        return shape[0], shape[1], shape[2]
    if channel_axis == 1:
        return shape[1], shape[0], shape[2]
    return shape[2], shape[0], shape[1]


def _scene_shape(path: Path) -> Tuple[int, int, int]:
    """
    Return scene shape as (C, H, W) without loading full scene values.
    """
    suffix = path.suffix.lower()
    if suffix in {".nc", ".h5", ".hdf5"}:
        if not HAS_H5PY:
            raise ImportError("h5py is required to read NetCDF/HDF5 PRISMA scenes.")
        with h5py.File(path, "r") as f:
            if "sr" not in f:
                raise KeyError(f"Missing 'sr' dataset in {path}")
            shape = tuple(f["sr"].shape)
        if len(shape) != 3:
            raise ValueError(f"'sr' dataset in {path} must be 3D, got {shape}")
        return _shape_to_chw(shape)

    if suffix in {".tif", ".tiff"}:
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to read GeoTIFF PRISMA scenes.")
        with rasterio.open(path) as src:
            return src.count, src.height, src.width

    raise ValueError(f"Unsupported scene format for {path}. Expected .nc/.h5/.tif")


@lru_cache(maxsize=5)
def _load_scene(path: Path) -> np.ndarray:
    """
    Load a PRISMA scene and return a float32 array (C, H, W) in [0, 1].
    Uses LRU cache to avoid redundant disk I/O when extracting multiple patches.
    """
    suffix = path.suffix.lower()
    if suffix in {".nc", ".h5", ".hdf5"}:
        if not HAS_H5PY:
            raise ImportError("h5py is required to read NetCDF/HDF5 PRISMA scenes.")
        with h5py.File(path, "r") as f:
            if "sr" not in f:
                raise KeyError(f"Missing 'sr' dataset in {path}")
            arr = f["sr"][()]
    elif suffix in {".tif", ".tiff"}:
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to read GeoTIFF PRISMA scenes.")
        with rasterio.open(path) as src:
            arr = src.read()
    else:
        raise ValueError(f"Unsupported scene format for {path}. Expected .nc/.h5/.tif")

    return _normalise(_to_chw(arr))


def _patch_indices(h: int, w: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Return (row, col) top-left corners for all valid non-overlapping patches."""
    rows = range(0, h - patch_size + 1, stride)
    cols = range(0, w - patch_size + 1, stride)
    return [(r, c) for r in rows for c in cols]


# ────────────────────────────── main dataset ──────────────────────────────────

class MUMUCDPatchDataset(Dataset):
    """
    Iterates over 128×128 non-overlapping patches extracted from MUMUCD PRISMA scenes.

    Args:
        data_root: directory containing scenes.
        patch_size: spatial size of each patch (default 128).
        stride: stride for patch extraction (default == patch_size → non-overlapping).
        cache_path: if set, patches are cached to an HDF5 file on first call.
        transform: optional callable applied to the patch tensor (e.g. augmentations).
        max_scenes: if set, load only this many scenes (useful for quick tests).
    """

    def __init__(
        self,
        data_root: str | Path,
        patch_size: int = 128,
        stride: Optional[int] = None,
        cache_path: Optional[str | Path] = None,
        transform=None,
        max_scenes: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.transform = transform

        # Discover PRISMA scenes (NetCDF and GeoTIFF support).
        scene_candidates = set()
        for pattern in ("*prs*.nc", "*prs*.tif", "*prs*.tiff", "*prisma*.tif", "*prisma*.tiff"):
            scene_candidates.update(self.data_root.rglob(pattern))
        self.scene_paths = sorted(scene_candidates)
        if max_scenes is not None:
            self.scene_paths = self.scene_paths[:max_scenes]

        if not self.scene_paths:
            raise FileNotFoundError(
                f"No PRISMA scenes found under {self.data_root}. "
                "Expected files matching '*prs*.nc' or '*prisma*.tif'."
            )

        # Build the index: list of (scene_idx, row, col)
        self._index: List[Tuple[int, int, int]] = []
        self._scene_shapes: List[Tuple[int, int, int]] = []  # (C, H, W) per scene

        self._build_index()

        # Optionally build/load HDF5 cache
        self._cache_ds = None
        if self.cache_path is not None:
            self._init_cache()

    # ── Index ──────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Probe each scene file to record its shape, then enumerate patch coords."""
        meta_cache = self.data_root / ".patch_index.json"

        # Re-use saved index if dataset hasn't changed
        if meta_cache.exists():
            try:
                with open(meta_cache) as f:
                    saved = json.load(f)
                current_scene_paths = [str(p.relative_to(self.data_root)) for p in self.scene_paths]
                if saved.get("patch_size") == self.patch_size and \
                   saved.get("stride") == self.stride and \
                   saved.get("n_scenes") == len(self.scene_paths) and \
                   saved.get("scene_paths") == current_scene_paths:
                    self._index = [tuple(x) for x in saved["index"]]
                    self._scene_shapes = [tuple(x) for x in saved["shapes"]]
                    return
            except Exception:
                pass  # rebuild on any error

        print(f"[MUMUCDPatchDataset] Building patch index for {len(self.scene_paths)} scenes …")
        for s_idx, path in enumerate(self.scene_paths):
            c, h, w = _scene_shape(path)
            if self._scene_shapes and c != self._scene_shapes[0][0]:
                raise ValueError(
                    f"Inconsistent channel count across scenes: {path} has {c}, "
                    f"expected {self._scene_shapes[0][0]}"
                )
            self._scene_shapes.append((c, h, w))
            for r, col in _patch_indices(h, w, self.patch_size, self.stride):
                self._index.append((s_idx, r, col))

        # Persist index
        try:
            with open(meta_cache, "w") as f:
                json.dump({
                    "patch_size": self.patch_size,
                    "stride": self.stride,
                    "n_scenes": len(self.scene_paths),
                    "scene_paths": [str(p.relative_to(self.data_root)) for p in self.scene_paths],
                    "index": self._index,
                    "shapes": self._scene_shapes,
                }, f)
        except OSError:
            pass

        print(f"[MUMUCDPatchDataset] {len(self._index)} patches indexed.")

    # ── HDF5 cache ────────────────────────────────────────────────────────────

    def _init_cache(self):
        if not HAS_H5PY:
            raise ImportError("h5py is required when cache_path is set.")

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self._index)
        # Infer C from first scene shape; fall back to config default
        c = self._scene_shapes[0][0] if self._scene_shapes else 230
        shape = (n, c, self.patch_size, self.patch_size)

        if self.cache_path.exists():
            # Validate existing cache
            try:
                with h5py.File(self.cache_path, "r") as f:
                    if "patches" in f and f["patches"].shape == shape:
                        print(f"[MUMUCDPatchDataset] Using existing cache: {self.cache_path}")
                        self._cache_ds = "valid"
                        return
                    if "patches" not in f:
                        print("[MUMUCDPatchDataset] Cache missing 'patches' dataset — rebuilding.")
                    else:
                        print("[MUMUCDPatchDataset] Cache shape mismatch — rebuilding.")
            except (OSError, ValueError) as exc:
                print(f"[MUMUCDPatchDataset] Cache unreadable ({exc}) — rebuilding.")

            try:
                self.cache_path.unlink()
            except OSError as exc:
                raise OSError(
                    f"Failed to remove invalid cache file {self.cache_path}: {exc}"
                ) from exc

        print(f"[MUMUCDPatchDataset] Building HDF5 cache → {self.cache_path} …")
        try:
            with h5py.File(self.cache_path, "w") as f:
                ds = f.create_dataset(
                    "patches", shape=shape, dtype="float32",
                    chunks=(1, c, self.patch_size, self.patch_size),
                    compression="lzf",
                )
                current_scene_idx = None
                current_scene = None
                for i, (s_idx, r, col) in enumerate(self._index):
                    if s_idx != current_scene_idx:
                        current_scene = _load_scene(self.scene_paths[s_idx])
                        current_scene_idx = s_idx
                    p = self.patch_size
                    ds[i] = current_scene[:, r:r + p, col:col + p]
                    if (i + 1) % 500 == 0:
                        print(f"  cached {i + 1}/{n} patches")
        except (OSError, RuntimeError) as exc:
            try:
                if self.cache_path.exists():
                    self.cache_path.unlink()
            except OSError as cleanup_exc:
                raise OSError(
                    f"Failed while building cache {self.cache_path}: {exc}. "
                    f"Also failed to remove partial cache file: {cleanup_exc}"
                ) from cleanup_exc
            raise OSError(
                f"Failed while building cache {self.cache_path}: {exc}. "
                "Free disk space or disable cache."
            ) from exc
        self._cache_ds = "valid"
        print("[MUMUCDPatchDataset] Cache built.")

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s_idx, r, c = self._index[idx]
        p = self.patch_size

        if self._cache_ds == "valid" and self.cache_path is not None and HAS_H5PY:
            with h5py.File(self.cache_path, "r") as f:
                patch = f["patches"][idx]           # float32 numpy
        else:
            scene = _load_scene(self.scene_paths[s_idx])
            patch = scene[:, r:r + p, c:c + p]

        patch = torch.from_numpy(patch.copy())      # (C, H, W) float32

        if self.transform is not None:
            patch = self.transform(patch)

        return patch
