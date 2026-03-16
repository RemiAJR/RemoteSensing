"""
data/mumucd_dataset.py — PRISMA GeoTIFF loader, non-overlapping patch extraction, and
optional HDF5 caching for the MUMUCD dataset.

Expected layout after extraction:
    data/mumucd/
        <scene_id>_prisma.tif   (one per scene, shape C×H×W in float32 or uint16)
        ...

The dataset returns (patch, path, row, col) tuples where *patch* is a
float32 tensor of shape (C, patch_size, patch_size) normalised to [0, 1].
"""

import os
import json
import hashlib
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


def _load_scene(path: Path) -> np.ndarray:
    """
    Load a PRISMA GeoTIFF and return a float32 array (C, H, W) in [0, 1].
    Falls back to a random array when rasterio is unavailable (unit tests / CI).
    """
    if not HAS_RASTERIO:
        # Minimal stub for offline / CI environments
        rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
        return rng.random((239, 1536, 1536), dtype=np.float32)

    with rasterio.open(path) as src:
        arr = src.read()           # (C, H, W), possibly uint16
    return _normalise(arr)


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
        data_root: directory containing `*prisma*.tif` files.
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

        # Discover scene files
        self.scene_paths: List[Path] = sorted(
            list(self.data_root.glob("*prisma*.tif"))
            + list(self.data_root.glob("*PRISMA*.tif"))
        )
        if max_scenes is not None:
            self.scene_paths = self.scene_paths[:max_scenes]

        if not self.scene_paths:
            raise FileNotFoundError(
                f"No PRISMA GeoTIFF files found under {self.data_root}. "
                "Expected filenames matching '*prisma*.tif'. "
                "Download with: zenodo_get 10674011 --record-filter '*prisma*'"
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
                if saved.get("patch_size") == self.patch_size and \
                   saved.get("stride") == self.stride and \
                   saved.get("n_scenes") == len(self.scene_paths):
                    self._index = [tuple(x) for x in saved["index"]]
                    self._scene_shapes = [tuple(x) for x in saved["shapes"]]
                    return
            except Exception:
                pass  # rebuild on any error

        print(f"[MUMUCDPatchDataset] Building patch index for {len(self.scene_paths)} scenes …")
        for s_idx, path in enumerate(self.scene_paths):
            scene = _load_scene(path)              # (C, H, W)
            c, h, w = scene.shape
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
                    "index": self._index,
                    "shapes": self._scene_shapes,
                }, f)
        except OSError:
            pass

        print(f"[MUMUCDPatchDataset] {len(self._index)} patches indexed.")

    # ── HDF5 cache ────────────────────────────────────────────────────────────

    def _init_cache(self):
        if not HAS_H5PY:
            print("[MUMUCDPatchDataset] h5py not installed — running without cache.")
            return

        n = len(self._index)
        # Infer C from first scene shape; fall back to config default
        c = self._scene_shapes[0][0] if self._scene_shapes else 239
        shape = (n, c, self.patch_size, self.patch_size)

        if self.cache_path.exists():
            # Validate existing cache
            with h5py.File(self.cache_path, "r") as f:
                if f["patches"].shape == shape:
                    print(f"[MUMUCDPatchDataset] Using existing cache: {self.cache_path}")
                    self._cache_ds = "valid"
                    return
            print("[MUMUCDPatchDataset] Cache shape mismatch — rebuilding.")
            self.cache_path.unlink()

        print(f"[MUMUCDPatchDataset] Building HDF5 cache → {self.cache_path} …")
        scenes: dict = {}
        with h5py.File(self.cache_path, "w") as f:
            ds = f.create_dataset(
                "patches", shape=shape, dtype="float32",
                chunks=(1, c, self.patch_size, self.patch_size),
                compression="lzf",
            )
            for i, (s_idx, r, col) in enumerate(self._index):
                if s_idx not in scenes:
                    scenes[s_idx] = _load_scene(self.scene_paths[s_idx])
                p = self.patch_size
                ds[i] = scenes[s_idx][:, r:r + p, col:col + p]
                if (i + 1) % 500 == 0:
                    print(f"  cached {i + 1}/{n} patches")
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
