"""PyTorch Dataset for loading cell crops from the pipeline's HDF5 output.

The dataset is a pure reader: it returns crops + metadata. Augmentation
runs as a batched GPU op in ``train.py:_augment_batch_gpu``.

Crops can be loaded fully into RAM at construction time (the default when
the data fits within ``MAX_CACHE_BYTES``). This avoids the random-access
penalty of HDF5 chunks: per-row chunking would be 16 MB per chunk for
256-row chunks of 1×128×128 floats, so each shuffled read pulls in 256×
more data than it uses. With the cache, ``__getitem__`` is just a numpy
slice and the GPU stays fed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Module-level cache so train_ds and val_ds (different instances over the
# same file) share one in-memory copy — keyed by (path, channels, size).
_CROP_CACHE: dict[tuple, np.ndarray] = {}

# Hard ceiling on how much RAM the cache may use. Anything above this and
# the dataset falls back to streaming from HDF5 — slower, but won't OOM
# on huge libraries. Override via the ``MYCOPREP_CROP_CACHE_BYTES`` env var.
_DEFAULT_MAX_CACHE_BYTES = 12 * 1024**3  # 12 GB


def _max_cache_bytes() -> int:
    raw = os.environ.get("MYCOPREP_CROP_CACHE_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return _DEFAULT_MAX_CACHE_BYTES


class HDF5CropDataset(Dataset):
    """Reads cell crops from ``all_crops.h5`` files (RAM-cached when small).

    Selects imaging channels (dropping mask) and resizes legacy 96px crops
    to the target size. Augmentation is the training loop's responsibility.
    """

    def __init__(
        self,
        h5_paths: Sequence[Path],
        *,
        target_size: int = 128,
        in_channels: int = 1,
        # Kept for API compatibility — ignored. Augmentation is done on the
        # GPU per batch in train.py for efficiency.
        augment: bool = False,
        rotation: bool = True,
        flip: bool = True,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
        gaussian_blur_sigma: float = 1.5,
        indices: Optional[np.ndarray] = None,
        cache_in_memory: Optional[bool] = None,
    ) -> None:
        import h5py

        self.target_size = target_size
        self.in_channels = in_channels

        # Build an index mapping global_idx → (file_idx, local_idx)
        self._h5_paths = [Path(p) for p in h5_paths]
        self._file_offsets: list[int] = []
        self._file_lengths: list[int] = []
        self._channel_indices: list[list[int]] = []
        self._crop_sizes: list[int] = []

        total = 0
        total_bytes = 0
        for path in self._h5_paths:
            with h5py.File(str(path), "r") as f:
                n = f["crops"].shape[0]
                crop_sz = int(f.attrs.get("crop_size", f["crops"].shape[-1]))
                channel_names = list(f.attrs.get("channel_names", []))

                # Select imaging channels (drop mask)
                ch_idx = self._select_channels(channel_names, in_channels)

                self._file_offsets.append(total)
                self._file_lengths.append(n)
                self._channel_indices.append(ch_idx)
                self._crop_sizes.append(crop_sz)
                total += n
                total_bytes += n * len(ch_idx) * target_size * target_size * 4

        self._total = total

        # Optional subset (for train/val splits)
        if indices is not None:
            self._indices = indices
        else:
            self._indices = np.arange(total)

        # Auto-decide caching: cache when total fits the budget; cache
        # holds the FULL files (so train+val datasets share one copy).
        budget = _max_cache_bytes()
        if cache_in_memory is None:
            self._use_cache = total_bytes <= budget
        else:
            self._use_cache = bool(cache_in_memory) and total_bytes <= budget
        self._cache_bytes = total_bytes
        self._cached: list[Optional[np.ndarray]] = [None] * len(self._h5_paths)
        if self._use_cache:
            self._populate_cache()

        # Lazy file handles (opened on first access per worker; only used
        # when caching is off).
        self._h5_files: list = [None] * len(self._h5_paths)

    def _populate_cache(self) -> None:
        """Load each file's crops (selected channels, resized) and metadata into RAM."""
        import h5py

        # Per-file metadata caches: path → dict[name → ndarray-of-strings/ints].
        # Keyed only by path (not channels/size) since the metadata is per-cell
        # and independent of how the imagery is sliced.
        self._cached_meta: list[dict] = []
        for fi, path in enumerate(self._h5_paths):
            ch_idx = self._channel_indices[fi]
            cache_key = (path.resolve(), tuple(ch_idx), self.target_size)
            if cache_key in _CROP_CACHE:
                self._cached[fi] = _CROP_CACHE[cache_key]
            else:
                with h5py.File(str(path), "r") as f:
                    arr = f["crops"][:, ch_idx]  # (N, C, H, W) float32
                arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
                if arr.shape[-1] != self.target_size:
                    arr = self._batched_resize(arr, self.target_size)
                _CROP_CACHE[cache_key] = arr
                self._cached[fi] = arr

            # Cache metadata too — tiny compared to crops, but necessary
            # downstream (groupby condition_label etc).
            meta_key = ("meta", path.resolve())
            if meta_key in _CROP_CACHE:
                self._cached_meta.append(_CROP_CACHE[meta_key])
                continue
            meta_dict: dict = {}
            with h5py.File(str(path), "r") as f:
                if "condition_labels" in f:
                    meta_dict["condition_labels"] = np.array(
                        [s.decode() if isinstance(s, bytes) else str(s)
                         for s in f["condition_labels"][:]]
                    )
                if "cell_uids" in f:
                    meta_dict["cell_uids"] = np.array(
                        [s.decode() if isinstance(s, bytes) else str(s)
                         for s in f["cell_uids"][:]]
                    )
                if "fov_ids" in f:
                    meta_dict["fov_ids"] = f["fov_ids"][:].astype(np.int64)
            _CROP_CACHE[meta_key] = meta_dict  # type: ignore[assignment]
            self._cached_meta.append(meta_dict)

    @staticmethod
    def _batched_resize(arr: np.ndarray, target: int) -> np.ndarray:
        """Resize (N,C,H,W) to (N,C,target,target) via torch interpolate."""
        # Process in chunks to avoid spiking memory on large arrays.
        out_chunks = []
        chunk = 1024
        for i in range(0, arr.shape[0], chunk):
            t = torch.from_numpy(arr[i : i + chunk])
            t = F.interpolate(
                t, size=target, mode="bilinear", align_corners=False
            )
            out_chunks.append(t.numpy())
        return np.concatenate(out_chunks, axis=0)

    @staticmethod
    def _select_channels(channel_names: list[str], in_channels: int) -> list[int]:
        """Pick the first ``in_channels`` imaging channels (skip mask)."""
        imaging = []
        for i, name in enumerate(channel_names):
            if "mask" in name.lower():
                continue
            imaging.append(i)
        if not imaging:
            imaging = list(range(in_channels))
        return imaging[:in_channels]

    def _get_h5(self, file_idx: int):
        import h5py

        if self._h5_files[file_idx] is None:
            self._h5_files[file_idx] = h5py.File(
                str(self._h5_paths[file_idx]), "r", swmr=True
            )
        return self._h5_files[file_idx]

    def __len__(self) -> int:
        return len(self._indices)

    def _global_to_file(self, global_idx: int) -> tuple[int, int]:
        """Map a global index to (file_idx, local_idx)."""
        for fi in range(len(self._file_offsets) - 1, -1, -1):
            if global_idx >= self._file_offsets[fi]:
                return fi, global_idx - self._file_offsets[fi]
        return 0, global_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        global_idx = int(self._indices[idx])
        file_idx, local_idx = self._global_to_file(global_idx)

        if self._use_cache:
            arr = self._cached[file_idx]
            crop = torch.from_numpy(arr[local_idx])  # (C, H, W) float32
            meta: dict = {"global_idx": global_idx, "file_idx": file_idx}
            mdict = self._cached_meta[file_idx] if hasattr(self, "_cached_meta") else {}
            if "condition_labels" in mdict:
                meta["condition_label"] = str(mdict["condition_labels"][local_idx])
            if "cell_uids" in mdict:
                meta["cell_uid"] = str(mdict["cell_uids"][local_idx])
            if "fov_ids" in mdict:
                meta["fov_id"] = int(mdict["fov_ids"][local_idx])
            return crop, meta

        f = self._get_h5(file_idx)
        ch_idx = self._channel_indices[file_idx]

        # Read only the needed channels
        crop = f["crops"][local_idx, ch_idx]  # (C, H, W) float32
        crop = torch.from_numpy(crop.astype(np.float32))

        # Resize if needed
        if crop.shape[-1] != self.target_size:
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Metadata
        meta = {"global_idx": global_idx, "file_idx": file_idx}
        if "condition_labels" in f:
            meta["condition_label"] = f["condition_labels"][local_idx].decode()
        if "cell_uids" in f:
            meta["cell_uid"] = f["cell_uids"][local_idx].decode()
        if "fov_ids" in f:
            meta["fov_id"] = int(f["fov_ids"][local_idx])

        return crop, meta

    def close(self) -> None:
        for f in self._h5_files:
            if f is not None:
                f.close()
        self._h5_files = [None] * len(self._h5_paths)


def split_by_fov(
    h5_paths: Sequence[Path],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by FOV to avoid data leakage.

    Returns (train_indices, val_indices) as global index arrays.
    """
    import h5py

    rng = np.random.default_rng(seed)

    all_fov_ids: list[int] = []
    offset = 0

    for path in h5_paths:
        with h5py.File(str(path), "r") as f:
            n = f["crops"].shape[0]
            if "fov_ids" in f:
                fov_ids = f["fov_ids"][:] + offset * 10000
            else:
                fov_ids = np.full(n, offset, dtype=np.int32)
            all_fov_ids.append(fov_ids)
            offset += 1

    fov_arr = np.concatenate(all_fov_ids)
    unique_fovs = np.unique(fov_arr)
    rng.shuffle(unique_fovs)

    n_val = max(1, int(len(unique_fovs) * val_fraction))
    val_fovs = set(unique_fovs[:n_val].tolist())

    all_idx = np.arange(len(fov_arr))
    val_mask = np.isin(fov_arr, list(val_fovs))

    return all_idx[~val_mask], all_idx[val_mask]
