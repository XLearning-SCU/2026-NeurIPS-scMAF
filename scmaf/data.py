"""Dataset loading and preprocessing for multi-omics and multi-view data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import Dataset
import torch

LOGGER = logging.getLogger(__name__)


class MultiOmicsDataset(Dataset[dict[str, Tensor | int]]):
    def __init__(
        self,
        x1: np.ndarray,
        raw1: np.ndarray,
        sf1: np.ndarray,
        x2: np.ndarray,
        raw2: np.ndarray,
        sf2: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.x1 = torch.as_tensor(x1, dtype=torch.float32)
        self.raw1 = torch.as_tensor(raw1, dtype=torch.float32)
        self.sf1 = torch.as_tensor(sf1, dtype=torch.float32)
        self.x2 = torch.as_tensor(x2, dtype=torch.float32)
        self.raw2 = torch.as_tensor(raw2, dtype=torch.float32)
        self.sf2 = torch.as_tensor(sf2, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.x1.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        return {
            "x1": self.x1[index],
            "raw1": self.raw1[index],
            "sf1": self.sf1[index],
            "x2": self.x2[index],
            "raw2": self.raw2[index],
            "sf2": self.sf2[index],
            "label": self.labels[index],
            "idx": index,
        }


class MultiViewDataset(Dataset[dict[str, Tensor | int]]):
    def __init__(self, x1: np.ndarray, x2: np.ndarray, labels: np.ndarray) -> None:
        self.x1 = torch.as_tensor(x1, dtype=torch.float32)
        self.x2 = torch.as_tensor(x2, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.x1.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        return {
            "x1": self.x1[index],
            "x2": self.x2[index],
            "label": self.labels[index],
            "idx": index,
        }


@dataclass(frozen=True)
class DataBundle:
    dataset: MultiOmicsDataset | MultiViewDataset
    input_dims: tuple[int, int]
    n_classes: int
    labels: np.ndarray
    modality_types: tuple[str, str] | None = None


def _as_pair(value: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return value[0], value[1]


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("The label array is empty")
    _, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(np.int64, copy=False)


def select_features_by_dropout(
    data: np.ndarray,
    n_features: int,
    threshold: float = 0.0,
    at_least: int = 10,
    y_offset: float = 0.02,
    decay: float = 1.5,
    block_size: int = 1024,
) -> np.ndarray:
    """Select informative count features using the DCA-style dropout curve.

    The statistics are computed in feature blocks to avoid allocating another
    matrix the size of a large RNA count matrix.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if n_features >= data.shape[1]:
        return np.ones(data.shape[1], dtype=bool)

    detected_counts = np.zeros(data.shape[1], dtype=np.int64)
    mean_log_expression = np.full(data.shape[1], np.nan, dtype=np.float64)
    for start in range(0, data.shape[1], block_size):
        stop = min(start + block_size, data.shape[1])
        block = data[:, start:stop]
        detected = block > threshold
        counts = detected.sum(axis=0)
        detected_counts[start:stop] = counts
        safe_values = np.where(detected, block, 1.0)
        log_sums = np.where(detected, np.log2(safe_values), 0.0).sum(axis=0)
        np.divide(
            log_sums,
            counts,
            out=mean_log_expression[start:stop],
            where=counts > 0,
        )

    zero_rate = 1.0 - detected_counts / data.shape[0]
    invalid = detected_counts < at_least
    zero_rate[invalid] = np.nan
    mean_log_expression[invalid] = np.nan
    valid = ~np.isnan(zero_rate)
    if valid.sum() < n_features:
        raise ValueError(
            f"Requested {n_features} features, but only {valid.sum()} pass the detection threshold"
        )

    lower, upper, x_offset = 0.0, 10.0, 5.0
    selected = np.zeros(data.shape[1], dtype=bool)
    for _ in range(100):
        selected[:] = False
        selected[valid] = zero_rate[valid] > (
            np.exp(-decay * (mean_log_expression[valid] - x_offset)) + y_offset
        )
        count = int(selected.sum())
        if count == n_features:
            break
        if count < n_features:
            upper = x_offset
        else:
            lower = x_offset
        x_offset = (lower + upper) / 2.0

    if selected.sum() != n_features:
        # Ties may prevent the curve from selecting exactly n features. Rank by
        # vertical distance to the final curve to make the result deterministic.
        score = np.full(data.shape[1], -np.inf)
        score[valid] = zero_rate[valid] - (
            np.exp(-decay * (mean_log_expression[valid] - x_offset)) + y_offset
        )
        chosen = np.argpartition(score, -n_features)[-n_features:]
        selected[:] = False
        selected[chosen] = True

    LOGGER.info("Selected %d of %d features", selected.sum(), data.shape[1])
    return selected


def _normalize_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Library-size normalize, log1p transform, and standardize count data."""
    raw = np.asarray(counts, dtype=np.float32)
    library_sizes = raw.sum(axis=1, dtype=np.float64)
    positive = library_sizes > 0
    if not positive.all():
        raise ValueError("Zero-count cells must be removed before normalization")
    median_library_size = float(np.median(library_sizes))
    if median_library_size <= 0:
        raise ValueError("The median library size must be positive")

    size_factors = (library_sizes / median_library_size).astype(np.float32).reshape(-1, 1)
    normalized = raw / size_factors
    np.log1p(normalized, out=normalized)
    means = normalized.mean(axis=0, dtype=np.float64)
    standard_deviations = normalized.std(axis=0, ddof=1, dtype=np.float64)
    standard_deviations[standard_deviations == 0] = 1.0
    normalized = ((normalized - means) / standard_deviations).astype(np.float32)
    return normalized, raw, size_factors


def _limit_samples(
    x1: np.ndarray,
    x2: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(labels):
        return x1, x2, labels
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(labels), size=max_samples, replace=False))
    return x1[indices], x2[indices], labels[indices]


def load_multiomics(config: Any) -> DataBundle:
    path = Path(config.data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Multi-omics data file not found: {path}")
    view_keys = _as_pair(getattr(config, "view_keys", ["X1", "X2"]), "view_keys")
    label_key = getattr(config, "label_key", "Y")

    with h5py.File(path, "r") as handle:
        missing = [key for key in (*view_keys, label_key) if key not in handle]
        if missing:
            raise KeyError(f"Missing HDF5 dataset(s) {missing}; available keys: {list(handle.keys())}")
        labels = np.asarray(handle[label_key]).reshape(-1)
        max_samples = getattr(config, "max_samples", None)
        if max_samples is not None and int(max_samples) < len(labels):
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive")
            rng = np.random.default_rng(int(config.seed))
            indices = np.sort(rng.choice(len(labels), size=int(max_samples), replace=False))
            x1 = handle[view_keys[0]].astype(np.float32)[indices, :]
            x2 = handle[view_keys[1]].astype(np.float32)[indices, :]
            labels = labels[indices]
        else:
            x1 = handle[view_keys[0]].astype(np.float32)[:]
            x2 = handle[view_keys[1]].astype(np.float32)[:]

    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError(f"Both modalities must be matrices, got {x1.shape} and {x2.shape}")
    if x1.shape[0] != x2.shape[0] or x1.shape[0] != labels.shape[0]:
        raise ValueError("The two modalities and labels must contain the same number of samples")

    LOGGER.info("Loaded multi-omics data: view1=%s, view2=%s", x1.shape, x2.shape)

    if getattr(config, "filter1", True):
        x1 = x1[:, select_features_by_dropout(x1, int(config.f1))]
    if getattr(config, "filter2", False):
        x2 = x2[:, select_features_by_dropout(x2, int(config.f2))]

    # Match the original Scanpy QC behavior: remove empty features in each view,
    # then retain only cells that are non-empty in both views.
    x1 = x1[:, x1.sum(axis=0) > 0]
    x2 = x2[:, x2.sum(axis=0) > 0]
    common_cells = (x1.sum(axis=1) > 0) & (x2.sum(axis=1) > 0)
    x1, x2, labels = x1[common_cells], x2[common_cells], labels[common_cells]
    if len(labels) == 0:
        raise ValueError("No common non-empty cells remain after quality control")

    labels = _encode_labels(labels)
    normalized1, raw1, sf1 = _normalize_counts(x1)
    normalized2, raw2, sf2 = _normalize_counts(x2)
    data_types = tuple(str(value).upper() for value in _as_pair(config.data_type, "data_type"))
    dataset = MultiOmicsDataset(normalized1, raw1, sf1, normalized2, raw2, sf2, labels)
    LOGGER.info(
        "Prepared %d cells, input dimensions %s, %d classes",
        len(dataset),
        (normalized1.shape[1], normalized2.shape[1]),
        len(np.unique(labels)),
    )
    return DataBundle(
        dataset=dataset,
        input_dims=(normalized1.shape[1], normalized2.shape[1]),
        n_classes=len(np.unique(labels)),
        labels=labels,
        modality_types=data_types,
    )


def _resolve_multiview_path(config: Any) -> Path:
    path = Path(config.data_path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        known_names = {
            "NUSWIDE": "nuswide_deep_2_view.mat",
            "XMediaNet": "xmedia_deep_2_view.mat",
            "umpc_food101": "umpc_food101.mat",
        }
        filename = getattr(config, "data_file", known_names.get(config.dataset))
        if filename:
            candidate = path / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Multi-view data file not found: {path}")


def load_multiview(config: Any) -> DataBundle:
    path = _resolve_multiview_path(config)
    view_keys = _as_pair(getattr(config, "view_keys", ["Img", "Txt"]), "view_keys")
    label_key = getattr(config, "label_key", "label")
    contents = sio.loadmat(path)
    missing = [key for key in (*view_keys, label_key) if key not in contents]
    if missing:
        available = [key for key in contents if not key.startswith("__")]
        raise KeyError(f"Missing MAT variable(s) {missing}; available keys: {available}")

    # Standardize in the source precision, then cast once to float32. Scaling
    # float32 arrays directly can trigger avoidable cancellation warnings.
    x1 = np.asarray(contents[view_keys[0]])
    x2 = np.asarray(contents[view_keys[1]])
    labels = np.asarray(contents[label_key]).reshape(-1)
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError(f"Both views must be matrices, got {x1.shape} and {x2.shape}")
    if x1.shape[0] != x2.shape[0] or x1.shape[0] != labels.shape[0]:
        raise ValueError("The two views and labels must contain the same number of samples")

    x1, x2, labels = _limit_samples(
        x1, x2, labels, getattr(config, "max_samples", None), int(config.seed)
    )
    normalization = str(getattr(config, "data_norm", "standard")).lower()
    normalizers = {
        "standard": preprocessing.scale,
        "l2-norm": preprocessing.normalize,
        "min-max": preprocessing.minmax_scale,
        "none": lambda value: value,
    }
    if normalization not in normalizers:
        raise ValueError(f"Unknown data normalization: {normalization!r}")
    x1 = np.asarray(normalizers[normalization](x1), dtype=np.float32)
    x2 = np.asarray(normalizers[normalization](x2), dtype=np.float32)
    labels = _encode_labels(labels)
    dataset = MultiViewDataset(x1, x2, labels)
    LOGGER.info(
        "Prepared %d samples, input dimensions %s, %d classes",
        len(dataset),
        (x1.shape[1], x2.shape[1]),
        len(np.unique(labels)),
    )
    return DataBundle(
        dataset=dataset,
        input_dims=(x1.shape[1], x2.shape[1]),
        n_classes=len(np.unique(labels)),
        labels=labels,
    )


def load_data(config: Any) -> DataBundle:
    dataset_type = str(config.dataset_type).lower().replace("-", "_")
    if dataset_type == "multi_omics":
        return load_multiomics(config)
    if dataset_type == "multi_view":
        return load_multiview(config)
    raise ValueError(f"Unsupported dataset_type: {config.dataset_type!r}")
