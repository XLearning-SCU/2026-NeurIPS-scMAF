"""Clustering metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def align_cluster_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map cluster identifiers to class labels with the Hungarian algorithm."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    true_labels = np.unique(y_true)
    predicted_labels = np.unique(y_pred)

    contingency = np.zeros((len(true_labels), len(predicted_labels)), dtype=np.int64)
    true_index = {label: index for index, label in enumerate(true_labels)}
    pred_index = {label: index for index, label in enumerate(predicted_labels)}
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        contingency[true_index[true_label], pred_index[predicted_label]] += 1

    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = {
        predicted_labels[column]: true_labels[row]
        for row, column in zip(row_indices, column_indices, strict=True)
    }
    fallback = true_labels[0]
    return np.asarray([mapping.get(label, fallback) for label in y_pred])


def clustering_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the clustering scores reported in the paper experiments."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    aligned = align_cluster_labels(y_true, y_pred)
    return {
        "nmi": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "acc": float(metrics.accuracy_score(y_true, aligned)),
    }
