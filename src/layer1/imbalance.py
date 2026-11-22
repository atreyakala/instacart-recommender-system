"""Threshold tuning for imbalanced classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> float:
    """Grid-search threshold maximizing the given metric on validation data."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)
    return best_threshold
