"""Evaluation metrics for Layer 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def f1_on_basket(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> float:
    """
    Per-user basket F1, macro-averaged.
    y_true: binary labels (reordered or not)
    y_pred: binary predictions (thresholded probabilities)
    user_ids: user id per row
    """
    df = pd.DataFrame({"user_id": user_ids, "y_true": y_true, "y_pred": y_pred})
    f1s = []
    for _, group in df.groupby("user_id"):
        pred_set = set(group.loc[group["y_pred"] == 1].index.tolist())  # not product ids, but row positions
        # Use actual product membership via y_true==1 positions
        actual_pos = set(group.loc[group["y_true"] == 1].index.tolist())
        pred_pos = set(group.loc[group["y_pred"] == 1].index.tolist())
        tp = len(actual_pos & pred_pos)
        fp = len(pred_pos - actual_pos)
        fn = len(actual_pos - pred_pos)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def lift_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.1) -> float:
    """Precision@k divided by baseline positive rate."""
    n_top = max(1, int(len(y_proba) * k))
    top_idx = np.argsort(y_proba)[-n_top:]
    precision_at_k = y_true[top_idx].mean()
    baseline_rate = y_true.mean()
    return float(precision_at_k / baseline_rate) if baseline_rate > 0 else 0.0


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray, user_ids: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_on_basket": f1_on_basket(y_true, y_pred, user_ids),
        "lift_at_k": lift_at_k(y_true, y_proba, k=0.1),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": float(threshold),
    }
