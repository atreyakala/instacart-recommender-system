"""Evaluation metrics: basket F1, per-model metrics, threshold optimization."""

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


def f1_on_basket(y_true_sets, y_pred_sets) -> dict:
    """
    Per-user basket F1, macro-averaged.
    y_true_sets: list of sets of product_ids (actual next basket per user)
    y_pred_sets: list of sets of product_ids (predicted basket per user)
    """
    f1s = []
    total_tp = total_fp = total_fn = 0
    for actual, pred in zip(y_true_sets, y_pred_sets):
        tp = len(actual & pred)
        fp = len(pred - actual)
        fn = len(actual - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        f1s.append(f1)
    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0, "micro_f1": float(micro_f1), "per_user_f1": f1s}


def lift_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.1) -> float:
    n_top = max(1, int(len(y_proba) * k))
    top_idx = np.argsort(y_proba)[-n_top:]
    precision_at_k = y_true[top_idx].mean()
    baseline_rate = y_true.mean()
    return float(precision_at_k / baseline_rate) if baseline_rate > 0 else 0.0


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "lift_at_k": lift_at_k(y_true, y_proba, k=0.1),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": float(threshold),
    }
