"""Business impact: lift analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_business_lift(y_true: np.ndarray, y_prob_model: np.ndarray, y_prob_baseline: np.ndarray, k_values=None) -> pd.DataFrame:
    if k_values is None:
        k_values = [0.01, 0.05, 0.1, 0.2]
    baseline_rate = y_true.mean()
    results = []
    for k in k_values:
        n_top = int(len(y_prob_model) * k)
        model_top = np.argsort(y_prob_model)[-n_top:]
        model_prec = y_true[model_top].mean()
        baseline_top = np.argsort(y_prob_baseline)[-n_top:]
        baseline_prec = y_true[baseline_top].mean()
        lift = model_prec / baseline_prec if baseline_prec > 0 else 0.0
        results.append({
            "k": k,
            "top_n": n_top,
            "model_precision": float(model_prec),
            "baseline_precision": float(baseline_prec),
            "lift": float(lift),
            "baseline_rate": float(baseline_rate),
        })
    return pd.DataFrame(results)
