import os

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a core dependency
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover - visualization deps
    plt = None
    sns = None
    matplotlib = None

from .model_metrics import ModelEvaluator


class ModelComparison:
    """Evaluate multiple models and compare their metrics side by side."""

    def __init__(self, evaluator=None):
        self.evaluator = evaluator or ModelEvaluator()

    def evaluate_all(self, models, X_val, y_val, X_test, y_test,
                     threshold_optimizer, metric='f1'):
        """Evaluate every model and return a DataFrame of per-model test metrics.

        For each model:
          1. Tune the threshold on the validation set using ``threshold_optimizer``.
          2. Compute metrics on the held-out test set at that threshold.

        Parameters
        ----------
        models : dict of str -> estimator
            Fitted models exposing ``predict_proba``.
        X_val, y_val : validation features/labels for threshold tuning.
        X_test, y_test : held-out test features/labels for final metrics.
        threshold_optimizer : ThresholdOptimizer
        metric : str
            Metric to optimize the threshold on (default 'f1').

        Returns
        -------
        pandas.DataFrame
            One row per model with columns: model, threshold, accuracy, precision,
            recall, f1, lift@0.1, roc_auc, pr_auc.
        """
        if pd is None:
            raise ImportError("pandas is required for ModelComparison.evaluate_all")
        if np is None:
            raise ImportError("numpy is required for ModelComparison.evaluate_all")

        y_val = np.asarray(y_val).ravel()
        y_test = np.asarray(y_test).ravel()

        rows = []
        for name, model in models.items():
            prob_val = self._predict_proba(model, X_val)
            best_threshold = threshold_optimizer.find_optimal_threshold(
                y_val, prob_val, metric=metric
            )
            prob_test = self._predict_proba(model, X_test)
            metrics = self.evaluator.compute_metrics(
                y_test, prob_test, threshold=best_threshold
            )
            row = {"model": name, "threshold": float(best_threshold)}
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def _predict_proba(self, model, X):
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                f"Model {model!r} does not implement predict_proba"
            )
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    def plot_comparison(self, results_df, save_dir, metrics=None):
        """Bar charts of each metric per model, saved as PNGs into ``save_dir``."""
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required for plot_comparison")
        if pd is None:
            raise ImportError("pandas is required for plot_comparison")

        os.makedirs(save_dir, exist_ok=True)
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1",
                       "lift@0.1", "roc_auc", "pr_auc"]
        metrics = [m for m in metrics if m in results_df.columns]

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=results_df, x="model", y=metric, palette="viridis")
            plt.title(f"{metric} by model")
            plt.ylabel(metric)
            plt.xlabel("model")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out_path = os.path.join(save_dir, f"{metric.replace('@', '_at_')}_by_model.png")
            plt.savefig(out_path)
            plt.close()
