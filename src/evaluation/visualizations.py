try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover - visualization deps
    plt = None
    sns = None
    matplotlib = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None


def _require_viz():
    if plt is None or sns is None:
        raise ImportError("matplotlib and seaborn are required for visualizations")
    if np is None:
        raise ImportError("numpy is required for visualizations")


def plot_roc_curves(y_true, probs_dict, save_path=None, title="ROC Curves"):
    """Plot ROC curves for multiple models.

    probs_dict : dict of str -> array-like of predicted probabilities.
    """
    _require_viz()
    from sklearn.metrics import roc_curve, auc

    y_true = np.asarray(y_true).ravel()
    plt.figure(figsize=(8, 6))
    for name, y_prob in probs_dict.items():
        y_prob = np.asarray(y_prob).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_pr_curves(y_true, probs_dict, save_path=None, title="Precision-Recall Curves"):
    """Plot precision-recall curves for multiple models."""
    _require_viz()
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_true = np.asarray(y_true).ravel()
    plt.figure(figsize=(8, 6))
    for name, y_prob in probs_dict.items():
        y_prob = np.asarray(y_prob).ravel()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_threshold_vs_metric(y_true, y_prob, metric='f1', save_path=None,
                             thresholds=None, title=None):
    """Plot a metric as a function of the classification threshold."""
    _require_viz()
    from .threshold_optimizer import ThresholdOptimizer

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    scores = []
    optimizer = ThresholdOptimizer()
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        scores.append(optimizer._score(y_true, y_pred, metric))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, scores, marker="o")
    best_idx = int(np.argmax(scores))
    plt.axvline(thresholds[best_idx], color="r", linestyle="--",
                label=f"Best = {thresholds[best_idx]:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.title(title or f"{metric} vs threshold")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_lift_curve(y_true, y_prob, save_path=None, title="Lift Curve"):
    """Plot cumulative lift: precision at each percentile of predicted probability
    divided by the baseline positive rate."""
    _require_viz()

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    baseline_rate = float(np.mean(y_true)) if n > 0 else 0.0

    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]

    percentiles = np.arange(1, n + 1) / n
    cumulative_precision = np.cumsum(y_sorted) / np.arange(1, n + 1)
    lift = cumulative_precision / baseline_rate if baseline_rate > 0 else np.zeros_like(cumulative_precision)

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, lift, label="Model lift")
    plt.axhline(1.0, color="k", linestyle="--", label="Baseline (lift = 1)")
    plt.xlabel("Fraction of population (top-ranked by probability)")
    plt.ylabel("Lift (precision / baseline rate)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None, labels=None,
                          title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    _require_viz()
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = [0, 1]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_shap_summary(shap_values, X, feature_names=None, save_path=None,
                      title="SHAP Summary", max_display=20):
    """Plot a SHAP summary (beeswarm) plot.

    shap_values : array-like or list (multi-class) of SHAP values.
    X : array-like or DataFrame used to compute shap_values.
    feature_names : list of str, optional.
    """
    _require_viz()
    import shap

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
