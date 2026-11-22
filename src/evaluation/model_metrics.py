try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency
    np = None

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )
except ImportError:  # pragma: no cover - scikit-learn is a core dependency
    roc_auc_score = None
    average_precision_score = None


class ModelEvaluator:
    """Binary classification metrics for (user, product) reorder probability."""

    def compute_metrics(self, y_true_binary, y_prob, threshold=0.5):
        """Compute accuracy, precision, recall, F1, lift@0.1, roc_auc, pr_auc.

        Parameters
        ----------
        y_true_binary : array-like of int
            Ground-truth reorder labels (0/1).
        y_prob : array-like of float
            Predicted reorder probabilities.
        threshold : float
            Decision threshold for converting probabilities to binary predictions.

        Returns
        -------
        dict
        """
        if np is None:
            raise ImportError("numpy is required for ModelEvaluator.compute_metrics")
        y_true = np.asarray(y_true_binary).ravel()
        y_prob = np.asarray(y_prob).ravel()
        if y_true.shape != y_prob.shape:
            raise ValueError("y_true_binary and y_prob must have the same shape")

        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        lift_at_k = self._compute_lift(y_true, y_prob, k=0.1)

        if roc_auc_score is None or average_precision_score is None:
            raise ImportError("scikit-learn is required for roc_auc/pr_auc")
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = float("nan")

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "lift@0.1": float(lift_at_k),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": float(threshold),
        }

    def _compute_lift(self, y_true, y_prob, k=0.1):
        """Lift = precision@k / baseline positive rate.

        Top-k fraction of users/products ranked by predicted probability.
        """
        n = len(y_true)
        if n == 0:
            return 0.0
        n_top = max(1, int(n * k))
        top_idx = np.argsort(-y_prob, kind="mergesort")[:n_top]
        precision_at_k = float(np.mean(y_true[top_idx]))
        baseline_rate = float(np.mean(y_true))
        if baseline_rate <= 0:
            return 0.0
        return precision_at_k / baseline_rate
