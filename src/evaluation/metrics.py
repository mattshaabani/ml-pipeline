"""
src/evaluation/metrics.py

Evaluation metrics for binary classification.
Computes comprehensive metrics beyond just accuracy.

Usage:
    from src.evaluation.metrics import ClassificationMetrics
    metrics = ClassificationMetrics()
    report  = metrics.compute(y_true, y_pred, y_pred_prob)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationMetrics:
    """
    Comprehensive binary classification metrics.

    Why not just use accuracy?
        With 76% negative class, a model predicting ALL negatives
        gets 76% accuracy but is completely useless.

        ROC-AUC measures ability to distinguish classes regardless
        of threshold. F1 balances precision and recall.
        These are the metrics that actually matter.

    ROC-AUC:
        Area under the Receiver Operating Characteristic curve.
        Plots True Positive Rate vs False Positive Rate at all thresholds.
        AUC=0.5 → random, AUC=1.0 → perfect, AUC=0.9 → excellent.

    PR-AUC (Average Precision):
        Area under Precision-Recall curve.
        More informative than ROC-AUC on imbalanced datasets.
    """

    def compute(
        self,
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        y_pred_prob: np.ndarray,
        prefix:      str = "",
    ) -> dict:
        """
        Compute all classification metrics.

        Args:
            y_true:      Ground truth labels (0 or 1)
            y_pred:      Binary predictions (0 or 1)
            y_pred_prob: Predicted probabilities for class 1
            prefix:      Optional prefix for metric names (e.g. "test_")

        Returns:
            Dict of metric_name → value
        """
        p = f"{prefix}_" if prefix else ""

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            f"{p}roc_auc":           round(roc_auc_score(y_true, y_pred_prob), 4),
            f"{p}avg_precision":     round(average_precision_score(y_true, y_pred_prob), 4),
            f"{p}accuracy":          round(accuracy_score(y_true, y_pred), 4),
            f"{p}f1":                round(f1_score(y_true, y_pred), 4),
            f"{p}precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
            f"{p}recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
            f"{p}true_positives":    int(tp),
            f"{p}false_positives":   int(fp),
            f"{p}true_negatives":    int(tn),
            f"{p}false_negatives":   int(fn),
            f"{p}specificity":       round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        }

        logger.info(f"Metrics computed", extra={
            "roc_auc":  metrics.get(f"{p}roc_auc"),
            "f1":       metrics.get(f"{p}f1"),
            "accuracy": metrics.get(f"{p}accuracy"),
        })

        return metrics

    def find_optimal_threshold(
        self,
        y_true:      np.ndarray,
        y_pred_prob: np.ndarray,
        metric:      str = "f1",
    ) -> float:
        """
        Find the probability threshold that maximizes the given metric.

        Default threshold (0.5) is rarely optimal for imbalanced datasets.
        Optimizing for F1 balances precision and recall.

        Returns:
            Optimal threshold value between 0 and 1.
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = -1.0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f"Optimal threshold found", extra={
            "threshold": round(best_threshold, 2),
            "best_score": round(best_score, 4),
            "metric": metric,
        })

        return float(best_threshold)