"""
src/evaluation/evaluator.py

Evaluates trained models on test data and logs results to MLflow.

Usage:
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    report    = evaluator.evaluate(model, X_test, y_test, model_name)
"""

import os
import json
import numpy as np
import mlflow
from src.evaluation.metrics import ClassificationMetrics
from src.features.feature_store import FeatureStore
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates models on held-out test data and logs to MLflow.

    Why evaluate on test set separately from validation?
        Validation set was used during hyperparameter tuning —
        so the model has "seen" it indirectly through tuning decisions.
        Test set is truly held-out — never touched until final evaluation.
        Test metrics are the honest estimate of production performance.
    """

    def __init__(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        self.metrics_calculator = ClassificationMetrics()
        self.feature_store      = FeatureStore()

    def evaluate(
        self,
        model,
        X:          np.ndarray,
        y:          np.ndarray,
        model_name: str,
        run_id:     str = None,
        split:      str = "test",
    ) -> dict:
        """
        Evaluate a model and log results to MLflow.

        Args:
            model:      Trained sklearn-compatible model
            X:          Feature matrix
            y:          True labels
            model_name: Name for logging
            run_id:     Existing MLflow run to log to (optional)
            split:      Which split this is ("val", "test")

        Returns:
            Dict of metrics.
        """
        import pandas as pd

        # Use DataFrame with feature names for compatibility
        feature_names = self.feature_store.get_feature_names()
        if len(feature_names) == X.shape[1]:
            X_input = pd.DataFrame(X, columns=feature_names)
        else:
            X_input = X

        y_pred      = model.predict(X_input)
        y_pred_prob = model.predict_proba(X_input)[:, 1]

        metrics = self.metrics_calculator.compute(
            y_true=y,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            prefix=split,
        )

        # Find optimal threshold
        optimal_threshold = self.metrics_calculator.find_optimal_threshold(
            y, y_pred_prob, metric="f1"
        )
        metrics[f"{split}_optimal_threshold"] = optimal_threshold

        # Log to MLflow
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
        else:
            with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("split", split)
                mlflow.log_metrics(metrics)

        logger.info(f"Evaluation complete", extra={
            "model":    model_name,
            "split":    split,
            "roc_auc":  metrics.get(f"{split}_roc_auc"),
            "f1":       metrics.get(f"{split}_f1"),
        })

        return metrics

    def evaluate_all_splits(
        self,
        model,
        model_name: str,
        run_id:     str = None,
    ) -> dict:
        """
        Evaluate on all three splits and return combined report.
        Loads data directly from feature store.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.feature_store.get_training_features()
        )

        results = {}
        for split, X, y in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            results[split] = self.evaluate(model, X, y, model_name, run_id, split)

        self._print_report(model_name, results)
        return results

    def _print_report(self, model_name: str, results: dict) -> None:
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print(f"{'-'*60}")

        key_metrics = ["roc_auc", "accuracy", "f1", "precision", "recall"]
        for metric in key_metrics:
            train_val = results.get("train", {}).get(f"train_{metric}", 0)
            val_val   = results.get("val",   {}).get(f"val_{metric}", 0)
            test_val  = results.get("test",  {}).get(f"test_{metric}", 0)
            print(f"{metric:<25} {train_val:>10.4f} {val_val:>10.4f} {test_val:>10.4f}")

        print(f"{'='*60}")