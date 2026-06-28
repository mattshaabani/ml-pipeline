"""
src/training/trainer.py

Main training pipeline — trains all models, runs hyperparameter
optimization, and logs everything to MLflow.

Usage:
    from src.training.trainer import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train_all()
"""

import os
import time
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from src.training.models import get_model, MODEL_REGISTRY
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.features.feature_store import FeatureStore
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains all models with hyperparameter optimization and MLflow tracking.

    For each model:
        1. Run Bayesian hyperparameter optimization (Optuna)
        2. Train final model with best params on full training set
        3. Evaluate on validation set
        4. Log params, metrics, and model artifact to MLflow
        5. Register best model in MLflow Model Registry

    After training all models, compare and select the winner.
    """

    def __init__(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)

        self.feature_store = FeatureStore()

        logger.info(f"Initialized ModelTrainer", extra={
            "experiment": settings.mlflow.experiment_name,
            "models":     settings.training.models,
        })

    def _compute_metrics(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = "",
        feature_names: list = None,
    ) -> dict:
        """Compute all evaluation metrics for a model on a dataset."""
        import pandas as pd
        if feature_names and len(feature_names) == X.shape[1]:
            X = pd.DataFrame(X, columns=feature_names)

        y_pred      = model.predict(X)
        y_pred_prob = model.predict_proba(X)[:, 1]

        prefix = f"{prefix}_" if prefix else ""
        return {
            f"{prefix}roc_auc":  round(roc_auc_score(y, y_pred_prob), 4),
            f"{prefix}accuracy": round(accuracy_score(y, y_pred), 4),
            f"{prefix}f1":       round(f1_score(y, y_pred), 4),
        }

    def train_model(
        self,
        model_name:  str,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        n_tuning_trials: int = 20,
    ) -> dict:
        """
        Train one model with hyperparameter optimization.
        Logs everything to MLflow.

        Returns:
            Dict with model, metrics, and MLflow run_id.
        """
        logger.info(f"Training {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Log model name
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])

            # Step 1: Hyperparameter optimization
            tuner       = HyperparameterTuner(model_name)
            best_params = tuner.optimize(
                X_train, y_train,
                n_trials=n_tuning_trials,
                cv_folds=settings.training.cv_folds,
            )

            # Log best hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_best_roc_auc", tuner.best_score)

            # Step 2: Train final model on full training set
            import pandas as pd
            try:
                feature_names = self.feature_store.get_feature_names()
                if len(feature_names) == X_train.shape[1]:
                    X_train_fit = pd.DataFrame(X_train, columns=feature_names)
                    X_val_fit = pd.DataFrame(X_val, columns=feature_names)
                else:
                    X_train_fit = X_train
                    X_val_fit = X_val
            except Exception:
                X_train_fit = X_train
                X_val_fit = X_val
        
            start_time = time.time()
            model      = get_model(model_name, **best_params)
            model.fit(X_train_fit, y_train)
            train_time = time.time() - start_time

            mlflow.log_metric("train_time_seconds", round(train_time, 2))

            # Step 3: Evaluate on train and val
            train_metrics = self._compute_metrics(model, X_train, y_train, "train", feature_names)
            val_metrics   = self._compute_metrics(model, X_val, y_val, "val", feature_names)

            all_metrics = {**train_metrics, **val_metrics}
            mlflow.log_metrics(all_metrics)

            # Step 4: Log model artifact
            skops_trusted_types = [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "collections.OrderedDict",
            ]
            mlflow.sklearn.log_model(
                model,
                name=model_name,
                registered_model_name=None,
                skops_trusted_types=skops_trusted_types,
            )

            run_id = mlflow.active_run().info.run_id

            logger.info(f"Model trained", extra={
                "model":         model_name,
                "val_roc_auc":   val_metrics["val_roc_auc"],
                "train_time":    round(train_time, 2),
                "run_id":        run_id,
            })

            return {
                "model":      model,
                "model_name": model_name,
                "params":     best_params,
                "metrics":    all_metrics,
                "run_id":     run_id,
            }

    def train_all(self, n_tuning_trials: int = 20) -> dict:
        """
        Train all models and return comparison results.

        Returns:
            Dict mapping model_name → results dict.
            Also prints a comparison table.
        """
        # Load features from store
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.feature_store.get_training_features()
        )

        logger.info(f"Starting full training run", extra={
            "models":    settings.training.models,
            "n_trials":  n_tuning_trials,
            "X_train":   X_train.shape,
        })

        all_results = {}

        for model_name in settings.training.models:
            try:
                result = self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_tuning_trials=n_tuning_trials,
                )
                all_results[model_name] = result
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Training failed for {model_name}", extra={
                    "error": str(e)
                })

        # Print comparison table
        self._print_comparison(all_results)

        # Register best model
        best_model_name = self._find_best_model(all_results)
        if best_model_name:
            self._register_best_model(all_results[best_model_name])

        return all_results

    def _print_comparison(self, results: dict) -> None:
        """Print a formatted model comparison table."""
        print("\n" + "="*65)
        print("MODEL COMPARISON RESULTS")
        print("="*65)
        print(f"{'Model':<22} {'Val ROC-AUC':>12} {'Val Acc':>10} {'Val F1':>10}")
        print("-"*65)

        for name, result in results.items():
            m = result["metrics"]
            print(
                f"{name:<22} "
                f"{m.get('val_roc_auc', 0):>12.4f} "
                f"{m.get('val_accuracy', 0):>10.4f} "
                f"{m.get('val_f1', 0):>10.4f}"
            )

        print("="*65)

    def _find_best_model(self, results: dict) -> str:
        """Find the model with the highest validation ROC-AUC."""
        if not results:
            return None

        best = max(
            results.keys(),
            key=lambda k: results[k]["metrics"].get("val_roc_auc", 0)
        )
        logger.info(f"Best model: {best}", extra={
            "val_roc_auc": results[best]["metrics"].get("val_roc_auc")
        })
        return best

    def _register_best_model(self, result: dict) -> None:
        """Register the best model in MLflow Model Registry."""
        try:
            model_uri = f"runs:/{result['run_id']}/{result['model_name']}"
            mlflow.register_model(
                model_uri=model_uri,
                name=settings.mlflow.model_registry_name,
            )
            logger.info(f"Best model registered", extra={
                "model":    result["model_name"],
                "registry": settings.mlflow.model_registry_name,
            })
        except Exception as e:
            logger.warning(f"Model registration failed", extra={"error": str(e)})