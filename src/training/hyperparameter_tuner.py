"""
src/training/hyperparameter_tuner.py

Bayesian hyperparameter optimization using Optuna.

Optuna uses Tree-structured Parzen Estimators (TPE) to intelligently
search the hyperparameter space rather than trying combinations blindly.

Usage:
    from src.training.hyperparameter_tuner import HyperparameterTuner
    tuner  = HyperparameterTuner("xgboost")
    params = tuner.optimize(X_train, y_train, n_trials=30)
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from src.training.models import get_model
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization for any model in MODEL_REGISTRY.

    How TPE works:
        1. Sample n_startup_trials randomly (exploration phase)
        2. For each subsequent trial:
            a. Fit two density estimators:
               l(x) = p(hyperparams | good results)  ← better than threshold
               g(x) = p(hyperparams | bad results)   ← worse than threshold
            b. Choose hyperparams that maximize l(x) / g(x)
            c. Evaluate, update models
        3. Return the hyperparams that gave the best score
    """

    def __init__(self, model_name: str):
        self.model_name    = model_name
        self.best_params   = {}
        self.best_score    = -np.inf
        self.study         = None

    def _suggest_params(self, trial, model_name: str) -> dict:
        """
        Suggest hyperparameters for the given model.
        Optuna's suggest_* methods define the search space
        and use TPE to choose values intelligently.
        """
        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            }

        elif model_name == "random_forest":
            return {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 300),
                "max_depth":       trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            }

        elif model_name == "xgboost":
            return {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

        elif model_name == "lightgbm":
            return {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            }

        else:
            raise ValueError(f"No search space defined for {model_name}")

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials:  int = 20,
        cv_folds:  int = 3,
        metric:    str = "roc_auc",
        timeout:   int = 300,
    ) -> dict:
        """
        Run Bayesian optimization.

        Args:
            X_train:  Training features
            y_train:  Training labels
            n_trials: Number of hyperparameter combinations to try
            cv_folds: Cross-validation folds per trial
            metric:   Sklearn scoring metric to optimize
            timeout:  Max seconds to run (safety cap)

        Returns:
            Best hyperparameters found.
        """
        import optuna
        import pandas as pd
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"Starting hyperparameter optimization", extra={
            "model":    self.model_name,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "metric":   metric,
        })

        # Convert numpy array to DataFrame with feature names
        # This fixes LightGBM and XGBoost feature name warnings
        try:
            from src.features.feature_store import FeatureStore
            store         = FeatureStore()
            feature_names = store.get_feature_names()
            if len(feature_names) == X_train.shape[1]:
                X_train_input = pd.DataFrame(X_train, columns=feature_names)
            else:
                X_train_input = X_train
        except Exception:
            X_train_input = X_train

        def objective(trial) -> float:
            params = self._suggest_params(trial, self.model_name)
            model  = get_model(self.model_name, **params)

            try:
                scores = cross_val_score(
                    model, X_train_input, y_train,
                    cv=cv_folds,
                    scoring=metric,
                    n_jobs=-1,
                )
                return float(scores.mean())
            except Exception as e:
                logger.warning(f"Trial failed", extra={"error": str(e)})
                return 0.0

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        self.best_score  = self.study.best_value

        logger.info(f"Optimization complete", extra={
            "model":       self.model_name,
            "best_score":  round(self.best_score, 4),
            "best_params": self.best_params,
            "n_trials":    len(self.study.trials),
        })

        return self.best_params