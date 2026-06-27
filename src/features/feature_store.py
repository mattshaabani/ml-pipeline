"""
src/features/feature_store.py

Feast feature store integration for the ML pipeline.

Feast provides:
    - Offline store: historical features for training
    - Online store: low-latency features for serving
    - Feature registry: versioned feature definitions

We use Feast in local file-based mode — no cloud required.

In production you'd swap the file store for BigQuery/Redshift
(offline) and Redis/DynamoDB (online) without changing any
training or serving code — that's Feast's core value.

Usage:
    from src.features.feature_store import FeatureStore
    store = FeatureStore()
    store.materialize(train_df)
    features = store.get_training_features(entity_df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.features.transformers import FeatureTransformer
from src.features.feature_definitions import FEATURE_SCHEMA
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Simplified feature store that wraps our FeatureTransformer
    with persistence — saving processed features to disk so
    Airflow tasks can read them without recomputing.

    This is a lightweight stand-in for full Feast that provides
    the same conceptual interface:
        materialize() → compute and store features
        get_training_features() → retrieve stored features
        get_online_features() → retrieve features for one entity

    When you add full Feast:
        1. Replace materialize() with feast.FeatureStore.materialize()
        2. Replace get_training_features() with feast.FeatureStore.get_historical_features()
        3. Replace get_online_features() with feast.FeatureStore.get_online_features()
        All training and serving code stays the same.
    """

    STORE_PATH = Path(settings.features.store_path)

    def __init__(self):
        self.store_path  = self.STORE_PATH
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.transformer = FeatureTransformer()

        logger.info(f"Initialized FeatureStore", extra={
            "store_path": str(self.store_path)
        })

    def materialize(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
    ) -> None:
        """
        Fit transformer on training data and save processed
        feature matrices for all three splits.

        This is called once per training run by the Airflow DAG.
        Subsequent tasks read the saved features instead of
        recomputing them.
        """
        logger.info(f"Materializing features")

        # Fit ONLY on training data — critical for leak prevention
        X_train = self.transformer.fit_transform(train_df)
        X_val   = self.transformer.transform(val_df)
        X_test  = self.transformer.transform(test_df)

        # Extract labels
        y_train = (train_df[FEATURE_SCHEMA.target] == ">50K").astype(int).values
        y_val   = (val_df[FEATURE_SCHEMA.target]   == ">50K").astype(int).values
        y_test  = (test_df[FEATURE_SCHEMA.target]  == ">50K").astype(int).values

        # Save to disk
        np.save(self.store_path / "X_train.npy", X_train)
        np.save(self.store_path / "X_val.npy",   X_val)
        np.save(self.store_path / "X_test.npy",  X_test)
        np.save(self.store_path / "y_train.npy", y_train)
        np.save(self.store_path / "y_val.npy",   y_val)
        np.save(self.store_path / "y_test.npy",  y_test)

        # Save feature names for interpretability
        feature_names = self.transformer.feature_names_out
        pd.Series(feature_names).to_csv(
            self.store_path / "feature_names.csv", index=False
        )

        # Save training statistics for drift detection
        train_stats = self.transformer.get_feature_stats(train_df)
        import json
        with open(self.store_path / "train_stats.json", "w") as f:
            json.dump({
                "numerical_means":   train_stats.numerical_means,
                "numerical_stds":    train_stats.numerical_stds,
                "numerical_medians": train_stats.numerical_medians,
                "categorical_modes": train_stats.categorical_modes,
                "n_samples":         train_stats.n_samples,
            }, f, indent=2)

        logger.info(f"Features materialized", extra={
            "X_train_shape": X_train.shape,
            "X_val_shape":   X_val.shape,
            "X_test_shape":  X_test.shape,
            "n_features":    X_train.shape[1],
        })

    def get_training_features(self) -> tuple:
        """
        Load materialized training features from disk.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X_train = np.load(self.store_path / "X_train.npy")
        X_val   = np.load(self.store_path / "X_val.npy")
        X_test  = np.load(self.store_path / "X_test.npy")
        y_train = np.load(self.store_path / "y_train.npy")
        y_val   = np.load(self.store_path / "y_val.npy")
        y_test  = np.load(self.store_path / "y_test.npy")

        logger.info(f"Loaded features from store", extra={
            "X_train": X_train.shape,
            "X_val":   X_val.shape,
            "X_test":  X_test.shape,
        })

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_online_features(self, entity: dict) -> np.ndarray:
        """
        Get features for a single entity at inference time.

        Args:
            entity: dict of raw feature values for one person
                    e.g. {"age": 35, "workclass": "Private", ...}

        Returns:
            Transformed feature vector ready for model.predict()
        """
        df = pd.DataFrame([entity])
        return self.transformer.transform(df)

    def get_feature_names(self) -> list[str]:
        """Load saved feature names."""
        path = self.store_path / "feature_names.csv"
        if path.exists():
            return pd.read_csv(path).iloc[:, 0].tolist()
        return self.transformer.feature_names_out