"""
src/features/transformers.py

Feature transformation pipeline for the Adult Income dataset.

Handles:
    - Missing value imputation
    - Numerical scaling (StandardScaler)
    - Categorical encoding (OneHotEncoder)
    - Feature selection (drop irrelevant columns)

Uses sklearn Pipeline for clean, leak-free transformations.
The pipeline is fit ONLY on training data, then applied to
val and test — this prevents data leakage.

Why sklearn Pipeline instead of manual transforms?
    If you fit the scaler on the full dataset (including test),
    test set information leaks into the scaler's mean/std.
    Pipeline.fit(X_train) → Pipeline.transform(X_test) is
    the correct leak-free pattern.

Usage:
    from src.features.transformers import FeatureTransformer
    transformer = FeatureTransformer()
    transformer.fit(X_train)
    X_train_transformed = transformer.transform(X_train)
    X_test_transformed  = transformer.transform(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.features.feature_definitions import FEATURE_SCHEMA, FeatureStats
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureTransformer:
    """
    Transforms raw DataFrame into model-ready feature matrix.

    The transformer builds a sklearn ColumnTransformer pipeline:

        Numerical branch:
            1. SimpleImputer(strategy="median")   ← fill missing with median
            2. StandardScaler()                    ← zero mean, unit variance

        Categorical branch:
            1. SimpleImputer(strategy="most_frequent") ← fill missing with mode
            2. OneHotEncoder(handle_unknown="ignore")  ← binary columns

    These two branches run in parallel on their respective columns,
    then outputs are concatenated into the final feature matrix.
    """

    def __init__(self):
        self.schema      = FEATURE_SCHEMA
        self.pipeline    = None
        self.is_fitted   = False
        self.feature_names_out: list[str] = []

        # Adjust numerical features — remove dropped ones
        self.numerical_features = [
            f for f in self.schema.numerical_features
            if f not in self.schema.features_to_drop
        ]
        self.categorical_features = [
            f for f in self.schema.categorical_features
            if f not in self.schema.features_to_drop
        ]

    def _build_pipeline(self) -> ColumnTransformer:
        """Build the sklearn ColumnTransformer pipeline."""

        # Numerical: impute → scale
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        # Categorical: impute → encode
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("numerical",   numerical_pipeline,   self.numerical_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",   # drop any columns not listed above
        )

        return transformer

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """
        Fit the transformer on training data.

        CRITICAL: only call this on TRAINING data.
        Calling on test data would leak test statistics into the pipeline.
        """
        X = self._prepare_input(df)

        logger.info(f"Fitting feature transformer", extra={
            "n_samples":     len(X),
            "n_numerical":   len(self.numerical_features),
            "n_categorical": len(self.categorical_features),
        })

        self.pipeline  = self._build_pipeline()
        self.pipeline.fit(X)
        self.is_fitted = True

        # Capture output feature names for interpretability
        self.feature_names_out = self._get_feature_names()

        logger.info(f"Feature transformer fitted", extra={
            "output_features": len(self.feature_names_out),
        })

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a DataFrame using the fitted pipeline.
        Can be called on train, val, or test data.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        X = self._prepare_input(df)
        return self.pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step — convenience for training."""
        return self.fit(df).transform(df)

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop irrelevant columns and target before transforming."""
        cols_to_drop = self.schema.features_to_drop + [self.schema.target]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def _get_feature_names(self) -> list[str]:
        """
        Get output feature names after transformation.
        Numerical features keep their names.
        Categorical features get names like 'workclass_Private'.
        """
        names = list(self.numerical_features)

        # Get one-hot encoded names from the categorical encoder
        cat_encoder = self.pipeline.named_transformers_["categorical"]["encoder"]
        for i, col in enumerate(self.categorical_features):
            for category in cat_encoder.categories_[i]:
                names.append(f"{col}_{category}")

        return names

    def get_feature_stats(self, df: pd.DataFrame) -> FeatureStats:
        """
        Compute and return statistics from a DataFrame.
        Saved alongside the model for monitoring and drift detection.
        """
        stats = FeatureStats(n_samples=len(df))

        for col in self.numerical_features:
            if col in df.columns:
                stats.numerical_means[col]   = float(df[col].mean())
                stats.numerical_stds[col]    = float(df[col].std())
                stats.numerical_medians[col] = float(df[col].median())

        for col in self.categorical_features:
            if col in df.columns:
                stats.categorical_modes[col] = str(df[col].mode()[0])
                stats.categorical_vocab[col] = sorted(df[col].dropna().unique().tolist())

        return stats