"""
src/data/validation.py

Data validation before training.
Catches data quality issues early — before they silently corrupt models.

Validation checks:
    - Required columns present
    - No unexpected columns
    - Numerical columns have correct types
    - Missing value rates within acceptable bounds
    - Target column has correct values
    - Row count above minimum threshold

Usage:
    from src.data.validation import DataValidator
    validator = DataValidator()
    report    = validator.validate(df)
    if not report["passed"]:
        raise ValueError(report["errors"])
"""

import pandas as pd
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

VALID_TARGET_VALUES = ["<=50K", ">50K"]
MIN_ROW_COUNT       = 100
MAX_MISSING_RATE    = 0.15   # max 15% missing values per column


class DataValidator:
    """
    Validates a DataFrame before it enters the training pipeline.

    Why validate?
        Silent data quality issues are the #1 cause of mysterious
        model degradation in production. A schema change upstream,
        a new NULL pattern, an unexpected categorical value — all
        of these can corrupt training without raising any error.

    Validation catches these issues at the pipeline boundary,
    before they propagate downstream.
    """

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Run all validation checks.

        Returns:
            {
                "passed": bool,
                "errors": list of error strings,
                "warnings": list of warning strings,
                "stats": summary statistics,
            }
        """
        errors   = []
        warnings = []

        # Check 1: minimum row count
        if len(df) < MIN_ROW_COUNT:
            errors.append(f"Too few rows: {len(df)} < {MIN_ROW_COUNT}")

        # Check 2: required columns present
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check 3: no completely unexpected columns
        extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
        if extra_cols:
            warnings.append(f"Extra columns (will be ignored): {extra_cols}")

        # Check 4: missing value rates
        for col in df.columns:
            missing_rate = df[col].isna().mean()
            if missing_rate > MAX_MISSING_RATE:
                errors.append(
                    f"Column '{col}' has {missing_rate:.1%} missing values "
                    f"(max allowed: {MAX_MISSING_RATE:.1%})"
                )
            elif missing_rate > 0.05:
                warnings.append(f"Column '{col}' has {missing_rate:.1%} missing values")

        # Check 5: target column values
        if "income" in df.columns:
            actual_values = set(df["income"].dropna().unique())
            invalid_values = actual_values - set(VALID_TARGET_VALUES)
            if invalid_values:
                errors.append(f"Invalid target values: {invalid_values}")

        # Check 6: numerical column ranges
        range_checks = {
            "age":            (0, 120),
            "hours_per_week": (0, 168),
            "education_num":  (0, 20),
        }
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    warnings.append(
                        f"Column '{col}' has {out_of_range} values outside "
                        f"expected range [{min_val}, {max_val}]"
                    )

        # Compute summary stats
        stats = {
            "row_count":        len(df),
            "column_count":     len(df.columns),
            "missing_values":   int(df.isna().sum().sum()),
            "duplicate_rows":   int(df.duplicated().sum()),
            "target_distribution": df["income"].value_counts(normalize=True).to_dict()
                                   if "income" in df.columns else {},
        }

        passed = len(errors) == 0

        if passed:
            logger.info(f"Data validation passed", extra={
                "rows":     stats["row_count"],
                "warnings": len(warnings),
            })
        else:
            logger.error(f"Data validation FAILED", extra={
                "errors":   len(errors),
                "warnings": len(warnings),
            })

        for w in warnings:
            logger.warning(f"Validation warning: {w}")
        for e in errors:
            logger.error(f"Validation error: {e}")

        return {
            "passed":   passed,
            "errors":   errors,
            "warnings": warnings,
            "stats":    stats,
        }