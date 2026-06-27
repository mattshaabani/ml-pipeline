"""
src/data/splitter.py

Creates reproducible stratified train/val/test splits.

Why stratified splitting?
    The Adult dataset is class-imbalanced (~75% <=50K, ~25% >50K).
    Random splitting might give you a test set with 40% >50K by chance,
    making your evaluation metrics misleading.

    Stratified splitting preserves the class ratio in every split.

Usage:
    from src.data.splitter import DataSplitter
    splitter = DataSplitter()
    train, val, test = splitter.split(df)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Creates stratified train/val/test splits with reproducible seeds.

    Split sizes from config:
        test_size = 0.2   → 20% test
        val_size  = 0.1   → 10% validation (of original data)
        train     = 0.7   → 70% training
    """

    def __init__(self):
        self.test_size    = settings.data.test_size
        self.val_size     = settings.data.val_size
        self.random_state = settings.data.random_state
        self.target       = settings.data.target_column

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split df into (train, val, test) DataFrames.

        Strategy:
            1. First split off test set (stratified)
            2. Then split remaining into train + val (stratified)

        Both splits are stratified on the target column.
        """
        # Step 1: split off test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target],
        )

        # Step 2: split train_val into train + val
        # val_size relative to original data, so we adjust:
        # if original is 100%, test=20%, remaining=80%
        # we want val=10% of original = 10/80 = 12.5% of remaining
        val_size_adjusted = self.val_size / (1 - self.test_size)

        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_val[self.target],
        )

        logger.info(f"Data split complete", extra={
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        })

        # Verify class ratios are preserved
        self._log_class_ratios(train, val, test)

        return train, val, test

    def _log_class_ratios(
        self,
        train: pd.DataFrame,
        val:   pd.DataFrame,
        test:  pd.DataFrame,
    ) -> None:
        """Log class ratio in each split to verify stratification worked."""
        for name, df in [("train", train), ("val", val), ("test", test)]:
            ratio = (df[self.target] == ">50K").mean()
            logger.info(f"Class ratio in {name}", extra={
                "split":   name,
                "gt50k":   round(ratio, 4),
                "lte50k":  round(1 - ratio, 4),
            })

    def save_splits(
        self,
        train: pd.DataFrame,
        val:   pd.DataFrame,
        test:  pd.DataFrame,
        output_dir: str = "data/processed",
    ) -> None:
        """Save splits to CSV for reproducibility and Airflow task handoff."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        train.to_csv(f"{output_dir}/train.csv", index=False)
        val.to_csv(f"{output_dir}/val.csv",   index=False)
        test.to_csv(f"{output_dir}/test.csv",  index=False)

        logger.info(f"Splits saved", extra={"dir": output_dir})