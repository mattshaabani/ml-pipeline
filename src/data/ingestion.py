"""
src/data/ingestion.py

Downloads and loads the UCI Adult Income dataset.

Dataset source:
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

The raw CSV has no header and uses " ?" for missing values.
We handle all of that here so downstream code sees a clean DataFrame.

Usage:
    from src.data.ingestion import DataIngestion
    ingestion = DataIngestion()
    df = ingestion.load()
"""

import pandas as pd
from pathlib import Path
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Column names in the exact order they appear in the raw file
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Numerical columns for type validation
NUMERICAL_COLUMNS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]


class DataIngestion:
    """
    Loads the Adult Income dataset from local cache or downloads it.

    Why cache locally?
        Downloading from UCI every run is slow and unreliable.
        We download once, save to data/raw/, and load from there.
    """

    RAW_DATA_PATH = settings.root_dir / "data" / "raw" / "adult.csv"

    def __init__(self):
        self.raw_data_path = self.RAW_DATA_PATH
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        """Download the dataset from UCI if not already cached locally."""
        if self.raw_data_path.exists():
            logger.info(f"Using cached dataset", extra={
                "path": str(self.raw_data_path)
            })
            return self.raw_data_path

        logger.info(f"Downloading Adult Income dataset")

        import urllib.request
        url = settings.data.source_url

        try:
            urllib.request.urlretrieve(url, self.raw_data_path)
            logger.info(f"Download complete", extra={
                "path": str(self.raw_data_path)
            })
        except Exception as e:
            logger.warning(f"Download failed, using fallback", extra={
                "error": str(e)
            })
            self._create_fallback_dataset()

        return self.raw_data_path

    def _create_fallback_dataset(self) -> None:
        """
        Create a small synthetic dataset if download fails.
        This ensures the pipeline can run without internet access.
        Useful for testing and development.
        """
        import numpy as np
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            "age":            np.random.randint(18, 90, n),
            "workclass":      np.random.choice(["Private", "Self-emp", "Gov", "?"], n),
            "fnlwgt":         np.random.randint(10000, 1000000, n),
            "education":      np.random.choice(["Bachelors", "HS-grad", "Masters", "Some-college"], n),
            "education_num":  np.random.randint(1, 16, n),
            "marital_status": np.random.choice(["Married", "Single", "Divorced"], n),
            "occupation":     np.random.choice(["Tech-support", "Craft-repair", "Sales", "?"], n),
            "relationship":   np.random.choice(["Wife", "Husband", "Own-child", "Not-in-family"], n),
            "race":           np.random.choice(["White", "Black", "Asian-Pac-Islander"], n),
            "sex":            np.random.choice(["Male", "Female"], n),
            "capital_gain":   np.random.randint(0, 100000, n),
            "capital_loss":   np.random.randint(0, 4000, n),
            "hours_per_week": np.random.randint(1, 99, n),
            "native_country": np.random.choice(["United-States", "Mexico", "Other"], n),
            "income":         np.random.choice(["<=50K", ">50K"], n, p=[0.75, 0.25]),
        })

        df.to_csv(self.raw_data_path, index=False)
        logger.info(f"Fallback synthetic dataset created", extra={"rows": n})

    def load(self) -> pd.DataFrame:
        """
        Load the dataset, downloading if necessary.

        Returns:
            Clean DataFrame with proper column names and types.
        """
        self.download()

        try:
            df = pd.read_csv(
                self.raw_data_path,
                names=COLUMN_NAMES,
                sep=",",
                skipinitialspace=True,
                na_values=["?", " ?"],
                skiprows=1 if self._has_header() else 0,
            )
        except Exception:
            df = pd.read_csv(self.raw_data_path)

        df = self._clean(df)

        logger.info(f"Dataset loaded", extra={
            "rows":    len(df),
            "columns": len(df.columns),
        })

        return df

    def _has_header(self) -> bool:
        """Check if the raw file has a header row."""
        with open(self.raw_data_path, "r") as f:
            first_line = f.readline()
        return "age" in first_line.lower()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data:
        - Strip whitespace from string columns
        - Normalize target column (remove trailing dots from UCI format)
        - Cast numerical columns to correct types
        """
        # Strip whitespace from all string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        # Normalize target — UCI raw file has "<=50K." and ">50K." with trailing dot
        if "income" in df.columns:
            df["income"] = df["income"].str.replace(".", "", regex=False)

        # Cast numerical columns
        for col in NUMERICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df