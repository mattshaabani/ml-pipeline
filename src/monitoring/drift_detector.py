"""
src/monitoring/drift_detector.py

Data drift detection using KS test, PSI, and Wasserstein distance.

Drift = the statistical properties of production data diverge
from training data, causing model performance to degrade silently.

Usage:
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector()
    report   = detector.detect_drift(reference_df, production_df)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from src.features.feature_definitions import FEATURE_SCHEMA
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Detects data drift between reference (training) and
    production data distributions.

    Three methods for three feature types:
        KS test          → continuous numerical features
        PSI              → categorical features + binned numericals
        Wasserstein dist → continuous features (alternative to KS)
    """

    def __init__(self):
        self.drift_threshold    = settings.monitoring.drift_threshold
        self.numerical_features = FEATURE_SCHEMA.numerical_features
        self.categorical_features = FEATURE_SCHEMA.categorical_features

    def ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
    ) -> dict:
        """
        Kolmogorov-Smirnov test for continuous feature drift.

        Tests H0: the two samples come from the same distribution.
        If p_value < 0.05, we reject H0 → distributions differ → drift.

        KS statistic = max|F1(x) - F2(x)| where F1, F2 are CDFs.
        Range: 0 (identical) to 1 (completely different).
        """
        statistic, p_value = stats.ks_2samp(reference, production)
        drift_detected = p_value < 0.05

        return {
            "ks_statistic":   round(float(statistic), 4),
            "p_value":        round(float(p_value), 4),
            "drift_detected": drift_detected,
            "method":         "ks_test",
        }

    def psi(
        self,
        reference:  np.ndarray,
        production: np.ndarray,
        n_bins:     int = 10,
    ) -> dict:
        """
        Population Stability Index for distribution shift.

        PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        Thresholds:
            PSI < 0.1:  no significant change
            PSI < 0.2:  moderate change
            PSI >= 0.2: significant shift → action required
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)
        bin_edges[0]  -= 1e-10
        bin_edges[-1] += 1e-10

        ref_counts  = np.histogram(reference, bins=bin_edges)[0]
        prod_counts = np.histogram(production, bins=bin_edges)[0]

        # Convert to proportions — add small epsilon to avoid log(0)
        ref_pct  = (ref_counts  + 1e-10) / len(reference)
        prod_pct = (prod_counts + 1e-10) / len(production)

        psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        drift_detected = psi_value >= self.drift_threshold * 2   # PSI threshold ~0.2

        return {
            "psi":            round(psi_value, 4),
            "drift_detected": drift_detected,
            "method":         "psi",
        }

    def wasserstein_distance(
        self,
        reference:  np.ndarray,
        production: np.ndarray,
    ) -> dict:
        """
        Wasserstein distance (Earth Mover's Distance).

        Measures the minimum "work" to transform reference into production.
        Scale-dependent — normalized by reference standard deviation.

        Normalized distance > drift_threshold → drift detected.
        """
        distance = float(stats.wasserstein_distance(reference, production))

        # Normalize by reference std for scale independence
        ref_std = np.std(reference)
        normalized = distance / ref_std if ref_std > 0 else distance
        drift_detected = normalized > self.drift_threshold

        return {
            "wasserstein_distance":   round(distance, 4),
            "normalized_distance":    round(normalized, 4),
            "drift_detected":         drift_detected,
            "method":                 "wasserstein",
        }

    def categorical_drift(
        self,
        reference:  pd.Series,
        production: pd.Series,
    ) -> dict:
        """
        Chi-square test for categorical feature drift.

        Tests if observed category frequencies differ significantly
        from expected frequencies.
        """
        ref_counts  = reference.value_counts(normalize=True)
        prod_counts = production.value_counts(normalize=True)

        # Align on same categories
        all_cats = ref_counts.index.union(prod_counts.index)
        ref_aligned  = ref_counts.reindex(all_cats, fill_value=1e-10)
        prod_aligned = prod_counts.reindex(all_cats, fill_value=1e-10)

        psi_value = float(np.sum(
            (prod_aligned - ref_aligned) * np.log(prod_aligned / ref_aligned)
        ))
        drift_detected = psi_value >= 0.2

        return {
            "psi":            round(psi_value, 4),
            "drift_detected": drift_detected,
            "method":         "categorical_psi",
        }

    def detect_drift(
        self,
        reference_df:  pd.DataFrame,
        production_df: pd.DataFrame,
    ) -> dict:
        """
        Run drift detection on all features.

        Args:
            reference_df:  Training/reference data
            production_df: New production/incoming data

        Returns:
            Full drift report with per-feature results and summary.
        """
        logger.info(f"Running drift detection", extra={
            "reference_rows":  len(reference_df),
            "production_rows": len(production_df),
        })

        feature_results = {}
        drifted_features = []

        # Numerical features — KS test + Wasserstein
        for feature in self.numerical_features:
            if feature not in reference_df.columns:
                continue
            if feature not in production_df.columns:
                continue

            ref  = reference_df[feature].dropna().values
            prod = production_df[feature].dropna().values

            ks_result   = self.ks_test(ref, prod)
            psi_result  = self.psi(ref, prod)
            wass_result = self.wasserstein_distance(ref, prod)

            any_drift = (
                ks_result["drift_detected"] or
                wass_result["drift_detected"]
            )

            feature_results[feature] = {
                "type":          "numerical",
                "ks_test":       ks_result,
                "psi":           psi_result,
                "wasserstein":   wass_result,
                "drift_detected": any_drift,
            }

            if any_drift:
                drifted_features.append(feature)

        # Categorical features — PSI
        for feature in self.categorical_features:
            if feature not in reference_df.columns:
                continue
            if feature not in production_df.columns:
                continue

            result = self.categorical_drift(
                reference_df[feature],
                production_df[feature],
            )

            feature_results[feature] = {
                "type":           "categorical",
                "psi":            result,
                "drift_detected": result["drift_detected"],
            }

            if result["drift_detected"]:
                drifted_features.append(feature)

        # Overall drift summary
        total_features  = len(feature_results)
        drifted_count   = len(drifted_features)
        drift_rate      = drifted_count / total_features if total_features > 0 else 0
        overall_drift   = drift_rate > self.drift_threshold

        report = {
            "overall_drift_detected": overall_drift,
            "drift_rate":             round(drift_rate, 4),
            "drifted_features":       drifted_features,
            "total_features_checked": total_features,
            "feature_results":        feature_results,
            "should_retrain":         overall_drift and settings.monitoring.retrain_on_drift,
        }

        if overall_drift:
            logger.warning(f"DRIFT DETECTED", extra={
                "drift_rate":      round(drift_rate, 4),
                "n_drifted":       drifted_count,
                "should_retrain":  report["should_retrain"],
            })
        else:
            logger.info(f"No significant drift detected", extra={
                "drift_rate": round(drift_rate, 4),
            })

        return report

    def save_report(self, report: dict, path: str = "data/drift_reports/report.json") -> None:
        """Save drift report to disk for Airflow task handoff."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable types
        clean_report = json.loads(json.dumps(report, default=str))

        with open(path, "w") as f:
            json.dump(clean_report, f, indent=2)

        logger.info(f"Drift report saved", extra={"path": path})