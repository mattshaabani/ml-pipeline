import numpy as np
import pandas as pd
from src.data.ingestion import DataIngestion
from src.data.splitter import DataSplitter
from src.features.feature_store import FeatureStore
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import ClassificationMetrics
from src.monitoring.drift_detector import DriftDetector
from src.training.models import get_model

# Load features
store = FeatureStore()
X_train, X_val, X_test, y_train, y_val, y_test = store.get_training_features()
feature_names = store.get_feature_names()

# Train a quick LightGBM model
import pandas as pd
model = get_model("lightgbm", n_estimators=100)
model.fit(pd.DataFrame(X_train, columns=feature_names), y_train)

# Evaluate on test set
evaluator = ModelEvaluator()
results   = evaluator.evaluate_all_splits(model, "lightgbm_test")

# Test drift detection
print("\n=== DRIFT DETECTION TEST ===")
ingestion = DataIngestion()
df        = ingestion.load()

splitter = DataSplitter()
train, val, test = splitter.split(df)

detector = DriftDetector()

# Simulate NO drift — compare train to val (same distribution)
print("\n1. No drift scenario (train vs val):")
report_no_drift = detector.detect_drift(train, val)
print(f"   Drift detected: {report_no_drift['overall_drift_detected']}")
print(f"   Drift rate: {report_no_drift['drift_rate']}")
print(f"   Drifted features: {report_no_drift['drifted_features'][:3]}")

# Simulate DRIFT — compare young people vs old people (very different distributions)
print("\n2. Drift scenario (young vs old population):")
young = df[df["age"] < 30].copy()
old   = df[df["age"] > 55].copy()
report_drift = detector.detect_drift(young, old)
print(f"   Drift detected: {report_drift['overall_drift_detected']}")
print(f"   Drift rate: {report_drift['drift_rate']}")
print(f"   Drifted features: {report_drift['drifted_features'][:5]}")
print(f"   Should retrain: {report_drift['should_retrain']}")

# Save report
detector.save_report(report_drift)
print("\nDrift report saved to data/drift_reports/report.json")