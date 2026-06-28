"""
dags/monitoring_pipeline.py

Hourly monitoring pipeline DAG.
Checks for data drift and model performance degradation.
Triggers retraining DAG if drift is detected.

    fetch_production_data
            ↓
    detect_data_drift
            ↓
    check_model_performance
            ↓
    trigger_retraining (conditional — only if drift detected)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_ARGS = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}


def task_fetch_production_data(**context):
    """
    Fetch recent production data for drift monitoring.
    In production: query your database for last N hours of predictions.
    Here: simulate by sampling from the dataset with slight modifications.
    """
    from src.data.ingestion import DataIngestion
    import numpy as np

    ingestion = DataIngestion()
    df        = ingestion.load()

    # Simulate production data — sample 500 rows with slight noise
    production_sample = df.sample(n=500, random_state=int(datetime.now().timestamp()))

    # Save production sample for drift detection task
    production_sample.to_csv("data/processed/production_sample.csv", index=False)

    context["ti"].xcom_push(key="production_rows", value=len(production_sample))
    print(f"Fetched {len(production_sample)} production rows")


def task_detect_drift(**context):
    """Run drift detection between training data and production sample."""
    import pandas as pd
    from src.data.ingestion import DataIngestion
    from src.data.splitter import DataSplitter
    from src.monitoring.drift_detector import DriftDetector

    # Load reference (training) data
    ingestion        = DataIngestion()
    df               = ingestion.load()
    splitter         = DataSplitter()
    train, _, _      = splitter.split(df)

    # Load production sample
    production = pd.read_csv("data/processed/production_sample.csv")

    # Run drift detection
    detector = DriftDetector()
    report   = detector.detect_drift(train, production)
    detector.save_report(report)

    context["ti"].xcom_push(key="drift_detected",  value=report["overall_drift_detected"])
    context["ti"].xcom_push(key="drift_rate",      value=report["drift_rate"])
    context["ti"].xcom_push(key="should_retrain",  value=report["should_retrain"])

    print(f"Drift detected: {report['overall_drift_detected']}")
    print(f"Drift rate: {report['drift_rate']:.2%}")
    print(f"Drifted features: {report['drifted_features']}")


def task_check_model_performance(**context):
    """
    Check if model performance has degraded on recent data.
    Compares current performance against the registered threshold.
    """
    from src.utils.config import settings

    # In production: evaluate model on labeled production data
    # Here: simulate a performance check
    import random
    random.seed(int(datetime.now().timestamp()))
    simulated_roc_auc = random.uniform(0.85, 0.95)

    threshold          = settings.monitoring.performance_threshold
    performance_ok     = simulated_roc_auc >= threshold

    context["ti"].xcom_push(key="current_roc_auc",  value=simulated_roc_auc)
    context["ti"].xcom_push(key="performance_ok",    value=performance_ok)

    print(f"Current ROC-AUC: {simulated_roc_auc:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Performance OK: {performance_ok}")


def decide_retraining(**context):
    """
    Branch operator — decides whether to trigger retraining.
    Returns task_id of the next task to run.
    """
    drift_detected = context["ti"].xcom_pull(
        task_ids="detect_drift", key="drift_detected"
    )
    performance_ok = context["ti"].xcom_pull(
        task_ids="check_model_performance", key="performance_ok"
    )

    if drift_detected or not performance_ok:
        print("Triggering retraining pipeline")
        return "trigger_retraining"
    else:
        print("No retraining needed")
        return "no_retraining_needed"


with DAG(
    dag_id="monitoring_pipeline",
    default_args=DEFAULT_ARGS,
    description="Hourly monitoring for data drift and model performance",
    schedule_interval="0 * * * *",
    catchup=False,
    tags=["ml", "monitoring"],
) as dag:

    fetch_production_data = PythonOperator(
        task_id="fetch_production_data",
        python_callable=task_fetch_production_data,
    )

    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=task_detect_drift,
    )

    check_model_performance = PythonOperator(
        task_id="check_model_performance",
        python_callable=task_check_model_performance,
    )

    retraining_decision = BranchPythonOperator(
        task_id="retraining_decision",
        python_callable=decide_retraining,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="retraining_pipeline",
        wait_for_completion=False,
    )

    no_retraining_needed = EmptyOperator(
        task_id="no_retraining_needed",
    )

    # DAG structure
    fetch_production_data >> detect_drift >> check_model_performance >> retraining_decision
    retraining_decision >> [trigger_retraining, no_retraining_needed]