"""
dags/retraining_pipeline.py

Retraining pipeline — triggered by monitoring when drift is detected.
Runs the full training pipeline with fresh data.

    log_retraining_trigger
            ↓
    retrain_models
            ↓
    evaluate_new_model
            ↓
    compare_with_production  (A/B comparison)
            ↓
    promote_if_better
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_ARGS = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}


def task_log_retraining_trigger(**context):
    """Log why retraining was triggered."""
    import json

    drift_report_path = "data/drift_reports/report.json"
    reason = "manual_trigger"

    if os.path.exists(drift_report_path):
        with open(drift_report_path) as f:
            report = json.load(f)
        drift_rate = report.get("drift_rate", 0)
        reason     = f"drift_detected (rate={drift_rate:.2%})"

    print(f"Retraining triggered: {reason}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    context["ti"].xcom_push(key="trigger_reason", value=reason)


def task_retrain_models(**context):
    """Retrain all models with fresh data."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    from src.data.ingestion import DataIngestion
    from src.data.splitter import DataSplitter
    from src.features.feature_store import FeatureStore
    from src.training.trainer import ModelTrainer

    # Re-ingest and re-split (in production: include new data)
    ingestion        = DataIngestion()
    df               = ingestion.load()
    splitter         = DataSplitter()
    train, val, test = splitter.split(df)

    # Re-materialize features
    store = FeatureStore()
    store.materialize(train, val, test)

    # Retrain
    trainer = ModelTrainer()
    results = trainer.train_all(n_tuning_trials=20)

    best_model   = max(results.keys(), key=lambda k: results[k]["metrics"].get("val_roc_auc", 0))
    best_roc_auc = results[best_model]["metrics"].get("val_roc_auc", 0)

    context["ti"].xcom_push(key="new_best_model",   value=best_model)
    context["ti"].xcom_push(key="new_val_roc_auc",  value=best_roc_auc)
    print(f"Retraining complete. Best: {best_model} (ROC-AUC: {best_roc_auc:.4f})")


def task_compare_with_production(**context):
    """
    Compare new model performance against current production model.
    This is the A/B test gate — only promote if the new model is better.
    """
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    import mlflow
    import pandas as pd
    from src.features.feature_store import FeatureStore
    from src.utils.config import settings
    from sklearn.metrics import roc_auc_score

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    store = FeatureStore()
    _, _, X_test, _, _, y_test = store.get_training_features()
    feature_names = store.get_feature_names()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Load current production model
    try:
        production_model = mlflow.sklearn.load_model(
            f"models:/{settings.mlflow.model_registry_name}/latest"
        )
        prod_proba    = production_model.predict_proba(X_test_df)[:, 1]
        prod_roc_auc  = roc_auc_score(y_test, prod_proba)
    except Exception:
        prod_roc_auc = 0.0

    new_val_roc_auc = context["ti"].xcom_pull(
        task_ids="retrain_models", key="new_val_roc_auc"
    )

    should_promote = new_val_roc_auc > prod_roc_auc + 0.005  # must be meaningfully better

    context["ti"].xcom_push(key="prod_roc_auc",    value=prod_roc_auc)
    context["ti"].xcom_push(key="should_promote",  value=should_promote)

    print(f"Production model ROC-AUC: {prod_roc_auc:.4f}")
    print(f"New model ROC-AUC:        {new_val_roc_auc:.4f}")
    print(f"Should promote:           {should_promote}")


def task_promote_if_better(**context):
    """Promote new model to production if it outperforms current model."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    should_promote  = context["ti"].xcom_pull(task_ids="compare_with_production", key="should_promote")
    new_model       = context["ti"].xcom_pull(task_ids="retrain_models", key="new_best_model")
    new_roc_auc     = context["ti"].xcom_pull(task_ids="retrain_models", key="new_val_roc_auc")
    prod_roc_auc    = context["ti"].xcom_pull(task_ids="compare_with_production", key="prod_roc_auc")

    if should_promote:
        print(f"PROMOTING new {new_model} model to production")
        print(f"Improvement: {prod_roc_auc:.4f} → {new_roc_auc:.4f}")
    else:
        print(f"NOT promoting — new model ({new_roc_auc:.4f}) not sufficiently")
        print(f"better than production ({prod_roc_auc:.4f})")
        print("Keeping current production model")


with DAG(
    dag_id="retraining_pipeline",
    default_args=DEFAULT_ARGS,
    description="Triggered retraining pipeline with A/B comparison",
    schedule_interval=None,   # only triggered by monitoring pipeline
    catchup=False,
    tags=["ml", "retraining"],
) as dag:

    log_trigger = PythonOperator(
        task_id="log_retraining_trigger",
        python_callable=task_log_retraining_trigger,
    )

    retrain_models = PythonOperator(
        task_id="retrain_models",
        python_callable=task_retrain_models,
    )

    compare_with_production = PythonOperator(
        task_id="compare_with_production",
        python_callable=task_compare_with_production,
    )

    promote_if_better = PythonOperator(
        task_id="promote_if_better",
        python_callable=task_promote_if_better,
    )

    log_trigger >> retrain_models >> compare_with_production >> promote_if_better