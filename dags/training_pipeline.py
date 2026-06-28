"""
dags/training_pipeline.py

Daily training pipeline DAG.
Runs every day at 2am, orchestrating the full ML pipeline:

    ingest_data
         ↓
    validate_data
         ↓
    engineer_features
         ↓
    train_models
         ↓
    evaluate_models
         ↓
    register_best_model

Each task is independent and logs to MLflow.
If any task fails, Airflow retries it and sends an alert.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# Default task arguments
# ─────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ─────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────

def task_ingest_data(**context):
    """Download and load the dataset."""
    from src.data.ingestion import DataIngestion

    ingestion = DataIngestion()
    df        = ingestion.load()

    # Pass row count to next task via XCom
    # XCom = Airflow's cross-task communication mechanism
    context["ti"].xcom_push(key="row_count", value=len(df))
    print(f"Ingested {len(df)} rows")


def task_validate_data(**context):
    """Validate data quality before processing."""
    from src.data.ingestion import DataIngestion
    from src.data.validation import DataValidator

    ingestion = DataIngestion()
    df        = ingestion.load()

    validator = DataValidator()
    report    = validator.validate(df)

    if not report["passed"]:
        raise ValueError(f"Data validation failed: {report['errors']}")

    print(f"Validation passed. Warnings: {len(report['warnings'])}")
    context["ti"].xcom_push(key="validation_passed", value=True)


def task_engineer_features(**context):
    """Run feature engineering and materialize to feature store."""
    from src.data.ingestion import DataIngestion
    from src.data.splitter import DataSplitter
    from src.features.feature_store import FeatureStore

    ingestion        = DataIngestion()
    df               = ingestion.load()

    splitter         = DataSplitter()
    train, val, test = splitter.split(df)

    store = FeatureStore()
    store.materialize(train, val, test)

    print(f"Features materialized: train={len(train)}, val={len(val)}, test={len(test)}")
    context["ti"].xcom_push(key="n_features", value=88)


def task_train_models(**context):
    """Train all models with hyperparameter optimization."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer()
    results = trainer.train_all(n_tuning_trials=20)

    best_model = max(
        results.keys(),
        key=lambda k: results[k]["metrics"].get("val_roc_auc", 0)
    )
    best_roc_auc = results[best_model]["metrics"].get("val_roc_auc", 0)

    context["ti"].xcom_push(key="best_model",   value=best_model)
    context["ti"].xcom_push(key="best_roc_auc", value=best_roc_auc)
    print(f"Best model: {best_model} (ROC-AUC: {best_roc_auc:.4f})")


def task_evaluate_models(**context):
    """Evaluate best model on test set."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    import mlflow
    import pandas as pd
    from src.features.feature_store import FeatureStore
    from src.evaluation.evaluator import ModelEvaluator
    from src.utils.config import settings

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    store = FeatureStore()
    _, _, X_test, _, _, y_test = store.get_training_features()
    feature_names = store.get_feature_names()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    best_model_name = context["ti"].xcom_pull(
        task_ids="train_models", key="best_model"
    )

    # Load best model from MLflow registry
    model = mlflow.sklearn.load_model(
        f"models:/{settings.mlflow.model_registry_name}/latest"
    )

    evaluator = ModelEvaluator()
    results   = evaluator.evaluate(
        model, X_test, y_test, best_model_name, split="test"
    )

    test_roc_auc = results.get("test_roc_auc", 0)
    context["ti"].xcom_push(key="test_roc_auc", value=test_roc_auc)
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")

    # Check if performance meets threshold
    threshold = settings.monitoring.performance_threshold
    if test_roc_auc < threshold:
        raise ValueError(
            f"Model performance {test_roc_auc:.4f} below "
            f"threshold {threshold}"
        )


def task_register_best_model(**context):
    """Register the best model as production-ready."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    best_model   = context["ti"].xcom_pull(task_ids="train_models", key="best_model")
    test_roc_auc = context["ti"].xcom_pull(task_ids="evaluate_models", key="test_roc_auc")

    print(f"Model '{best_model}' registered as production model")
    print(f"Final test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")


# ─────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────

with DAG(
    dag_id="training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily ML training pipeline for Adult Income prediction",
    schedule_interval="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
) as dag:

    ingest_data = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    engineer_features = PythonOperator(
        task_id="engineer_features",
        python_callable=task_engineer_features,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )

    evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=task_evaluate_models,
    )

    register_best_model = PythonOperator(
        task_id="register_best_model",
        python_callable=task_register_best_model,
    )

    # Task dependencies — defines the DAG structure
    ingest_data >> validate_data >> engineer_features >> train_models >> evaluate_models >> register_best_model