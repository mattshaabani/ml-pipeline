# End-to-End MLOps Pipeline

A production-grade ML pipeline with feature engineering, multi-model
training, Bayesian hyperparameter optimization, drift detection, and
automated retraining — orchestrated with Airflow and tracked with MLflow.

---

## Overview

This project builds the complete MLOps lifecycle: a model deployed today
degrades tomorrow as the world changes. This pipeline detects that
degradation automatically and retrains the model without human intervention,
while keeping a full audit trail of every experiment.

**Dataset:** UCI Adult Income — predict whether a person earns >$50K/year
from census data. Real class imbalance (76%/24%), mixed feature types,
genuine drift patterns across demographics.

---

## Architecture

    Raw Data (UCI Adult Income)
            |
    Data Validation (schema, missing values, ranges)
            |
    Stratified Train/Val/Test Split
            |
    Feature Engineering (impute, scale, one-hot encode)
            |
    Feature Store (materialized, versioned)
            |
    Multi-Model Training (4 models × Optuna Bayesian tuning)
            |
    MLflow Experiment Tracking + Model Registry
            |
    Test Set Evaluation
            |
    Drift Detection (KS test, PSI, Wasserstein distance)
            |
    Automated Retraining (triggered by drift)
            |
    A/B Comparison Gate (promote only if better)
            |
    Airflow Orchestration (training / monitoring / retraining DAGs)

---

## Tech Stack

| Component | Technology |
|---|---|
| Feature engineering | scikit-learn Pipeline + ColumnTransformer |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Hyperparameter tuning | Optuna (Bayesian / TPE) |
| Experiment tracking | MLflow (SQLite backend) |
| Model registry | MLflow Model Registry |
| Drift detection | scipy (KS test, Wasserstein), custom PSI |
| Orchestration | Apache Airflow (Docker + PostgreSQL) |
| Containerization | Docker + Docker Compose |
| Environment | Conda + ipykernel |

---

## Project Structure

    ml-pipeline/
    |-- src/
    |   |-- data/
    |   |   |-- ingestion.py         # download, load, clean raw data
    |   |   |-- validation.py        # schema and quality checks
    |   |   └-- splitter.py          # stratified train/val/test split
    |   |-- features/
    |   |   |-- feature_definitions.py  # feature schema (source of truth)
    |   |   |-- transformers.py         # impute, scale, encode pipeline
    |   |   └-- feature_store.py        # materialize + serve features
    |   |-- training/
    |   |   |-- models.py               # 4 model factory functions
    |   |   |-- hyperparameter_tuner.py # Optuna Bayesian optimization
    |   |   └-- trainer.py              # full training orchestration
    |   |-- evaluation/
    |   |   |-- metrics.py              # ROC-AUC, F1, precision, recall
    |   |   └-- evaluator.py            # test set evaluation + MLflow
    |   |-- monitoring/
    |   |   └-- drift_detector.py       # KS test, PSI, Wasserstein
    |   └-- utils/
    |       |-- config.py
    |       └-- logger.py
    |-- dags/
    |   |-- training_pipeline.py     # daily: ingest -> train -> register
    |   |-- monitoring_pipeline.py   # hourly: drift check -> trigger retrain
    |   └-- retraining_pipeline.py   # triggered: retrain -> A/B -> promote
    |-- notebooks/
    |   |-- 01_data_exploration.ipynb
    |   |-- 02_feature_engineering.ipynb
    |   |-- 03_model_comparison.ipynb
    |   └-- 04_drift_analysis.ipynb
    |-- configs/
    |   |-- pipeline_config.yaml
    |   |-- model_config.yaml       # Optuna search spaces
    |   |-- feature_config.yaml
    |   └-- airflow_config.yaml
    |-- data/
    |   |-- raw/
    |   |-- processed/
    |   |-- features/
    |   └-- drift_reports/
    |-- docker-compose.yml           # Airflow + PostgreSQL
    |-- environment.yml
    └-- .env.example

---

## Quick Start

**1. Clone and set up environment**

    git clone https://github.com/MattShaabani/ml-pipeline.git
    cd ml-pipeline

    conda env create -f environment.yml
    conda activate ml-pipeline
    pip install -e .

**2. Run the full pipeline locally**

    python scripts/test_data.py          # ingest, validate, split
    python scripts/test_features.py      # feature engineering
    python scripts/test_training.py      # train 4 models with Optuna
    python scripts/test_evaluation.py    # evaluate + drift detection

**3. View MLflow experiment tracking**

    mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db

Open http://localhost:5000

**4. Run the Airflow orchestration stack**

    docker-compose up airflow-webserver postgres -d

Open http://localhost:8080 (login: admin/admin)

---

## Model Comparison Results

Trained on 22,792 samples, 88 engineered features, evaluated on 3,256
validation samples with Optuna Bayesian hyperparameter optimization
(20 trials per model, 3-fold cross-validation):

| Model | Val ROC-AUC | Val Accuracy | Val F1 |
|---|---|---|---|
| Logistic Regression | 0.9014 | 84.77% | 0.657 |
| Random Forest | 0.9115 | 85.72% | 0.662 |
| XGBoost | 0.9233 | 86.86% | 0.708 |
| **LightGBM** | **0.9250** | **87.29%** | **0.714** |

**Test set confirmation (LightGBM):** ROC-AUC 0.9299, Accuracy 87.52%,
F1 0.7186 — closely matches validation performance, confirming no
overfitting. Registered as production model in MLflow Model Registry.

---

## Drift Detection Results

| Scenario | Drift Detected | Drift Rate | Drifted Features |
|---|---|---|---|
| Train vs Validation (same distribution) | No | 0.0% | none |
| Young (<30) vs Old (>55) population | **Yes** | **78.6%** | age, education_num, capital_gain, capital_loss, fnlwgt |

This confirms the drift detector correctly distinguishes stable
distributions from genuinely shifted ones — critical for avoiding
false-positive retraining triggers in production.

---

## The Math Behind the Pipeline

**Gradient Boosting (XGBoost/LightGBM):**

    F_m(x) = F_{m-1}(x) + eta * tree_m(x)

Each tree corrects the residual errors of all previous trees. XGBoost
uses second-order Taylor expansion (gradient + hessian) for more precise
split decisions than first-order gradient boosting.

**Bayesian Hyperparameter Optimization (TPE):**

Models p(hyperparams | good results) and p(hyperparams | bad results)
separately, then selects hyperparameters that maximize their ratio.
Finds strong hyperparameters in ~20 trials vs 100+ for grid search.

**KS Test for Drift:**

    KS_statistic = max|F_reference(x) - F_production(x)|

Measures the maximum distance between two cumulative distribution
functions. p-value < 0.05 indicates statistically significant drift.

**Population Stability Index (PSI):**

    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )

    PSI < 0.10: stable
    PSI < 0.20: moderate shift, monitor
    PSI >= 0.20: significant shift, investigate

---

## Airflow DAGs

**training_pipeline** (daily, 2am):
ingest_data -> validate_data -> engineer_features -> train_models -> evaluate_models -> register_best_model

**monitoring_pipeline** (hourly):
fetch_production_data -> detect_drift -> check_model_performance -> [trigger_retraining | no_retraining_needed]

**retraining_pipeline** (triggered by monitoring):
log_retraining_trigger -> retrain_models -> compare_with_production -> promote_if_better

The retraining pipeline includes an A/B comparison gate — new models
are only promoted to production if they meaningfully outperform
(>0.5% ROC-AUC improvement) the current production model. This prevents
unnecessary model churn from noisy retraining results.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| APP_ENV | development or production | development |
| LOG_LEVEL | INFO, DEBUG, WARNING | INFO |
| MLFLOW_TRACKING_URI | MLflow backend | sqlite:///mlflow.db |
| AIRFLOW_HOME | Airflow home directory | ./airflow |

---

## License

MIT