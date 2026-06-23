"""
src/utils/config.py

Central configuration loader for the ML pipeline.
"""

from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent


def load_yaml(filename: str) -> dict:
    path = ROOT_DIR / "configs" / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


_pipeline_cfg = load_yaml("pipeline_config.yaml")
_model_cfg    = load_yaml("model_config.yaml")
_feature_cfg  = load_yaml("feature_config.yaml")
_airflow_cfg  = load_yaml("airflow_config.yaml")


class DataConfig:
    dataset_name:   str   = _pipeline_cfg["data"]["dataset_name"]
    source_url:     str   = _pipeline_cfg["data"]["source_url"]
    target_column:  str   = _pipeline_cfg["data"]["target_column"]
    test_size:      float = _pipeline_cfg["data"]["test_size"]
    val_size:       float = _pipeline_cfg["data"]["val_size"]
    random_state:   int   = _pipeline_cfg["data"]["random_state"]


class FeatureConfig:
    numerical:    list = _pipeline_cfg["features"]["numerical"]
    categorical:  list = _pipeline_cfg["features"]["categorical"]
    target:       str  = _pipeline_cfg["features"]["target"]
    scaling:      dict = _feature_cfg["scaling"]
    encoding:     dict = _feature_cfg["encoding"]
    store_path:   str  = _feature_cfg["feature_store"]["store_path"]


class TrainingConfig:
    models:           list = _pipeline_cfg["training"]["models"]
    cv_folds:         int  = _pipeline_cfg["training"]["cv_folds"]
    scoring_metric:   str  = _pipeline_cfg["training"]["scoring_metric"]
    random_state:     int  = _pipeline_cfg["training"]["random_state"]
    model_params:     dict = _model_cfg


class MLflowConfig:
    experiment_name:    str = _pipeline_cfg["mlflow"]["experiment_name"]
    tracking_uri:       str = _pipeline_cfg["mlflow"]["tracking_uri"]
    model_registry_name: str = _pipeline_cfg["mlflow"]["model_registry_name"]


class MonitoringConfig:
    drift_threshold:       float = _pipeline_cfg["retraining"]["drift_threshold"]
    performance_threshold: float = _pipeline_cfg["retraining"]["performance_threshold"]
    retrain_on_drift:      bool  = _pipeline_cfg["retraining"]["retrain_on_drift"]


class AirflowConfig:
    training_dag:   dict = _airflow_cfg["training_pipeline"]
    monitoring_dag: dict = _airflow_cfg["monitoring_pipeline"]
    retraining_dag: dict = _airflow_cfg["retraining_pipeline"]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_env:              str = Field(default="development")
    log_level:            str = Field(default="INFO")
    mlflow_tracking_uri:  str = Field(default="sqlite:///mlflow.db")
    airflow_home:         str = Field(default="./airflow")


class Settings:
    data:       DataConfig       = DataConfig()
    features:   FeatureConfig    = FeatureConfig()
    training:   TrainingConfig   = TrainingConfig()
    mlflow:     MLflowConfig     = MLflowConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    airflow:    AirflowConfig    = AirflowConfig()
    env:        EnvSettings      = EnvSettings()
    root_dir:   Path             = ROOT_DIR


settings = Settings()