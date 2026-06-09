"""Orkestrasi training+predict sebagai Prefect flow (re-use stack ETL).

Opsional — pipeline tetap bisa dijalankan langsung via train.py/predict.py.
"""
from __future__ import annotations

from prefect import flow, task

from climate_ml.config import load_model_config
from climate_ml.data.loaders import load_bmkg
from climate_ml.pipelines.predict import predict_uc1, write_predictions
from climate_ml.pipelines.train import train_uc1


@task
def task_train_uc1(config_path: str) -> dict:
    return train_uc1(load_model_config(config_path))


@task
def task_predict_uc1(artifact_path: str) -> int:
    df = load_bmkg()
    return write_predictions(predict_uc1(df, artifact_path))


@flow(name="climate-ml-uc1", log_prints=True)
def climate_ml_uc1(config_path: str = "config/models/uc1_weather_clf.yaml") -> None:
    result = task_train_uc1(config_path)
    task_predict_uc1(result["artifact"])


if __name__ == "__main__":
    climate_ml_uc1()
