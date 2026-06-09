"""Batch predict → DataFrame hasil sesuai skema ml_predictions.

Penulisan ke PostGIS dilakukan terpisah (butuh DB) agar fungsi inti tetap teruji
tanpa database.
"""
from __future__ import annotations

import argparse

import pandas as pd

from climate_ml.features.build import prepare_uc1_frame
from climate_ml.utils.io import load_artifact
from climate_ml.utils.logging import get_logger

log = get_logger()


def predict_uc1(df: pd.DataFrame, artifact_path: str) -> pd.DataFrame:
    """Inference UC-1 → DataFrame kolom skema ml_predictions."""
    pipeline, meta = load_artifact(artifact_path)
    prepared = prepare_uc1_frame(df)
    preds = pipeline.predict(prepared)
    proba = pipeline.predict_proba(prepared).max(axis=1)

    return pd.DataFrame({
        "use_case": "UC1",
        "model_name": meta.get("model_name", "unknown"),
        "model_version": meta.get("model_version", "dev"),
        "target": meta.get("target", "cuaca"),
        "predicted": preds,
        "proba": proba,
        "lat": df["lat"].to_numpy(),
        "lon": df["lon"].to_numpy(),
    })


def write_predictions(result: pd.DataFrame, engine=None) -> int:
    """Tulis hasil ke tabel ml_predictions (butuh DB). Mengembalikan jumlah baris."""
    from geoalchemy2 import Geometry  # noqa: F401  (registrasi tipe)
    from shapely.geometry import Point

    from climate_ml.data.db import get_engine

    engine = engine or get_engine()
    gdf = result.copy()
    gdf["geom"] = [Point(lon, lat).wkt for lon, lat in zip(gdf["lon"], gdf["lat"], strict=True)]
    gdf = gdf.drop(columns=["lat", "lon"])
    gdf.to_sql("ml_predictions", engine, if_exists="append", index=False)
    return len(gdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", default="UC1")
    parser.add_argument("--model", default="models/UC1_weather_clf_latest.joblib")
    args = parser.parse_args()

    from climate_ml.data.loaders import load_bmkg

    df = load_bmkg()
    result = predict_uc1(df, args.model)
    n = write_predictions(result)
    log.info("Ditulis %d prediksi ke ml_predictions", n)


if __name__ == "__main__":
    main()
