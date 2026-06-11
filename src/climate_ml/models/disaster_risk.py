"""UC-5 ML: prediksi risiko bencana per wilayah dari profil iklim.

Training data: bnpb_disaster (jumlah_kejadian per wilayah per tahun) di-join
ke titik NASA POWER terdekat → fitur iklim tahunan sebagai prediktor.

Model: GradientBoostingRegressor memprediksi avg_kejadian_per_tahun.
Output risk_level dikategorikan dari prediksi (Rendah/Sedang/Tinggi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from climate_ml.config import get_settings


_CLIMATE_FEATURES = ["avg_t2m", "avg_precip", "avg_rh2m", "avg_tmax", "lat", "lon"]


def _nearest_climate(bnpb_row: pd.Series, climate_pts: pd.DataFrame) -> pd.Series:
    """Kembalikan baris climate_pts terdekat ke koordinat bnpb_row."""
    d = (climate_pts["lat"] - bnpb_row["lat"]) ** 2 + \
        (climate_pts["lon"] - bnpb_row["lon"]) ** 2
    return climate_pts.loc[d.idxmin()]


def build_uc5_training_data(
    bnpb_df: pd.DataFrame,
    nasa_df: pd.DataFrame,
) -> pd.DataFrame:
    """Gabungkan bnpb_disaster dengan profil iklim NASA POWER.

    Untuk setiap wilayah BNPB, cari titik NASA POWER terdekat dan ambil
    rata-rata tahunan iklimnya sebagai fitur.

    Return DataFrame dengan kolom: lat, lon, avg_t2m, avg_precip, avg_rh2m,
    avg_tmax, avg_kejadian_per_tahun.
    """
    # Profil iklim tahunan per titik NASA POWER
    climate_profile = (
        nasa_df.groupby(["location_label", "lat", "lon"])
        .agg(
            avg_t2m=("t2m", "mean"),
            avg_tmax=("t2m_max", "mean"),
            avg_precip=("prectotcorr", "mean"),
            avg_rh2m=("rh2m", "mean"),
        )
        .reset_index()
    )

    # Agregasi BNPB: avg kejadian per tahun per wilayah
    bnpb_agg = (
        bnpb_df.groupby(["kode_wilayah", "nama_wilayah", "lat", "lon"])
        .agg(avg_kejadian_per_tahun=("jumlah_kejadian", "mean"))
        .reset_index()
    )

    # Nearest-point join
    rows = []
    for _, row in bnpb_agg.iterrows():
        cp = _nearest_climate(row, climate_profile)
        rows.append({
            "lat": row["lat"],
            "lon": row["lon"],
            "nama_wilayah": row["nama_wilayah"],
            "avg_t2m": cp["avg_t2m"],
            "avg_tmax": cp["avg_tmax"],
            "avg_precip": cp["avg_precip"],
            "avg_rh2m": cp["avg_rh2m"],
            "nearest_climate": cp["location_label"],
            "avg_kejadian_per_tahun": row["avg_kejadian_per_tahun"],
        })

    return pd.DataFrame(rows)


def build_uc5_pipeline(**hp) -> Pipeline:
    """Gradient Boosting untuk prediksi frekuensi bencana dari iklim."""
    seed = get_settings().random_seed
    reg = GradientBoostingRegressor(
        n_estimators=hp.get("n_estimators", 200),
        max_depth=hp.get("max_depth", 3),
        learning_rate=hp.get("learning_rate", 0.05),
        random_state=seed,
    )
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", reg),
    ])


def build_uc5_baseline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", DummyRegressor(strategy="mean")),
    ])


def risk_level_from_score(score: float, q33: float, q66: float) -> str:
    if score <= q33:
        return "Rendah"
    if score <= q66:
        return "Sedang"
    return "Tinggi"
