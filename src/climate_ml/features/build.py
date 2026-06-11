"""Membangun preprocessing sklearn (ColumnTransformer) per use case.

Semua transformasi hidup di dalam Pipeline agar fit hanya pada train fold
(anti-leakage) dan identik antara training & serving (anti-skew).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from climate_ml.features.spatial import add_spatial_features
from climate_ml.features.temporal import add_temporal_features


class SinCosTransformer(BaseEstimator, TransformerMixin):
    """Encoding siklik → [sin, cos]. Kelas level-modul agar artefak picklable."""

    def __init__(self, period: float = 360.0):
        self.period = period

    def fit(self, X, y=None):
        # Tandai sebagai fitted (atribut ber-underscore) agar check_is_fitted lolos.
        # Tanpa ini, sklearn >=1.8 menganggap transformer belum di-fit dan error.
        self.n_features_in_ = np.asarray(X).shape[1] if np.asarray(X).ndim > 1 else 1
        self.fitted_ = True
        return self

    def transform(self, X):
        rad = 2 * np.pi * np.asarray(X, dtype=float) / self.period
        return np.concatenate([np.sin(rad), np.cos(rad)], axis=1)


def _sin_cos(period: float) -> SinCosTransformer:
    return SinCosTransformer(period=period)


def prepare_uc1_frame(df):
    """Turunkan fitur temporal & spasial mentah untuk UC-1 sebelum ColumnTransformer."""
    df = add_temporal_features(df)
    df = add_spatial_features(df)
    return df


def build_uc1_preprocessor() -> ColumnTransformer:
    """Preprocessor klasifikasi cuaca (UC-1)."""
    numeric = ["suhu_c", "kelembaban_pct", "kecepatan_angin_kmh", "tutupan_awan_pct",
               "lat", "lon"]
    cyclical = ["arah_angin_deg", "hour", "month"]
    categorical = ["musim"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cyclical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("encode", _sin_cos(360)),  # arah_angin; hour/month sudah kecil tapi tetap aman dimodulo
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cyc", cyclical_pipe, cyclical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


def enrich_uc2_with_worldcover(df: pd.DataFrame, wc_df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom landcover_class ke df NASA POWER via nearest-point join.

    Setiap titik NASA POWER mendapat kelas tutupan lahan ESA WorldCover
    dari titik WorldCover terdekat.
    """
    if wc_df is None or wc_df.empty:
        return df
    wc_pts = wc_df[["lat", "lon", "landcover_class"]].copy()
    classes = []
    for _, row in df.iterrows():
        d = (wc_pts["lat"] - row["lat"]) ** 2 + (wc_pts["lon"] - row["lon"]) ** 2
        classes.append(int(wc_pts.loc[d.idxmin(), "landcover_class"]))
    df = df.copy()
    df["landcover_class"] = classes
    return df


def build_uc2_preprocessor(use_landcover: bool = False) -> ColumnTransformer:
    """Preprocessor regresi iklim bulanan (UC-2).

    t2m_max/t2m_min sengaja TIDAK dipakai sebagai fitur untuk target t2m —
    keduanya hampir menentukan t2m secara langsung (leakage trivial). Model
    belajar pola dari musim (month) + lokasi (lat/lon) + parameter lain.
    use_landcover=True bila kolom landcover_class tersedia dari WorldCover.
    """
    numeric = ["rh2m", "ws2m", "allsky_sfc_sw_dwn", "lat", "lon"]
    cyclical = ["month"]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [
        ("num", numeric_pipe, numeric),
        ("cyc", _sin_cos(12), cyclical),
    ]
    if use_landcover:
        transformers.append(
            ("lc", OneHotEncoder(handle_unknown="ignore"), ["landcover_class"])
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_uc4_frame(df):
    """Pilih kolom fitur UC-4. Mendukung era5_monthly dan era5_land_monthly.
    Fitur tambahan soil_temp_c dipakai bila tersedia (era5_land)."""
    target_col = "t2m_celsius" if "t2m_celsius" in df.columns else None
    feature_cols = ["lat", "lon", "month"]
    if "soil_temp_c" in df.columns:
        feature_cols.append("soil_temp_c")
    if target_col:
        feature_cols.append(target_col)
    return df[feature_cols].copy()


def build_uc4_preprocessor(use_soil: bool = False) -> ColumnTransformer:
    """Preprocessor interpolasi spasial (UC-4): lokasi + bulan + soil_temp opsional."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    num_cols = ["lat", "lon", "soil_temp_c"] if use_soil else ["lat", "lon"]
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cyc", _sin_cos(12), ["month"]),
        ],
        remainder="drop",
    )
