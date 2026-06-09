"""UC-3: deteksi anomali / quality control — dua lapis (rule + IsolationForest)."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import IsolationForest

from climate_ml.config import get_config, get_settings
from climate_ml.data.validation import validate_ranges

log = logging.getLogger(__name__)

# Kolom fitur default untuk IsolationForest
UC3_FEATURE_COLS = ["suhu_c", "kelembaban_pct", "curah_hujan_mm", "kecepatan_angin_kmh"]


def rule_based_flags(df: pd.DataFrame, contract: dict | None = None) -> pd.Series:
    """True bila baris melanggar range fisik. Reason memakai format qc_flag laporan
    v4.1 (Bab 16.2): pipe-separated `RANGE:col=val|...`; 'OK' bila lolos."""
    contract = contract or get_config()["contract"]
    flags = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)
    for col, rule in contract.items():
        if not isinstance(rule, dict) or col not in df.columns:
            continue
        lo, hi = rule["min"], rule["max"]
        bad = ((df[col] < lo) | (df[col] > hi)).fillna(False)
        for idx in df.index[bad]:
            token = f"RANGE:{col}={df.at[idx, col]:g}"
            reasons[idx] = token if not reasons[idx] else f"{reasons[idx]}|{token}"
        flags |= bad
    reasons[~flags] = "OK"
    rule_based_flags.last_reasons = reasons  # disimpan untuk dipakai pemanggil
    return flags


def build_iso_model(df: pd.DataFrame, feature_cols: list[str], contamination: float = 0.05) -> IsolationForest:
    """Latih IsolationForest dari dataset referensi distribusi normal (training offline)."""
    feats = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    iso = IsolationForest(contamination=contamination, random_state=get_settings().random_seed)
    iso.fit(feats)
    return iso


def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    iso_model: IsolationForest | None = None,
) -> pd.DataFrame:
    """Gabungkan rule-based + IsolationForest. Mengembalikan DataFrame flag+score+reason.

    iso_model: model yang sudah dilatih offline (UC3_anomaly_detector_latest.joblib).
    Jika None, pakai fallback fit_predict — tidak akurat untuk 1 baris, log warning.
    """
    rules = rule_based_flags(df)
    reasons = getattr(rule_based_flags, "last_reasons", pd.Series("", index=df.index)).copy()

    feats = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    if iso_model is not None:
        # Path benar: inference pakai distribusi yang dipelajari dari 418 baris BMKG
        pred = iso_model.predict(feats)           # -1 = outlier
        score = -iso_model.score_samples(feats)   # makin tinggi makin anomali
    else:
        # Fallback: fit_predict pada input saja — hasil tidak bermakna untuk 1 baris
        log.warning(
            "UC3: iso_model tidak tersedia — fallback fit_predict (tidak akurat). "
            "Jalankan: python -m climate_ml.pipelines.train --use-case UC3 --config config/config.yaml"
        )
        iso = IsolationForest(contamination=contamination, random_state=get_settings().random_seed)
        pred = iso.fit_predict(feats)
        score = -iso.score_samples(feats)

    iso_flag = pred == -1
    reasons[iso_flag & (reasons == "")] = "isolation_forest"

    is_anomaly = rules | iso_flag
    return pd.DataFrame(
        {"is_anomaly": is_anomaly, "anomaly_score": score, "reason": reasons},
        index=df.index,
    )


def validate(df: pd.DataFrame):
    """Shortcut: hasil validasi range untuk pelaporan QC."""
    return validate_ranges(df)
