"""Evaluasi model + quality gate (Bab 9.4 dokumen teknis)."""
from __future__ import annotations

from sklearn.model_selection import cross_val_predict

from climate_ml.features.build import prepare_uc1_frame, prepare_uc4_frame
from climate_ml.models.climate_regressor import build_uc2_baseline, build_uc2_pipeline
from climate_ml.models.spatial_interpolator import build_uc4_baseline, build_uc4_pipeline
from climate_ml.models.weather_classifier import build_uc1_baseline, build_uc1_pipeline
from climate_ml.utils.metrics import classification_metrics, regression_metrics, skill_score


def evaluate_uc1(df, target: str = "cuaca", n_splits: int = 5, margin: float = 0.10) -> dict:
    """Cross-val macro-F1 model vs baseline. Mengembalikan metrik + status gate."""
    df = prepare_uc1_frame(df)
    X, y = df, df[target]

    # n_splits aman untuk dataset kecil
    splits = min(n_splits, int(y.value_counts().min()), 5)
    splits = max(splits, 2)

    model_pred = cross_val_predict(build_uc1_pipeline(), X, y, cv=splits)
    base_pred = cross_val_predict(build_uc1_baseline(), X, y, cv=splits)

    model_m = classification_metrics(y, model_pred)
    base_m = classification_metrics(y, base_pred)
    passed = model_m["macro_f1"] >= base_m["macro_f1"] + margin
    return {
        "model": model_m,
        "baseline": base_m,
        "quality_gate_passed": bool(passed),
        "margin": margin,
    }


def evaluate_uc4(df, target: str = "t2m_celsius", n_splits: int = 5) -> dict:
    """Cross-val interpolasi spasial ERA5 vs baseline (mean). Gate: skill_score > 0."""
    df = prepare_uc4_frame(df)
    X, y = df, df[target]
    splits = max(2, min(n_splits, len(df) // 10))

    model_pred = cross_val_predict(build_uc4_pipeline(), X, y, cv=splits)
    base_pred = cross_val_predict(build_uc4_baseline(), X, y, cv=splits)

    model_m = regression_metrics(y, model_pred)
    base_m = regression_metrics(y, base_pred)
    skill = skill_score(model_m["mae"], base_m["mae"])
    return {
        "model": model_m,
        "baseline": base_m,
        "skill_score": skill,
        "quality_gate_passed": bool(skill > 0),
    }


def evaluate_uc2(df, target: str = "t2m", n_splits: int = 5) -> dict:
    """Cross-val regresi vs baseline klimatologi (mean). Gate: skill_score > 0."""
    X, y = df, df[target]
    splits = max(2, min(n_splits, len(df) // 4))

    model_pred = cross_val_predict(build_uc2_pipeline(), X, y, cv=splits)
    base_pred = cross_val_predict(build_uc2_baseline(), X, y, cv=splits)

    model_m = regression_metrics(y, model_pred)
    base_m = regression_metrics(y, base_pred)
    skill = skill_score(model_m["mae"], base_m["mae"])
    return {
        "model": model_m,
        "baseline": base_m,
        "skill_score": skill,
        "quality_gate_passed": bool(skill > 0),
    }
