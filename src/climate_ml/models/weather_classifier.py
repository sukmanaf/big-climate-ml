"""UC-1: klasifikasi kondisi cuaca dari fitur meteorologi."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from climate_ml.config import get_settings
from climate_ml.features.build import build_uc1_preprocessor
from climate_ml.models.registry import make_estimator


def build_uc1_pipeline(model_name: str = "random_forest", **hp) -> Pipeline:
    """Pipeline lengkap preprocessing + classifier untuk UC-1."""
    seed = get_settings().random_seed
    estimator = make_estimator(model_name, random_state=seed, **hp)
    return Pipeline([
        ("pre", build_uc1_preprocessor()),
        ("clf", estimator),
    ])


def build_uc1_baseline() -> Pipeline:
    """Baseline DummyClassifier (most_frequent) untuk quality gate."""
    return Pipeline([
        ("pre", build_uc1_preprocessor()),
        ("clf", make_estimator("dummy_clf", strategy="most_frequent")),
    ])
