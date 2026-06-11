"""UC-2: regresi parameter iklim bulanan (suhu / curah hujan)."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from climate_ml.config import get_settings
from climate_ml.features.build import build_uc2_preprocessor
from climate_ml.models.registry import make_estimator


def build_uc2_pipeline(model_name: str = "gradient_boosting",
                       use_landcover: bool = False, **hp) -> Pipeline:
    seed = get_settings().random_seed
    estimator = make_estimator(model_name, random_state=seed, **hp)
    return Pipeline([
        ("pre", build_uc2_preprocessor(use_landcover=use_landcover)),
        ("reg", estimator),
    ])


def build_uc2_baseline(use_landcover: bool = False) -> Pipeline:
    """Baseline klimatologi: prediksi rata-rata target (DummyRegressor mean)."""
    return Pipeline([
        ("pre", build_uc2_preprocessor(use_landcover=use_landcover)),
        ("reg", make_estimator("dummy_reg", strategy="mean")),
    ])
