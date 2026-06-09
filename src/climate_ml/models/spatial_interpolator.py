"""UC-4: interpolasi spasial suhu ERA5 (downscaling lat/lon/month → t2m_celsius).

Baseline: RandomForest(lat, lon, month). Roadmap: Kriging + kovariat DEM.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from climate_ml.config import get_settings
from climate_ml.features.build import build_uc4_preprocessor


def build_uc4_pipeline(**hp) -> Pipeline:
    """Pipeline lengkap preprocessor + RF untuk UC-4."""
    seed = get_settings().random_seed
    reg = RandomForestRegressor(random_state=seed, **hp)
    return Pipeline([
        ("pre", build_uc4_preprocessor()),
        ("reg", reg),
    ])


def build_uc4_baseline() -> Pipeline:
    """Baseline: DummyRegressor mean (rata-rata suhu semua grid)."""
    from sklearn.dummy import DummyRegressor
    return Pipeline([
        ("pre", build_uc4_preprocessor()),
        ("reg", DummyRegressor(strategy="mean")),
    ])
