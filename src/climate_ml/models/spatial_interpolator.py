"""UC-4: interpolasi spasial suhu ERA5 (downscaling lat/lon/month → t2m_celsius).

Baseline: RandomForest(lat, lon, month). Roadmap: Kriging + kovariat DEM.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from climate_ml.config import get_settings
from climate_ml.features.build import build_uc4_preprocessor


def build_uc4_pipeline(use_soil: bool = False, **hp) -> Pipeline:
    """Pipeline lengkap preprocessor + RF untuk UC-4.
    use_soil=True bila data dari era5_land_monthly (ada kolom soil_temp_c)."""
    seed = get_settings().random_seed
    reg = RandomForestRegressor(random_state=seed, **hp)
    return Pipeline([
        ("pre", build_uc4_preprocessor(use_soil=use_soil)),
        ("reg", reg),
    ])


def build_uc4_baseline(use_soil: bool = False) -> Pipeline:
    """Baseline: DummyRegressor mean."""
    from sklearn.dummy import DummyRegressor
    return Pipeline([
        ("pre", build_uc4_preprocessor(use_soil=use_soil)),
        ("reg", DummyRegressor(strategy="mean")),
    ])
