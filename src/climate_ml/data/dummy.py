"""Loader data dummy dari JSON — alternatif PostGIS untuk demo/testing.

Mengembalikan DataFrame dengan kolom identik dengan loaders.py (lat/lon sudah ada),
sehingga pipeline ML tidak peduli sumbernya DB atau JSON.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from climate_ml.config import PROJECT_ROOT

DUMMY_DIR = PROJECT_ROOT / "data" / "dummy"


def _load(name: str) -> pd.DataFrame:
    path = DUMMY_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada. Jalankan: python scripts/generate_dummy_data.py"
        )
    return pd.read_json(path)


def load_bmkg_dummy() -> pd.DataFrame:
    return _load("bmkg_forecast.json")


def load_nasa_power_dummy() -> pd.DataFrame:
    return _load("nasa_power_monthly.json")


def load_chirps_dummy() -> pd.DataFrame:
    """Curah hujan satelit CHIRPS grid Yogyakarta (Phase 4)."""
    return _load("chirps_monthly.json")


def load_rdtr_dummy() -> pd.DataFrame:
    """Zona tata ruang RDTR Kota Yogyakarta (Phase 3)."""
    return _load("rdtr_pola_ruang.json")
