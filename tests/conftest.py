"""Fixture bersama. Data sintetis agar test berjalan tanpa database."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip test ber-marker `db` bila DATABASE_URL tak diset."""
    if os.getenv("DATABASE_URL"):
        return
    skip_db = pytest.mark.skip(reason="DATABASE_URL tidak diset")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)


@pytest.fixture
def bmkg_df() -> pd.DataFrame:
    """Data BMKG sintetis menyerupai struktur tabel bmkg_forecast.

    Dibuat agar `cuaca` punya hubungan dengan fitur (kelembaban/tutupan awan
    tinggi → Hujan) supaya model bisa mengalahkan baseline.
    """
    rng = np.random.default_rng(42)
    n = 240
    kelembaban = rng.uniform(50, 100, n)
    tutupan = rng.uniform(0, 100, n)
    suhu = rng.uniform(22, 34, n)
    # Label deterministik + sedikit noise
    cuaca = np.where(
        (kelembaban > 85) & (tutupan > 70), "Hujan Ringan",
        np.where(tutupan > 50, "Berawan",
                 np.where(tutupan > 20, "Cerah Berawan", "Cerah")),
    )
    hours = rng.choice([2, 5, 8, 11, 14, 17, 20, 23], n)
    months = rng.integers(1, 13, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "suhu_c": suhu,
        "kelembaban_pct": kelembaban,
        "curah_hujan_mm": np.where(cuaca == "Hujan Ringan", rng.uniform(1, 10, n), 0.0),
        "kecepatan_angin_kmh": rng.uniform(0, 20, n),
        "arah_angin_deg": rng.uniform(0, 360, n),
        "tutupan_awan_pct": tutupan,
        "cuaca": cuaca,
        "lat": rng.uniform(-6.9, -3.0, n),
        "lon": rng.uniform(107.6, 120.3, n),
        "datetime_local": [
            f"2023-{m:02d}-15 {h:02d}:00:00" for m, h in zip(months, hours, strict=True)
        ],
    })


@pytest.fixture
def nasa_df() -> pd.DataFrame:
    """NASA POWER sintetis dengan pola musim + ketinggian (dataran tinggi lebih dingin)."""
    rng = np.random.default_rng(7)
    rows = []
    locs = {
        "Makassar": (-5.14, 119.42, 27.0), "Bone": (-4.54, 120.33, 27.0),
        "Toraja": (-3.05, 119.85, 22.0), "Bandung": (-6.91, 107.61, 22.0),
    }
    for loc, (lat, lon, base) in locs.items():
        for month in range(1, 13):
            t2m = base + 1.5 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 0.3)
            rows.append({
                "location_label": loc, "lat": lat, "lon": lon,
                "year": 2023, "month": month,
                "t2m": t2m, "t2m_max": t2m + 4, "t2m_min": t2m - 5,
                "rh2m": rng.uniform(64, 91), "ws2m": rng.uniform(0.4, 5.7),
                "allsky_sfc_sw_dwn": rng.uniform(14, 26),
                "prectotcorr": rng.uniform(0, 16),
            })
    return pd.DataFrame(rows)
