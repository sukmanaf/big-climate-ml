"""Fitur spasial. Loader sudah mengekstrak lat/lon dari geom; di sini turunannya."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah proxy geografis sederhana.

    Catatan: lat<-5.5 & lon~119-120 di dataset PoC cenderung dataran tinggi
    (Toraja) yang lebih dingin (lihat laporan ETL Bab 7.8). Proxy ini akan
    digantikan kovariat DEM/ketinggian sebenarnya saat data tersedia.
    """
    out = df.copy()
    # Jarak kasar dari garis pantai selatan Sulsel sebagai proxy (placeholder).
    out["abs_lat"] = np.abs(out["lat"])
    return out
