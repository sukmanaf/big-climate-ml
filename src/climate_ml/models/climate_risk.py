"""UC-5: Climate Risk Score per zona RDTR (Phase 3 laporan PoC v4.1).

Mereplikasi analisis overlay PostGIS (Bab 14 laporan) di sisi Python:
setiap zona RDTR diberi data iklim dari titik NASA POWER terdekat (nearest point
ke centroid), lalu dihitung composite risk score ternormalisasi:

    risk_score = 0.5 * norm(avg_tmax) + 0.3 * norm(avg_precip) + 0.2 * norm(avg_humidity)

norm(x) = (x - min) / (max - min) lintas zona. Bobot 0.5/0.3/0.2 sesuai laporan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W_TMAX, W_PRECIP, W_HUMID = 0.5, 0.3, 0.2


def _nearest_point(lat: float, lon: float, points: pd.DataFrame) -> pd.Series:
    d = (points["lat"] - lat) ** 2 + (points["lon"] - lon) ** 2
    return points.loc[d.idxmin()]


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def compute_zone_risk(rdtr_df: pd.DataFrame, nasa_df: pd.DataFrame,
                      provinsi: str = "DI Yogyakarta") -> pd.DataFrame:
    """Hitung climate risk score per zona. Mengembalikan DataFrame terurut desc."""
    grid = nasa_df[nasa_df["provinsi"] == provinsi]
    if grid.empty:
        raise ValueError(f"Tidak ada titik NASA POWER untuk provinsi '{provinsi}'")
    locs = grid.groupby(["lat", "lon"]).agg(
        avg_tmax=("t2m_max", "mean"),
        avg_precip=("prectotcorr", "mean"),
        avg_humidity=("rh2m", "mean"),
    ).reset_index()

    rows = []
    for _, z in rdtr_df.iterrows():
        np_pt = _nearest_point(z["centroid_lat"], z["centroid_lon"], locs)
        rows.append({
            "nama_zona": z["nama_zona"],
            "kecamatan": z["kecamatan"],
            "kategori_zona": z["kategori_zona"],
            "luas_ha": z["luas_ha"],
            "centroid_lat": z["centroid_lat"],
            "centroid_lon": z["centroid_lon"],
            "avg_tmax": round(float(np_pt["avg_tmax"]), 1),
            "avg_precip": round(float(np_pt["avg_precip"]), 2),
            "avg_humidity": round(float(np_pt["avg_humidity"]), 0),
        })
    out = pd.DataFrame(rows)
    out["risk_score"] = (
        W_TMAX * _norm(out["avg_tmax"])
        + W_PRECIP * _norm(out["avg_precip"])
        + W_HUMID * _norm(out["avg_humidity"])
    ).round(3)
    out["risk_level"] = pd.cut(
        out["risk_score"], bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Rendah", "Sedang", "Tinggi"],
    ).astype(str)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
