"""Fitur temporal: encoding siklik + musim (iklim tropis Indonesia)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cyclical_encode(values: pd.Series, period: float) -> pd.DataFrame:
    """Encode nilai periodik menjadi sin/cos agar batas (mis. Des↔Jan) kontinu."""
    radians = 2 * np.pi * values.astype(float) / period
    return pd.DataFrame(
        {f"{values.name}_sin": np.sin(radians), f"{values.name}_cos": np.cos(radians)},
        index=values.index,
    )


def add_temporal_features(df: pd.DataFrame, datetime_col: str = "datetime_local") -> pd.DataFrame:
    """Turunkan hour, month (siklik), dan musim dari kolom datetime string/Timestamp."""
    out = df.copy()
    dt = pd.to_datetime(out[datetime_col], errors="coerce")
    out["hour"] = dt.dt.hour
    out["month"] = dt.dt.month
    out = pd.concat([out, cyclical_encode(out["hour"].rename("hour"), 24)], axis=1)
    out = pd.concat([out, cyclical_encode(out["month"].rename("month"), 12)], axis=1)
    # Musim: basah (Nov-Apr) vs kemarau (Mei-Okt)
    out["musim"] = np.where(out["month"].isin([11, 12, 1, 2, 3, 4]), "basah", "kemarau")
    return out
