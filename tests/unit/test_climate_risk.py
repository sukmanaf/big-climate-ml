"""UC-5: climate risk score per zona RDTR."""
import pandas as pd

from climate_ml.models.climate_risk import compute_zone_risk


def _nasa_yogya():
    rows = []
    for lat in (-7.76, -7.79, -7.82):
        for lon in (110.35, 110.38, 110.41):
            for month in range(1, 13):
                rows.append({"provinsi": "DI Yogyakarta", "lat": lat, "lon": lon,
                             "month": month, "t2m_max": 30 + lat,  # variasi spasial
                             "prectotcorr": 5.0, "rh2m": 75.0})
    return pd.DataFrame(rows)


def _rdtr():
    return pd.DataFrame([
        {"nama_zona": "Z1", "kecamatan": "A", "kategori_zona": "Permukiman",
         "luas_ha": 100, "centroid_lat": -7.76, "centroid_lon": 110.35},
        {"nama_zona": "Z2", "kecamatan": "B", "kategori_zona": "RTH",
         "luas_ha": 50, "centroid_lat": -7.82, "centroid_lon": 110.41},
    ])


def test_risk_terurut_dan_lengkap():
    df = compute_zone_risk(_rdtr(), _nasa_yogya())
    assert list(df.columns).count("risk_score") == 1
    assert len(df) == 2
    # terurut desc
    assert df.iloc[0]["risk_score"] >= df.iloc[1]["risk_score"]
    assert set(df["risk_level"]) <= {"Rendah", "Sedang", "Tinggi"}


def test_risk_score_dalam_0_1():
    df = compute_zone_risk(_rdtr(), _nasa_yogya())
    assert df["risk_score"].between(0, 1).all()


def test_provinsi_tak_ada_error():
    import pytest
    nasa = _nasa_yogya().assign(provinsi="Sulawesi Selatan")
    with pytest.raises(ValueError, match="DI Yogyakarta"):
        compute_zone_risk(_rdtr(), nasa)
