import numpy as np
import pandas as pd

from climate_ml.features.temporal import add_temporal_features, cyclical_encode


def test_cyclical_encode_kontinu_di_batas():
    """Bulan 12 dan bulan 0 harus dekat di ruang sin/cos (kontinuitas siklik)."""
    s = pd.Series([0, 12, 6], name="month")
    enc = cyclical_encode(s, 12)
    # month 0 dan 12 identik
    assert np.isclose(enc.iloc[0]["month_sin"], enc.iloc[1]["month_sin"])
    assert np.isclose(enc.iloc[0]["month_cos"], enc.iloc[1]["month_cos"])
    # month 6 (setengah periode) cos = -1
    assert np.isclose(enc.iloc[2]["month_cos"], -1.0)


def test_add_temporal_menurunkan_jam_bulan_musim():
    df = pd.DataFrame({"datetime_local": ["2023-01-15 20:00:00", "2023-07-15 08:00:00"]})
    out = add_temporal_features(df)
    assert out.loc[0, "hour"] == 20
    assert out.loc[0, "month"] == 1
    assert out.loc[0, "musim"] == "basah"
    assert out.loc[1, "musim"] == "kemarau"


def test_lat_lon_tidak_tertukar(bmkg_df):
    """lat selalu negatif (Indonesia selatan khatulistiwa di sampel); lon positif besar."""
    assert (bmkg_df["lat"] < 0).all()
    assert (bmkg_df["lon"] > 100).all()
