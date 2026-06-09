import pandas as pd

from climate_ml.models.anomaly_detector import detect_anomalies, rule_based_flags

CONTRACT = {
    "suhu_c": {"min": -50, "max": 60},
    "kelembaban_pct": {"min": 0, "max": 100},
}


def test_rule_based_menangkap_pelanggaran():
    df = pd.DataFrame({"suhu_c": [28, 999], "kelembaban_pct": [80, 50]})
    flags = rule_based_flags(df, CONTRACT)
    assert not flags.iloc[0]
    assert flags.iloc[1]  # suhu 999 = anomali


def test_recall_rule_based_satu(bmkg_df):
    """Quality gate UC-3: nilai di luar range fisik HARUS tertangkap (recall=1)."""
    df = bmkg_df.copy()
    df.loc[0, "suhu_c"] = 200  # tanam anomali jelas
    df.loc[1, "kelembaban_pct"] = -10
    flags = rule_based_flags(df, CONTRACT)
    assert flags.iloc[0] and flags.iloc[1]


def test_detect_menggabungkan_rule_dan_isolation(bmkg_df):
    res = detect_anomalies(
        bmkg_df, feature_cols=["suhu_c", "kelembaban_pct", "tutupan_awan_pct"]
    )
    assert {"is_anomaly", "anomaly_score", "reason"} <= set(res.columns)
    assert len(res) == len(bmkg_df)
