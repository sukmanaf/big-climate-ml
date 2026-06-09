import pandas as pd

from climate_ml.data.validation import check_min_rows, filter_qc_clean, validate_ranges


def test_filter_qc_clean_buang_flagged():
    df = pd.DataFrame({"suhu_c": [28, 99, 30], "qc_flag": ["OK", "RANGE:suhu_c=99", "OK"]})
    out = filter_qc_clean(df)
    assert len(out) == 2
    assert (out["qc_flag"] == "OK").all()


def test_filter_qc_clean_tanpa_kolom_aman():
    df = pd.DataFrame({"suhu_c": [28, 30]})
    assert len(filter_qc_clean(df)) == 2

CONTRACT = {
    "suhu_c": {"min": -50, "max": 60},
    "kelembaban_pct": {"min": 0, "max": 100},
    "min_rows_for_training": 50,
}


def test_range_menolak_suhu_tidak_wajar():
    df = pd.DataFrame({"suhu_c": [28, 999, -100], "kelembaban_pct": [80, 50, 60]})
    res = validate_ranges(df, CONTRACT)
    assert not res.ok
    assert any("suhu_c" in v for v in res.violations)


def test_range_meloloskan_data_wajar():
    df = pd.DataFrame({"suhu_c": [28, 30], "kelembaban_pct": [80, 90]})
    assert validate_ranges(df, CONTRACT).ok


def test_range_menolak_kelembaban_di_atas_100():
    df = pd.DataFrame({"suhu_c": [28], "kelembaban_pct": [150]})
    assert not validate_ranges(df, CONTRACT).ok


def test_min_rows_gagal_cepat():
    df = pd.DataFrame({"suhu_c": [28] * 10})
    res = check_min_rows(df, CONTRACT)
    assert not res.ok


def test_raise_if_invalid_melempar():
    df = pd.DataFrame({"suhu_c": [999]})
    try:
        validate_ranges(df, CONTRACT).raise_if_invalid()
        raise AssertionError("seharusnya melempar ValueError")
    except ValueError:
        pass
