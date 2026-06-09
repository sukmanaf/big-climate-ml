"""Validasi kontrak data pada data NYATA di PostGIS.

Butuh DB → ditandai marker `db`, otomatis di-skip bila DATABASE_URL tak diset.
"""
import pytest

from climate_ml.data.loaders import load_bmkg
from climate_ml.data.validation import check_min_rows, validate_ranges


@pytest.mark.db
def test_bmkg_memenuhi_kontrak():
    df = load_bmkg()
    assert validate_ranges(df).ok, "data BMKG melanggar range fisik"
    assert "cuaca" in df.columns


@pytest.mark.db
def test_bmkg_cukup_untuk_training():
    assert check_min_rows(load_bmkg()).ok
