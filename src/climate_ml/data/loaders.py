"""Memuat data dari tabel PostGIS ke pandas DataFrame.

Kolom geometri di-ekstrak menjadi lat/lon eksplisit (ST_Y=lat, ST_X=lon) —
lihat gotcha urutan koordinat POINT(lon, lat) di laporan PoC ETL Bab 12.5.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from climate_ml.data.db import get_engine


def _read(sql: str, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def load_bmkg(engine: Engine | None = None) -> pd.DataFrame:
    """Prakiraan cuaca BMKG per desa/kecamatan (UC-1, UC-3). Termasuk qc_flag."""
    sql = """
        SELECT id, provinsi, kotkab, kecamatan, desa, kode_adm4,
               datetime_utc, datetime_local,
               suhu_c, kelembaban_pct, curah_hujan_mm, kecepatan_angin_kmh,
               arah_angin_deg, tutupan_awan_pct, cuaca, cuaca_en, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM bmkg_forecast
    """
    return _read(sql, engine)


def load_nasa_power(engine: Engine | None = None) -> pd.DataFrame:
    """Parameter iklim bulanan NASA POWER (UC-2). Termasuk qc_flag."""
    sql = """
        SELECT id, location_label, provinsi, year, month,
               t2m, t2m_max, t2m_min, allsky_sfc_sw_dwn, rh2m, ws2m, prectotcorr, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM nasa_power_monthly
    """
    return _read(sql, engine)


def load_chirps(engine: Engine | None = None) -> pd.DataFrame:
    """Curah hujan satelit CHIRPS bulanan grid (Phase 4)."""
    sql = """
        SELECT id, year, month, precipitation_mm, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM chirps_monthly
    """
    return _read(sql, engine)


def load_rdtr(engine: Engine | None = None) -> pd.DataFrame:
    """Zona tata ruang RDTR (Phase 3). Centroid diekstrak dari polygon."""
    sql = """
        SELECT id, kota, kecamatan, kelurahan, kode_zona, nama_zona,
               kategori_zona, luas_ha, sumber_data,
               ST_Y(ST_Centroid(geom)) AS centroid_lat,
               ST_X(ST_Centroid(geom)) AS centroid_lon
        FROM rdtr_pola_ruang
    """
    return _read(sql, engine)


def load_era5(engine: Engine | None = None) -> pd.DataFrame:
    """Suhu bulanan grid ERA5 (UC-2, UC-4)."""
    sql = """
        SELECT id, year, month, t2m_kelvin, t2m_celsius,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM era5_monthly
    """
    return _read(sql, engine)
