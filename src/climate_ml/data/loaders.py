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
    """Suhu bulanan grid ERA5 (UC-4). Fallback ke era5_land jika kosong."""
    df = _read("""
        SELECT id, year, month, t2m_kelvin, t2m_celsius,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM era5_monthly
    """, engine)
    if df.empty:
        return load_era5_land(engine)
    return df


def load_era5_land(engine: Engine | None = None) -> pd.DataFrame:
    """ERA5-Land bulanan — resolusi lebih tinggi, ada soil features.
    Kolom distandarkan ke skema era5_monthly agar kompatibel dengan train_uc4."""
    df = _read("""
        SELECT id, year, month,
               t2m_c          AS t2m_celsius,
               total_precip_m AS precip_m,
               soil_temp_c,
               soil_moisture,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM era5_land_monthly
    """, engine)
    if not df.empty:
        df["t2m_kelvin"] = df["t2m_celsius"] + 273.15
    return df


def load_noaa_ghcn(engine: Engine | None = None) -> pd.DataFrame:
    """Observasi harian NOAA GHCN (tmax, tmin, tavg, prcp) per stasiun."""
    return _read("""
        SELECT id, station_id, station_name,
               EXTRACT(YEAR FROM date::date)::int AS year,
               date, tmax_c, tmin_c, tavg_c, prcp_mm, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM noaa_ghcn_daily
    """, engine)


def load_noaa_ghcn_monthly(engine: Engine | None = None) -> pd.DataFrame:
    """Observasi NOAA GHCN diagregasi ke bulanan — kompatibel dengan nasa_power_monthly."""
    return _read("""
        SELECT station_id AS location_label,
               station_name,
               EXTRACT(YEAR FROM date::date)::int  AS year,
               EXTRACT(MONTH FROM date::date)::int AS month,
               AVG(tavg_c)  AS t2m,
               MAX(tmax_c)  AS t2m_max,
               MIN(tmin_c)  AS t2m_min,
               SUM(prcp_mm) AS prectotcorr,
               ST_Y(geom) AS lat, ST_X(geom) AS lon,
               MIN(qc_flag) AS qc_flag
        FROM noaa_ghcn_daily
        WHERE tavg_c IS NOT NULL
        GROUP BY station_id, station_name, year, month, geom
        ORDER BY station_id, year, month
    """, engine)


def load_bnpb_disaster(engine: Engine | None = None) -> pd.DataFrame:
    """Data kejadian bencana BNPB per wilayah per tahun (UC-5 ML)."""
    return _read("""
        SELECT kode_wilayah, nama_wilayah, level, year,
               jumlah_kejadian, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM bnpb_disaster
        WHERE qc_flag = 'OK' OR qc_flag IS NULL
        ORDER BY kode_wilayah, year
    """, engine)


def load_worldcover(engine: Engine | None = None) -> pd.DataFrame:
    """Tutupan lahan ESA WorldCover per titik grid."""
    return _read("""
        SELECT id, year, landcover_class, landcover_name, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM worldcover_landuse
    """, engine)


def load_worldpop(engine: Engine | None = None) -> pd.DataFrame:
    """Populasi per titik grid (WorldPop)."""
    return _read("""
        SELECT id, year, population, qc_flag,
               ST_Y(geom) AS lat, ST_X(geom) AS lon
        FROM worldpop_population
    """, engine)
