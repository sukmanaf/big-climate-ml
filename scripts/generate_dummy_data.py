"""Generate data dummy JSON yang menyerupai struktur tabel PoC ETL v4.1.

Selaras dengan laporan-poc-etl-climate-data.md v4.1 (Phase 1-4):
  - bmkg_forecast      : prakiraan cuaca per desa/kecamatan (+ Yogyakarta) + qc_flag
  - nasa_power_monthly : 7 param iklim bulanan (+ grid Yogyakarta) + qc_flag
  - chirps_monthly     : curah hujan satelit grid Yogyakarta (Phase 4) + qc_flag
  - rdtr_pola_ruang    : 14 zona tata ruang Kota Yogyakarta (Phase 3, polygon)
  - qc_log             : ringkasan quality control per sumber (Phase 4)

Dipakai untuk demo ML + frontend tanpa PostgreSQL (data real masih diproses).
Kolom lat/lon sudah diekstrak (seperti output loaders.py).

Jalankan: python scripts/generate_dummy_data.py  → data/dummy/*.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dummy"

# ---------------------------------------------------------------------------
# BMKG: desa/kecamatan (kode, nama, kecamatan, kota, provinsi, lat, lon, base°C)
# 8 desa Phase 1 (Makassar/Bandung) + kecamatan Kota Yogyakarta (Phase 4).
# ---------------------------------------------------------------------------
DESA = [
    ("73.71.01.1001", "Bontorannu", "Mariso", "Makassar", "Sulawesi Selatan", -5.1622, 119.4043, 28.0),
    ("73.71.01.1002", "Kampung Buyang", "Mariso", "Makassar", "Sulawesi Selatan", -5.1480, 119.4100, 28.0),
    ("73.71.01.1003", "Lette", "Mariso", "Makassar", "Sulawesi Selatan", -5.1550, 119.4080, 28.0),
    ("73.71.03.1001", "Bontomakkio", "Rappocini", "Makassar", "Sulawesi Selatan", -5.1700, 119.4350, 28.0),
    ("73.71.04.1001", "Mangasa", "Tamalate", "Makassar", "Sulawesi Selatan", -5.1850, 119.4200, 28.0),
    ("32.73.01.1001", "Babakan", "Babakan Ciparay", "Bandung", "Jawa Barat", -6.9300, 107.5800, 24.0),
    ("32.73.02.1001", "Cihaurgeulis", "Cibeunying Kaler", "Bandung", "Jawa Barat", -6.8950, 107.6300, 24.0),
    ("32.73.06.1001", "Sukapada", "Cibeunying Kidul", "Bandung", "Jawa Barat", -6.9050, 107.6500, 24.0),
    # Kota Yogyakarta (dataran rendah ~26°C) — Phase 3/4
    ("34.71.07.1001", "Umbulharjo", "Umbulharjo", "Kota Yogyakarta", "DI Yogyakarta", -7.8010, 110.3850, 26.0),
    ("34.71.08.1001", "Kotagede", "Kotagede", "Kota Yogyakarta", "DI Yogyakarta", -7.8230, 110.4000, 26.0),
    ("34.71.06.1001", "Gondokusuman", "Gondokusuman", "Kota Yogyakarta", "DI Yogyakarta", -7.7820, 110.3800, 26.0),
    ("34.71.14.1001", "Tegalrejo", "Tegalrejo", "Kota Yogyakarta", "DI Yogyakarta", -7.7820, 110.3550, 26.0),
    ("34.71.04.1001", "Mergangsan", "Mergangsan", "Kota Yogyakarta", "DI Yogyakarta", -7.8150, 110.3700, 26.0),
    ("34.71.13.1001", "Jetis", "Jetis", "Kota Yogyakarta", "DI Yogyakarta", -7.7780, 110.3650, 26.0),
]

# ---------------------------------------------------------------------------
# NASA POWER: (lokasi, provinsi, lat, lon, base°C). 7 kota + grid 3x3 Yogyakarta.
# ---------------------------------------------------------------------------
NASA_LOC = [
    ("Makassar", "Sulawesi Selatan", -5.14, 119.42, 27.0),
    ("Bone", "Sulawesi Selatan", -4.54, 120.33, 27.0),
    ("Toraja", "Sulawesi Selatan", -3.05, 119.85, 22.0),
    ("Bandung", "Jawa Barat", -6.91, 107.61, 22.5),
    ("Kota Batu", "Jawa Timur", -7.87, 112.52, 20.0),
    ("Jakarta", "DKI Jakarta", -6.21, 106.85, 28.0),
    ("Medan", "Sumatera Utara", 3.59, 98.67, 27.0),
]


def _yogya_nasa_grid() -> list[tuple]:
    """Grid 3x3 NASA POWER Kota Yogyakarta (lon 110.35/38/41 x lat -7.76/79/82)."""
    names = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
    lats, lons = [-7.76, -7.79, -7.82], [110.35, 110.38, 110.41]
    out = []
    k = 0
    for la in lats:
        for lo in lons:
            out.append((f"Yogya_{names[k]}", "DI Yogyakarta", la, lo, 26.5))
            k += 1
    return out


NASA_LOC = NASA_LOC + _yogya_nasa_grid()

# ---------------------------------------------------------------------------
# RDTR: 14 zona Kota Yogyakarta. (kecamatan, kode_zona, kategori, luas_ha, lat, lon)
# Kategori dari prefix kode: R=Permukiman, K=Komersial, H=RTH.
# Centroid disebar dalam extent [110.345..110.405, -7.840..-7.767].
# ---------------------------------------------------------------------------
RDTR_ZONA = [
    ("Mantrijeron", "R.2", "Permukiman", 261.0, -7.8230, 110.3600),
    ("Kraton", "K.1", "Komersial", 140.0, -7.8050, 110.3620),
    ("Mergangsan", "R.2", "Permukiman", 231.0, -7.8150, 110.3720),
    ("Umbulharjo", "R.3", "Permukiman", 812.0, -7.8050, 110.3870),
    ("Kotagede", "R.2", "Permukiman", 307.0, -7.8230, 110.3990),
    ("Gondokusuman", "K.2", "Komersial", 399.0, -7.7820, 110.3830),
    ("Danurejan", "K.1", "Komersial", 110.0, -7.7900, 110.3700),
    ("Pakualaman", "R.2", "Permukiman", 63.0, -7.8000, 110.3760),
    ("Gondomanan", "K.1", "Komersial", 112.0, -7.8020, 110.3680),
    ("Ngampilan", "R.3", "Permukiman", 82.0, -7.8000, 110.3560),
    ("Wirobrajan", "R.2", "Permukiman", 176.0, -7.8030, 110.3480),
    ("Gedongtengen", "K.2", "Komersial", 96.0, -7.7900, 110.3600),
    ("Jetis", "R.2", "Permukiman", 170.0, -7.7780, 110.3650),
    ("Tegalrejo", "H.1", "RTH", 291.0, -7.7820, 110.3520),
]

# CHIRPS: grid 4 lat x 5 lon di bbox Yogyakarta (lon 110.275..110.475, lat -7.675..-7.825)
CHIRPS_LATS = [-7.675, -7.725, -7.775, -7.825]
CHIRPS_LONS = [110.275, 110.325, 110.375, 110.425, 110.475]


def _label_cuaca(kelembaban: float, tutupan: float) -> str:
    if kelembaban > 85 and tutupan > 70:
        return "Hujan Ringan"
    if tutupan > 50:
        return "Berawan"
    if tutupan > 20:
        return "Cerah Berawan"
    return "Cerah"


def gen_bmkg(rng: np.random.Generator, hari: int = 3) -> list[dict]:
    """Prakiraan 3-jam-an selama `hari` hari per lokasi (struktur bmkg_forecast)."""
    rows: list[dict] = []
    jam = [0, 3, 6, 9, 12, 15, 18, 21]
    rid = 0
    for kode, desa, kec, kota, prov, lat, lon, base_t in DESA:
        for d in range(hari):
            for h in jam:
                kelembaban = float(np.clip(rng.normal(82, 10), 50, 100))
                tutupan = float(np.clip(rng.normal(60, 30), 0, 100))
                diurnal = -2.0 if h in (0, 3, 21) else (2.0 if h in (9, 12, 15) else 0.0)
                suhu = round(float(base_t + diurnal + rng.normal(0, 1.0)), 1)
                cuaca = _label_cuaca(kelembaban, tutupan)
                rows.append({
                    "id": rid,
                    "provinsi": prov, "kotkab": kota, "kecamatan": kec, "desa": desa,
                    "kode_adm4": kode,
                    "datetime_utc": f"2026-06-{1 + d:02d}T{h:02d}:00:00Z",
                    "datetime_local": f"2026-06-{1 + d:02d} {(h + 8) % 24:02d}:00:00",
                    "suhu_c": suhu,
                    "kelembaban_pct": round(kelembaban, 1),
                    "curah_hujan_mm": round(rng.uniform(1, 12), 1) if cuaca == "Hujan Ringan" else 0.0,
                    "kecepatan_angin_kmh": round(float(np.clip(rng.normal(8, 4), 0, 30)), 1),
                    "arah_angin_deg": round(rng.uniform(0, 360), 0),
                    "tutupan_awan_pct": round(tutupan, 1),
                    "cuaca": cuaca,
                    "cuaca_en": {"Cerah": "Clear Skies", "Cerah Berawan": "Partly Cloudy",
                                 "Berawan": "Mostly Cloudy", "Hujan Ringan": "Light Rain"}[cuaca],
                    "lat": lat, "lon": lon,
                    "qc_flag": "OK",
                })
                rid += 1
    # Sisipkan beberapa baris bermasalah (untuk demo QC/UC-3): suhu/kelembaban di luar range
    for idx in rng.choice(len(rows), size=3, replace=False):
        rows[idx]["suhu_c"] = 99.0
        rows[idx]["qc_flag"] = "RANGE:suhu_c=99"
    return rows


def gen_nasa(rng: np.random.Generator) -> list[dict]:
    """7 parameter iklim bulanan 2023 per lokasi (struktur nasa_power_monthly)."""
    rows: list[dict] = []
    rid = 0
    for loc, prov, lat, lon, base in NASA_LOC:
        for month in range(1, 13):
            seasonal = 1.5 * np.sin(2 * np.pi * (month - 1) / 12)
            t2m = round(float(base + seasonal + rng.normal(0, 0.3)), 2)
            # curah hujan musiman: basah Nov-Apr tinggi, kemarau Jun-Sep rendah
            wet = 1 if month in (11, 12, 1, 2, 3, 4) else 0
            rows.append({
                "id": rid, "location_label": loc, "provinsi": prov,
                "year": 2023, "month": month,
                "t2m": t2m, "t2m_max": round(t2m + rng.uniform(3, 5), 2),
                "t2m_min": round(t2m - rng.uniform(4, 6), 2),
                "allsky_sfc_sw_dwn": round(rng.uniform(14, 26), 2),
                "rh2m": round(rng.uniform(70, 92) if wet else rng.uniform(60, 78), 2),
                "ws2m": round(rng.uniform(0.4, 5.7), 2),
                "prectotcorr": round(rng.uniform(6, 16) if wet else rng.uniform(0, 5), 2),
                "lat": lat, "lon": lon,
                "qc_flag": "OK",
            })
            rid += 1
    return rows


def gen_chirps(rng: np.random.Generator) -> list[dict]:
    """Curah hujan satelit bulanan grid Yogyakarta (struktur chirps_monthly)."""
    rows: list[dict] = []
    rid = 0
    for lat in CHIRPS_LATS:
        for lon in CHIRPS_LONS:
            for month in range(1, 13):
                # pola monsun: puncak Des-Feb (~400mm), kering Jul-Sep (~10mm)
                seasonal = np.cos(2 * np.pi * (month - 1) / 12)  # 1 di Jan, -1 di Jul
                precip = max(0.0, 205 + 195 * seasonal + rng.normal(0, 20))
                rows.append({
                    "id": rid, "lat": lat, "lon": lon, "year": 2023, "month": month,
                    "precipitation_mm": round(float(precip), 1),
                    "qc_flag": "OK",
                })
                rid += 1
    return rows


def gen_rdtr() -> list[dict]:
    """14 zona tata ruang Kota Yogyakarta (struktur rdtr_pola_ruang, centroid)."""
    rows = []
    for i, (kec, kode, kat, luas, lat, lon) in enumerate(RDTR_ZONA):
        # polygon kotak kecil ~0.01° di sekitar centroid (WKT) untuk representasi
        d = 0.005
        wkt = (f"POLYGON(({lon-d} {lat-d}, {lon+d} {lat-d}, "
               f"{lon+d} {lat+d}, {lon-d} {lat+d}, {lon-d} {lat-d}))")
        rows.append({
            "id": i, "kota": "Kota Yogyakarta", "kecamatan": kec,
            "kelurahan": None, "kode_zona": kode, "nama_zona": f"Kecamatan {kec}",
            "kategori_zona": kat, "luas_ha": luas,
            "centroid_lat": lat, "centroid_lon": lon, "geom_wkt": wkt,
            "sumber_data": "Lokal:dummy",
        })
    return rows


def gen_qc_log(n_bmkg: int, n_nasa: int, n_chirps: int) -> list[dict]:
    """Ringkasan QC per sumber (struktur qc_log)."""
    def entry(source, total, flagged):
        return {"source": source, "run_timestamp": "2026-06-05T00:00:00Z",
                "total_rows": total, "clean_rows": total - flagged,
                "flagged_rows": flagged,
                "pct_clean": round(100.0 * (total - flagged) / total, 1)}
    return [
        entry("bmkg", n_bmkg, 3),
        entry("nasa_power", n_nasa, 0),
        entry("chirps", n_chirps, 0),
    ]


def main() -> None:
    rng = np.random.default_rng(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bmkg = gen_bmkg(rng)
    nasa = gen_nasa(rng)
    chirps = gen_chirps(rng)
    rdtr = gen_rdtr()
    qc_log = gen_qc_log(len(bmkg), len(nasa), len(chirps))

    for name, data in [("bmkg_forecast", bmkg), ("nasa_power_monthly", nasa),
                       ("chirps_monthly", chirps), ("rdtr_pola_ruang", rdtr),
                       ("qc_log", qc_log)]:
        (OUT_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))

    print(f"OK → {OUT_DIR}")
    print(f"  bmkg_forecast      : {len(bmkg)} rows ({len(DESA)} lokasi, 3 flagged)")
    print(f"  nasa_power_monthly : {len(nasa)} rows ({len(NASA_LOC)} lokasi)")
    print(f"  chirps_monthly     : {len(chirps)} rows ({len(CHIRPS_LATS)*len(CHIRPS_LONS)} grid)")
    print(f"  rdtr_pola_ruang    : {len(rdtr)} zona")
    print(f"  qc_log             : {len(qc_log)} entri")


if __name__ == "__main__":
    main()
