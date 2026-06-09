# Progress: Migrasi Data Dummy → Data Real (PostGIS)

**Tanggal:** 8 Juni 2026  
**Status:** ✅ SELESAI — data real aktif di port 8000 (Docker) dan port 8001 (host)  
**Database real:** `100.96.101.0:5433 / climate_etl`

---

## 1. Hasil Analisis Database Real

### 1.1 Kondisi Tabel (per 8 Juni 2026)

| Tabel | Rows | Status | Keterangan |
|-------|------|--------|------------|
| `bmkg_forecast` | **418** | ✅ Berisi data | Makassar (5 desa) + Bandung (3 desa) + Yogyakarta (14 kecamatan), prakiraan per 3 jam, updated 5–6 Jun 2026 |
| `nasa_power_monthly` | **156** | ✅ Berisi data | 4 lokasi Sulsel (Makassar, Bone, Toraja, Bandung) + 9 grid Yogyakarta, tahun 2023 (12 bln × 13 lok) |
| `chirps_monthly` | **240** | ✅ Berisi data | 20 grid points area Yogyakarta (0.05° resolusi), Januari–Desember 2023 |
| `rdtr_pola_ruang` | **14** | ✅ Berisi data | 14 kecamatan Kota Yogyakarta (MULTIPOLYGON, kategori Administratif) |
| `qc_log` | **3** | ✅ Berisi data | 3 log QC: BMKG, NASA POWER, CHIRPS — semua 100% clean |
| `era5_monthly` | **2652** | ✅ Berisi data | 12 bln × 17 lat × 13 lon, Sulawesi Selatan 2023. Dimuat setelah analisis awal |

### 1.2 Sampel Data Real

**bmkg_forecast** (prakiraan aktual):
```
Bontorannu, Makassar | 2026-06-05 16:00 | suhu=30°C | kelembaban=65% | hujan=0mm | OK
Bontorannu, Makassar | 2026-06-05 19:00 | suhu=28°C | kelembaban=70% | hujan=0mm | OK
```

**nasa_power_monthly** (iklim bulanan 2023):
```
Makassar | Jan 2023 | t2m=26.61°C | t2m_max=28.69 | precip=10.85 mm/hari | rh2m=86.29% | OK
Makassar | Feb 2023 | t2m=26.37°C | t2m_max=28.63 | precip=16.04 mm/hari | rh2m=86.79% | OK
```

**chirps_monthly** (curah hujan grid Yogyakarta):
```
Jan 2023 | lat=-7.675, lon=110.275 | precipitation=391.62 mm | OK
Jan 2023 | lat=-7.675, lon=110.325 | precipitation=402.40 mm | OK
```

**rdtr_pola_ruang** (zona tata ruang Yogyakarta):
```
Kota Yogyakarta | KOTAGEDE  | ADM | Kecamatan KOTAGEDE  | Administratif | 298.60 ha
Kota Yogyakarta | UMBULHARJO| ADM | Kecamatan UMBULHARJO| Administratif | 824.42 ha
```

### 1.3 Kesesuaian Schema

Schema DB real vs query di `loaders.py`:

| Tabel | Kolom DB | Kolom Query Loader | Match? |
|-------|----------|--------------------|--------|
| `bmkg_forecast` | 20 kolom (id, provinsi, kotkab, kecamatan, desa, kode_adm4, datetime_utc, datetime_local, suhu_c, kelembaban_pct, curah_hujan_mm, kecepatan_angin_kmh, arah_angin, arah_angin_deg, tutupan_awan_pct, cuaca, cuaca_en, geom, fetched_at, qc_flag) | 19 kolom di SELECT (semua ada) | ✅ Full match |
| `nasa_power_monthly` | id, location_label, year, month, t2m, t2m_max, t2m_min, allsky_sfc_sw_dwn, rh2m, ws2m, prectotcorr, geom, qc_flag + provinsi | Semua ada di SELECT | ✅ Full match |
| `chirps_monthly` | id, year, month, precipitation_mm, geom, qc_flag | Semua ada di SELECT | ✅ Full match |
| `rdtr_pola_ruang` | id, kota, kecamatan, kelurahan, kode_zona, nama_zona, kategori_zona, luas_ha, geom, sumber_data, fetched_at | Semua ada di SELECT + ST_Centroid | ✅ Full match |
| `era5_monthly` | id, lat, lon, year, month, t2m_kelvin, t2m_celsius, geom, fetched_at | Semua ada di SELECT | ✅ Schema match (tapi kosong) |

> **Kesimpulan:** Tidak ada perubahan schema yang dibutuhkan. Semua kolom yang diquery loaders.py sudah ada di DB real.

---

## 2. Identifikasi Titik Dummy di Codebase

### 2.1 File yang Perlu Diubah

| File | Baris | Kode Dummy | Pengganti |
|------|-------|------------|-----------|
| `src/climate_ml/config.py` | 20 | `database_url = "...localhost:5432..."` | `.env` file dengan URL ke `100.96.101.0:5433` |
| `src/climate_ml/serving/api.py` | 130–132 | `load_bmkg_dummy()` di `_bmkg_sample_df()` | `load_bmkg()` dari loaders |
| `src/climate_ml/serving/api.py` | 159–162 | `load_nasa_power_dummy()` + `load_rdtr_dummy()` di `/v1/risk/zones` | `load_nasa_power()` + `load_rdtr()` |

### 2.2 File yang TIDAK Perlu Diubah

| File | Alasan |
|------|--------|
| `src/climate_ml/pipelines/train.py` | Default `source="db"` sudah benar; sudah pakai `load_bmkg()` / `load_nasa_power()` |
| `src/climate_ml/data/loaders.py` | Query SQL sudah benar dan schema-match dengan DB real |
| `src/climate_ml/data/dummy.py` | Tetap ada sebagai fallback/testing |
| `src/climate_ml/pipelines/flow.py` | Sudah pakai `load_bmkg()` langsung |

### 2.3 Caveats yang Perlu Diperhatikan

| Caveat | Detail |
|--------|--------|
| **`era5_monthly` kosong** | UC-4 (spatial interpolator ERA5) tidak akan punya data training. Perlu re-run ETL ERA5 atau mock skip di endpoint |
| **BMKG = prakiraan, bukan observasi** | Field `cuaca` dipakai sebagai label UC-1 (classifier). Data prakiraan valid untuk classification task |
| **`_SAMPLE_CACHE` di api.py** | Cache harus di-reset saat pertama start — sudah OK karena initialize `None` |
| **`provinsi` di `nasa_power_monthly`** | Loaders.py SELECT sudah include `provinsi`, DB real punya kolom ini |
| **`qc_flag` filter** | `train.py` sudah menjalankan `filter_qc_clean(df)` — data 100% OK di DB real |

---

## 3. Rencana Eksekusi

### Step 1 — Buat `.env` file (koneksi DB real)
**File:** `climate-ml/.env`  
**Aksi:** Buat file `.env` dengan `DATABASE_URL` ke server Tailscale

### Step 2 — Update `api.py`: ganti dummy → loaders real
**File:** `src/climate_ml/serving/api.py`  
**Aksi:**
- `_bmkg_sample_df()`: ganti `load_bmkg_dummy()` → `load_bmkg()`
- `/v1/risk/zones`: ganti `load_nasa_power_dummy()` + `load_rdtr_dummy()` → `load_nasa_power()` + `load_rdtr()`

### Step 3 — Verifikasi koneksi & data flow
**Aksi:** Test koneksi Python ke DB real, cek row count via SQLAlchemy

### Step 4 — Re-train model dengan data real (opsional tapi direkomendasikan)
**Aksi:** Jalankan `train.py --source db` untuk UC-1 dan UC-2 agar model dilatih pakai data real

### Step 5 — Jalankan API + smoke test endpoint
**Aksi:** Start uvicorn, test semua endpoint: `/v1/sample/weather`, `/v1/risk/zones`, `/v1/predict/weather`

---

## 4. Checklist Progress

- [x] **Step 1** — Buat `.env` dengan `DATABASE_URL=postgresql+psycopg2://climate_user:climate_poc_2026@100.96.101.0:5433/climate_etl`
- [x] **Step 2a** — Update `api.py`: `_bmkg_sample_df()` pakai `load_bmkg()`
- [x] **Step 2b** — Update `api.py`: `/v1/risk/zones` pakai `load_nasa_power()` + `load_rdtr()`
- [x] **Step 2c** — Tambah endpoint `/v1/sample/climate` untuk auto-fill UC-2 dari `nasa_power_monthly`
- [x] **Step 2d** — Update `index.html`: header "data real", CITIES 4 kota (sesuai DB), tombol "📥 Isi dari DB" UC-2
- [x] **Step 3** — Verifikasi koneksi Python → DB real: ✅ BMKG 418, NASA POWER 156, RDTR 14, CHIRPS 240
- [x] **Step 4a** — Re-train UC-1 dengan data BMKG real: macro-F1 CV=0.874, final=0.949 ✅
- [x] **Step 4b** — Re-train UC-2 dengan data NASA POWER real: MAE=1.28, skill_score=0.179 ✅
- [x] **Step 5** — Smoke test semua endpoint API ✅ (port 8001, semua OK)
- [x] **Step 6** — Server port 8001 dijalankan persistent dengan `--reload` (nohup). Log: `/tmp/climate-ml-8001.log`
- [x] **Step 7** — ERA5 ternyata ada 2652 rows (dimuat di server). Endpoint `/v1/era5/status` dikonfirmasi ✅
- [x] **Step 8** — UC-4 card ditambahkan ke frontend dengan live status ERA5 availability
- [x] **Step 9** — Docker container port 8000 di-rebuild dengan data real. `docker compose up --build -d` ✅
  - Tambah `DATABASE_URL` ke `docker-compose.yml`
  - Bind mount `src/`, `web/`, `config/`, `models/` → perubahan kode langsung aktif tanpa rebuild
  - Update `requirements-serve.txt` tambah `sqlalchemy`, `psycopg2-binary`, `geoalchemy2`, `shapely`
  - Update `docker-entrypoint.sh` → auto-detect DB real vs dummy

### Hasil Smoke Test (8 Juni 2026)

| Endpoint | Method | Status | Catatan |
|----------|--------|--------|---------|
| `/healthz` | GET | ✅ 200 | `model_loaded: true` |
| `/v1/model/info` | GET | ✅ 200 | RF, 4 kelas cuaca (Berawan/Cerah/Cerah Berawan/Udara Kabur) |
| `/v1/sample/weather` | GET | ✅ 200 | Data real BMKG — Sukarasa (Bandung), suhu=26°C |
| `/v1/predict/weather` | POST | ✅ 200 | Prediksi "Cerah" proba=87.1% |
| `/v1/sample/climate` | GET | ✅ 200 | Bone bln 3: rh2m=76.95%, rad=20.25 MJ |
| `/v1/predict/climate` | POST | ✅ 200 | Prediksi t2m=27.23°C |
| `/v1/risk/zones` | GET | ✅ 200 | 14 zona RDTR real Yogyakarta |
| `/v1/anomaly/check` | POST | ✅ 200 | suhu=999 → `is_anomaly: true`, reason=RANGE:suhu_c=999 |
| `/v1/era5/status` | GET | ✅ 200 | `available: true`, 2652 rows, 2023, Sulawesi Selatan |

### Temuan Tambahan dari Verifikasi
- `nasa_power_monthly` punya 13 `location_label`: Makassar, Bone, Toraja, Bandung + 9 grid Yogyakarta (Yogya_NW, Yogya_N, dst)
- `rdtr_pola_ruang` berisi 14 kecamatan Kota Yogyakarta (KOTAGEDE, MERGANGSAN, UMBULHARJO, dll)
- Semua data `qc_flag = OK` (418/418 BMKG)

---

## 5. Dampak per Use Case

| Use Case | Tabel Utama | Data Tersedia | Dampak Migrasi |
|----------|-------------|--------------|----------------|
| **UC-1** Weather Classifier | `bmkg_forecast` | ✅ 418 rows | Langsung bisa train & predict |
| **UC-2** Climate Regressor | `nasa_power_monthly` | ✅ 156 rows | Langsung bisa train & predict |
| **UC-3** Anomaly Detector | `bmkg_forecast` | ✅ 418 rows | Rule-based, tidak perlu train |
| **UC-4** Spatial Interpolator | `era5_monthly` | ✅ 2652 rows | Data tersedia. Training pipeline UC-4 belum ada di CLI — roadmap |
| **UC-5** Climate Risk Score | `rdtr_pola_ruang` + `nasa_power_monthly` | ✅ 14 + 156 rows | Langsung bisa kalkulasi |

---

*Dibuat: 8 Juni 2026 — berdasarkan analisis dokumen-etl-climate-data.md v3.1 + laporan-poc-etl-climate-data.md v4.1 + koneksi langsung ke DB real*
