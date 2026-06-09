# Climate ML

Lapisan **Machine Learning** untuk data iklim BIG, dibangun di atas database PostGIS hasil
PoC ETL (`bmkg_forecast`, `era5_monthly`, `nasa_power_monthly`).

📄 Desain lengkap: [`docs/dokumen-teknis-ml-climate.md`](docs/dokumen-teknis-ml-climate.md) · PDF via `make pdf`

## Use Case

Selaras laporan PoC ETL **v4.1** (Phase 1-4: ETL core, RDTR×Climate overlay, QC, CHIRPS, Docker).
ML bersifat **QC-aware** (hanya melatih baris `qc_flag='OK'`).

| Kode | Use Case | Tabel | Tipe |
|------|----------|-------|------|
| UC-1 | Klasifikasi kondisi cuaca | `bmkg_forecast` | Klasifikasi |
| UC-2 | Prediksi parameter iklim bulanan | `nasa_power_monthly` | Regresi |
| UC-3 | Deteksi anomali / Quality Control | semua (+`qc_flag`) | Unsupervised + rule |
| UC-4 | Interpolasi spasial suhu | `era5_monthly` | Regresi spasial |
| UC-5 | Climate risk score per zona RDTR | `rdtr_pola_ruang` × `nasa_power_monthly` | Overlay komposit |

## 🚀 Demo cepat (TANPA database — pakai data dummy JSON)

Untuk membuktikan ML jalan + mencoba frontend, **tidak perlu PostgreSQL**. Data dummy
JSON sudah cukup (data real masih diproses):

```bash
make install                 # buat venv + install deps
make demo                    # generate dummy JSON + latih model UC-1 (tanpa DB)
make serve                   # jalankan API + frontend
```

Lalu buka **http://localhost:8000/ui/** — halaman tester untuk:
- **UC-1** klasifikasi cuaca (input parameter → prediksi `cuaca` + bar probabilitas)
- **UC-2** prediksi suhu bulanan (lokasi + bulan → `t2m` °C)
- **UC-3** deteksi anomali (coba suhu `999` → tertangkap)
- Status model real-time

API docs interaktif: http://localhost:8000/docs

### Atau pakai Docker (paling ringkas, tanpa setup Python sama sekali)

```bash
docker compose up --build      # lalu buka http://localhost:8000/ui/
```

Container otomatis generate dummy + latih UC-1 & UC-2 saat pertama start.

> **JSON vs PostgreSQL?** Untuk demo & test ML/FE → **JSON cukup**. PostgreSQL hanya
> diperlukan bila ingin *batch predict menulis balik ke DB* + query spasial (UC-3 skala penuh).

## Setup penuh (dengan PostGIS, opsional)

```bash
cp .env.example .env         # isi DATABASE_URL (re-use kredensial PoC ETL)
make sql                     # buat tabel ml_predictions & ml_anomalies
make train UC=UC1            # latih dari PostGIS → models/*.joblib + log MLflow
make predict UC=UC1          # batch predict → tulis ke PostGIS
make test                    # verifikasi scaffold (tanpa perlu DB)
```

## Testing

```bash
make test                    # semua test, skip yang butuh DB
make test-unit               # unit test saja (cepat)
make test-db                 # test yang butuh PostGIS
pytest --cov=climate_ml      # dengan coverage
```

Test default **tidak butuh database** (pakai data sintetis/fixture). Test ber-marker `db`
otomatis di-skip bila `DATABASE_URL` tidak diset.

## Struktur

```
src/climate_ml/   data · features · models · pipelines · serving · utils
web/              frontend tester (index.html, dilayani di /ui)
scripts/          generate_dummy_data.py
data/dummy/       dummy JSON (bmkg_forecast.json, nasa_power_monthly.json)
config/           config.yaml + hyperparameter per use case
sql/              DDL tabel hasil ML
tests/            unit · integration · api · data
```
