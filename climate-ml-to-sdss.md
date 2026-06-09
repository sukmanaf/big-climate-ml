# Pemetaan Climate-ML → SDSS
## Katalog Fitur SDSS Detail v2.2

**Proyek:** Geospatial Data Utilization Platform for Climate Action (BIG)  
**Tanggal:** 8 Juni 2026

---

## Ringkasan Pemetaan

| Use Case | Fitur SDSS | Bagian |
|----------|-----------|--------|
| UC-1 · Klasifikasi Kondisi Cuaca | **6.3** Predictive Modeling Framework | Bagian 6 — AI/ML Layer |
| UC-2 · Prediksi Suhu Bulanan | **6.3** Predictive Modeling Framework | Bagian 6 — AI/ML Layer |
| UC-3 · Deteksi Anomali / QC | **6.2** Anomaly Detection & Early Warning | Bagian 6 — AI/ML Layer |
| UC-4 · Interpolasi Spasial ERA5 | **2.1** Advanced Climate Modeling | Bagian 2 — Advanced Modeling |
| UC-5 · Climate Risk Score RDTR | **4.1** Spatial Planning Support Toolbox + **3.1** Multi-Criteria Vulnerability | Bagian 4 + 3 |

---

## UC-1 · Klasifikasi Kondisi Cuaca → Fitur 6.3

**Fitur SDSS:** `6.3 Predictive Modeling Framework`

Fitur 6.3 adalah framework yang menampung semua model prediktif berbasis observasi — dari ARIMA/Prophet hingga LSTM dan Transformer. UC-1 adalah **implementasi pertama yang siap** di framework ini: Random Forest Classifier yang memprediksi kategori cuaca (Cerah / Berawan / Cerah Berawan / Udara Kabur) dari data BMKG real.

**Kontribusi UC-1 ke Fitur 6.3:**
- Model terlatih tersedia via REST endpoint `POST /v1/predict/weather`
- Model registry via joblib + metadata JSON (cikal bakal MLflow registry SDSS)
- Quality gate otomatis (macro-F1 vs baseline) — pola yang bisa distandarisasi untuk semua model di framework ini
- Pipeline sklearn (preprocessing + model) yang anti-leakage, bisa dijadikan template

**Gap menuju Fitur 6.3 penuh:**
- Belum ada LSTM/GRU untuk time-series cuaca (saat ini hanya RF)
- Belum integrasi MLflow tracking
- Belum ada AutoML (PyCaret/Optuna)

---

## UC-2 · Prediksi Suhu Bulanan → Fitur 6.3

**Fitur SDSS:** `6.3 Predictive Modeling Framework`

UC-2 (Gradient Boosting Regressor dari NASA POWER) adalah contoh **regression model spasial-temporal** yang juga dinaungi fitur 6.3. Bersama UC-1, keduanya membentuk pasangan klasifikasi + regresi sebagai proof-of-concept framework prediktif SDSS.

**Kontribusi UC-2 ke Fitur 6.3:**
- Endpoint `POST /v1/predict/climate` + `GET /v1/sample/climate`
- Baseline klimatologi otomatis (DummyRegressor mean) sebagai pembanding
- Skill score sebagai metrik yang relevan untuk model iklim (lebih bermakna dari MAE absolut)

**Gap menuju Fitur 6.3 penuh:**
- Belum ada model time-series eksplisit (SARIMA, Prophet) untuk tren bulanan
- Dataset hanya 1 tahun (2023) — perlu backfill historis ERA5/NASA POWER untuk model yang lebih kuat

---

## UC-3 · Deteksi Anomali / QC → Fitur 6.2

**Fitur SDSS:** `6.2 Anomaly Detection & Early Warning System`

Ini pemetaan paling langsung. Fitur 6.2 di katalog SDSS menyebut secara eksplisit: rule-based (Z-score, IQR) + ML-based (Isolation Forest, One-Class SVM) + severity scoring + alert routing. UC-3 mengimplementasikan **dua dari tiga lapis tersebut** dan bisa langsung diklaim sebagai **versi v1** fitur 6.2.

**Kontribusi UC-3 ke Fitur 6.2:**
- Lapis 1 rule-based: range check fisik per parameter (`config/config.yaml`)
- Lapis 2 ML: Isolation Forest (contamination=5%)
- Format `reason` selaras dengan `qc_flag` di database ETL (RANGE:col=val)
- Endpoint `POST /v1/anomaly/check` siap dikonsumsi SDSS

**Gap menuju Fitur 6.2 penuh:**
- Belum ada lapis 3: Autoencoder / LSTM-based untuk anomali temporal sekuensial
- Belum ada alert routing (Kafka, SMS, email)
- Belum ada One-Class SVM sebagai alternatif Isolation Forest
- Belum ada real-time stream consumer

---

## UC-4 · Interpolasi Spasial ERA5 → Fitur 2.1

**Fitur SDSS:** `2.1 Advanced Climate Modeling`

Fitur 2.1 mencakup "statistical & dynamical downscaling dari GCM/RCM, ERA5 sebagai input observasi". UC-4 adalah **implementasi awal downscaling ERA5** — Random Forest yang menginterpolasi suhu dari 221 titik grid 0.25° (ERA5 Sulawesi Selatan) ke titik arbitrari. Ini fondasi yang nanti dikembangkan menjadi downscaling penuh di SDSS.

**Kontribusi UC-4 ke Fitur 2.1:**
- Pipeline interpolasi spasial ERA5 (lat, lon, month → t2m_celsius)
- Endpoint `POST /v1/predict/era5` + `GET /v1/era5/status`
- Metrik kuat: R²=0.905, MAE=0.54°C, skill_score=0.703
- Validasi fisis konsisten: Toraja (dataran tinggi) ~6.6°C lebih dingin dari Makassar (pesisir)

**Gap menuju Fitur 2.1 penuh:**
- Saat ini hanya 1 variabel (suhu) — perlu tambah curah hujan, kelembaban, angin
- Model RF akan diganti Kriging + kovariat DEM (DEMNAS) untuk hasil geostatistik yang lebih akurat
- Cakupan saat ini hanya Sulawesi Selatan — perlu diperluas ke 18 provinsi target
- Belum ada ensemble + uncertainty quantification (BMA/bayesian)

---

## UC-5 · Climate Risk Score RDTR → Fitur 4.1 + 3.1

**Fitur SDSS:** `4.1 Spatial Planning Support Toolbox (RDTR)` + `3.1 Multi-Criteria Vulnerability Assessment`

UC-5 adalah yang paling langsung relevan dengan output SDSS untuk penyusunan kebijakan. Fitur 4.1 adalah "climate-informed RDTR toolbox: hazard overlay + risk scoring per zona". UC-5 mengimplementasikan overlay RDTR × iklim (NASA POWER + rdtr_pola_ruang) dan menghasilkan composite risk score per kecamatan — ini adalah **inti dari fitur 4.1**.

Selain itu, logika pembobotan UC-5 (0.5 × norm(tmax) + 0.3 × norm(precip) + 0.2 × norm(humidity)) identik dengan metodologi **fitur 3.1** (WSM/normalisasi min-max lintas zona).

**Kontribusi UC-5 ke Fitur 4.1 + 3.1:**
- Overlay spasial RDTR × iklim via nearest-point (centroid zona → titik NASA POWER terdekat)
- Composite risk score ternormalisasi + klasifikasi Rendah/Sedang/Tinggi
- Endpoint `GET /v1/risk/zones` — 14 zona Yogyakarta siap dikonsumsi peta SDSS
- Pattern dapat diperluas ke 18 provinsi target dengan ganti bbox + sumber RDTR

**Gap menuju Fitur 4.1 penuh:**
- Saat ini hanya 1 pilot area (Kota Yogyakarta) — perlu replikasi ke kota/kabupaten lain
- Bobot 0.5/0.3/0.2 masih hardcoded — fitur 4.1 butuh bobot yang bisa dikonfigurasi user (AHP, entropy weighting)
- Belum ada analisis konflik spasial atau output paket NSPK-compliant
- Nearest-point sederhana — bisa ditingkatkan dengan spatial join + weighted average

---

## Gap yang Perlu Diisi

Satu fitur SDSS yang belum disentuh sama sekali oleh climate-ml saat ini:

**Fitur 6.9 — Explainable AI (XAI) Service**

SDSS membutuhkan SHAP/LIME agar model bisa menjelaskan *kenapa* sebuah prediksi atau skor dihasilkan. Ini penting untuk akuntabilitas keputusan kebijakan:
- UC-1: kenapa diklasifikasikan "Berawan" bukan "Hujan"?
- UC-5: parameter mana (panas/hujan/lembab) yang paling drive risk score zona ini?

Implementasi: tambah `shap` ke `requirements.txt`, buat endpoint `POST /v1/explain/{uc}` yang mengembalikan feature importance per prediksi.

---

## Diagram Alur Integrasi

```
climate-ml (PoC)                    SDSS (Platform)
─────────────────                   ─────────────────────────────

bmkg_forecast ──→ UC-1 (RF)    ──→  6.3 Predictive Modeling
                  POST /v1/predict/weather

nasa_power    ──→ UC-2 (GB)    ──→  6.3 Predictive Modeling
                  POST /v1/predict/climate

bmkg_forecast ──→ UC-3 (Rule   ──→  6.2 Anomaly Detection
                       +IsoFr)      POST /v1/anomaly/check

era5_monthly  ──→ UC-4 (RF)    ──→  2.1 Advanced Climate Modeling
                  POST /v1/predict/era5

rdtr +        ──→ UC-5 (Overlay──→  4.1 RDTR Toolbox
nasa_power         +Scoring)        3.1 Vulnerability Assessment
                  GET /v1/risk/zones

                  [roadmap]    ──→  6.9 XAI Service
                  POST /v1/explain/{uc}
```

---

## Rekomendasi Urutan Integrasi ke SDSS

| Prioritas | Use Case | Fitur SDSS | Alasan |
|-----------|----------|-----------|--------|
| **1** | UC-3 (Anomali) | 6.2 | Paling siap, langsung plug-in, tidak butuh perubahan |
| **2** | UC-5 (RDTR Risk) | 4.1 + 3.1 | Output langsung relevan untuk pembuat kebijakan |
| **3** | UC-1 (Klasifikasi) | 6.3 | Model kuat (F1=0.874), endpoint sudah production-ready |
| **4** | UC-2 (Suhu Bulanan) | 6.3 | Perlu diperkuat dengan data historis lebih panjang dulu |
| **5** | UC-4 (ERA5 Downscaling) | 2.1 | Perlu upgrade ke Kriging + perluasan area sebelum produksi |
| **6** | XAI (belum ada) | 6.9 | Tambahkan SHAP setelah model stabil |

---

*Referensi: Katalog_Fitur_SDSS_Detail_v2.2.md · use-cases.md · laporan-poc-etl-climate-data.md v4.1*
