-- DDL tabel hasil ML — ditulis kembali ke database PostGIS hasil PoC ETL.
-- Jalankan: psql "$DATABASE_URL" -f sql/01_ml_tables.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- Hasil prediksi (UC-1 klasifikasi cuaca, UC-2 regresi iklim, UC-4 interpolasi)
CREATE TABLE IF NOT EXISTS ml_predictions (
    id            SERIAL PRIMARY KEY,
    use_case      VARCHAR(20)  NOT NULL,
    model_name    VARCHAR(100) NOT NULL,
    model_version VARCHAR(50)  NOT NULL,
    target        VARCHAR(50)  NOT NULL,
    predicted     VARCHAR(100) NOT NULL,
    proba         FLOAT,
    ref_year      INTEGER,
    ref_month     INTEGER,
    geom          geometry(Point, 4326),
    predicted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (use_case, model_version, target, ref_year, ref_month, geom)
);
CREATE INDEX IF NOT EXISTS idx_ml_pred_geom ON ml_predictions USING GIST (geom);

-- Hasil deteksi anomali (UC-3 quality control)
CREATE TABLE IF NOT EXISTS ml_anomalies (
    id            SERIAL PRIMARY KEY,
    source_table  VARCHAR(50)  NOT NULL,
    source_id     INTEGER      NOT NULL,
    is_anomaly    BOOLEAN      NOT NULL,
    anomaly_score FLOAT,
    reason        TEXT,
    geom          geometry(Point, 4326),
    detected_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ml_anom_geom ON ml_anomalies USING GIST (geom);
