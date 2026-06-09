"""Kontrak endpoint FastAPI (tanpa DB). Model dilatih ke tmp dir lalu di-load."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(bmkg_df, nasa_df, tmp_path, monkeypatch):
    from climate_ml import config as cfgmod
    from climate_ml.pipelines.train import train_uc1, train_uc2

    monkeypatch.setattr(cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False)
    train_uc1({"target": "cuaca", "model_name": "random_forest",
               "hyperparameters": {"n_estimators": 50}}, df=bmkg_df)
    train_uc2({"target": "t2m", "model_name": "gradient_boosting"}, df=nasa_df)

    # Import app setelah model_dir di-patch, lalu paksa muat model dari tmp
    from climate_ml.serving import api
    monkeypatch.setattr(api, "_model_path", lambda: tmp_path / "UC1_weather_clf_latest.joblib")
    monkeypatch.setattr(api, "_model_path_uc2", lambda: tmp_path / "UC2_climate_reg_latest.joblib")
    with TestClient(api.app) as c:  # memicu lifespan → load model
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["target"] == "cuaca"
    assert len(body["classes"]) >= 2


def test_predict_weather_label_valid(client):
    payload = {
        "suhu_c": 28.0, "kelembaban_pct": 95, "kecepatan_angin_kmh": 5.1,
        "arah_angin_deg": 146, "tutupan_awan_pct": 90,
        "lat": -5.16, "lon": 119.40, "datetime_local": "2023-01-20T20:00:00",
    }
    r = client.post("/v1/predict/weather", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted"], str)
    assert 0.0 <= body["proba"] <= 1.0
    # probabilitas semua kelas berjumlah ~1
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_climate(client):
    payload = {"lat": -5.14, "lon": 119.42, "month": 7,
               "rh2m": 80, "ws2m": 2, "allsky_sfc_sw_dwn": 20}
    r = client.post("/v1/predict/climate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "t2m"
    assert 15 < body["predicted"] < 35  # suhu wajar Indonesia
    assert body["unit"] == "°C"


def test_predict_climate_bulan_invalid_422(client):
    r = client.post("/v1/predict/climate", json={"lat": -5.14, "lon": 119.42, "month": 99})
    assert r.status_code == 422


def test_sample_weather(client):
    """Ambil sampel data → field meteorologi + label cuaca tercatat."""
    r = client.get("/v1/sample/weather")
    assert r.status_code == 200
    body = r.json()
    assert -50 <= body["suhu_c"] <= 60
    assert 0 <= body["kelembaban_pct"] <= 100
    assert body["actual_cuaca"]  # ada label tercatat


def test_sample_weather_terdekat(client):
    """Dengan lat/lon → mengembalikan record (lokasi terdekat tersedia)."""
    r = client.get("/v1/sample/weather", params={"lat": -5.16, "lon": 119.40})
    assert r.status_code == 200
    assert "desa" in r.json()


def test_risk_zones(client):
    """UC-5: endpoint risiko iklim per zona RDTR (pakai dummy)."""
    r = client.get("/v1/risk/zones")
    assert r.status_code == 200
    zones = r.json()
    assert len(zones) >= 1
    z = zones[0]
    assert {"nama_zona", "kategori_zona", "risk_score", "risk_level"} <= set(z)
    assert 0 <= z["risk_score"] <= 1
    # terurut risiko tertinggi dulu
    scores = [x["risk_score"] for x in zones]
    assert scores == sorted(scores, reverse=True)


def test_frontend_disajikan(client):
    """/ redirect ke /ui/, dan /ui/ menyajikan index.html."""
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "Climate ML" in r.text


def test_predict_weather_input_invalid_422(client):
    bad = {"suhu_c": 999, "kelembaban_pct": 95}  # suhu out of range + field kurang
    r = client.post("/v1/predict/weather", json=bad)
    assert r.status_code == 422


def test_anomaly_check(client):
    r = client.post("/v1/anomaly/check", json={"suhu_c": 999, "kelembaban_pct": 80})
    assert r.status_code == 200
    assert r.json()["is_anomaly"] is True
