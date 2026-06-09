"""UC-2 regresi iklim bulanan — train + evaluate pada data sintetis."""
from climate_ml.pipelines.evaluate import evaluate_uc2
from climate_ml.pipelines.train import train_uc2


def test_uc2_mengalahkan_baseline_klimatologi(nasa_df):
    """Suhu sintetis berpola musim+lokasi → model harus skill_score > 0."""
    res = evaluate_uc2(nasa_df, target="t2m")
    assert res["model"]["mae"] < res["baseline"]["mae"]
    assert res["skill_score"] > 0
    assert res["quality_gate_passed"]


def test_train_uc2_menghasilkan_artefak(nasa_df, tmp_path, monkeypatch):
    from climate_ml import config as cfgmod

    monkeypatch.setattr(cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False)
    cfg = {"target": "t2m", "model_name": "gradient_boosting",
           "hyperparameters": {"n_estimators": 100}}
    result = train_uc2(cfg, df=nasa_df)
    assert result["artifact"].endswith(".joblib")
    assert (tmp_path / "UC2_climate_reg_latest.joblib").exists()


def test_uc2_prediksi_dataran_tinggi_lebih_dingin(nasa_df, tmp_path, monkeypatch):
    """Sanity: Toraja (dataran tinggi) diprediksi lebih dingin dari Makassar (pesisir)."""
    import pandas as pd

    from climate_ml import config as cfgmod
    from climate_ml.utils.io import load_artifact

    monkeypatch.setattr(cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False)
    train_uc2({"target": "t2m", "model_name": "gradient_boosting"}, df=nasa_df)
    pipeline, _ = load_artifact(tmp_path / "UC2_climate_reg_latest.joblib")

    makassar = pd.DataFrame([{"lat": -5.14, "lon": 119.42, "month": 7,
                              "rh2m": 80, "ws2m": 2, "allsky_sfc_sw_dwn": 20}])
    toraja = pd.DataFrame([{"lat": -3.05, "lon": 119.85, "month": 7,
                            "rh2m": 80, "ws2m": 2, "allsky_sfc_sw_dwn": 20}])
    assert pipeline.predict(toraja)[0] < pipeline.predict(makassar)[0]
