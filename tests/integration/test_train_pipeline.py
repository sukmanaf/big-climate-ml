"""Training end-to-end pada data sintetis (tanpa DB)."""
from climate_ml.pipelines.evaluate import evaluate_uc1
from climate_ml.pipelines.train import train_uc1


def test_evaluate_uc1_mengalahkan_baseline(bmkg_df):
    """Karena cuaca berkorelasi dengan fitur, model harus lewat quality gate."""
    res = evaluate_uc1(bmkg_df, target="cuaca", margin=0.10)
    assert res["model"]["macro_f1"] > res["baseline"]["macro_f1"]
    assert res["quality_gate_passed"]


def test_train_uc1_menghasilkan_artefak(bmkg_df, tmp_path, monkeypatch):
    # Arahkan model_dir ke folder sementara
    from climate_ml import config as cfgmod

    monkeypatch.setattr(
        cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False
    )
    cfg = {
        "target": "cuaca", "model_name": "random_forest",
        "hyperparameters": {"n_estimators": 50},
        "quality_gate": {"margin_over_baseline": 0.10},
    }
    result = train_uc1(cfg, df=bmkg_df)
    assert result["artifact"].endswith(".joblib")
    assert (tmp_path / "UC1_weather_clf_latest.joblib").exists()
    assert (tmp_path / "UC1_weather_clf_latest.json").exists()


def test_determinisme(bmkg_df):
    """random_state tetap → metrik identik antar run (reproducibility)."""
    a = evaluate_uc1(bmkg_df, target="cuaca")
    b = evaluate_uc1(bmkg_df, target="cuaca")
    assert a["model"]["macro_f1"] == b["model"]["macro_f1"]
