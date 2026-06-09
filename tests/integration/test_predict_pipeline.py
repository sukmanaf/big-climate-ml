"""Predict end-to-end: latih → muat artefak → inference → cek skema output."""
from climate_ml.pipelines.predict import predict_uc1
from climate_ml.pipelines.train import train_uc1


def test_predict_menghasilkan_skema_ml_predictions(bmkg_df, tmp_path, monkeypatch):
    from climate_ml import config as cfgmod

    monkeypatch.setattr(cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False)
    cfg = {"target": "cuaca", "model_name": "random_forest",
           "hyperparameters": {"n_estimators": 50}}
    train_uc1(cfg, df=bmkg_df)

    artifact = tmp_path / "UC1_weather_clf_latest.joblib"
    result = predict_uc1(bmkg_df, str(artifact))

    expected = {"use_case", "model_name", "model_version", "target",
                "predicted", "proba", "lat", "lon"}
    assert expected <= set(result.columns)
    assert len(result) == len(bmkg_df)
    assert result["proba"].between(0, 1).all()
    assert (result["use_case"] == "UC1").all()


def test_no_training_serving_skew(bmkg_df, tmp_path, monkeypatch):
    """Prediksi dari artefak == prediksi dari pipeline in-memory (Pipeline sama)."""
    from climate_ml import config as cfgmod
    from climate_ml.features.build import prepare_uc1_frame
    from climate_ml.utils.io import load_artifact

    monkeypatch.setattr(cfgmod.get_settings(), "model_dir", str(tmp_path), raising=False)
    train_uc1({"target": "cuaca", "model_name": "random_forest",
               "hyperparameters": {"n_estimators": 50}}, df=bmkg_df)

    pipeline, _ = load_artifact(tmp_path / "UC1_weather_clf_latest.joblib")
    direct = pipeline.predict(prepare_uc1_frame(bmkg_df))
    served = predict_uc1(bmkg_df, str(tmp_path / "UC1_weather_clf_latest.joblib"))["predicted"]
    assert list(direct) == list(served)
