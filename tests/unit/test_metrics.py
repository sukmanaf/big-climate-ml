import numpy as np

from climate_ml.utils.metrics import (
    classification_metrics,
    regression_metrics,
    skill_score,
)


def test_classification_sempurna():
    y = ["a", "b", "a", "c"]
    assert classification_metrics(y, y)["macro_f1"] == 1.0


def test_regression_sempurna():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["r2"] == 1.0


def test_skill_score_positif_bila_lebih_baik():
    assert skill_score(mae_model=1.0, mae_baseline=2.0) == 0.5


def test_skill_score_negatif_bila_lebih_buruk():
    assert skill_score(mae_model=3.0, mae_baseline=2.0) < 0


def test_skill_score_baseline_nol_aman():
    assert skill_score(1.0, 0.0) == 0.0
