import pytest
from sklearn.ensemble import RandomForestClassifier

from climate_ml.models.registry import make_estimator


def test_membuat_estimator_dikenal():
    est = make_estimator("random_forest", n_estimators=10)
    assert isinstance(est, RandomForestClassifier)
    assert est.n_estimators == 10


def test_nama_tak_dikenal_error_jelas():
    with pytest.raises(KeyError, match="tidak dikenal"):
        make_estimator("model_ajaib")
