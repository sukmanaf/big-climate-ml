"""Factory estimator: nama (string dari config) → instance sklearn."""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "random_forest_reg": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "dummy_clf": DummyClassifier,
    "dummy_reg": DummyRegressor,
}


def make_estimator(name: str, **params) -> BaseEstimator:
    """Buat estimator dari nama. Error jelas bila nama tak dikenal."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' tidak dikenal. Pilihan: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**params)
