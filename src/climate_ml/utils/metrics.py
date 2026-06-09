"""Metrik evaluasi + skill score terhadap baseline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """1 - mae_model/mae_baseline. >0 berarti model lebih baik dari baseline."""
    if mae_baseline == 0:
        return 0.0
    return 1.0 - (mae_model / mae_baseline)
