"""Pipeline training (CLI).

Contoh:
    python -m climate_ml.pipelines.train --use-case UC1 \
        --config config/models/uc1_weather_clf.yaml
"""
from __future__ import annotations

import argparse

from climate_ml.config import get_settings, load_model_config
from climate_ml.data.validation import check_min_rows, filter_qc_clean, validate_ranges
from climate_ml.features.build import prepare_uc1_frame, prepare_uc4_frame
from climate_ml.models.anomaly_detector import UC3_FEATURE_COLS, build_iso_model
from climate_ml.models.climate_regressor import build_uc2_pipeline
from climate_ml.models.spatial_interpolator import build_uc4_pipeline
from climate_ml.models.weather_classifier import build_uc1_pipeline
from climate_ml.pipelines.evaluate import evaluate_uc1, evaluate_uc2, evaluate_uc4
from climate_ml.utils.io import save_artifact
from climate_ml.utils.logging import get_logger

log = get_logger()


def train_uc1(cfg: dict, df=None, source: str = "db") -> dict:
    """Latih UC-1 end-to-end.

    df bisa di-inject untuk test (tanpa DB). source: "db" (PostGIS) | "dummy" (JSON).
    """
    if df is None:
        if source == "dummy":
            from climate_ml.data.dummy import load_bmkg_dummy

            df = load_bmkg_dummy()
        else:
            from climate_ml.data.loaders import load_bmkg

            df = load_bmkg()

    df = filter_qc_clean(df)            # buang baris ber-flag QC (flag-and-load)
    check_min_rows(df).raise_if_invalid()
    validate_ranges(df).raise_if_invalid()

    target = cfg["target"]
    hp = cfg.get("hyperparameters", {})
    margin = cfg.get("quality_gate", {}).get("margin_over_baseline", 0.10)

    # Evaluasi + quality gate sebelum commit artefak
    eval_result = evaluate_uc1(df, target=target, margin=margin)
    log.info("Eval UC1: %s", eval_result)
    if not eval_result["quality_gate_passed"]:
        log.warning("Quality gate TIDAK lolos — artefak tetap disimpan dengan flag.")

    # Fit final pada seluruh data
    prepared = prepare_uc1_frame(df)
    pipeline = build_uc1_pipeline(cfg.get("model_name", "random_forest"), **hp)
    pipeline.fit(prepared, prepared[target])

    metadata = {
        "use_case": "UC1",
        "model_name": cfg.get("model_name", "random_forest"),
        "model_version": "dev",
        "target": target,
        "data_rows": len(df),
        "metrics": eval_result["model"],
        "baseline_metrics": eval_result["baseline"],
        "quality_gate_passed": eval_result["quality_gate_passed"],
    }
    artifact = save_artifact(
        pipeline, metadata, get_settings().model_dir, "UC1_weather_clf_latest"
    )
    log.info("Artefak disimpan: %s", artifact)
    return {"artifact": str(artifact), **eval_result}


def train_uc2(cfg: dict, df=None, source: str = "db") -> dict:
    """Latih UC-2 (regresi iklim bulanan). df bisa di-inject untuk test."""
    if df is None:
        if source == "dummy":
            from climate_ml.data.dummy import load_nasa_power_dummy

            df = load_nasa_power_dummy()
        else:
            from climate_ml.data.loaders import load_nasa_power

            df = load_nasa_power()

    df = filter_qc_clean(df)
    target = cfg.get("target", "t2m")
    hp = cfg.get("hyperparameters", {})

    eval_result = evaluate_uc2(df, target=target)
    log.info("Eval UC2: %s", eval_result)
    if not eval_result["quality_gate_passed"]:
        log.warning("Quality gate UC2 TIDAK lolos (skill_score <= 0).")

    pipeline = build_uc2_pipeline(cfg.get("model_name", "gradient_boosting"), **hp)
    pipeline.fit(df, df[target])

    metadata = {
        "use_case": "UC2",
        "model_name": cfg.get("model_name", "gradient_boosting"),
        "model_version": "dev",
        "target": target,
        "data_rows": len(df),
        "metrics": eval_result["model"],
        "baseline_metrics": eval_result["baseline"],
        "skill_score": eval_result["skill_score"],
        "quality_gate_passed": eval_result["quality_gate_passed"],
    }
    artifact = save_artifact(
        pipeline, metadata, get_settings().model_dir, "UC2_climate_reg_latest"
    )
    log.info("Artefak disimpan: %s", artifact)
    return {"artifact": str(artifact), **eval_result}


def train_uc3(cfg: dict = None, df=None, source: str = "db") -> dict:
    """Latih UC-3 IsolationForest offline dari dataset BMKG (hanya baris qc_flag=OK).

    Model disimpan ke UC3_anomaly_detector_latest.joblib — bukan sklearn Pipeline,
    melainkan IsolationForest langsung (tanpa preprocessor, fitur sudah numerik).
    """
    if df is None:
        if source == "dummy":
            from climate_ml.data.dummy import load_bmkg_dummy
            df = load_bmkg_dummy()
        else:
            from climate_ml.data.loaders import load_bmkg
            df = load_bmkg()

    df = filter_qc_clean(df)  # hanya data OK sebagai referensi distribusi normal
    check_min_rows(df).raise_if_invalid()

    available_cols = [c for c in UC3_FEATURE_COLS if c in df.columns]
    if not available_cols:
        raise RuntimeError(f"Tidak ada kolom fitur UC3 yang tersedia. Butuh salah satu dari: {UC3_FEATURE_COLS}")

    iso = build_iso_model(df, available_cols)

    metadata = {
        "use_case": "UC3",
        "model_name": "isolation_forest",
        "model_version": "dev",
        "feature_cols": available_cols,
        "data_rows": len(df),
        "contamination": 0.05,
    }
    artifact = save_artifact(iso, metadata, get_settings().model_dir, "UC3_anomaly_detector_latest")
    log.info("Artefak UC3 disimpan: %s | rows=%d | features=%s", artifact, len(df), available_cols)
    return {"artifact": str(artifact), "data_rows": len(df), "feature_cols": available_cols}


def train_uc4(cfg: dict, df=None, source: str = "db") -> dict:
    """Latih UC-4 (interpolasi spasial ERA5). df bisa di-inject untuk test."""
    if df is None:
        from climate_ml.data.loaders import load_era5
        df = load_era5()

    if df.empty:
        raise RuntimeError("Tabel era5_monthly kosong. Jalankan ETL ERA5 terlebih dahulu.")

    target = cfg.get("target", "t2m_celsius")
    hp = cfg.get("hyperparameters", {})

    eval_result = evaluate_uc4(df, target=target)
    log.info("Eval UC4: %s", eval_result)
    if not eval_result["quality_gate_passed"]:
        log.warning("Quality gate UC4 TIDAK lolos (skill_score <= 0).")

    prepared = prepare_uc4_frame(df)
    pipeline = build_uc4_pipeline(**hp)
    pipeline.fit(prepared, prepared[target])

    metadata = {
        "use_case": "UC4",
        "model_name": cfg.get("model_name", "random_forest"),
        "model_version": "dev",
        "target": target,
        "data_rows": len(df),
        "metrics": eval_result["model"],
        "baseline_metrics": eval_result["baseline"],
        "skill_score": eval_result["skill_score"],
        "quality_gate_passed": eval_result["quality_gate_passed"],
    }
    artifact = save_artifact(
        pipeline, metadata, get_settings().model_dir, "UC4_spatial_interp_latest"
    )
    log.info("Artefak disimpan: %s", artifact)
    return {"artifact": str(artifact), **eval_result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", default="UC1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default="db", choices=["db", "dummy"],
                        help="Sumber data: db (PostGIS) atau dummy (JSON)")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    uc = args.use_case.upper()
    if uc == "UC1":
        train_uc1(cfg, source=args.source)
    elif uc == "UC2":
        train_uc2(cfg, source=args.source)
    elif uc == "UC3":
        train_uc3(source=args.source)
    elif uc == "UC4":
        train_uc4(cfg, source=args.source)
    else:
        raise SystemExit(f"Use case {args.use_case} belum diimplementasikan di CLI ini.")


if __name__ == "__main__":
    main()
