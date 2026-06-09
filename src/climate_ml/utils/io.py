"""Persistensi artefak model + metadata sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import joblib


def save_artifact(pipeline, metadata: dict, model_dir: str | Path, basename: str) -> Path:
    """Simpan Pipeline (.joblib) + metadata (.json). Mengembalikan path artefak."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{basename}.joblib"
    joblib.dump(pipeline, artifact_path)
    (model_dir / f"{basename}.json").write_text(json.dumps(metadata, indent=2))
    return artifact_path


def load_artifact(path: str | Path):
    """Muat Pipeline + metadata sidecar (jika ada)."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, metadata
