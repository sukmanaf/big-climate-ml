"""Validasi kontrak data (Bab 3.4 dokumen teknis).

Mengembalikan daftar pelanggaran agar bisa dipakai untuk QC (UC-3) maupun
gagal-cepat di pipeline training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from climate_ml.config import get_config


@dataclass
class ValidationResult:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Validasi kontrak data gagal:\n- " + "\n- ".join(self.violations))


def validate_ranges(df: pd.DataFrame, contract: dict | None = None) -> ValidationResult:
    """Cek nilai di luar range fisik. Kolom yang tak ada di df dilewati."""
    contract = contract or get_config()["contract"]
    violations: list[str] = []
    for col, rule in contract.items():
        if not isinstance(rule, dict) or col not in df.columns:
            continue
        s = df[col].dropna()
        lo, hi = rule["min"], rule["max"]
        n_bad = int(((s < lo) | (s > hi)).sum())
        if n_bad:
            violations.append(f"{col}: {n_bad} nilai di luar [{lo}, {hi}]")
    return ValidationResult(ok=not violations, violations=violations)


def filter_qc_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil hanya baris lolos QC (qc_flag == 'OK' atau kolom tak ada).

    Selaras pendekatan flag-and-load laporan PoC v4.1: data bermasalah tetap
    tersimpan di DB tapi TIDAK dipakai untuk training ML.
    """
    if "qc_flag" not in df.columns:
        return df
    return df[df["qc_flag"].fillna("OK") == "OK"].reset_index(drop=True)


def check_min_rows(df: pd.DataFrame, contract: dict | None = None) -> ValidationResult:
    """Gagal-cepat bila data terlalu sedikit untuk training yang bermakna."""
    contract = contract or get_config()["contract"]
    minimum = contract.get("min_rows_for_training", 50)
    if len(df) < minimum:
        return ValidationResult(
            ok=False,
            violations=[f"hanya {len(df)} baris, minimum {minimum} untuk training"],
        )
    return ValidationResult(ok=True)
