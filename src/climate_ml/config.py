"""Pemuatan konfigurasi: env var (.env) + file YAML.

Nilai sensitif (DATABASE_URL) hanya dari environment; sisanya dari config/config.yaml.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Settings(BaseSettings):
    """Setting dari environment (.env)."""

    database_url: str = "postgresql+psycopg2://climate_user:climate_poc_2026@localhost:5432/climate_etl"
    random_seed: int = 42
    mlflow_tracking_uri: str = "file:./mlruns"
    model_dir: str = "./models"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(path: str | Path = CONFIG_PATH) -> dict:
    """Muat config.yaml sebagai dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> dict:
    """Muat konfigurasi spesifik use case (config/models/*.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)
