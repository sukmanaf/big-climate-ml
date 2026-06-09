"""Koneksi ke PostgreSQL/PostGIS (re-use kredensial PoC ETL)."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from climate_ml.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Engine SQLAlchemy ke database PostGIS."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)
