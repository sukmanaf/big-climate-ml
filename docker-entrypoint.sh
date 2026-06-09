#!/usr/bin/env bash
# Start: pastikan model ada & kompatibel, lalu jalankan API + frontend.
# Jika DATABASE_URL menunjuk ke DB real (non-localhost), latih dengan data real.
# Jika tidak, fallback ke data dummy (mode demo offline).
set -e

model_loads() {
  python - <<'PY' 2>/dev/null
import os, sys
from pathlib import Path
from climate_ml.utils.io import load_artifact
mdir = Path(os.getenv("MODEL_DIR", "/app/models"))
for name in ("UC1_weather_clf_latest.joblib", "UC2_climate_reg_latest.joblib"):
    p = mdir / name
    if not p.exists():
        sys.exit(1)
    load_artifact(p)
PY
}

# Tentukan sumber data: real DB atau dummy
if echo "${DATABASE_URL:-}" | grep -qv "localhost\|127.0.0.1"; then
  DATA_SOURCE="db"
  echo ">>> DATABASE_URL terdeteksi → mode data real (PostGIS)"
else
  DATA_SOURCE="dummy"
  echo ">>> Tidak ada DATABASE_URL → mode dummy (offline demo)"
  python scripts/generate_dummy_data.py
fi

if model_loads; then
  echo ">>> Model sudah ada & kompatibel — skip training."
else
  echo ">>> Model belum ada / tidak kompatibel — latih UC-1, UC-2, UC-4 (source=${DATA_SOURCE})..."
  python -m climate_ml.pipelines.train --use-case UC1 \
      --config config/models/uc1_weather_clf.yaml --source "${DATA_SOURCE}"
  python -m climate_ml.pipelines.train --use-case UC2 \
      --config config/models/uc2_climate_reg.yaml --source "${DATA_SOURCE}"
  if [ "${DATA_SOURCE}" = "db" ]; then
    python -m climate_ml.pipelines.train --use-case UC4 \
        --config config/models/uc4_spatial_interp.yaml || \
        echo ">>> UC-4 training gagal (ERA5 kosong?) — dilanjutkan tanpa UC-4."
  fi
fi

echo ">>> API berjalan di http://0.0.0.0:8000  (frontend → /ui/)"
exec uvicorn climate_ml.serving.api:app --host 0.0.0.0 --port 8000
