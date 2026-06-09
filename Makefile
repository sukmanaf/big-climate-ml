.PHONY: install test test-unit test-db lint demo dummy train-dummy train predict serve docker pdf sql clean

VENV := .venv
PY := $(VENV)/bin/python
UC ?= UC1

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:               ## Semua test (skip yang butuh DB bila DB absen)
	$(PY) -m pytest -m "not db" --cov=climate_ml

test-unit:          ## Hanya unit test (cepat)
	$(PY) -m pytest tests/unit -v

test-db:            ## Test yang butuh PostGIS (set DATABASE_URL dulu)
	$(PY) -m pytest -m db -v

lint:
	$(PY) -m ruff check src tests

sql:                ## Buat tabel ML di PostGIS
	psql "$$DATABASE_URL" -f sql/01_ml_tables.sql

demo: dummy train-dummy   ## Siapkan demo: dummy JSON + latih model UC-1 & UC-2 (TANPA database)
	@echo ">>> Demo siap. Jalankan: make serve  → buka http://localhost:8000/ui/"

dummy:              ## Generate data dummy JSON
	$(PY) scripts/generate_dummy_data.py

train-dummy:        ## Latih UC-1 & UC-2 dari dummy JSON (tanpa DB)
	$(PY) -m climate_ml.pipelines.train --use-case UC1 --config config/models/uc1_weather_clf.yaml --source dummy
	$(PY) -m climate_ml.pipelines.train --use-case UC2 --config config/models/uc2_climate_reg.yaml --source dummy

train:              ## make train UC=UC1 (dari PostGIS)
	$(PY) -m climate_ml.pipelines.train --use-case $(UC) --config config/models/uc1_weather_clf.yaml --source db

predict:
	$(PY) -m climate_ml.pipelines.predict --use-case $(UC)

serve:
	$(VENV)/bin/uvicorn climate_ml.serving.api:app --reload --port 8000

docker:             ## Build + jalankan demo via Docker (buka :8000/ui/)
	docker compose up --build

pdf:                ## Export dokumen teknis ke PDF (perlu weasyprint di PATH)
	$(PY) -m pip install -q markdown pygments
	$(PY) scripts/export_pdf.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.egg-info
