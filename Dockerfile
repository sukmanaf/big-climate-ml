# Image demo Climate ML — UC-1 & UC-2 + frontend, berbasis data dummy (tanpa DB).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    MLFLOW_TRACKING_URI=file:/app/mlruns

WORKDIR /app

# Install dependencies dulu (layer cache)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Salin kode
COPY src/ ./src/
COPY config/ ./config/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY docker-entrypoint.sh ./

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
