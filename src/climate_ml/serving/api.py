"""FastAPI serving untuk model Climate ML.

Jalankan: uvicorn climate_ml.serving.api:app --reload
Dokumentasi interaktif: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from climate_ml.config import PROJECT_ROOT, get_config, get_settings
from climate_ml.data.validation import validate_ranges
from climate_ml.features.build import prepare_uc1_frame
from climate_ml.models.anomaly_detector import detect_anomalies, rule_based_flags
from climate_ml.serving.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ClimateRequest,
    ClimateResponse,
    Era5Request,
    Era5Response,
    Health,
    ModelInfo,
    SampleWeather,
    WeatherRequest,
    WeatherResponse,
    ZoneRisk,
)
from climate_ml.utils.io import load_artifact

_MODEL = {"pipeline": None, "meta": {}}        # UC-1 (klasifikasi cuaca)
_MODEL_UC2 = {"pipeline": None, "meta": {}}    # UC-2 (regresi iklim)
_MODEL_UC3 = {"iso": None, "meta": {}}         # UC-3 (anomali — IsolationForest pre-trained)
_MODEL_UC4 = {"pipeline": None, "meta": {}}    # UC-4 (interpolasi spasial ERA5)


def _model_path() -> Path:
    return Path(get_settings().model_dir) / "UC1_weather_clf_latest.joblib"


def _model_path_uc2() -> Path:
    return Path(get_settings().model_dir) / "UC2_climate_reg_latest.joblib"


def _model_path_uc3() -> Path:
    return Path(get_settings().model_dir) / "UC3_anomaly_detector_latest.joblib"


def _model_path_uc4() -> Path:
    return Path(get_settings().model_dir) / "UC4_spatial_interp_latest.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _model_path()
    if path.exists():
        _MODEL["pipeline"], _MODEL["meta"] = load_artifact(path)
    path2 = _model_path_uc2()
    if path2.exists():
        _MODEL_UC2["pipeline"], _MODEL_UC2["meta"] = load_artifact(path2)
    path3 = _model_path_uc3()
    if path3.exists():
        _MODEL_UC3["iso"], _MODEL_UC3["meta"] = load_artifact(path3)
    path4 = _model_path_uc4()
    if path4.exists():
        _MODEL_UC4["pipeline"], _MODEL_UC4["meta"] = load_artifact(path4)
    yield


app = FastAPI(title="Climate ML API", version="1.0.0", lifespan=lifespan)

# CORS dibuka untuk dev agar FE bisa dibuka dari origin mana pun
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    return Health(status="ok", model_loaded=_MODEL["pipeline"] is not None)


@app.get("/v1/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    pipeline, meta = _MODEL["pipeline"], _MODEL["meta"]
    if pipeline is None:
        return ModelInfo(model_loaded=False)
    classes = [str(c) for c in getattr(pipeline.named_steps["clf"], "classes_", [])]
    return ModelInfo(
        model_loaded=True,
        model_name=meta.get("model_name", "-"),
        model_version=meta.get("model_version", "dev"),
        target=meta.get("target", "-"),
        classes=classes,
        metrics=meta.get("metrics", {}),
        data_rows=meta.get("data_rows", 0),
    )


@app.post("/v1/predict/weather", response_model=WeatherResponse)
def predict_weather(req: WeatherRequest) -> WeatherResponse:
    if _MODEL["pipeline"] is None:
        raise HTTPException(503, "Model UC-1 belum dilatih. Jalankan `make demo` atau `make train`.")
    df = prepare_uc1_frame(pd.DataFrame([req.model_dump()]))
    pipeline = _MODEL["pipeline"]
    proba_arr = pipeline.predict_proba(df)[0]
    classes = [str(c) for c in pipeline.named_steps["clf"].classes_]
    probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba_arr, strict=True)}
    pred = max(probabilities, key=probabilities.get)
    return WeatherResponse(
        predicted=pred, proba=probabilities[pred], probabilities=probabilities,
        model_version=_MODEL["meta"].get("model_version", "dev"),
    )


@app.post("/v1/predict/climate", response_model=ClimateResponse)
def predict_climate(req: ClimateRequest) -> ClimateResponse:
    if _MODEL_UC2["pipeline"] is None:
        raise HTTPException(503, "Model UC-2 belum dilatih. Jalankan `make demo`.")
    df = pd.DataFrame([req.model_dump()])
    value = float(_MODEL_UC2["pipeline"].predict(df)[0])
    target = _MODEL_UC2["meta"].get("target", "t2m")
    return ClimateResponse(
        target=target, predicted=round(value, 2),
        unit="°C" if target.startswith("t2m") else "mm/hari",
        model_version=_MODEL_UC2["meta"].get("model_version", "dev"),
    )


_SAMPLE_CACHE = {"df": None}


def _bmkg_sample_df():
    """Sumber sampel: data BMKG real dari PostGIS."""
    if _SAMPLE_CACHE["df"] is None:
        from climate_ml.data.loaders import load_bmkg

        _SAMPLE_CACHE["df"] = load_bmkg()
    return _SAMPLE_CACHE["df"]


@app.get("/v1/locations")
def get_locations():
    """Hierarki lokasi dari bmkg_forecast: provinsi → kotkab → kecamatan (dengan desa)."""
    df = _bmkg_sample_df()
    result = {}
    for prov in sorted(df["provinsi"].dropna().unique()):
        sub1 = df[df["provinsi"] == prov]
        result[prov] = {}
        for kab in sorted(sub1["kotkab"].dropna().unique()):
            sub2 = sub1[sub1["kotkab"] == kab]
            kec_list = []
            for kec in sorted(sub2["kecamatan"].dropna().unique()):
                sub3 = sub2[sub2["kecamatan"] == kec]
                desa_list = sorted(sub3["desa"].dropna().unique().tolist())
                kec_list.append({"kecamatan": kec, "desa": desa_list})
            result[prov][kab] = kec_list
    return result


def _agg_bmkg_rows(df) -> SampleWeather:
    """Agregasi banyak baris BMKG menjadi satu SampleWeather (rata-rata numerik)."""
    num = ["suhu_c", "kelembaban_pct", "kecepatan_angin_kmh", "arah_angin_deg",
           "tutupan_awan_pct", "curah_hujan_mm", "lat", "lon"]
    avg = df[num].mean()
    ref = df.iloc[0]
    return SampleWeather(
        suhu_c=round(float(avg["suhu_c"]), 1),
        kelembaban_pct=round(float(avg["kelembaban_pct"]), 1),
        kecepatan_angin_kmh=round(float(avg["kecepatan_angin_kmh"]), 1),
        arah_angin_deg=round(float(avg["arah_angin_deg"]), 0),
        tutupan_awan_pct=round(float(avg["tutupan_awan_pct"]), 1),
        curah_hujan_mm=round(float(avg["curah_hujan_mm"]), 1),
        lat=round(float(avg["lat"]), 4),
        lon=round(float(avg["lon"]), 4),
        datetime_local=str(ref["datetime_local"]),
        desa=str(ref.get("desa", "")),
        actual_cuaca=str(ref.get("cuaca", "")),
    )


@app.get("/v1/sample/weather", response_model=SampleWeather)
def sample_weather(
    lat: float | None = None,
    lon: float | None = None,
    provinsi: str | None = None,
    kotkab: str | None = None,
    kecamatan: str | None = None,
    desa: str | None = None,
) -> SampleWeather:
    """Ambil data cuaca dari DB. Filter bertingkat: provinsi→kotkab→kecamatan→desa.
    Jika filter menghasilkan banyak baris, nilai dirata-rata (representatif level itu).
    Fallback: terdekat ke lat/lon jika diberikan tanpa filter lokasi."""
    df = _bmkg_sample_df()

    if any([provinsi, kotkab, kecamatan, desa]):
        if provinsi:
            df = df[df["provinsi"] == provinsi]
        if kotkab:
            df = df[df["kotkab"] == kotkab]
        if kecamatan:
            df = df[df["kecamatan"] == kecamatan]
        if desa:
            df = df[df["desa"] == desa]
        if df.empty:
            raise HTTPException(404, "Tidak ada data untuk filter lokasi tersebut.")
        return _agg_bmkg_rows(df)

    if lat is not None and lon is not None:
        dist = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
        return _agg_bmkg_rows(df.loc[[dist.idxmin()]])

    return _agg_bmkg_rows(df.sample(1))


@app.get("/v1/risk/zones", response_model=list[ZoneRisk])
def risk_zones() -> list[ZoneRisk]:
    """UC-5: climate risk score per zona RDTR Kota Yogyakarta (Phase 3 laporan)."""
    from climate_ml.data.loaders import load_nasa_power, load_rdtr
    from climate_ml.models.climate_risk import compute_zone_risk

    df = compute_zone_risk(load_rdtr(), load_nasa_power())
    return df.to_dict(orient="records")


@app.post("/v1/predict/era5", response_model=Era5Response)
def predict_era5(req: Era5Request) -> Era5Response:
    """UC-4: prediksi suhu (t2m_celsius) di titik arbitrari via interpolasi spasial ERA5."""
    if _MODEL_UC4["pipeline"] is None:
        raise HTTPException(503, "Model UC-4 belum dilatih. Jalankan: python -m climate_ml.pipelines.train --use-case UC4 --config config/models/uc4_spatial_interp.yaml")
    import pandas as pd
    df = pd.DataFrame([{"lat": req.lat, "lon": req.lon, "month": req.month}])
    value = float(_MODEL_UC4["pipeline"].predict(df)[0])
    return Era5Response(
        lat=req.lat, lon=req.lon, month=req.month,
        predicted_t2m_celsius=round(value, 2),
        model_version=_MODEL_UC4["meta"].get("model_version", "dev"),
    )


@app.get("/v1/era5/status")
def era5_status():
    """UC-4: cek ketersediaan data ERA5 di database."""
    from climate_ml.data.loaders import load_era5

    df = load_era5()
    if df.empty:
        return {
            "available": False,
            "rows": 0,
            "message": "Tabel era5_monthly kosong. Jalankan ETL ERA5 di server Docker untuk mengisi data.",
        }
    return {
        "available": True,
        "rows": len(df),
        "year_range": f"{int(df['year'].min())}–{int(df['year'].max())}",
        "lat_range": f"{df['lat'].min():.2f} s/d {df['lat'].max():.2f}",
        "lon_range": f"{df['lon'].min():.2f} s/d {df['lon'].max():.2f}",
    }


@app.get("/v1/nasa-locations")
def nasa_locations():
    """Daftar lokasi unik dari nasa_power_monthly beserta koordinat rata-rata."""
    from climate_ml.data.loaders import load_nasa_power
    df = load_nasa_power()
    locs = (df.groupby("location_label")[["lat", "lon"]]
              .mean().round(4).reset_index()
              .rename(columns={"location_label": "label"}))
    return locs.to_dict(orient="records")


@app.get("/v1/sample/climate")
def sample_climate(location_label: str | None = None, month: int | None = None):
    """Ambil nilai iklim dari nasa_power_monthly (avg per lokasi+bulan) untuk UC-2."""
    from climate_ml.data.loaders import load_nasa_power

    df = load_nasa_power()
    if location_label:
        sub = df[df["location_label"].str.lower() == location_label.lower()]
        if sub.empty:
            sub = df
    else:
        sub = df
    if month:
        m = sub[sub["month"] == month]
        if not m.empty:
            sub = m

    avg = sub[["lat", "lon", "rh2m", "ws2m", "allsky_sfc_sw_dwn"]].mean()
    row = sub.iloc[0]
    return {
        "location_label": str(row.get("location_label", "")),
        "lat": round(float(avg["lat"]), 4),
        "lon": round(float(avg["lon"]), 4),
        "month": month or int(row["month"]),
        "rh2m": round(float(avg["rh2m"]), 1),
        "ws2m": round(float(avg["ws2m"]), 2),
        "allsky_sfc_sw_dwn": round(float(avg["allsky_sfc_sw_dwn"]), 2),
    }


@app.post("/v1/anomaly/check", response_model=AnomalyResponse)
def check_anomaly(req: AnomalyRequest) -> AnomalyResponse:
    """UC-3: dua lapis — Layer 1: rule-based range check, Layer 2: IsolationForest.

    IsolationForest memakai model yang dilatih offline dari 418 baris BMKG
    (UC3_anomaly_detector_latest.joblib). Jika model belum ada, jalankan:
        python -m climate_ml.pipelines.train --use-case UC3 --config config/config.yaml
    """
    df = pd.DataFrame([req.model_dump()])
    feature_cols = ["suhu_c", "kelembaban_pct", "curah_hujan_mm", "kecepatan_angin_kmh"]

    result = detect_anomalies(df, feature_cols=feature_cols, iso_model=_MODEL_UC3["iso"])
    row = result.iloc[0]

    rule_flag = rule_based_flags(df, get_config()["contract"]).iloc[0]
    iso_flag  = bool(row["is_anomaly"]) and not rule_flag

    if rule_flag and iso_flag:
        method = "both"
    elif rule_flag:
        method = "rule"
    elif iso_flag:
        method = "isolation_forest"
    else:
        method = "rule"

    return AnomalyResponse(
        is_anomaly=bool(row["is_anomaly"]),
        reason=str(row["reason"]),
        anomaly_score=round(float(row["anomaly_score"]), 4),
        method=method,
    )


# Sajikan frontend statis di /ui (mount terakhir agar tak menimpa route API)
_WEB_DIR = PROJECT_ROOT / "web"
if _WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")
