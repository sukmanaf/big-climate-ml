"""FastAPI serving untuk model Climate ML.

Jalankan: uvicorn climate_ml.serving.api:app --reload
Dokumentasi interaktif: http://localhost:8000/docs
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
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
    BatchClimateRequest,
    DisasterRiskZone,
    BatchClimateResponse,
    BatchWeatherRequest,
    BatchWeatherResponse,
    ClimateByKabResult,
    ClimateRequest,
    ClimateResponse,
    Era5Request,
    Era5Response,
    Health,
    ModelInfo,
    ProvinceClimateResponse,
    SampleWeather,
    ShapValue,
    WeatherRequest,
    WeatherResponse,
    ZoneRisk,
)
from climate_ml.utils.io import load_artifact

_PROV_CODE_MAP = {
    "diy":    "DI Yogyakarta",
    "sulsel": "Sulawesi Selatan",
    "jabar":  "Jawa Barat",
}

_MODEL = {"pipeline": None, "meta": {}}        # UC-1 (klasifikasi cuaca)
_MODEL_UC2 = {"pipeline": None, "meta": {}}    # UC-2 (regresi iklim)
_MODEL_UC3 = {"iso": None, "meta": {}}         # UC-3 (anomali — IsolationForest pre-trained)
_MODEL_UC4 = {"pipeline": None, "meta": {}}    # UC-4 (interpolasi spasial ERA5)
_MODEL_UC5 = {"pipeline": None, "meta": {}}    # UC-5 (disaster risk ML)
_EXPLAINER = {"uc1": None, "uc2": None}        # SHAP TreeExplainer cache

# Peta indeks fitur setelah ColumnTransformer → nama tampilan (digroup)
# UC-1: num[0-5] + cyc[6-11] (sin_arah,sin_hour,sin_month,cos_arah,cos_hour,cos_month) + cat[12-13] (musim)
_UC1_FEAT_MAP: dict[int, str] = {
    0: "suhu_c", 1: "kelembaban_pct", 2: "kecepatan_angin_kmh",
    3: "tutupan_awan_pct", 4: "koordinat", 5: "koordinat",
    6: "arah_angin", 7: "hour", 8: "bulan",
    9: "arah_angin", 10: "hour", 11: "bulan",
    12: "musim", 13: "musim",
}
# UC-2: num[0-4] (rh2m,ws2m,rad,lat,lon) + cyc[5-6] (sin/cos month)
_UC2_FEAT_MAP: dict[int, str] = {
    0: "kelembaban (RH)", 1: "kecepatan angin", 2: "radiasi surya",
    3: "koordinat", 4: "koordinat",
    5: "bulan", 6: "bulan",
}


def _model_path() -> Path:
    return Path(get_settings().model_dir) / "UC1_weather_clf_latest.joblib"


def _model_path_uc2() -> Path:
    return Path(get_settings().model_dir) / "UC2_climate_reg_latest.joblib"


def _model_path_uc3() -> Path:
    return Path(get_settings().model_dir) / "UC3_anomaly_detector_latest.joblib"


def _model_path_uc4() -> Path:
    return Path(get_settings().model_dir) / "UC4_spatial_interp_latest.joblib"


def _model_path_uc5() -> Path:
    return Path(get_settings().model_dir) / "UC5_disaster_risk_latest.joblib"


def _shap_list(shap_row: np.ndarray, feat_map: dict[int, str]) -> list[ShapValue]:
    """Grupkan raw SHAP (abs) per nama asli fitur, normalisasi, return top-5."""
    groups: dict[str, float] = defaultdict(float)
    for idx, name in feat_map.items():
        if idx < len(shap_row):
            groups[name] += float(abs(shap_row[idx]))
    total = sum(groups.values()) or 1.0
    ranked = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    return [ShapValue(f=name, w=round(raw / total, 4), raw=round(raw, 6)) for name, raw in ranked[:5]]


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
    path5 = _model_path_uc5()
    if path5.exists():
        _MODEL_UC5["pipeline"], _MODEL_UC5["meta"] = load_artifact(path5)

    # Inisialisasi SHAP TreeExplainer (lazy, opsional)
    try:
        import shap as _shap
        if _MODEL["pipeline"] is not None:
            _EXPLAINER["uc1"] = _shap.TreeExplainer(_MODEL["pipeline"].named_steps["clf"])
        if _MODEL_UC2["pipeline"] is not None:
            _EXPLAINER["uc2"] = _shap.TreeExplainer(_MODEL_UC2["pipeline"].named_steps["reg"])
    except Exception:
        pass  # SHAP opsional — prediksi tetap berjalan tanpa XAI

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

    shap_vals: list[ShapValue] = []
    if _EXPLAINER["uc1"] is not None:
        try:
            X_t = pipeline.named_steps["pre"].transform(df)
            sv = _EXPLAINER["uc1"].shap_values(X_t)
            # sv shapes:
            #   old shap: list[(n_samples, n_features)] per class
            #   new shap: ndarray (n_samples, n_features, n_classes)
            if isinstance(sv, list):
                row = np.abs(np.stack([s[0] for s in sv], axis=0)).sum(axis=0)
            elif sv.ndim == 3:
                row = np.abs(sv[0]).sum(axis=1)  # sum across classes → (n_features,)
            else:
                row = np.abs(sv[0])
            shap_vals = _shap_list(row, _UC1_FEAT_MAP)
        except Exception:
            pass

    return WeatherResponse(
        predicted=pred, proba=probabilities[pred], probabilities=probabilities,
        model_version=_MODEL["meta"].get("model_version", "dev"),
        shap=shap_vals,
    )


@app.post("/v1/predict/climate", response_model=ClimateResponse)
def predict_climate(req: ClimateRequest) -> ClimateResponse:
    if _MODEL_UC2["pipeline"] is None:
        raise HTTPException(503, "Model UC-2 belum dilatih. Jalankan `make demo`.")
    df = pd.DataFrame([req.model_dump()])
    pipeline = _MODEL_UC2["pipeline"]

    # Enrich dengan landcover_class bila model butuh fitur ini
    try:
        from climate_ml.data.loaders import load_worldcover
        from climate_ml.features.build import enrich_uc2_with_worldcover
        wc_df = load_worldcover()
        if not wc_df.empty:
            df = enrich_uc2_with_worldcover(df, wc_df)
    except Exception:
        df["landcover_class"] = 0

    value = float(pipeline.predict(df)[0])
    target = _MODEL_UC2["meta"].get("target", "t2m")

    # Conformal CI: ±ci_q90 disimpan saat training
    ci_q90 = _MODEL_UC2["meta"].get("ci_q90")
    ci_low  = round(value - ci_q90, 2) if ci_q90 is not None else None
    ci_high = round(value + ci_q90, 2) if ci_q90 is not None else None

    shap_vals: list[ShapValue] = []
    if _EXPLAINER["uc2"] is not None:
        try:
            X_t = pipeline.named_steps["pre"].transform(df)
            sv = _EXPLAINER["uc2"].shap_values(X_t)
            row = np.abs(sv[0]) if hasattr(sv, "__len__") and sv.ndim > 1 else np.abs(sv)
            shap_vals = _shap_list(row, _UC2_FEAT_MAP)
        except Exception:
            pass

    return ClimateResponse(
        target=target, predicted=round(value, 2),
        unit="°C" if target.startswith("t2m") else "mm/hari",
        model_version=_MODEL_UC2["meta"].get("model_version", "dev"),
        ci_low=ci_low, ci_high=ci_high, shap=shap_vals,
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


@app.get("/v1/risk/disaster", response_model=list[DisasterRiskZone])
def risk_disaster() -> list[DisasterRiskZone]:
    """UC-5 ML: prediksi risiko bencana per wilayah dari model GradientBoosting.
    Dilatih dari bnpb_disaster (510 wilayah, 2010–2024) + profil iklim NASA POWER."""
    if _MODEL_UC5["pipeline"] is None:
        raise HTTPException(503, "Model UC-5 ML belum dilatih. Jalankan train_uc5().")

    from climate_ml.data.loaders import load_bnpb_disaster, load_nasa_power
    from climate_ml.models.disaster_risk import build_uc5_training_data, risk_level_from_score

    bnpb_df  = load_bnpb_disaster()
    nasa_df  = load_nasa_power()
    train_df = build_uc5_training_data(bnpb_df, nasa_df)

    feature_cols = ["avg_t2m", "avg_tmax", "avg_precip", "avg_rh2m", "lat", "lon"]
    preds = _MODEL_UC5["pipeline"].predict(train_df[feature_cols])

    q33 = _MODEL_UC5["meta"].get("q33", float(preds.mean()))
    q66 = _MODEL_UC5["meta"].get("q66", float(preds.mean()) * 1.5)

    results = []
    for i, row in train_df.iterrows():
        pred = float(preds[i])
        results.append(DisasterRiskZone(
            nama_wilayah=str(row["nama_wilayah"]),
            lat=round(float(row["lat"]), 4),
            lon=round(float(row["lon"]), 4),
            nearest_climate=str(row.get("nearest_climate", "-")),
            predicted_kejadian=round(pred, 2),
            risk_level=risk_level_from_score(pred, q33, q66),
        ))

    return sorted(results, key=lambda x: x.predicted_kejadian, reverse=True)


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


@app.get("/v1/climate/stations/{prov}")
def climate_stations(prov: str):
    """Semua titik NASA POWER untuk satu provinsi dengan series bulanan.

    prov: kode frontend (diy / sulsel / jabar) atau nama provinsi di DB.
    Return: [{label, lat, lon, normalT2m, series: [{month, t2m, precip, rh, wind, rad}]}]
    """
    from climate_ml.data.loaders import load_nasa_power

    df = load_nasa_power()
    prov_name = _PROV_CODE_MAP.get(prov.lower(), prov)
    sub = df[df["provinsi"].str.lower() == prov_name.lower()]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data NASA POWER untuk provinsi '{prov}'")

    months_id = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    result = []
    for label, grp in sub.groupby("location_label"):
        monthly = (
            grp.groupby("month")[["t2m", "prectotcorr", "rh2m", "ws2m", "allsky_sfc_sw_dwn"]]
            .mean().round(2)
        )
        series = []
        for m in range(1, 13):
            if m in monthly.index:
                r = monthly.loc[m]
                series.append({
                    "month": months_id[m - 1],
                    "t2m": round(float(r["t2m"]), 1),
                    "precip": round(float(r["prectotcorr"]), 1),
                    "rh": round(float(r["rh2m"]), 1),
                    "wind": round(float(r["ws2m"]), 2),
                    "rad": round(float(r["allsky_sfc_sw_dwn"]), 2),
                })
        lat = round(float(grp["lat"].iloc[0]), 4)
        lon = round(float(grp["lon"].iloc[0]), 4)
        normal_t2m = round(float(monthly["t2m"].mean()), 1) if not monthly.empty else None
        result.append({"label": str(label), "lat": lat, "lon": lon, "normalT2m": normal_t2m, "series": series})

    return result


@app.get("/v1/climate/annual/{location_label}")
def climate_annual(location_label: str):
    """Agregasi tahunan NASA POWER per lokasi (rata-rata suhu, total hujan, dll).

    Return: { location_label, records: [{year, t2m, precip, rh, wind, rad}] }
    """
    from climate_ml.data.loaders import load_nasa_power

    df = load_nasa_power()
    sub = df[df["location_label"].str.lower() == location_label.lower()]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data NASA POWER untuk '{location_label}'")

    agg = (
        sub.groupby("year")
        .agg(
            t2m=("t2m", "mean"),
            precip=("prectotcorr", "sum"),
            rh=("rh2m", "mean"),
            wind=("ws2m", "mean"),
            rad=("allsky_sfc_sw_dwn", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("year")
    )
    records = [
        {
            "year": int(row["year"]),
            "t2m": round(float(row["t2m"]), 1),
            "precip": round(float(row["precip"]), 1),
            "rh": round(float(row["rh"]), 1),
            "wind": round(float(row["wind"]), 2),
            "rad": round(float(row["rad"]), 2),
        }
        for _, row in agg.iterrows()
    ]
    return {"location_label": location_label, "records": records}


@app.get("/v1/climate/daily/{location_label}")
def climate_daily(location_label: str, year: Optional[int] = None, month: Optional[int] = None):
    """Time-series harian NOAA GHCN untuk stasiun terdekat ke location_label.

    Return: { location_label, station_id, station_name, station_dist_km, year, records: [{date,tmax,tmin,tavg,prcp}] }
    """
    import math
    from climate_ml.data.loaders import load_nasa_power, load_noaa_ghcn

    nasa = load_nasa_power()
    loc = nasa[nasa["location_label"].str.lower() == location_label.lower()]
    if loc.empty:
        raise HTTPException(404, f"Lokasi '{location_label}' tidak ditemukan")
    ref_lat = float(loc["lat"].iloc[0])
    ref_lon = float(loc["lon"].iloc[0])

    ghcn = load_noaa_ghcn()
    if ghcn.empty:
        raise HTTPException(404, "Tidak ada data GHCN tersedia")

    stations = ghcn[["station_id", "station_name", "lat", "lon"]].drop_duplicates("station_id").copy()
    stations["dist"] = stations.apply(
        lambda r: math.sqrt((r["lat"] - ref_lat) ** 2 + (r["lon"] - ref_lon) ** 2), axis=1
    )
    nearest = stations.loc[stations["dist"].idxmin()]

    sub = ghcn[ghcn["station_id"] == nearest["station_id"]]
    if year is not None:
        sub = sub[sub["year"] == year]
    if month is not None:
        sub = sub[sub["date"].str[5:7].astype(int) == month]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data harian untuk '{nearest['station_name']}'")

    records = [
        {
            "date": str(row["date"]),
            "tmax": round(float(row["tmax_c"]), 1) if pd.notna(row["tmax_c"]) else None,
            "tmin": round(float(row["tmin_c"]), 1) if pd.notna(row["tmin_c"]) else None,
            "tavg": round(float(row["tavg_c"]), 1) if pd.notna(row["tavg_c"]) else None,
            "prcp": round(float(row["prcp_mm"]), 1) if pd.notna(row["prcp_mm"]) else None,
        }
        for _, row in sub.sort_values("date").iterrows()
    ]
    return {
        "location_label": location_label,
        "station_id": str(nearest["station_id"]),
        "station_name": str(nearest["station_name"]),
        "station_dist_km": round(float(nearest["dist"]) * 111, 1),
        "year": year,
        "records": records,
    }


@app.get("/v1/climate/years/{location_label}")
def climate_years(location_label: str):
    """Daftar tahun yang tersedia untuk satu lokasi NASA POWER."""
    from climate_ml.data.loaders import load_nasa_power

    df = load_nasa_power()
    sub = df[df["location_label"].str.lower() == location_label.lower()]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data NASA POWER untuk '{location_label}'")
    years = sorted(int(y) for y in sub["year"].dropna().unique())
    return {"location_label": location_label, "years": years}


@app.get("/v1/climate/series/{location_label}")
def climate_series(location_label: str, year: Optional[int] = None):
    """Time-series bulanan NASA POWER untuk satu lokasi.

    Query param ?year=2023 untuk filter tahun tertentu; tanpa year = rata-rata semua tahun.
    Return: { location_label, year, normalT2m, series: [{month,t2m,precip,rh,wind,rad}, ...] }
    """
    from climate_ml.data.loaders import load_nasa_power

    df = load_nasa_power()
    sub = df[df["location_label"].str.lower() == location_label.lower()]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data NASA POWER untuk '{location_label}'")

    if year is not None:
        sub = sub[sub["year"] == year]
        if sub.empty:
            raise HTTPException(404, f"Tidak ada data untuk '{location_label}' tahun {year}")

    months_id = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    monthly = (
        sub.groupby("month")[["t2m", "prectotcorr", "rh2m", "ws2m", "allsky_sfc_sw_dwn"]]
        .mean()
        .round(2)
    )
    series = []
    for m in range(1, 13):
        if m in monthly.index:
            row = monthly.loc[m]
            series.append({
                "month": months_id[m - 1],
                "t2m": round(float(row["t2m"]), 1),
                "precip": round(float(row["prectotcorr"]), 1),
                "rh": round(float(row["rh2m"]), 1),
                "wind": round(float(row["ws2m"]), 2),
                "rad": round(float(row["allsky_sfc_sw_dwn"]), 2),
            })
    normal_t2m = round(float(monthly["t2m"].mean()), 1) if not monthly.empty else None
    return {"location_label": location_label, "year": year, "normalT2m": normal_t2m, "series": series}


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


@app.post("/v1/predict/batch/weather", response_model=BatchWeatherResponse)
def predict_weather_batch(req: BatchWeatherRequest) -> BatchWeatherResponse:
    """UC-1 batch: prediksi kondisi cuaca untuk banyak titik sekaligus (maks 300).

    SHAP dinonaktifkan untuk efisiensi batch — gunakan /v1/predict/weather untuk XAI.
    """
    if len(req.items) > 300:
        raise HTTPException(400, "Maksimum 300 item per batch request.")
    if _MODEL["pipeline"] is None:
        raise HTTPException(503, "Model UC-1 belum dilatih.")

    results: list[WeatherResponse | None] = []
    pipeline = _MODEL["pipeline"]
    for item in req.items:
        try:
            df = prepare_uc1_frame(pd.DataFrame([item.model_dump()]))
            proba_arr = pipeline.predict_proba(df)[0]
            classes = [str(c) for c in pipeline.named_steps["clf"].classes_]
            probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba_arr, strict=True)}
            pred = max(probabilities, key=probabilities.get)
            results.append(WeatherResponse(
                predicted=pred, proba=probabilities[pred], probabilities=probabilities,
                model_version=_MODEL["meta"].get("model_version", "dev"),
            ))
        except Exception:
            results.append(None)
    return BatchWeatherResponse(results=results, count=len(results))


@app.post("/v1/predict/batch/climate", response_model=BatchClimateResponse)
def predict_climate_batch(req: BatchClimateRequest) -> BatchClimateResponse:
    """UC-2 batch: prediksi suhu bulanan untuk banyak titik sekaligus (maks 300).

    Termasuk CI conformal (ci_low/ci_high). SHAP dinonaktifkan untuk efisiensi batch.
    """
    if len(req.items) > 300:
        raise HTTPException(400, "Maksimum 300 item per batch request.")
    if _MODEL_UC2["pipeline"] is None:
        raise HTTPException(503, "Model UC-2 belum dilatih.")

    pipeline = _MODEL_UC2["pipeline"]
    target = _MODEL_UC2["meta"].get("target", "t2m")
    ci_q90 = _MODEL_UC2["meta"].get("ci_q90")
    model_ver = _MODEL_UC2["meta"].get("model_version", "dev")

    # Load worldcover once outside the loop for efficiency
    _wc_df = None
    try:
        from climate_ml.data.loaders import load_worldcover
        _wc_df = load_worldcover()
        if _wc_df.empty:
            _wc_df = None
    except Exception:
        pass

    results: list[ClimateResponse | None] = []
    for item in req.items:
        try:
            df = pd.DataFrame([item.model_dump()])
            if _wc_df is not None:
                try:
                    from climate_ml.features.build import enrich_uc2_with_worldcover
                    df = enrich_uc2_with_worldcover(df, _wc_df)
                except Exception:
                    df["landcover_class"] = 0
            else:
                df["landcover_class"] = 0
            value = float(pipeline.predict(df)[0])
            results.append(ClimateResponse(
                target=target, predicted=round(value, 2),
                unit="°C" if target.startswith("t2m") else "mm/hari",
                model_version=model_ver,
                ci_low=round(value - ci_q90, 2) if ci_q90 is not None else None,
                ci_high=round(value + ci_q90, 2) if ci_q90 is not None else None,
            ))
        except Exception:
            results.append(None)
    return BatchClimateResponse(results=results, count=len(results))


@app.get("/v1/predict/climate/by-province/{prov}", response_model=ProvinceClimateResponse)
def predict_climate_by_province(prov: str, month: int, year: Optional[int] = None) -> ProvinceClimateResponse:
    """UC-2 batch per kab/lokasi pada satu provinsi.

    Berbeda dengan ``/v1/predict/batch/climate`` (yang menerima items dengan fitur
    sembarang), endpoint ini mengambil fitur cuaca (rh2m, ws2m, allsky_sfc_sw_dwn)
    untuk SETIAP lokasi di provinsi dari NASA POWER, lalu memprediksi suhu per
    lokasi memakai fiturnya sendiri.

    Tujuan: menjamin tampilan "Jawa Barat" = kumpulan prediksi per kab dengan
    fitur kabnya masing-masing, bukan diseragamkan dengan fitur kab yang sedang
    dipilih di UI.

    Args:
        prov: kode provinsi (``jabar``/``sulsel``/``diy``) atau nama provinsi di DB.
        month: 1–12.
        year: opsional. Tanpa year = rata-rata semua tahun.
    """
    from climate_ml.data.loaders import load_nasa_power

    if not (1 <= month <= 12):
        raise HTTPException(400, "month harus 1–12")
    if _MODEL_UC2["pipeline"] is None:
        raise HTTPException(503, "Model UC-2 belum dilatih.")

    df = load_nasa_power()
    prov_name = _PROV_CODE_MAP.get(prov.lower(), prov)
    sub = df[df["provinsi"].str.lower() == prov_name.lower()]
    if sub.empty:
        raise HTTPException(404, f"Tidak ada data NASA POWER untuk provinsi '{prov}'")
    if year is not None:
        sub = sub[sub["year"] == year]
        if sub.empty:
            raise HTTPException(404, f"Tidak ada data untuk provinsi '{prov}' tahun {year}")

    sub_month = sub[sub["month"] == month]
    if sub_month.empty:
        raise HTTPException(404, f"Tidak ada data untuk provinsi '{prov}' bulan {month}")

    grouped = sub_month.groupby("location_label").agg({
        "lat": "first",
        "lon": "first",
        "rh2m": "mean",
        "ws2m": "mean",
        "allsky_sfc_sw_dwn": "mean",
    })

    pipeline = _MODEL_UC2["pipeline"]
    target = _MODEL_UC2["meta"].get("target", "t2m")
    ci_q90 = _MODEL_UC2["meta"].get("ci_q90")
    model_ver = _MODEL_UC2["meta"].get("model_version", "dev")

    # Worldcover dimuat sekali untuk enrichment fitur landcover_class.
    _wc_df = None
    try:
        from climate_ml.data.loaders import load_worldcover
        _wc_df = load_worldcover()
        if _wc_df.empty:
            _wc_df = None
    except Exception:
        _wc_df = None

    unit = "°C" if target.startswith("t2m") else "mm/hari"
    results: list[ClimateByKabResult] = []
    for label, row in grouped.iterrows():
        try:
            item_df = pd.DataFrame([{
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "month": int(month),
                "rh2m": float(row["rh2m"]),
                "ws2m": float(row["ws2m"]),
                "allsky_sfc_sw_dwn": float(row["allsky_sfc_sw_dwn"]),
            }])
            if _wc_df is not None:
                try:
                    from climate_ml.features.build import enrich_uc2_with_worldcover
                    item_df = enrich_uc2_with_worldcover(item_df, _wc_df)
                except Exception:
                    item_df["landcover_class"] = 0
            else:
                item_df["landcover_class"] = 0

            value = float(pipeline.predict(item_df)[0])
            results.append(ClimateByKabResult(
                label=str(label),
                lat=round(float(row["lat"]), 4),
                lon=round(float(row["lon"]), 4),
                rh2m=round(float(row["rh2m"]), 2),
                ws2m=round(float(row["ws2m"]), 2),
                allsky_sfc_sw_dwn=round(float(row["allsky_sfc_sw_dwn"]), 2),
                predicted=round(value, 2),
                unit=unit,
                ci_low=round(value - ci_q90, 2) if ci_q90 is not None else None,
                ci_high=round(value + ci_q90, 2) if ci_q90 is not None else None,
            ))
        except Exception:
            # skip lokasi yang gagal predict, tapi jangan jatuhkan response
            continue

    return ProvinceClimateResponse(
        prov=prov_name,
        month=month,
        year=year,
        target=target,
        model_version=model_ver,
        results=results,
        count=len(results),
    )


# Sajikan frontend statis di /ui (mount terakhir agar tak menimpa route API)
_WEB_DIR = PROJECT_ROOT / "web"
if _WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")
