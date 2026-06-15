"""Skema request/response API (Pydantic v2)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WeatherRequest(BaseModel):
    suhu_c: float = Field(..., ge=-50, le=60)
    kelembaban_pct: float = Field(..., ge=0, le=100)
    kecepatan_angin_kmh: float = Field(..., ge=0, le=500)
    arah_angin_deg: float = Field(..., ge=0, le=360)
    tutupan_awan_pct: float = Field(..., ge=0, le=100)
    lat: float = Field(..., ge=-11, le=6)
    lon: float = Field(..., ge=95, le=141)
    datetime_local: str


class ShapValue(BaseModel):
    f: str              # nama fitur (original, bukan nama setelah transform)
    w: float            # bobot dinormalisasi 0–1
    raw: float = 0.0    # abs SHAP sebelum normalisasi


class WeatherResponse(BaseModel):
    predicted: str
    proba: float
    probabilities: dict[str, float] = {}
    model_version: str
    shap: list[ShapValue] = []


class ModelInfo(BaseModel):
    model_loaded: bool
    model_name: str = "-"
    model_version: str = "-"
    target: str = "-"
    classes: list[str] = []
    metrics: dict = {}
    data_rows: int = 0


class ClimateRequest(BaseModel):
    lat: float = Field(..., ge=-11, le=6)
    lon: float = Field(..., ge=95, le=141)
    month: int = Field(..., ge=1, le=12)
    rh2m: float = Field(80.0, ge=0, le=100)
    ws2m: float = Field(2.0, ge=0, le=100)
    allsky_sfc_sw_dwn: float = Field(20.0, ge=0, le=50)


class ClimateResponse(BaseModel):
    target: str
    predicted: float
    unit: str
    model_version: str
    ci_low: float | None = None
    ci_high: float | None = None
    shap: list[ShapValue] = []


class BatchWeatherRequest(BaseModel):
    items: list[WeatherRequest]


class BatchClimateRequest(BaseModel):
    items: list[ClimateRequest]


class BatchWeatherResponse(BaseModel):
    results: list[WeatherResponse | None]
    count: int


class BatchClimateResponse(BaseModel):
    results: list[ClimateResponse | None]
    count: int


class ClimateByKabResult(BaseModel):
    """Hasil prediksi UC-2 untuk satu kab/lokasi di provinsi."""
    label: str            # location_label (mis. "Bandung", "Sukabumi")
    lat: float
    lon: float
    rh2m: float           # fitur cuaca aktual kab tersebut (untuk transparansi)
    ws2m: float
    allsky_sfc_sw_dwn: float
    predicted: float
    unit: str
    ci_low: float | None = None
    ci_high: float | None = None


class ProvinceClimateResponse(BaseModel):
    """Hasil predict UC-2 per kab di satu provinsi pada bulan tertentu."""
    prov: str
    month: int
    year: int | None = None
    target: str
    model_version: str
    results: list[ClimateByKabResult]
    count: int


class SampleWeather(BaseModel):
    suhu_c: float
    kelembaban_pct: float
    kecepatan_angin_kmh: float
    arah_angin_deg: float
    tutupan_awan_pct: float
    curah_hujan_mm: float = 0.0
    lat: float
    lon: float
    datetime_local: str
    desa: str = ""
    actual_cuaca: str = ""   # label tercatat — untuk dibandingkan dengan prediksi


class ZoneRisk(BaseModel):
    nama_zona: str
    kecamatan: str
    kategori_zona: str
    luas_ha: float
    avg_tmax: float
    avg_precip: float
    avg_humidity: float
    risk_score: float
    risk_level: str
    centroid_lat: float
    centroid_lon: float


class Era5Request(BaseModel):
    lat: float = Field(..., ge=-11, le=6, description="Latitude (WGS84)")
    lon: float = Field(..., ge=95, le=141, description="Longitude (WGS84)")
    month: int = Field(..., ge=1, le=12, description="Bulan (1-12)")


class Era5Response(BaseModel):
    lat: float
    lon: float
    month: int
    predicted_t2m_celsius: float
    unit: str = "°C"
    model_version: str


class AnomalyRequest(BaseModel):
    suhu_c: float
    kelembaban_pct: float
    curah_hujan_mm: float = 0.0
    kecepatan_angin_kmh: float = 0.0


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    reason: str
    anomaly_score: float = 0.0
    method: str = "rule"   # "rule" | "isolation_forest" | "both"


class DisasterRiskZone(BaseModel):
    nama_wilayah: str
    lat: float
    lon: float
    nearest_climate: str
    predicted_kejadian: float
    risk_level: str   # Rendah / Sedang / Tinggi


class Health(BaseModel):
    status: str
    model_loaded: bool
