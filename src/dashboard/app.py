"""
FastAPI Backend for Real Hardware Federated Predictive Maintenance Dashboard.
Real ThingSpeak telemetry ingestion, robust live analytics, pure-NumPy Autoencoder &
Isolation Forest inference, honest RUL forecasting, and AI diagnostics insights.

No simulated machinery, zero PyTorch imports in runtime path (Vercel serverless compatible).
"""

import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.thingspeak import ThingSpeakAdapter, FIELD_MAPPING
from src.dashboard.live_analytics import (
    compute_descriptive_stats,
    compute_robust_baseline,
    detect_anomalies,
    detect_anomaly_episodes,
    analyze_trends,
    compute_correlations,
    generate_data_quality_report,
)
from src.dashboard.live_rul import (
    compute_live_health_series,
    estimate_rul,
    DEFAULT_WARNING_THRESHOLD,
    DEFAULT_FAILURE_THRESHOLD,
)
from src.dashboard.live_ml import ml_engine
from src.dashboard.live_insights import get_diagnostics_insights

app = FastAPI(
    title="Real Hardware Federated Predictive Maintenance API",
    description="Live ThingSpeak telemetry, pure-NumPy edge Autoencoder & Isolation Forest, honest RUL, and AI insights",
    version="2.0.0-real",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

thingspeak_adapter = ThingSpeakAdapter()
PUBLIC_DIR = PROJECT_ROOT / "public"
STATIC_DIR = PROJECT_ROOT / "src" / "dashboard" / "static"


# Request / Response Schemas
class RetrainRequest(BaseModel):
    results: int = Field(default=100, ge=10, le=8000, description="Telemetry batch size for fine-tuning")


class InsightsRequest(BaseModel):
    force_refresh: bool = Field(default=False, description="Bypass 5-minute cache and re-query LLM")


def _get_live_hardware_payload(results: int = 100, window: str = "all") -> Dict[str, Any]:
    """Fetch, normalize, analyze, and enrich telemetry from ThingSpeak."""
    try:
        raw_feeds, channel_info, source = thingspeak_adapter.fetch_raw_feeds(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to retrieve telemetry from ThingSpeak upstream: {str(e)}",
        )

    norm_res = thingspeak_adapter.normalize(raw_feeds)
    samples = norm_res["samples"]
    health = norm_res["health"]

    if not samples:
        return {
            "status": "empty",
            "source": source,
            "channel_id": thingspeak_adapter.channel_id,
            "online": False,
            "age_seconds": None,
            "latest": None,
            "series": [],
            "stats": {},
            "baseline": {},
            "anomalies": [],
            "reconciliation": {},
            "events": [],
            "trends": {},
            "correlations": {},
            "data_quality": {},
            "twin": {"health_index": 100.0, "state": "UNKNOWN", "subsystems": {}},
            "rul": {"rul_status": "INSUFFICIENT_DATA", "rul_hours": None},
        }

    # Window filtering (1h, 24h, all)
    if window in ("1h", "24h") and len(samples) > 1:
        last_dt = datetime.fromisoformat(samples[-1]["timestamp"].replace("Z", "+00:00"))
        cutoff_seconds = 3600.0 if window == "1h" else 86400.0
        filtered = [
            s for s in samples
            if (last_dt - datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))).total_seconds() <= cutoff_seconds
        ]
        if len(filtered) >= 5:
            samples = filtered

    # Baseline & Stats
    baseline = compute_robust_baseline(samples)
    stats = compute_descriptive_stats(samples)

    # Anomaly Detection & Episodes
    anom_res = detect_anomalies(samples, baseline)
    anomalies = anom_res["anomalies"]
    reconciliation = anom_res["reconciliation"]
    episodes = detect_anomaly_episodes(samples, anomalies)

    # Trend & Correlation
    trends = analyze_trends(samples)
    correlations = compute_correlations(samples)

    # Data Quality Report
    data_quality = generate_data_quality_report(samples, health["gaps"])

    # Health Index Series & Subsystems
    health_series = compute_live_health_series(samples, baseline)

    enriched_samples = []
    for s, h in zip(samples, health_series):
        merged = dict(s)
        merged["health_index"] = h["health_index"]
        merged["vibration_health"] = h["vibration_health"]
        merged["thermal_health"] = h["thermal_health"]
        merged["environment_health"] = h["environment_health"]
        merged["electrical_health"] = h["electrical_health"]
        enriched_samples.append(merged)

    latest_sample = enriched_samples[-1]
    current_hi = latest_sample["health_index"]
    twin_state = "OPTIMAL" if current_hi >= 80.0 else "WARNING" if current_hi >= 50.0 else "CRITICAL"

    twin = {
        "machine_id": "ESP32_BENCH_01",
        "machine_name": "ESP32 Industrial IoT Bench Node",
        "health_index": current_hi,
        "state": twin_state,
        "subsystems": {
            "vibration": latest_sample["vibration_health"],
            "thermal": latest_sample["thermal_health"],
            "environment": latest_sample["environment_health"],
            "electrical": latest_sample["electrical_health"],
        },
        "esp32_hardware": {
            "anomaly_flag": latest_sample.get("esp32_anomaly", 0),
            "alert_flag": latest_sample.get("esp32_alert", 0),
            "status_text": latest_sample.get("status", ""),
        },
    }

    rul = estimate_rul(samples, baseline)

    return {
        "status": "success",
        "source": source,
        "channel_id": thingspeak_adapter.channel_id,
        "channel_info": {
            "name": channel_info.get("name", "Industrial IoT Bench Twin"),
            "description": channel_info.get("description", "ESP32 Live Telemetry"),
            "fields": {k: meta["label"] for k, meta in FIELD_MAPPING.items()},
        },
        "online": health["online"],
        "age_seconds": health["age_seconds"],
        "sample_interval_seconds": health["sample_interval_seconds"],
        "offline_threshold_seconds": health["offline_threshold_seconds"],
        "latest_timestamp": health["latest_timestamp"],
        "latest": latest_sample,
        "series": enriched_samples,
        "stats": stats,
        "baseline": baseline,
        "anomalies": anomalies,
        "reconciliation": reconciliation,
        "events": episodes,
        "trends": trends,
        "correlations": correlations,
        "data_quality": data_quality,
        "twin": twin,
        "rul": rul,
    }


# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/live/hardware")
def get_live_hardware(
    results: int = Query(100, ge=10, le=8000),
    window: str = Query("all", pattern="^(1h|24h|all)$"),
):
    """Real ThingSpeak telemetry + connection health + 5-min stats + alerts."""
    return _get_live_hardware_payload(results=results, window=window)


@app.get("/api/live/hardware/history")
def get_live_hardware_history(
    results: int = Query(200, ge=10, le=8000),
    window: str = Query("all", pattern="^(1h|24h|all)$"),
):
    """Windowed historical telemetry time-series for charts and export."""
    payload = _get_live_hardware_payload(results=results, window=window)
    return {
        "status": payload["status"],
        "channel_id": payload["channel_id"],
        "total_samples": len(payload.get("series", [])),
        "online": payload["online"],
        "series": payload.get("series", []),
    }


@app.get("/api/analysis/stats")
def get_analysis_stats(
    results: int = Query(200, ge=10, le=8000),
    window: str = Query("all", pattern="^(1h|24h|all)$"),
):
    """Full statistical analysis: MAD, robust z-scores, CUSUM drift, EWMA, correlations, gaps."""
    payload = _get_live_hardware_payload(results=results, window=window)
    return {
        "status": payload["status"],
        "window": window,
        "sample_count": len(payload.get("series", [])),
        "stats": payload.get("stats", {}),
        "baseline": payload.get("baseline", {}),
        "trends": payload.get("trends", {}),
        "correlations": payload.get("correlations", {}),
        "data_quality": payload.get("data_quality", {}),
        "reconciliation": payload.get("reconciliation", {}),
        "episodes": payload.get("events", []),
    }


@app.get("/api/analysis/model")
def get_analysis_model(
    results: int = Query(200, ge=10, le=8000),
):
    """ML anomaly scores, autoencoder reconstruction, honest RUL, and FL model card."""
    payload = _get_live_hardware_payload(results=results, window="all")
    series = payload.get("series", [])
    
    # Compute ML scores and honest RUL
    ml_scores = ml_engine.compute_anomaly_scores(series)
    honest_rul = ml_engine.estimate_honest_rul(series)
    model_meta = ml_engine.get_model_card_and_metrics()

    return {
        "status": "success",
        "sample_count": len(series),
        "ml_inference": ml_scores,
        "honest_rul": honest_rul,
        "model_card": model_meta.get("model_card", {}),
        "federated_benchmark": model_meta.get("federated_benchmark", {}),
        "thresholds": model_meta.get("thresholds", {}),
    }


@app.post("/api/analysis/retrain")
def retrain_federated_model(req: RetrainRequest):
    """Lightweight federated update round on current cached telemetry."""
    payload = _get_live_hardware_payload(results=req.results, window="all")
    series = payload.get("series", [])
    return ml_engine.retrain_federated_round(series)


@app.post("/api/analysis/insights")
def get_analysis_insights(req: InsightsRequest):
    """AI diagnostics summary via Claude API with rate limiting and rule-based fallback."""
    payload = _get_live_hardware_payload(results=100, window="all")
    series = payload.get("series", [])
    stats = payload.get("stats", {})
    twin = payload.get("twin", {})
    rul = payload.get("rul", {})
    trends = payload.get("trends", {})

    latest = payload.get("latest") or {}
    vibe_stats = stats.get("vibration_g", {})
    temp_stats = stats.get("temperature_c", {})
    hum_stats = stats.get("humidity_pct", {})
    vibe_trend = trends.get("vibration_g", {})

    summary = {
        "device_name": "ESP32 Industrial IoT Bench Node",
        "health_index": twin.get("health_index", 100.0),
        "operational_state": twin.get("state", "OPTIMAL"),
        "latest_vibration_g": latest.get("vibration_g", 0.02),
        "vibration_mean_g": vibe_stats.get("mean", 0.02),
        "vibration_max_g": vibe_stats.get("max", 0.02),
        "vibration_slope_per_hour": vibe_trend.get("slope_per_hour", 0.0),
        "latest_temp_c": latest.get("temperature_c", 24.0),
        "temp_mean_c": temp_stats.get("mean", 24.0),
        "latest_humidity_pct": latest.get("humidity_pct", 65.0),
        "humidity_mean_pct": hum_stats.get("mean", 65.0),
        "esp32_anomaly_flag": latest.get("esp32_anomaly", 0),
        "esp32_alert_flag": latest.get("esp32_alert", 0),
        "recent_anomaly_count": len(payload.get("events", [])),
        "rul_state": rul.get("rul_status", "STABLE"),
        "estimated_rul_hours": rul.get("rul_hours") or "> 1000 h",
    }

    return get_diagnostics_insights(summary, force_refresh=req.force_refresh)


@app.get("/api/live/export.csv")
def export_live_csv(
    results: int = Query(500, ge=10, le=8000),
):
    """Export analyzed telemetry time series and health metrics as CSV."""
    payload = _get_live_hardware_payload(results=results, window="all")
    series = payload.get("series", [])

    fields = [
        "timestamp",
        "entry_id",
        "temperature_c",
        "humidity_pct",
        "temp_secondary_c",
        "pressure_hpa",
        "vibration_g",
        "current_a",
        "esp32_anomaly",
        "esp32_alert",
        "health_index",
        "vibration_health",
        "thermal_health",
        "environment_health",
        "electrical_health",
        "status",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in series:
        writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="esp32_hardware_telemetry.csv"'},
    )


@app.get("/api/health")
def get_health():
    """Uptime health check."""
    return {
        "status": "healthy",
        "mode": "REAL_DATA_ONLY",
        "device": "ESP32 DevKit V1",
        "channel_id": thingspeak_adapter.channel_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Static files mount
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR), check_dir=False), name="static")
elif STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    for candidate in [PUBLIC_DIR / "index.html", STATIC_DIR / "index.html"]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>IIoT Real Edge Intelligence Dashboard</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.app:app", host="127.0.0.1", port=8000, reload=True)
