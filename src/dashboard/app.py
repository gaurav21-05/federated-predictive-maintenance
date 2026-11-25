"""
FastAPI Application for Industrial IoT Federated Predictive Maintenance Dashboard
Exposes RESTful APIs for telemetry monitoring, digital twin state sync, what-if stress sandbox,
prescriptive maintenance work orders, and interactive federated learning runs.
"""

import csv
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Resolve directories and ensure project root is in sys.path
DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
PROJECT_ROOT = DASHBOARD_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .service import DashboardService
    from .thingspeak import ThingSpeakAdapter, FIELD_MAPPING
    from .live_analytics import (
        compute_descriptive_stats,
        compute_robust_baseline,
        detect_anomalies,
        detect_anomaly_episodes,
        analyze_trends,
        compute_correlations,
        generate_data_quality_report,
    )
    from .live_rul import (
        compute_live_health_series,
        compute_sample_subsystems,
        compute_composite_health_index,
        estimate_rul,
        DEFAULT_WARNING_THRESHOLD,
        DEFAULT_FAILURE_THRESHOLD,
    )
except (ImportError, ValueError):
    from src.dashboard.service import DashboardService
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
        compute_sample_subsystems,
        compute_composite_health_index,
        estimate_rul,
        DEFAULT_WARNING_THRESHOLD,
        DEFAULT_FAILURE_THRESHOLD,
    )


app = FastAPI(
    title="IIoT Federated Predictive Maintenance & Digital Twin Dashboard",
    description="Privacy-Preserving Industrial Edge Predictive Maintenance & Physics-Grounded Digital Twin Monitoring",
    version="1.0.0",
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single service instance
service = DashboardService()


# Request Models
class TelemetrySource(str, Enum):
    ESP32_LIVE = "ESP32_LIVE"
    SIMULATED = "SIMULATED"


class TelemetryIngestRequest(BaseModel):
    machine_id: str = Field(..., min_length=1, max_length=64, description="Unique machine identifier (e.g. CNC_MILL_01)")
    temperature_c: float = Field(..., ge=-40.0, le=150.0, description="Temperature in Celsius (-40.0°C to 150.0°C)")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage (0.0% to 100.0%)")
    acceleration_x: float = Field(..., ge=-50.0, le=50.0, description="Acceleration along X-axis in g (-50.0g to 50.0g)")
    acceleration_y: float = Field(..., ge=-50.0, le=50.0, description="Acceleration along Y-axis in g (-50.0g to 50.0g)")
    acceleration_z: float = Field(..., ge=-50.0, le=50.0, description="Acceleration along Z-axis in g (-50.0g to 50.0g)")
    gyro_x: float = Field(..., ge=-2000.0, le=2000.0, description="Angular velocity around X-axis in deg/s (-2000.0 to 2000.0 deg/s)")
    gyro_y: float = Field(..., ge=-2000.0, le=2000.0, description="Angular velocity around Y-axis in deg/s (-2000.0 to 2000.0 deg/s)")
    gyro_z: float = Field(..., ge=-2000.0, le=2000.0, description="Angular velocity around Z-axis in deg/s (-2000.0 to 2000.0 deg/s)")
    vibration_g: float = Field(..., ge=0.0, le=50.0, description="Vibration amplitude/RMS in g (0.0g to 50.0g)")
    source: TelemetrySource = Field(..., description="Telemetry source attribution: 'ESP32_LIVE' or 'SIMULATED'")
    timestamp: Optional[str] = Field(default=None, description="ISO-8601 UTC timestamp string (optional, defaults to server UTC)")

    # Optional complementary parameters
    current_a: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Optional motor current in Amperes")
    rpm: Optional[float] = Field(default=None, ge=0.0, le=20000.0, description="Optional rotational speed in RPM")
    pressure_bar: Optional[float] = Field(default=None, ge=0.0, le=500.0, description="Optional hydraulic/pneumatic pressure in bar")

    @field_validator("machine_id")
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("machine_id must not be empty or whitespace only")
        return s

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str:
            return None
        try:
            datetime.fromisoformat(v_str.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"Invalid timestamp format '{v}'. Expected ISO-8601 formatted timestamp string.")
        return v_str


class StepRequest(BaseModel):
    client_id: Optional[str] = None
    steps: int = Field(default=1, ge=1, le=100)


class ResetRequest(BaseModel):
    client_id: Optional[str] = None


class WhatIfRequest(BaseModel):
    client_id: str
    scenario_key: Optional[str] = None
    rpm_multiplier: float = 1.0
    load_multiplier: float = 1.0
    ambient_temp_delta: float = 0.0
    horizon_minutes: float = 60.0


class WorkOrderStatusRequest(BaseModel):
    work_order_id: str
    new_status: str


class FederatedTrainRequest(BaseModel):
    strategy: str = "FedProx"
    num_rounds: int = Field(default=3, ge=1, le=20)
    local_epochs: int = Field(default=1, ge=1, le=10)
    learning_rate: float = Field(default=0.001, gt=0.0)
    enable_dp: bool = True
    clip_norm: float = Field(default=1.0, gt=0.0)
    noise_multiplier: float = Field(default=0.05, ge=0.0)
    enable_secagg: bool = True
    mu: float = Field(default=0.01, ge=0.0)


# REST Endpoints
@app.get("/api/overview")
def get_overview():
    """Returns high-level plant fleet health, active machines, and federated status."""
    return service.get_overview()


@app.get("/api/telemetry/stream")
def get_telemetry_stream(
    client_id: str = Query("client_1", pattern="^client_[1-3]$"),
    window: int = Query(50, ge=10, le=200),
):
    """Returns the current historical window of sensor telemetry for the selected client."""
    return service.get_telemetry_window(client_id, window)


@app.post("/api/telemetry/ingest")
def ingest_telemetry(payload: TelemetryIngestRequest):
    """
    Ingests sensor telemetry from physical edge hardware (ESP32_LIVE) or synthetic simulation (SIMULATED).
    Validates physical sensor bounds, routes reading to Digital Twin state engine,
    and returns ingestion status, machine ID, timestamp, and source attribution.
    """
    timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()
    data = payload.model_dump()
    data["timestamp"] = timestamp
    return service.ingest_telemetry(data)


@app.get("/api/telemetry/latest")
def get_latest_telemetry():
    """Returns the most recent live or simulated telemetry reading and in-memory buffer statistics."""
    return service.get_latest_ingested_sample()


@app.post("/api/digital-twin/step")
def step_simulation(req: StepRequest):
    """Advances telemetry playback and updates Digital Twin state indices."""
    return service.step(client_id=req.client_id, steps=req.steps)


@app.post("/api/digital-twin/reset")
def reset_simulation(req: ResetRequest):
    """Resets digital twin state and playback pointer."""
    return service.reset_simulation(client_id=req.client_id)


@app.post("/api/digital-twin/what-if")
def run_what_if_simulation(req: WhatIfRequest):
    """Executes a what-if stress scenario simulation."""
    res = service.run_what_if(
        client_id=req.client_id,
        scenario_key=req.scenario_key,
        rpm_multiplier=req.rpm_multiplier,
        load_multiplier=req.load_multiplier,
        ambient_temp_delta=req.ambient_temp_delta,
        horizon_minutes=req.horizon_minutes,
    )
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.get("/api/maintenance/orders")
def get_work_orders(status: Optional[str] = None):
    """Retrieves all maintenance work orders."""
    return service.get_work_orders(status=status)


@app.post("/api/maintenance/update")
def update_work_order(req: WorkOrderStatusRequest):
    """Updates work order status (OPEN, ASSIGNED, RESOLVED)."""
    res = service.update_work_order_status(req.work_order_id, req.new_status)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/api/federated/train")
def run_federated_training(req: FederatedTrainRequest):
    """Runs on-demand federated learning rounds with live metrics."""
    res = service.run_federated_training(
        strategy_name=req.strategy,
        num_rounds=req.num_rounds,
        local_epochs=req.local_epochs,
        learning_rate=req.learning_rate,
        enable_dp=req.enable_dp,
        clip_norm=req.clip_norm,
        noise_multiplier=req.noise_multiplier,
        enable_secagg=req.enable_secagg,
        mu=req.mu,
    )
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.get("/api/benchmark/summary")
def get_benchmark_summary():
    """Returns comparative benchmark results across Centralized, Isolated, FedAvg, FedProx, and DP."""
    report_file = PROJECT_ROOT / "docs" / "benchmark_report.md"
    content = ""
    if report_file.exists():
        content = report_file.read_text(encoding="utf-8")

    return {
        "status": "ready" if content else "pending",
        "has_figures": (PROJECT_ROOT / "docs" / "figures" / "fl_convergence.png").exists(),
        "report_markdown": content,
    }


# ============================================================================
# Live Hardware Telemetry & Real ESP32 Digital Twin Endpoints
# ============================================================================
thingspeak_adapter = ThingSpeakAdapter()


def _get_live_hardware_data(results: int = 100, window: str = "all") -> Dict[str, Any]:
    """
    Fetch, normalize, analyze, and compute RUL for live ESP32 hardware via ThingSpeak.
    Stateless and safe for Vercel Serverless.
    """
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
            "correlations": {},
            "data_quality": {},
            "twin": {"health_index": 100.0, "state": "UNKNOWN", "subsystems": {}},
            "rul": {"rul_status": "INSUFFICIENT_DATA", "rul_hours": None},
        }

    # Window filtering (1h, 24h, all)
    if window in ("1h", "24h") and len(samples) > 1:
        last_dt = datetime.fromisoformat(samples[-1]["timestamp"].replace("Z", "+00:00"))
        cutoff_seconds = 3600.0 if window == "1h" else 86400.0
        filtered_samples = [
            s for s in samples
            if (last_dt - datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))).total_seconds() <= cutoff_seconds
        ]
        # Use filtered if it has enough samples, else fall back to all
        if len(filtered_samples) >= 5:
            samples = filtered_samples

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
    
    # Merge health index and subsystem scores back into each sample
    enriched_samples = []
    for s, h in zip(samples, health_series):
        merged = dict(s)
        merged["health_index"] = h["health_index"]
        merged["vibration_health"] = h["vibration_health"]
        merged["thermal_health"] = h["thermal_health"]
        merged["environment_health"] = h["environment_health"]
        merged["electrical_health"] = h["electrical_health"]
        enriched_samples.append(merged)

    # Latest Digital Twin state
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

    # RUL Prediction
    rul = estimate_rul(samples, baseline)

    # Optional Idempotent Prescriptive Work Order
    # When health dips below warning threshold (60) or RUL < 1.0 hr, dispatch work order
    if current_hi < DEFAULT_WARNING_THRESHOLD and hasattr(service, "dispatcher"):
        episode_ref = episodes[-1]["episode_id"] if episodes else "EP-LIVE"
        wo_id = f"WO-ESP32-{episode_ref}"
        if wo_id not in service.dispatcher.work_orders:
            from src.digital_twin.maintenance_dispatcher import MaintenanceWorkOrder
            weak_sub = "vibration" if (latest_sample["vibration_health"] or 100) < 60 else "thermal"
            wo = MaintenanceWorkOrder(
                work_order_id=wo_id,
                machine_id="ESP32_BENCH_01",
                machine_name="ESP32 Bench Edge Node",
                timestamp=latest_sample["timestamp"],
                urgency="P1_CRITICAL" if current_hi < 35.0 else "P2_HIGH",
                affected_subsystem=weak_sub,
                machine_health_index=current_hi,
                estimated_rul_minutes=rul.get("rul_minutes") or 60.0,
                recommended_action=rul.get("recommended_action") or "Inspect bench motor mounting and MPU6050 wiring.",
                estimated_downtime_hours=0.5,
                required_spare_parts=["MPU6050 Vibration Sensor", "12V DC Motor", "N-MOSFET Driver"],
                technician_skill_level="Level 1 Edge Technician",
                safety_precautions=["Disconnect 12V DC motor power rail", "Inspect 3.3V GPIO lines"],
                status="OPEN",
            )
            service.dispatcher.work_orders[wo_id] = wo

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


@app.get("/api/live/hardware")
def get_live_hardware(
    results: int = Query(100, ge=10, le=8000),
    window: str = Query("all", pattern="^(1h|24h|all)$"),
):
    """
    Returns full telemetry stream, robust analytics, anomaly episodes,
    digital twin state, and RUL forecast for the real ESP32 edge device.
    """
    return _get_live_hardware_data(results=results, window=window)


@app.get("/api/live/rul")
def get_live_rul(
    results: int = Query(100, ge=10, le=8000),
    window: str = Query("all", pattern="^(1h|24h|all)$"),
):
    """
    Returns lightweight RUL prediction and digital twin state snapshot.
    """
    data = _get_live_hardware_data(results=results, window=window)
    return {
        "status": data["status"],
        "online": data["online"],
        "age_seconds": data["age_seconds"],
        "twin": data["twin"],
        "rul": data["rul"],
        "source": data["source"],
    }


@app.get("/api/live/export.csv")
def export_live_csv(
    results: int = Query(500, ge=10, le=8000),
):
    """
    Export the analyzed telemetry time series and health metrics as CSV.
    """
    data = _get_live_hardware_data(results=results, window="all")
    series = data.get("series", [])

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

    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in series:
        writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="live_hardware_telemetry.csv"'},
    )


# Health check
@app.get("/api/health")
def get_health():
    """Health check endpoint for uptime monitoring."""
    return {"status": "ok", "app": "IIoT Predictive Maintenance"}


# Static Files & SPA Root
try:
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")
except Exception:
    pass


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    public_path = PROJECT_ROOT / "public" / "index.html"
    if public_path.exists():
        return HTMLResponse(content=public_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>IIoT Edge Intelligence Dashboard Active</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.app:app", host="127.0.0.1", port=8000, reload=True)


