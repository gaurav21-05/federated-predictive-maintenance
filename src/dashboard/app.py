"""
FastAPI Application for Industrial IoT Federated Predictive Maintenance Dashboard
Exposes RESTful APIs for telemetry monitoring, digital twin state sync, what-if stress sandbox,
prescriptive maintenance work orders, and interactive federated learning runs.
"""

import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
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
except (ImportError, ValueError):
    from src.dashboard.service import DashboardService


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


