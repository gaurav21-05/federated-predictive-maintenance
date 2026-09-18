"""
FastAPI Application for Industrial IoT Federated Predictive Maintenance Dashboard
Exposes RESTful APIs for telemetry monitoring, digital twin state sync, what-if stress sandbox,
prescriptive maintenance work orders, and interactive federated learning runs.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import DashboardService

# Resolve directories
DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
PROJECT_ROOT = DASHBOARD_DIR.parent.parent

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


# Static Files & SPA Root
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Industrial IoT Dashboard API running. Static UI not found."})
