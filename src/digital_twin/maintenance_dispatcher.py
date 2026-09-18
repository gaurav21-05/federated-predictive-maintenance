"""
Prescriptive Maintenance Dispatcher Module
Translates Digital Twin health indices, failure probabilities, and RUL predictions
into actionable industrial maintenance work orders with technician instructions,
required spare parts, and downtime estimates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from .twin_engine import MachineDigitalTwin
from .rul_estimator import RULPrediction


@dataclass
class MaintenanceWorkOrder:
    """Standardized Industrial IoT Maintenance Work Order."""
    work_order_id: str
    machine_id: str
    machine_name: str
    timestamp: str
    urgency: str  # 'P1_CRITICAL', 'P2_HIGH', 'P3_MEDIUM', 'P4_LOW'
    affected_subsystem: str
    machine_health_index: float
    estimated_rul_minutes: float
    recommended_action: str
    estimated_downtime_hours: float
    required_spare_parts: List[str]
    technician_skill_level: str
    safety_precautions: List[str]
    status: str = "OPEN"  # 'OPEN', 'ASSIGNED', 'RESOLVED'

    def to_dict(self) -> Dict:
        return {
            "work_order_id": self.work_order_id,
            "machine_id": self.machine_id,
            "machine_name": self.machine_name,
            "timestamp": self.timestamp,
            "urgency": self.urgency,
            "affected_subsystem": self.affected_subsystem,
            "machine_health_index": round(self.machine_health_index, 2),
            "estimated_rul_minutes": round(self.estimated_rul_minutes, 1),
            "recommended_action": self.recommended_action,
            "estimated_downtime_hours": round(self.estimated_downtime_hours, 1),
            "required_spare_parts": self.required_spare_parts,
            "technician_skill_level": self.technician_skill_level,
            "safety_precautions": self.safety_precautions,
            "status": self.status,
        }


class PrescriptiveMaintenanceDispatcher:
    """
    Evaluates Digital Twin telemetry and dispatches prescriptive work orders.
    """

    ACTION_CATALOG = {
        "vibration": {
            "subsystem": "Mechanical Spindle & Bearings",
            "action": "Replace high-speed spindle ceramic bearings (SKF-7014), re-grease spindle cartridge, dynamic balance rotor.",
            "downtime_hours": 2.5,
            "parts": ["SKF-7014 Bearing Set", "Kluber Isoflex NBU 15 Grease", "Spindle Retention Knobs"],
            "skill": "Senior Mechanical Technician",
            "safety": ["Lockout / Tagout (LOTO) main drive", "Zero-speed verification", "Bearing puller protection"],
        },
        "thermal": {
            "subsystem": "Thermal Dissipation & Coolant Circuit",
            "action": "Flush heat-exchanger core, clean coolant chiller intake filter, verify spindle jacket flow rate (>18 L/min).",
            "downtime_hours": 1.0,
            "parts": ["Coolant Return Inline Filter", "Chiller Pump Seal Kit", "Synthetic Heat Transfer Fluid"],
            "skill": "Hydraulics / Facility Maintenance",
            "safety": ["Depressurize coolant lines", "Thermal gloves (hot fluid hazard)"],
        },
        "pressure": {
            "subsystem": "Hydraulic Power Unit & Actuators",
            "action": "Check accumulator nitrogen pre-charge (120 bar), replace cylinder rod polyurethane wiper seals, check proportional valve spool.",
            "downtime_hours": 3.0,
            "parts": ["PolyPak Hydraulic Seal Kit", "Rexroth Directional Valve Seal Ring", "ISO VG 46 Hydraulic Oil"],
            "skill": "Certified Fluid Power Specialist",
            "safety": ["Bleed hydraulic accumulator pressure to 0 bar", "Mechanical ram safety chocks"],
        },
        "electrical": {
            "subsystem": "Electrical Motor Drive & Power Inverter",
            "action": "Perform Megger insulation test on stator windings, inspect VFD inverter capacitor bank, re-torque power bus terminals.",
            "downtime_hours": 1.5,
            "parts": ["VFD Inverter Cooling Fan", "Motor Terminal Block", "Thermal Interface Paste"],
            "skill": "Certified Industrial Electrician",
            "safety": ["LOTO 480V main breaker", "Discharge DC bus capacitors (wait 10 min)", "Arc-flash PPE Category 2"],
        },
    }

    def __init__(self, dispatch_cooldown_ticks: int = 50):
        """
        Args:
            dispatch_cooldown_ticks: Minimum time ticks between duplicate work orders for same machine.
        """
        self.dispatch_cooldown_ticks = dispatch_cooldown_ticks
        self.work_orders: List[MaintenanceWorkOrder] = []
        self.last_dispatch_tick: Dict[str, int] = {}
        self.wo_counter = 1

    def evaluate_twin(
        self,
        twin: MachineDigitalTwin,
        rul_pred: Optional[RULPrediction] = None,
        current_tick: Optional[int] = None,
    ) -> Optional[MaintenanceWorkOrder]:
        """
        Evaluates current Twin health and failure risk. Dispatches a work order if thresholds are breached.
        """
        tick = current_tick if current_tick is not None else twin.total_ticks_processed
        cid = twin.client_id

        # Check cooldown to prevent alert storms
        last_tick = self.last_dispatch_tick.get(cid, -9999)
        if (tick - last_tick) < self.dispatch_cooldown_ticks:
            return None

        mhi = twin.overall_health_index
        prob = twin.failure_probability
        rul_min = rul_pred.rul_minutes if rul_pred else 180.0

        # Trigger logic
        if mhi < 45.0 or prob > 0.70 or rul_min < 15.0:
            urgency = "P1_CRITICAL"
        elif mhi < 65.0 or prob > 0.40 or rul_min < 45.0:
            urgency = "P2_HIGH"
        elif mhi < 80.0 or rul_min < 90.0:
            urgency = "P3_MEDIUM"
        else:
            # Machine is healthy, no work order needed
            return None

        # Identify primary degrading subsystem
        weakest_subsystem = min(twin.subsystem_health, key=twin.subsystem_health.get)
        spec = self.ACTION_CATALOG.get(weakest_subsystem, self.ACTION_CATALOG["vibration"])

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wo_id = f"WO-{cid.upper()}-{now_str[:10].replace('-', '')}-{self.wo_counter:04d}"
        self.wo_counter += 1

        order = MaintenanceWorkOrder(
            work_order_id=wo_id,
            machine_id=cid,
            machine_name=twin.machine_name,
            timestamp=now_str,
            urgency=urgency,
            affected_subsystem=spec["subsystem"],
            machine_health_index=mhi,
            estimated_rul_minutes=rul_min,
            recommended_action=spec["action"],
            estimated_downtime_hours=spec["downtime_hours"],
            required_spare_parts=spec["parts"],
            technician_skill_level=spec["skill"],
            safety_precautions=spec["safety"],
        )

        self.work_orders.append(order)
        self.last_dispatch_tick[cid] = tick
        return order

    def get_all_work_orders(self) -> List[Dict]:
        """Returns all dispatched work orders as dictionary list."""
        return [wo.to_dict() for wo in self.work_orders]

    def get_active_work_orders(self) -> List[Dict]:
        """Returns currently unresolved work orders."""
        return [wo.to_dict() for wo in self.work_orders if wo.status != "RESOLVED"]
