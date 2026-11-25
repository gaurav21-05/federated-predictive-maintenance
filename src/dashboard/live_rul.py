"""
Hardware-Specific Digital Twin Health Index & Remaining Useful Life (RUL) Estimator.
Calibrated for physical bench edge nodes (ESP32 + MPU6050 + DHT11 + ACS712 + DC Motor).

Features:
- Subsystem health breakdown: vibration (primary), thermal, environment, electrical.
- Automatic graceful activation of the electrical subsystem when ACS712 current data arrives.
- Robust Health Index (HI) time series generation.
- Statistically honest RUL forecasting: linear vs exponential degradation models (AIC selection).
- Robust smoothing (median filter) to prevent transient anomaly spikes from collapsing RUL.
- Honesty rules: returns STABLE_NO_DEGRADATION or INSUFFICIENT_DATA with null RUL when wear is negligible.
- Prediction confidence intervals, operating hours counter, and rule-based prescriptive actions.

Pure Python & lightweight NumPy. Zero PyTorch or heavy physics dependencies (Vercel-safe).
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Subsystem Weight Configurations
# ============================================================================
DEFAULT_WEIGHTS_NO_CURRENT = {
    "vibration": 0.55,
    "thermal": 0.30,
    "environment": 0.15,
}

DEFAULT_WEIGHTS_WITH_CURRENT = {
    "vibration": 0.45,
    "thermal": 0.25,
    "environment": 0.10,
    "electrical": 0.20,
}

# Critical thresholds
DEFAULT_WARNING_THRESHOLD = 60.0
DEFAULT_FAILURE_THRESHOLD = 30.0
DEFAULT_MIN_SAMPLES_FOR_RUL = 30


# ============================================================================
# Subsystem Health Evaluators
# ============================================================================
def _score_from_z(z: float, soft_limit: float = 2.0, hard_limit: float = 6.0) -> float:
    """
    Map a positive deviation z-score to a 0 - 100 health index score.
    z <= soft_limit -> 100%
    z >= hard_limit -> 0%
    Smooth parabolic / linear decay between soft_limit and hard_limit.
    """
    if z <= soft_limit:
        return 100.0
    if z >= hard_limit:
        return 0.0
    ratio = (z - soft_limit) / (hard_limit - soft_limit)
    # Smooth quadratic descent
    return max(0.0, min(100.0, 100.0 * (1.0 - ratio ** 1.5)))


def compute_sample_subsystems(
    sample: Dict[str, Any],
    baseline: Dict[str, Dict[str, float]],
    has_active_current: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Compute individual subsystem health scores (0 - 100) for a single telemetry sample.
    """
    subsystems: Dict[str, Optional[float]] = {
        "vibration": 100.0,
        "thermal": 100.0,
        "environment": 100.0,
        "electrical": None,
    }

    # 1. Vibration Health (MPU6050)
    v_vib = sample.get("vibration_g")
    b_vib = baseline.get("vibration_g")
    if v_vib is not None and b_vib:
        z_vib = max(0.0, (v_vib - b_vib["median"]) / max(1e-5, b_vib["sigma_mad"]))
        # Additional penalty for high absolute g
        vib_score = _score_from_z(z_vib, soft_limit=2.5, hard_limit=8.0)
        if v_vib > 0.4:
            vib_score = min(vib_score, 15.0)
        elif v_vib > 0.2:
            vib_score = min(vib_score, 40.0)
        subsystems["vibration"] = round(vib_score, 1)

    # 2. Thermal Health (DHT11 temp + secondary temp if available)
    v_temp = sample.get("temperature_c")
    b_temp = baseline.get("temperature_c")
    if v_temp is not None and b_temp:
        z_temp = max(0.0, (v_temp - b_temp["median"]) / max(1e-5, b_temp["sigma_mad"]))
        temp_score = _score_from_z(z_temp, soft_limit=3.0, hard_limit=7.0)
        if v_temp > 50.0:  # Absolute high ambient temp
            temp_score = min(temp_score, 30.0)
        subsystems["thermal"] = round(temp_score, 1)

    # 3. Environmental Health (Humidity + Pressure)
    v_hum = sample.get("humidity_pct")
    b_hum = baseline.get("humidity_pct")
    v_press = sample.get("pressure_hpa")
    b_press = baseline.get("pressure_hpa")

    hum_score = 100.0
    if v_hum is not None and b_hum:
        z_hum = max(0.0, abs(v_hum - b_hum["median"]) / max(1e-5, b_hum["sigma_mad"]))
        hum_score = _score_from_z(z_hum, soft_limit=3.0, hard_limit=6.0)

    press_score = 100.0
    if v_press is not None and b_press:
        z_press = max(0.0, abs(v_press - b_press["median"]) / max(1e-5, b_press["sigma_mad"]))
        press_score = _score_from_z(z_press, soft_limit=3.0, hard_limit=6.0)

    subsystems["environment"] = round(0.6 * hum_score + 0.4 * press_score, 1)

    # 4. Electrical Health (ACS712 current) - only active if current data is present
    v_curr = sample.get("current_a")
    if has_active_current and v_curr is not None:
        b_curr = baseline.get("current_a")
        if b_curr:
            z_curr = max(0.0, (v_curr - b_curr["median"]) / max(1e-5, b_curr["sigma_mad"]))
            curr_score = _score_from_z(z_curr, soft_limit=2.5, hard_limit=6.0)
        else:
            # Nominal bench 12V DC motor current bounds: 0.1A to 2.5A
            curr_score = 100.0 if v_curr <= 2.0 else max(0.0, 100.0 - (v_curr - 2.0) * 40.0)
        subsystems["electrical"] = round(curr_score, 1)
    else:
        subsystems["electrical"] = None

    return subsystems


def compute_composite_health_index(
    subsystems: Dict[str, Optional[float]],
    esp32_anomaly: int = 0,
    esp32_alert: int = 0,
) -> float:
    """
    Calculate composite Health Index (HI) 0 - 100 with dynamic weight normalization
    and ESP32 hardware alert capping.
    """
    has_electrical = subsystems.get("electrical") is not None
    weights = DEFAULT_WEIGHTS_WITH_CURRENT if has_electrical else DEFAULT_WEIGHTS_NO_CURRENT

    # Normalize weights over available subsystems
    total_weight = sum(weights[k] for k in weights if subsystems.get(k) is not None)
    if total_weight <= 0:
        composite = 100.0
    else:
        composite = sum(
            subsystems[k] * (weights[k] / total_weight)
            for k in weights
            if subsystems.get(k) is not None
        )

    # ESP32 hardware flag capping:
    # If the firmware fired the buzzer/LED (alert_flag == 1), health cannot exceed 45% (WARNING/CRITICAL)
    if esp32_alert == 1:
        composite = min(composite, 42.0)
    elif esp32_anomaly == 1:
        composite = min(composite, 65.0)

    return max(0.0, min(100.0, round(composite, 1)))


# ============================================================================
# RUL & Trend Degradation Forecasting
# ============================================================================
def compute_live_health_series(
    samples: List[Dict[str, Any]],
    baseline: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Compute subsystem breakdown and Health Index time series for every sample.
    """
    # Detect if any valid current readings exist in the sample set
    has_active_current = any(s.get("current_a") is not None for s in samples)

    enriched_series: List[Dict[str, Any]] = []

    for s in samples:
        sub = compute_sample_subsystems(s, baseline, has_active_current)
        hi = compute_composite_health_index(
            sub,
            esp32_anomaly=s.get("esp32_anomaly", 0) or 0,
            esp32_alert=s.get("esp32_alert", 0) or 0,
        )
        enriched_series.append({
            "entry_id": s.get("entry_id"),
            "timestamp": s.get("timestamp"),
            "health_index": hi,
            "vibration_health": sub["vibration"],
            "thermal_health": sub["thermal"],
            "environment_health": sub["environment"],
            "electrical_health": sub["electrical"],
            "esp32_anomaly": s.get("esp32_anomaly", 0),
            "esp32_alert": s.get("esp32_alert", 0),
        })

    return enriched_series


def _robust_smooth(values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply a 1D median filter smoothing window to protect RUL regression
    from collapsing due to transient single-sample anomaly spikes.
    """
    if len(values) < window_size:
        return list(values)
    
    half = window_size // 2
    smoothed: List[float] = []
    n = len(values)

    for i in range(n):
        low = max(0, i - half)
        high = min(n, i + half + 1)
        sub = values[low:high]
        smoothed.append(float(np.median(sub)))

    return smoothed


def compute_operating_hours(samples: List[Dict[str, Any]]) -> float:
    """
    Compute total cumulative operating hours, summing sample intervals
    excluding offline gaps longer than 5 minutes (300 seconds).
    """
    if len(samples) < 2:
        return 0.0

    total_seconds = 0.0
    for i in range(1, len(samples)):
        t_prev = datetime.fromisoformat(samples[i - 1]["timestamp"].replace("Z", "+00:00"))
        t_curr = datetime.fromisoformat(samples[i]["timestamp"].replace("Z", "+00:00"))
        diff = (t_curr - t_prev).total_seconds()
        # Only count if consecutive gap <= 300s
        if 0 < diff <= 300.0:
            total_seconds += diff

    return round(total_seconds / 3600.0, 2)


def estimate_rul(
    samples: List[Dict[str, Any]],
    baseline: Dict[str, Dict[str, float]],
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES_FOR_RUL,
) -> Dict[str, Any]:
    """
    Statistically honest RUL forecasting from physical telemetry.
    Selects between Linear and Exponential wear models via AIC.
    Enforces honesty rules (returns STABLE_NO_DEGRADATION or INSUFFICIENT_DATA when appropriate).
    """
    n = len(samples)
    op_hours = compute_operating_hours(samples)

    if n < min_samples:
        return {
            "rul_status": "INSUFFICIENT_DATA",
            "rul_hours": None,
            "rul_minutes": None,
            "ci_lower_hours": None,
            "ci_upper_hours": None,
            "operating_hours": op_hours,
            "time_to_warning_hours": None,
            "failure_probability": 0.01,
            "confidence": "LOW",
            "method": "insufficient_data",
            "samples_used": n,
            "warning_threshold": warning_threshold,
            "failure_threshold": failure_threshold,
            "recommended_action": f"Collecting telemetry. Need at least {min_samples} operational samples (currently {n}).",
            "model_fit": None,
        }

    # Generate Health Index series
    hi_series = compute_live_health_series(samples, baseline)
    raw_his = [item["health_index"] for item in hi_series]
    
    # Apply robust smoothing to filter transient spikes
    smoothed_his = _robust_smooth(raw_his, window_size=5)
    current_hi = smoothed_his[-1]

    # Time coordinate in hours from the first sample
    dt0 = datetime.fromisoformat(samples[0]["timestamp"].replace("Z", "+00:00"))
    t_hours = np.array([
        (datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")) - dt0).total_seconds() / 3600.0
        for s in samples
    ])
    t_current = t_hours[-1]
    y_hi = np.array(smoothed_his)

    # 1. Linear Degradation Fit: HI(t) = m * t + b
    t_mean = np.mean(t_hours)
    y_mean = np.mean(y_hi)
    ss_tt = np.sum((t_hours - t_mean) ** 2)
    ss_ty = np.sum((t_hours - t_mean) * (y_hi - y_mean))

    if ss_tt <= 1e-12:
        slope_linear = 0.0
        b_linear = y_mean
        r2_linear = 0.0
        rss_linear = float(np.sum((y_hi - y_mean) ** 2))
    else:
        slope_linear = float(ss_ty / ss_tt)
        b_linear = float(y_mean - slope_linear * t_mean)
        y_pred_lin = slope_linear * t_hours + b_linear
        rss_linear = float(np.sum((y_hi - y_pred_lin) ** 2))
        ss_tot = np.sum((y_hi - y_mean) ** 2)
        r2_linear = float(1.0 - (rss_linear / max(1e-12, ss_tot)))

    # Failure probability proxy (logistic curve centered at 50% HI)
    failure_prob = round(1.0 / (1.0 + math.exp((current_hi - 50.0) / 10.0)), 3)

    # Recommended diagnostic action based on weakest subsystem
    latest_sub = compute_sample_subsystems(samples[-1], baseline, any(s.get("current_a") is not None for s in samples))
    recommended_action = _generate_recommended_action(latest_sub, current_hi)

    # Honesty Rule: If trend slope is not significantly negative, machine is STABLE
    # (Wear must be decreasing at least 0.01% HI per hour to project a collapse)
    if slope_linear >= -0.01:
        return {
            "rul_status": "STABLE_NO_DEGRADATION",
            "rul_hours": None,
            "rul_minutes": None,
            "ci_lower_hours": None,
            "ci_upper_hours": None,
            "operating_hours": op_hours,
            "time_to_warning_hours": None,
            "failure_probability": failure_prob,
            "confidence": "HIGH",
            "method": "stable_baseline",
            "samples_used": n,
            "warning_threshold": warning_threshold,
            "failure_threshold": failure_threshold,
            "health_slope_per_hour": round(slope_linear, 5),
            "recommended_action": recommended_action,
            "model_fit": {
                "r_squared": round(r2_linear, 4),
                "degradation_rate_pct_hr": round(abs(slope_linear), 4),
            },
        }

    # 2. Exponential Wear Fit: ln(HI(t)) = ln(HI_0) - lambda * t
    # Valid only if HI values are positive
    valid_exp = all(h > 1.0 for h in y_hi)
    rss_exp = float("inf")
    lambda_exp = 0.0
    hi0_exp = 100.0
    r2_exp = 0.0

    if valid_exp:
        log_y = np.log(y_hi)
        log_y_mean = np.mean(log_y)
        ss_t_logy = np.sum((t_hours - t_mean) * (log_y - log_y_mean))
        if ss_tt > 1e-12:
            m_exp = float(ss_t_logy / ss_tt)
            b_exp = float(log_y_mean - m_exp * t_mean)
            if m_exp < 0:  # Valid decay
                lambda_exp = -m_exp
                hi0_exp = math.exp(b_exp)
                y_pred_exp = hi0_exp * np.exp(-lambda_exp * t_hours)
                rss_exp = float(np.sum((y_hi - y_pred_exp) ** 2))
                ss_tot_log = np.sum((log_y - log_y_mean) ** 2)
                r2_exp = float(1.0 - (np.sum((log_y - (m_exp * t_hours + b_exp)) ** 2) / max(1e-12, ss_tot_log)))

    # Model Selection via Residual Sum of Squares (AIC equivalent for equal parameter count)
    use_exponential = (rss_exp < rss_linear) and (lambda_exp > 1e-5)

    if use_exponential:
        method = "exponential_wear"
        r2 = r2_exp
        # RUL: time from current when HI decays to failure_threshold
        # failure_threshold = current_hi * exp(-lambda * RUL)
        # RUL = (ln(current_hi) - ln(failure_threshold)) / lambda
        if current_hi <= failure_threshold:
            rul_hours = 0.0
            time_to_warn = 0.0
        else:
            rul_hours = max(0.0, (math.log(current_hi) - math.log(failure_threshold)) / lambda_exp)
            time_to_warn = max(0.0, (math.log(current_hi) - math.log(warning_threshold)) / lambda_exp) if current_hi > warning_threshold else 0.0
        
        se_rul = max(0.1, 0.15 * rul_hours)
    else:
        method = "linear_extrapolation"
        r2 = r2_linear
        # failure_threshold = slope * (t_current + RUL) + b
        # RUL = (failure_threshold - (slope * t_current + b)) / slope
        current_pred = slope_linear * t_current + b_linear
        if current_hi <= failure_threshold or current_pred <= failure_threshold:
            rul_hours = 0.0
            time_to_warn = 0.0
        else:
            rul_hours = max(0.0, (failure_threshold - current_pred) / slope_linear)
            time_to_warn = max(0.0, (warning_threshold - current_pred) / slope_linear) if current_hi > warning_threshold else 0.0
        
        # Standard prediction interval for linear regression
        mse = rss_linear / max(1, n - 2)
        se_slope = math.sqrt(mse / max(1e-12, ss_tt))
        se_rul = max(0.1, abs(rul_hours) * (se_slope / max(1e-5, abs(slope_linear))))

    # 95% Confidence Bounds
    ci_lower = max(0.0, round(float(rul_hours - 1.96 * se_rul), 1))
    ci_upper = round(float(rul_hours + 1.96 * se_rul), 1)

    # Confidence label
    if r2 >= 0.8 and n >= 40:
        conf_label = "HIGH"
    elif r2 >= 0.4 or n >= 30:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    return {
        "rul_status": "DEGRADING",
        "rul_hours": round(float(rul_hours), 1),
        "rul_minutes": round(float(rul_hours * 60.0), 1),
        "ci_lower_hours": ci_lower,
        "ci_upper_hours": ci_upper,
        "operating_hours": float(op_hours),
        "time_to_warning_hours": round(float(time_to_warn), 1),
        "failure_probability": float(failure_prob),
        "confidence": conf_label,
        "method": method,
        "samples_used": int(n),
        "warning_threshold": float(warning_threshold),
        "failure_threshold": float(failure_threshold),
        "health_slope_per_hour": round(float(slope_linear), 5),
        "recommended_action": recommended_action,
        "model_fit": {
            "r_squared": round(float(r2), 4),
            "degradation_rate_pct_hr": round(float(abs(slope_linear)), 4),
            "fit_method": method,
        },
    }


def _generate_recommended_action(
    subsystems: Dict[str, Optional[float]],
    health_index: float,
) -> str:
    """
    Generate prescriptive diagnostics based on the lowest subsystem score.
    """
    if health_index >= 85.0:
        return "All edge hardware subsystems are operating within nominal baseline parameters. Routine observation advised."

    # Find the subsystem with the lowest health score
    valid_subs = {k: v for k, v in subsystems.items() if v is not None}
    if not valid_subs:
        return "Inspect edge device connections and physical wiring."

    weakest_sub = min(valid_subs, key=valid_subs.get)
    weak_score = valid_subs[weakest_sub]

    if weakest_sub == "vibration":
        return (
            f"Vibration subsystem score degraded to {weak_score:.0f}%. "
            "Inspect DC motor mounting bracket for mechanical looseness, shaft rotor imbalance, and bearing wear."
        )
    elif weakest_sub == "thermal":
        return (
            f"Thermal subsystem score degraded to {weak_score:.0f}%. "
            "Inspect motor enclosure ventilation, ambient airflow, and inspect for friction overheating."
        )
    elif weakest_sub == "electrical":
        return (
            f"Electrical subsystem score degraded to {weak_score:.0f}%. "
            "Inspect DC motor current draw (ACS712), verify N-MOSFET gate drive signal, and check power supply voltage stability."
        )
    else:
        return (
            f"Environmental subsystem score degraded to {weak_score:.0f}%. "
            "Inspect ambient relative humidity and barometric conditions; isolate electronics enclosure from moisture."
        )
