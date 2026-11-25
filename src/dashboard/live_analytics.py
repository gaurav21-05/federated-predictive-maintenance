"""
Live Hardware Analytics Engine for ESP32 & ThingSpeak Telemetry.
Implements descriptive statistics, robust baseline estimation (Median/MAD),
tri-method anomaly detection (Robust Z-Score, EWMA, CUSUM), trend significance testing,
bivariate correlation (Pearson/Spearman), anomaly episode clustering, and data quality diagnostics.

Pure Python & lightweight NumPy. No PyTorch or heavy dependencies (Vercel-safe).
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Statistical Utilities (Pure Python / Light NumPy)
# ============================================================================
def _median(vals: List[float]) -> float:
    """Compute exact median."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0


def _percentile(vals: List[float], p: float) -> float:
    """Compute p-th percentile (0 <= p <= 100)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1


def _t_to_p_value(t_stat: float, df: int) -> float:
    """
    Approximate two-tailed p-value for Student's t-distribution with df degrees of freedom.
    Uses Hill's approximation / normal tail for large df.
    """
    if df <= 0:
        return 1.0
    x = abs(t_stat)
    if df >= 30:
        # Standard normal approximation
        z = x
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return max(0.0, min(1.0, p))
    
    # Accurate continued fraction / incomplete beta approximation for small df
    # For Student's t: I_{df / (df + t^2)}(df/2, 1/2)
    # Using Hill (1970) approximation
    w = x / math.sqrt(df)
    y = 1.0 / (1.0 + w * w)
    z = 1.0
    for i in range(df - 2, 0, -2):
        z = 1.0 + (i - 1) * y * z / i
    if df % 2 == 1:
        p = 1.0 - (2.0 / math.pi) * (math.atan(w) + w * y * z)
    else:
        p = 1.0 - w * math.sqrt(y) * z
    return max(0.0, min(1.0, p))


# ============================================================================
# Core Analytics Functions
# ============================================================================
def compute_descriptive_stats(
    samples: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive descriptive statistics across sensors:
    mean, std, min, max, p25, p50, p75, p95, and hourly rate of change.
    """
    if not keys:
        keys = ["vibration_g", "temperature_c", "humidity_pct", "temp_secondary_c", "pressure_hpa", "current_a"]

    stats: Dict[str, Any] = {}
    total_samples = len(samples)

    if total_samples == 0:
        return {k: {"status": "no_data"} for k in keys}

    # Extract time span in hours
    first_dt = datetime.fromisoformat(samples[0]["timestamp"].replace("Z", "+00:00"))
    last_dt = datetime.fromisoformat(samples[-1]["timestamp"].replace("Z", "+00:00"))
    time_span_hours = max(1e-4, (last_dt - first_dt).total_seconds() / 3600.0)

    for k in keys:
        vals = [s[k] for s in samples if s.get(k) is not None]
        valid_count = len(vals)
        missing_count = total_samples - valid_count
        missing_pct = round((missing_count / total_samples) * 100.0, 1)

        if valid_count == 0:
            stats[k] = {
                "sensor_key": k,
                "connected": False,
                "valid_count": 0,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
                "rate_of_change_per_hour": None,
            }
            continue

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if valid_count > 1 else 0.0
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        p25 = _percentile(vals, 25.0)
        p50 = _median(vals)
        p75 = _percentile(vals, 75.0)
        p95 = _percentile(vals, 95.0)

        # Rate of change: delta / hours
        delta = vals[-1] - vals[0]
        rate_of_change = delta / time_span_hours

        stats[k] = {
            "sensor_key": k,
            "connected": True,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p75": round(p75, 4),
            "p95": round(p95, 4),
            "rate_of_change_per_hour": round(rate_of_change, 4),
        }

    return stats


def compute_robust_baseline(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute robust baseline (Median & MAD) from ESP32-normal samples (where anomaly & alert are 0).
    Falls back to all samples if fewer than 5 normal samples exist.
    """
    sensors = ["vibration_g", "temperature_c", "humidity_pct", "temp_secondary_c", "pressure_hpa", "current_a"]
    
    # Filter for normal operational state
    normal_samples = [
        s for s in samples
        if s.get("esp32_anomaly", 0) == 0 and s.get("esp32_alert", 0) == 0
    ]
    if len(normal_samples) < 5:
        normal_samples = samples

    baselines: Dict[str, Dict[str, float]] = {}

    for sensor in sensors:
        vals = [s[sensor] for s in normal_samples if s.get(sensor) is not None]
        if not vals:
            continue
        
        med = _median(vals)
        deviations = [abs(v - med) for v in vals]
        mad = _median(deviations)
        # Normal consistency scale factor: 1.4826 * MAD
        sigma_mad = max(1e-5, 1.4826 * mad)

        baselines[sensor] = {
            "median": round(med, 5),
            "mad": round(mad, 5),
            "sigma_mad": round(sigma_mad, 5),
            "lower_3sigma": round(med - 3.0 * sigma_mad, 5),
            "upper_3sigma": round(med + 3.0 * sigma_mad, 5),
            "sample_count": len(vals),
        }

    return baselines


def detect_anomalies(
    samples: List[Dict[str, Any]],
    baseline: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Detect anomalies per sample using 3 complementary methods:
    1. Robust Z-score (|z| > 3.0)
    2. EWMA control chart (alpha = 0.3, 3-sigma limits)
    3. CUSUM drift detector (k = 0.5, h = 4.0)

    Reconciles detections with ESP32 field7 and field8 hardware flags.
    """
    if not samples or not baseline:
        return {
            "anomalies": [],
            "reconciliation": {
                "agreement_rate": 100.0,
                "esp32_flagged_count": 0,
                "engine_flagged_count": 0,
                "both_flagged_count": 0,
            },
        }

    target_sensor = "vibration_g"  # primary vibration monitoring channel
    b_vib = baseline.get(target_sensor)
    if not b_vib:
        return {"anomalies": [], "reconciliation": {}}

    med = b_vib["median"]
    sigma = b_vib["sigma_mad"]

    # EWMA parameters
    alpha = 0.3
    ewma_val = med
    ewma_sigma = sigma * math.sqrt(alpha / (2.0 - alpha))
    ewma_ucl = med + 3.0 * ewma_sigma
    ewma_lcl = max(0.0, med - 3.0 * ewma_sigma)

    # CUSUM parameters (with winsorized z to isolate drift from single huge spikes)
    k_cusum = 0.5
    h_cusum = 4.0
    s_high = 0.0
    s_low = 0.0

    anomalies: List[Dict[str, Any]] = []
    engine_flags: List[bool] = []
    esp32_flags: List[bool] = []

    for idx, s in enumerate(samples):
        v = s.get(target_sensor)
        esp32_flag = (s.get("esp32_anomaly", 0) == 1) or (s.get("esp32_alert", 0) == 1)
        esp32_flags.append(esp32_flag)

        if v is None:
            engine_flags.append(False)
            continue

        # 1. Robust Z-Score
        z = (v - med) / sigma
        s["_z_score_vib"] = round(z, 2)

        methods_fired = []

        if abs(z) >= 3.0:
            methods_fired.append("robust_zscore")

        # 2. EWMA update (with winsorized input so single spike doesn't pin EWMA)
        v_clipped = min(max(v, med - 4.0 * sigma), med + 4.0 * sigma)
        ewma_val = (alpha * v_clipped) + ((1.0 - alpha) * ewma_val)
        if ewma_val > ewma_ucl or ewma_val < ewma_lcl:
            methods_fired.append("ewma_control_chart")

        # 3. CUSUM update with reset on trigger and clamped z
        z_clamped = min(max(z, -4.0), 4.0)
        s_high = max(0.0, s_high + (z_clamped - k_cusum))
        s_low = max(0.0, s_low - (z_clamped + k_cusum))
        if s_high > h_cusum or s_low > h_cusum:
            methods_fired.append("cusum_drift")
            # Standard CUSUM reset after alarm trigger
            s_high = 0.0
            s_low = 0.0

        is_engine_anomaly = len(methods_fired) > 0
        engine_flags.append(is_engine_anomaly)

        if is_engine_anomaly:
            severity = "CRITICAL" if abs(z) >= 4.5 or s.get("esp32_alert", 0) == 1 else "WARNING"
            anomalies.append({
                "entry_id": s.get("entry_id"),
                "timestamp": s.get("timestamp"),
                "sensor": target_sensor,
                "value": round(v, 4),
                "z_score": round(z, 2),
                "severity": severity,
                "methods_fired": methods_fired,
                "esp32_agreed": esp32_flag,
                "status_text": s.get("status", ""),
            })

    # Reconciliation Metrics
    total = len(samples)
    both = sum(1 for e, p in zip(engine_flags, esp32_flags) if e and p)
    e_total = sum(1 for e in engine_flags if e)
    p_total = sum(1 for p in esp32_flags if p)
    agreements = sum(1 for e, p in zip(engine_flags, esp32_flags) if e == p)
    agreement_rate = round((agreements / max(1, total)) * 100.0, 1)

    reconciliation = {
        "total_samples": total,
        "agreement_rate_pct": agreement_rate,
        "engine_flagged_count": e_total,
        "esp32_flagged_count": p_total,
        "both_flagged_count": both,
        "false_positive_vs_esp32": sum(1 for e, p in zip(engine_flags, esp32_flags) if e and not p),
        "false_negative_vs_esp32": sum(1 for e, p in zip(engine_flags, esp32_flags) if not e and p),
    }

    return {
        "anomalies": anomalies,
        "reconciliation": reconciliation,
    }


def detect_anomaly_episodes(
    samples: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cluster individual anomalous samples into contiguous anomaly episodes.
    """
    if not anomalies:
        return []

    # Map entry_id to anomaly metadata
    anom_by_entry = {a["entry_id"]: a for a in anomalies if a.get("entry_id") is not None}
    
    episodes: List[Dict[str, Any]] = []
    current_episode: Optional[Dict[str, Any]] = None

    for s in samples:
        eid = s.get("entry_id")
        is_anom = eid in anom_by_entry

        if is_anom:
            a_meta = anom_by_entry[eid]
            ts = s["timestamp"]
            vib = s.get("vibration_g", 0.0) or 0.0
            alert = s.get("esp32_alert", 0) == 1

            if current_episode is None:
                # Start new episode
                current_episode = {
                    "episode_id": f"EP-{len(episodes) + 1:03d}",
                    "start_time": ts,
                    "end_time": ts,
                    "start_entry": eid,
                    "end_entry": eid,
                    "sample_count": 1,
                    "peak_vibration_g": round(vib, 4),
                    "severities": {a_meta["severity"]},
                    "methods_fired": set(a_meta["methods_fired"]),
                    "esp32_alerted": alert,
                }
            else:
                # Extend current episode
                current_episode["end_time"] = ts
                current_episode["end_entry"] = eid
                current_episode["sample_count"] += 1
                current_episode["peak_vibration_g"] = max(current_episode["peak_vibration_g"], round(vib, 4))
                current_episode["severities"].add(a_meta["severity"])
                current_episode["methods_fired"].update(a_meta["methods_fired"])
                current_episode["esp32_alerted"] = current_episode["esp32_alerted"] or alert
        else:
            if current_episode is not None:
                # Close episode
                _finalize_episode(current_episode)
                episodes.append(current_episode)
                current_episode = None

    if current_episode is not None:
        _finalize_episode(current_episode)
        episodes.append(current_episode)

    return episodes


def _finalize_episode(ep: Dict[str, Any]):
    """Compute duration and convert sets to lists for JSON serialization."""
    dt_start = datetime.fromisoformat(ep["start_time"].replace("Z", "+00:00"))
    dt_end = datetime.fromisoformat(ep["end_time"].replace("Z", "+00:00"))
    duration = max(0.0, (dt_end - dt_start).total_seconds())
    ep["duration_seconds"] = round(duration, 1)
    ep["severities"] = list(ep["severities"])
    ep["methods_fired"] = list(ep["methods_fired"])
    ep["primary_severity"] = "CRITICAL" if "CRITICAL" in ep["severities"] else "WARNING"


def analyze_trends(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform linear regression and EWMA slope trend analysis per sensor with
    statistical significance tests (p-values).
    """
    sensors = ["vibration_g", "temperature_c", "humidity_pct", "pressure_hpa"]
    trends: Dict[str, Any] = {}

    if len(samples) < 5:
        return {s: {"status": "insufficient_data", "n": len(samples)} for s in sensors}

    # Reference time coordinates in hours from start
    dt0 = datetime.fromisoformat(samples[0]["timestamp"].replace("Z", "+00:00"))
    t_hours = [
        (datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")) - dt0).total_seconds() / 3600.0
        for s in samples
    ]

    for sensor in sensors:
        pairs = [(t, s[sensor]) for t, s in zip(t_hours, samples) if s.get(sensor) is not None]
        if len(pairs) < 5:
            trends[sensor] = {"status": "insufficient_data", "n": len(pairs)}
            continue

        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        n = len(x)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Linear regression slope: sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx <= 1e-12:
            slope = 0.0
            r2 = 0.0
            se_slope = 0.0
            p_val = 1.0
        else:
            slope = float(ss_xy / ss_xx)
            r2 = float((ss_xy ** 2) / (ss_xx * ss_yy)) if ss_yy > 1e-12 else 0.0
            
            # Standard error of slope
            y_pred = y_mean + slope * (x - x_mean)
            residuals = y - y_pred
            mse = np.sum(residuals ** 2) / max(1, n - 2)
            se_slope = math.sqrt(mse / ss_xx)
            
            t_stat = slope / max(1e-12, se_slope)
            p_val = _t_to_p_value(t_stat, n - 2)

        # EWMA slope over the last 10 points
        if n >= 10:
            sub_y = y[-10:]
            ewma_slope = float((sub_y[-1] - sub_y[0]) / max(1e-4, x[-1] - x[-10]))
        else:
            ewma_slope = slope

        is_significant = p_val < 0.05
        direction = "rising" if slope > 0 else "falling"
        if abs(slope) < 1e-4:
            direction = "stable"

        trends[sensor] = {
            "slope_per_hour": round(slope, 6),
            "ewma_slope_recent": round(ewma_slope, 6),
            "r_squared": round(r2, 4),
            "p_value": round(p_val, 4),
            "is_significant": is_significant,
            "direction": direction,
            "interpretation": f"{sensor} is {direction} at {slope:+.4f} units/hr (p={p_val:.3f}, {'significant' if is_significant else 'not significant'}).",
        }

    return trends


def compute_correlations(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Pearson & Spearman rank correlations between sensor pairs.
    """
    pairs_to_eval = [
        ("temperature_c", "vibration_g"),
        ("humidity_pct", "vibration_g"),
        ("temperature_c", "pressure_hpa"),
        ("current_a", "vibration_g"),
    ]

    results: Dict[str, Any] = {}

    for var1, var2 in pairs_to_eval:
        data = [
            (s[var1], s[var2])
            for s in samples
            if s.get(var1) is not None and s.get(var2) is not None
        ]
        key = f"{var1}__vs__{var2}"

        if len(data) < 5:
            results[key] = {
                "var1": var1,
                "var2": var2,
                "status": "insufficient_data",
                "sample_size": len(data),
            }
            continue

        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        n = len(x)

        # Pearson correlation
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < 1e-8 or y_std < 1e-8:
            r_pearson = 0.0
            p_pearson = 1.0
        else:
            r_pearson = float(np.corrcoef(x, y)[0, 1])
            # t-statistic for Pearson r
            t_r = r_pearson * math.sqrt(max(0, n - 2) / max(1e-8, 1.0 - r_pearson ** 2))
            p_pearson = _t_to_p_value(t_r, n - 2)

        # Spearman rank correlation
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        if np.std(x_rank) < 1e-8 or np.std(y_rank) < 1e-8:
            r_spearman = 0.0
            p_spearman = 1.0
        else:
            r_spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
            t_s = r_spearman * math.sqrt(max(0, n - 2) / max(1e-8, 1.0 - r_spearman ** 2))
            p_spearman = _t_to_p_value(t_s, n - 2)

        results[key] = {
            "var1": var1,
            "var2": var2,
            "status": "computed",
            "sample_size": n,
            "pearson_r": round(r_pearson, 4),
            "pearson_p": round(p_pearson, 4),
            "spearman_rho": round(r_spearman, 4),
            "spearman_p": round(p_spearman, 4),
            "interpretation": (
                f"{'Strong' if abs(r_pearson) > 0.6 else 'Moderate' if abs(r_pearson) > 0.3 else 'Weak'} "
                f"{'positive' if r_pearson > 0 else 'negative'} correlation (r={r_pearson:.3f}, p={p_pearson:.3f})"
            ),
        }

    return results


def generate_data_quality_report(
    samples: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate hardware diagnostic and data quality report:
    missing rates, flat-line detection, gap summary, and DHT11 quantisation note.
    """
    sensors = ["temperature_c", "humidity_pct", "vibration_g", "temp_secondary_c", "pressure_hpa", "current_a"]
    n = len(samples)

    missing_rates: Dict[str, float] = {}
    flatline_detected: Dict[str, bool] = {}

    for s_name in sensors:
        vals = [s.get(s_name) for s in samples]
        valid_vals = [v for v in vals if v is not None]
        missing_rate = round(((n - len(valid_vals)) / max(1, n)) * 100.0, 1)
        missing_rates[s_name] = missing_rate

        # Flatline detection: 10 consecutive ticks with zero variance
        flatline = False
        if len(valid_vals) >= 10:
            for i in range(len(valid_vals) - 9):
                sub = valid_vals[i:i + 10]
                if max(sub) - min(sub) < 1e-6:
                    flatline = True
                    break
        flatline_detected[s_name] = flatline

    return {
        "total_records": n,
        "missing_rates_pct": missing_rates,
        "flatline_detected": flatline_detected,
        "gap_count": len(gaps),
        "total_gap_duration_seconds": sum(g.get("duration_seconds", 0) for g in gaps),
        "dht11_quantisation_note": (
            "DHT11 has 1.0°C resolution (or 0.1°C via DHT22 emulation) and 1% RH resolution with ±2°C accuracy. "
            "Step-wise discretization in temperature and humidity readings is physically expected."
        ),
        "acs712_status": "DISCONNECTED_RESERVED" if missing_rates.get("current_a", 100.0) == 100.0 else "ACTIVE",
    }
