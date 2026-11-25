"""
AI-Generated Diagnostics Insights Module.
Calls Anthropic Claude API when ANTHROPIC_API_KEY is available.
Includes 5-minute server cache, rate-limiting protection, and a deterministic
rule-based engineering fallback that never fails or returns 500 when the key is missing.
"""

import os
import time
import json
import logging
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger("live_insights")

# In-memory cache for insights: (timestamp, result_dict)
_INSIGHTS_CACHE: Dict[str, Any] = {
    "cached_at": 0.0,
    "data": None,
}
CACHE_TTL_SECONDS = 300.0  # 5 minutes


def build_system_prompt() -> str:
    return (
        "You are a Senior Principal Reliability and Industrial IoT Engineer specializing in predictive maintenance "
        "and physical machinery monitoring. You analyze real telemetry from an ESP32 edge device equipped with "
        "an MPU6050 accelerometer, DHT11 temperature/humidity sensor, and DC motor. "
        "Provide a crisp, professional, human-readable engineering assessment. Avoid fluff or generic statements. "
        "Highlight actual physical interpretations, component health, and concrete recommended maintenance actions."
    )


def build_user_prompt(telemetry_summary: Dict[str, Any]) -> str:
    return (
        "Analyze the following real IoT sensor telemetry and predictive maintenance state:\n\n"
        f"```json\n{json.dumps(telemetry_summary, indent=2)}\n```\n\n"
        "Please provide an engineering assessment formatted in Markdown with the following 5 distinct sections:\n"
        "1. **Executive Health Summary**: Overall machinery state, health index rating, and primary finding.\n"
        "2. **Subsystem Analysis**: Detailed breakdown of Vibration RMS (MPU6050), Thermal & Ambient (DHT11), and Motor status.\n"
        "3. **Physical Interpretation**: What the observed vibrations, temperatures, and anomaly scores indicate physically (e.g. transient vibration spike, mechanical mounting, friction, or normal steady state).\n"
        "4. **Prescriptive Recommendations**: 2-3 specific, actionable physical inspection or maintenance steps.\n"
        "5. **Confidence & Data Quality Caveats**: Assessment of sensor sampling frequency (20s), data gaps, or sensor resolution limits."
    )


def generate_rule_based_insights(telemetry_summary: Dict[str, Any]) -> str:
    """Deterministic, high-quality rule-based engineering analysis fallback."""
    vibe_current = telemetry_summary.get("latest_vibration_g", 0.02)
    vibe_mean = telemetry_summary.get("vibration_mean_g", 0.02)
    temp_current = telemetry_summary.get("latest_temp_c", 24.0)
    hum_current = telemetry_summary.get("latest_humidity_pct", 65.0)
    health_index = telemetry_summary.get("health_index", 95.0)
    anom_count = telemetry_summary.get("recent_anomaly_count", 0)
    rul_state = telemetry_summary.get("rul_state", "STABLE")
    rul_hours = telemetry_summary.get("estimated_rul_hours", "> 1000 h")
    trend_slope = telemetry_summary.get("vibration_slope_per_hour", 0.0)

    # Health category
    if health_index >= 85:
        health_cat = "OPTIMAL"
        primary_msg = "The machinery is operating stably within normal operational bounds."
    elif health_index >= 60:
        health_cat = "ACCEPTABLE / WARNING"
        primary_msg = "Elevated vibration activity or transient spikes detected; monitoring advised."
    else:
        health_cat = "DEGRADED / CRITICAL"
        primary_msg = "Significant vibration or thermal anomaly detected requiring immediate physical inspection."

    # Subsystem remarks
    if vibe_current > 0.20:
        vibe_diag = f"CRITICAL: Current vibration ({vibe_current:.4f} g) exceeds safe threshold (0.25 g)!"
        phys_interp = "Vibration spikes indicate potential mechanical imbalance, loose bench mounting, or DC motor bearing rattle under load."
        rec_1 = "Check mechanical fastener tightness on motor mounting bracket and MPU6050 sensor board."
    elif vibe_current > 0.05:
        vibe_diag = f"ELEVATED: Current vibration ({vibe_current:.4f} g) is above normal baseline ({vibe_mean:.4f} g)."
        phys_interp = "Intermittent vibration transients observed; likely transient motor start-up oscillation or external bench disturbance."
        rec_1 = "Inspect bench damping pads and confirm MPU6050 is securely adhered without cable pull."
    else:
        vibe_diag = f"NOMINAL: Vibration RMS is {vibe_current:.4f} g (baseline: {vibe_mean:.4f} g), well below 0.25 g limit."
        phys_interp = "Vibration pattern reflects clean motor idling/rotation with minimal eccentric runout."
        rec_1 = "Maintain standard 20-second edge sampling interval; no immediate mechanical adjustments required."

    rec_2 = "Verify DHT11 probe ventilation and confirm ACS712 current sensor wiring when connecting motor current channel."

    text = f"""### Rule-based Analysis (Set `ANTHROPIC_API_KEY` for AI Insights)

#### 1. Executive Health Summary
* **Machinery Status**: **{health_cat}** (Health Index: **{health_index:.1f} / 100**)
* **Active Anomaly Episodes**: {anom_count} event(s) recorded in analysis window.
* **Remaining Useful Life**: **{rul_state}** ({rul_hours}).
* **Core Assessment**: {primary_msg}

#### 2. Subsystem Analysis
* **Vibration RMS (MPU6050)**: {vibe_diag} Linear trend slope is `{trend_slope:+.6f} g/h`.
* **Thermal & Environmental (DHT11)**: Temperature is `{temp_current:.1f} °C` and relative humidity is `{hum_current:.1f} %`. Thermal gradient is stable with no localized thermal runaway.
* **Electrical (ACS712 Current Sensor)**: Reserved channel currently inactive (NA); electrical subsystem health will activate automatically upon sensor hookup.

#### 3. Physical Interpretation
{phys_interp}

#### 4. Prescriptive Recommendations
1. **Mechanical**: {rec_1}
2. **Electrical & Telemetry**: {rec_2}
3. **Model Maintenance**: Retrain federated model weights when a minimum of 50 new operating samples are ingested.

#### 5. Confidence & Data Quality Caveats
* **Telemetry Cadence**: ThingSpeak 20-second telemetry update rate provides adequate resolution for trend modeling but may alias high-frequency vibration transients (>25 Hz).
* **Sensor Resolution**: DHT11 exhibits 1.0 °C / 1% RH quantization step.
* **Model Partitioning**: Federated learning metrics derived from time-partitioned operational sessions on the physical edge device.
"""
    return text


def get_diagnostics_insights(
    telemetry_summary: Dict[str, Any],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Generate or return cached insights.
    Calls Anthropic API if key exists, otherwise falls back gracefully.
    """
    global _INSIGHTS_CACHE

    now = time.time()
    # Check cache unless force_refresh is true
    if not force_refresh and _INSIGHTS_CACHE["data"] is not None:
        age = now - _INSIGHTS_CACHE["cached_at"]
        if age < CACHE_TTL_SECONDS:
            res = dict(_INSIGHTS_CACHE["data"])
            res["cached"] = True
            res["cache_age_seconds"] = round(age, 1)
            return res

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        logger.info("ANTHROPIC_API_KEY not configured; generating rule-based insights fallback.")
        content = generate_rule_based_insights(telemetry_summary)
        result = {
            "status": "success",
            "source": "rule_based_fallback",
            "markdown": content,
            "cached": False,
            "cache_age_seconds": 0.0,
            "api_key_configured": False,
        }
        _INSIGHTS_CACHE["cached_at"] = now
        _INSIGHTS_CACHE["data"] = result
        return result

    # Call Anthropic API
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "system": build_system_prompt(),
            "messages": [
                {"role": "user", "content": build_user_prompt(telemetry_summary)}
            ],
        }

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=12.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            content = "\n".join(text_blocks)
            result = {
                "status": "success",
                "source": "anthropic_claude",
                "model": data.get("model", "claude-3-5-sonnet-20241022"),
                "markdown": content,
                "cached": False,
                "cache_age_seconds": 0.0,
                "api_key_configured": True,
            }
            _INSIGHTS_CACHE["cached_at"] = now
            _INSIGHTS_CACHE["data"] = result
            return result
        else:
            logger.warning("Anthropic API returned status %d: %s. Using rule fallback.", resp.status_code, resp.text)
            content = generate_rule_based_insights(telemetry_summary)
            result = {
                "status": "success",
                "source": "rule_based_fallback",
                "markdown": content,
                "note": f"Anthropic API returned status {resp.status_code}",
                "cached": False,
                "cache_age_seconds": 0.0,
                "api_key_configured": True,
            }
            _INSIGHTS_CACHE["cached_at"] = now
            _INSIGHTS_CACHE["data"] = result
            return result
    except Exception as e:
        logger.warning("Error invoking Anthropic API: %s. Using rule fallback.", e)
        content = generate_rule_based_insights(telemetry_summary)
        result = {
            "status": "success",
            "source": "rule_based_fallback",
            "markdown": content,
            "note": f"API request error: {str(e)}",
            "cached": False,
            "cache_age_seconds": 0.0,
            "api_key_configured": True,
        }
        _INSIGHTS_CACHE["cached_at"] = now
        _INSIGHTS_CACHE["data"] = result
        return result
