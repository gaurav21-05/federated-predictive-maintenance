"""
Industrial IoT Real-Data Service Layer.
Connects directly to ThingSpeak live hardware telemetry and local feeds.csv fallback.
Stateless and safe for Vercel Serverless (zero PyTorch dependencies).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
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


class LiveDashboardService:
    """
    Stateless service coordinating physical ThingSpeak telemetry retrieval,
    signal analysis, dynamic twin state synchronization, and RUL estimation.
    """

    def __init__(self, adapter: Optional[ThingSpeakAdapter] = None):
        self.adapter = adapter or ThingSpeakAdapter()

    def get_hardware_telemetry(
        self, results: int = 150, window: str = "all"
    ) -> Dict[str, Any]:
        """Fetch, normalize, and analyze live edge device telemetry."""
        raw_feeds, channel_info, source = self.adapter.fetch_raw_feeds(results=results)
        norm_res = self.adapter.normalize(raw_feeds)
        samples = norm_res["samples"]
        health = norm_res["health"]

        if not samples:
            return {
                "status": "empty",
                "source": source,
                "channel_id": self.adapter.channel_id,
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

        # Window filtering
        if window in ("1h", "24h") and len(samples) > 1:
            from datetime import datetime
            last_dt = datetime.fromisoformat(samples[-1]["timestamp"].replace("Z", "+00:00"))
            cutoff_seconds = 3600.0 if window == "1h" else 86400.0
            filtered = [
                s for s in samples
                if (last_dt - datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))).total_seconds() <= cutoff_seconds
            ]
            if len(filtered) >= 5:
                samples = filtered

        # Statistical analysis
        baseline = compute_robust_baseline(samples)
        stats = compute_descriptive_stats(samples)

        # Anomaly detection & clustering
        anom_res = detect_anomalies(samples, baseline)
        anomalies = anom_res["anomalies"]
        reconciliation = anom_res["reconciliation"]
        episodes = detect_anomaly_episodes(samples, anomalies)

        # Trends & correlations
        trends = analyze_trends(samples)
        correlations = compute_correlations(samples)

        # Data quality report
        data_quality = generate_data_quality_report(samples, health["gaps"])

        # Health series
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

        # RUL estimation
        rul = estimate_rul(samples, baseline)

        return {
            "status": "success",
            "source": source,
            "channel_id": self.adapter.channel_id,
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


# Alias for backward compatibility
DashboardService = LiveDashboardService

