"""
ThingSpeak Ingestion & Normalization Layer for ESP32 Live Hardware Telemetry.
Provides robust fetching with exponential backoff, 10s caching, physical bounds validation,
connection health tracking, and fallback support for offline/demo operation.

Stateless and safe for Vercel Serverless (NO PyTorch or heavy dependencies).
"""

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("thingspeak_adapter")

# Safe project root resolution without heavy imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Attempt to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# Fallback lightweight manual .env loader
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

# ============================================================================
# Field Mapping Configuration (Single Editable Dictionary)
# ============================================================================
FIELD_MAPPING = {
    "field1": {
        "key": "temperature_c",
        "label": "Ambient Temperature",
        "unit": "°C",
        "sensor": "DHT11",
        "confirmed": True,
        "type": float,
        "valid_range": (-20.0, 80.0),
    },
    "field2": {
        "key": "humidity_pct",
        "label": "Relative Humidity",
        "unit": "%",
        "sensor": "DHT11",
        "confirmed": True,
        "type": float,
        "valid_range": (0.0, 100.0),
    },
    "field3": {
        "key": "temp_secondary_c",
        "label": "Field 3 (temp 2 - unconfirmed)",
        "unit": "°C",
        "sensor": "Aux Sensor",
        "confirmed": False,
        "type": float,
        "valid_range": (-20.0, 80.0),
    },
    "field4": {
        "key": "pressure_hpa",
        "label": "Field 4 (pressure? - unconfirmed)",
        "unit": "hPa",
        "sensor": "Barometer",
        "confirmed": False,
        "type": float,
        "valid_range": (800.0, 1200.0),
    },
    "field5": {
        "key": "vibration_g",
        "label": "Motor Dynamic Vibration RMS",
        "unit": "g",
        "sensor": "MPU6050",
        "confirmed": True,
        "type": float,
        "valid_range": (0.0, 20.0),
    },
    "field6": {
        "key": "current_a",
        "label": "Motor Current",
        "unit": "A",
        "sensor": "ACS712",
        "confirmed": True,
        "type": float,
        "valid_range": (0.0, 30.0),
    },
    "field7": {
        "key": "esp32_anomaly",
        "label": "ESP32 Anomaly Flag",
        "unit": "flag",
        "sensor": "ESP32",
        "confirmed": True,
        "type": int,
        "valid_range": (0, 1),
    },
    "field8": {
        "key": "esp32_alert",
        "label": "ESP32 Buzzer/LED Alert",
        "unit": "flag",
        "sensor": "ESP32",
        "confirmed": True,
        "type": int,
        "valid_range": (0, 1),
    },
}

# ============================================================================
# Cache State
# ============================================================================
_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "results_key": None,
    "data": None,
    "last_successful_payload": None,
}
CACHE_TTL_SECONDS = 10.0


def _parse_numeric(val: Any, target_type: type = float) -> Optional[float]:
    """Parse raw value to float/int, handling NA, null, empty strings safely."""
    if val is None:
        return None
    s = str(val).strip().upper()
    if s in ("", "NA", "NAN", "NULL", "NONE"):
        return None
    try:
        if target_type is int:
            return int(float(s))
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_timestamp(ts_str: Any) -> Optional[datetime]:
    """Parse ISO-8601 UTC timestamp string to datetime object."""
    if not ts_str:
        return None
    s = str(ts_str).strip()
    if not s:
        return None
    try:
        # Replace Z with +00:00 for fromisoformat compatibility
        clean = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class ThingSpeakAdapter:
    """
    Connects to ThingSpeak channel or local CSV fallback to retrieve and normalize
    raw telemetry feeds.
    """

    DEFAULT_CHANNEL_ID = "3591448"
    NOMINAL_SAMPLE_INTERVAL_SECONDS = 20.0

    def __init__(
        self,
        channel_id: Optional[str] = None,
        read_key: Optional[str] = None,
        fallback_csv_path: Optional[Path] = None,
        timeout: float = 8.0,
    ):
        self.channel_id = channel_id or os.environ.get("TS_CHANNEL_ID", self.DEFAULT_CHANNEL_ID)
        self.read_key = read_key or os.environ.get("TS_READ_KEY", "")
        
        # Determine fallback CSV
        env_fallback = os.environ.get("LIVE_DATA_FALLBACK_CSV")
        if env_fallback:
            self.fallback_csv_path = Path(env_fallback)
        elif fallback_csv_path:
            self.fallback_csv_path = Path(fallback_csv_path)
        else:
            default_path = PROJECT_ROOT / "feeds.csv"
            self.fallback_csv_path = default_path if default_path.exists() else None

        self.timeout = timeout

    def fetch_raw_feeds(self, results: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """
        Fetch feeds from ThingSpeak with 10s in-memory caching and exponential backoff retry.
        If credentials are unset or network fails, falls back to CSV fallback or cached data.
        
        Returns:
            (feeds_list, channel_metadata, source_str)
        """
        global _CACHE
        now = time.time()
        cache_key = f"{self.channel_id}_{results}"

        # 1. Check in-memory cache
        if (
            _CACHE["data"] is not None
            and _CACHE["results_key"] == cache_key
            and (now - _CACHE["timestamp"]) < CACHE_TTL_SECONDS
        ):
            return _CACHE["data"][0], _CACHE["data"][1], "CACHE_HIT"

        # 2. Check if we should use local CSV fallback
        read_key = self.read_key or os.environ.get("TS_READ_KEY", "")
        if not read_key:
            if self.fallback_csv_path and self.fallback_csv_path.exists():
                feeds, channel_info = self._load_from_csv(self.fallback_csv_path, results)
                _CACHE["timestamp"] = now
                _CACHE["results_key"] = cache_key
                _CACHE["data"] = (feeds, channel_info)
                _CACHE["last_successful_payload"] = (feeds, channel_info)
                return feeds, channel_info, "LOCAL_CSV_FALLBACK"
            raise RuntimeError(
                "ThingSpeak read key not configured (TS_READ_KEY is missing) and no fallback CSV found."
            )

        # 3. Network Fetch with exponential retry and 429 handling
        channel_id = self.channel_id or os.environ.get("TS_CHANNEL_ID", self.DEFAULT_CHANNEL_ID)
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
        params = {"api_key": read_key, "results": results}
        
        retries = 3
        backoff = 0.5
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                # Log without exposing API key
                logger.info(f"Fetching ThingSpeak feeds (attempt {attempt}/{retries}) for channel {self.channel_id}")
                resp = requests.get(url, params=params, timeout=self.timeout)
                
                if resp.status_code == 200:
                    payload = resp.json()
                    feeds = payload.get("feeds", [])
                    channel_info = payload.get("channel", {})
                    
                    # Update cache
                    _CACHE["timestamp"] = now
                    _CACHE["results_key"] = cache_key
                    _CACHE["data"] = (feeds, channel_info)
                    _CACHE["last_successful_payload"] = (feeds, channel_info)
                    return feeds, channel_info, "THINGSPEAK_LIVE"
                
                elif resp.status_code == 429:
                    # Rate limit encountered (15s rule)
                    logger.warning("ThingSpeak rate limit (429) hit. Backing off...")
                    time.sleep(backoff * 2)
                    backoff *= 2
                    continue
                else:
                    logger.warning(f"ThingSpeak responded with HTTP {resp.status_code}")
                    last_error = f"Upstream HTTP error {resp.status_code}"
            except requests.RequestException as e:
                logger.warning(f"Network error contacting ThingSpeak: {type(e).__name__}")
                last_error = f"Network connection failed: {type(e).__name__}"
            
            time.sleep(backoff)
            backoff *= 1.5

        # If fetch failed, serve last good cache if available
        if _CACHE["last_successful_payload"] is not None:
            logger.warning("Serving stale cached ThingSpeak payload due to upstream failure.")
            feeds, channel_info = _CACHE["last_successful_payload"]
            return feeds, channel_info, "STALE_CACHE_FALLBACK"

        # Or serve local CSV fallback if available
        if self.fallback_csv_path and self.fallback_csv_path.exists():
            logger.warning("Serving local CSV fallback after network failure.")
            feeds, channel_info = self._load_from_csv(self.fallback_csv_path, results)
            return feeds, channel_info, "LOCAL_CSV_FALLBACK"

        raise RuntimeError(
            f"Unable to reach ThingSpeak API after {retries} retries: {last_error}"
        )

    def _load_from_csv(self, csv_path: Path, results: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load feeds from local CSV file (e.g. feeds.csv)."""
        feeds = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feeds.append(row)
        
        # Limit to the most recent 'results' entries
        if len(feeds) > results:
            feeds = feeds[-results:]

        channel_info = {
            "id": self.channel_id,
            "name": "Industrial IoT Bench Twin (ESP32)",
            "description": "DHT11 + MPU6050 + ACS712 edge predictive maintenance telemetry",
            "field1": "temperature_c",
            "field2": "humidity_pct",
            "field3": "temp_secondary_c",
            "field4": "pressure_hpa",
            "field5": "vibration_g",
            "field6": "current_a",
            "field7": "esp32_anomaly",
            "field8": "esp32_alert",
            "latitude": "0.0",
            "longitude": "0.0",
            "elevation": "0.0",
        }
        return feeds, channel_info

    def normalize(
        self,
        raw_feeds: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Clean, sort, type-cast, range-validate, and compute connection health metrics.
        """
        cleaned_samples: List[Dict[str, Any]] = []
        seen_entry_ids = set()

        for raw in raw_feeds:
            entry_id_raw = raw.get("entry_id")
            entry_id = _parse_numeric(entry_id_raw, target_type=int)
            if entry_id is not None and entry_id in seen_entry_ids:
                continue
            if entry_id is not None:
                seen_entry_ids.add(entry_id)

            ts_raw = raw.get("created_at") or raw.get("timestamp")
            dt = parse_timestamp(ts_raw)
            if dt is None:
                continue

            sample: Dict[str, Any] = {
                "entry_id": entry_id,
                "timestamp": dt.isoformat(),
                "created_at": dt.isoformat(),
                "_datetime": dt,
                "status": raw.get("status") or "",
                "out_of_range": [],
            }

            # Map fields
            for field_key, meta in FIELD_MAPPING.items():
                name = meta["key"]
                raw_val = raw.get(field_key)
                val = _parse_numeric(raw_val, target_type=meta["type"])
                
                # Physical range validation
                if val is not None and "valid_range" in meta:
                    low, high = meta["valid_range"]
                    if not (low <= val <= high):
                        sample["out_of_range"].append(name)
                
                sample[name] = val

            cleaned_samples.append(sample)

        # Sort chronologically
        cleaned_samples.sort(key=lambda s: s["_datetime"])

        # Health & Gap Metrics
        health = self._compute_health_and_gaps(cleaned_samples, reference_time)

        # Remove internal datetime objects for clean serialization
        for s in cleaned_samples:
            s.pop("_datetime", None)

        return {
            "samples": cleaned_samples,
            "count": len(cleaned_samples),
            "health": health,
        }

    def _compute_health_and_gaps(
        self,
        samples: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Compute online/offline status, gaps, and data quality score."""
        if not samples:
            return {
                "online": False,
                "age_seconds": None,
                "sample_interval_seconds": self.NOMINAL_SAMPLE_INTERVAL_SECONDS,
                "gaps": [],
                "quality_score": 0.0,
                "total_samples": 0,
                "latest_timestamp": None,
            }

        latest_dt = samples[-1]["_datetime"]
        now_dt = reference_time or datetime.now(timezone.utc)
        age_seconds = max(0.0, (now_dt - latest_dt).total_seconds())

        # Determine median sample interval
        intervals = []
        for i in range(1, len(samples)):
            diff = (samples[i]["_datetime"] - samples[i - 1]["_datetime"]).total_seconds()
            if diff > 0:
                intervals.append(diff)
        
        sample_interval = float(sorted(intervals)[len(intervals) // 2]) if intervals else self.NOMINAL_SAMPLE_INTERVAL_SECONDS

        # Offline rule: newest sample older than 3x the sample interval
        # Note: If running against historical CSV where age > 1 day, offline is True relative to now
        offline_threshold = max(60.0, 3.0 * sample_interval)
        online = age_seconds <= offline_threshold

        # Gap detection
        gaps = []
        for i in range(1, len(samples)):
            prev_s = samples[i - 1]
            curr_s = samples[i]
            
            dt_diff = (curr_s["_datetime"] - prev_s["_datetime"]).total_seconds()
            
            entry_gap = 0
            if curr_s.get("entry_id") and prev_s.get("entry_id"):
                entry_gap = curr_s["entry_id"] - prev_s["entry_id"] - 1

            if dt_diff > (sample_interval * 1.75) or entry_gap > 0:
                gaps.append({
                    "from_entry": prev_s.get("entry_id"),
                    "to_entry": curr_s.get("entry_id"),
                    "missing_entries": max(0, entry_gap),
                    "from_time": prev_s["timestamp"],
                    "to_time": curr_s["timestamp"],
                    "duration_seconds": round(dt_diff, 1),
                })

        # Data quality score calculation (0 - 100)
        # Factors:
        # 1. Essential field completeness (temp, humidity, vibration): 50 pts
        # 2. Out of range penalty: 20 pts
        # 3. Gap penalty: 30 pts
        total = len(samples)
        essential_missing = 0
        out_of_range_count = 0
        for s in samples:
            for k in ("temperature_c", "humidity_pct", "vibration_g"):
                if s.get(k) is None:
                    essential_missing += 1
            out_of_range_count += len(s.get("out_of_range", []))

        essential_ratio = 1.0 - (essential_missing / max(1, total * 3))
        range_ratio = 1.0 - min(1.0, (out_of_range_count / max(1, total)))
        gap_penalty = min(30.0, len(gaps) * 5.0)
        
        quality_score = max(0.0, round((essential_ratio * 50.0) + (range_ratio * 20.0) + (30.0 - gap_penalty), 1))

        return {
            "online": online,
            "age_seconds": round(age_seconds, 1),
            "sample_interval_seconds": round(sample_interval, 1),
            "offline_threshold_seconds": round(offline_threshold, 1),
            "gaps": gaps,
            "gap_count": len(gaps),
            "quality_score": quality_score,
            "total_samples": total,
            "latest_timestamp": latest_dt.isoformat(),
        }

