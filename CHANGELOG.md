# Changelog

All notable changes to the **Industrial IoT Federated Predictive Maintenance & Digital Twin** platform are documented here.

## [Unreleased] - 2026-11-25

### Added
- **ThingSpeak Live Ingestion Adapter (`src/dashboard/thingspeak.py`)**:
  - Configurable 8-field telemetry ingestion contract matching private channel `3591448`.
  - In-memory 10-second server cache to respect ThingSpeak API limits.
  - Exponential backoff retry logic (with HTTP 429 rate-limit handling).
  - Physical parameter bounds validation with non-destructive `out_of_range` flagging.
  - Offline connection detection (offline when newest sample is older than $3\times$ the sampling interval).
  - Graceful fallback to `feeds.csv` via `LIVE_DATA_FALLBACK_CSV` when credentials are unset.
  - Safe error handling: returns HTTP 502 with sanitized message, never leaking API keys or secrets in logs or responses.
- **Live Hardware Analytics Engine (`src/dashboard/live_analytics.py`)**:
  - Descriptive statistics (mean, std, min, max, p25, p50, p75, p95, hourly rate of change).
  - Robust Median and MAD (Median Absolute Deviation $\times 1.4826$) baseline computed from normal edge states.
  - Tri-method anomaly detection: Robust Z-Score ($|z| > 3.0$), EWMA control chart, and CUSUM drift detection.
  - Hardware reconciliation: correlates engine anomalies against ESP32 firmware alert/anomaly flags.
  - Contiguous anomaly episode clustering (with duration, peak vibration, severity, and buzzer trigger tracking).
  - Linear and EWMA trend slope analysis with two-tailed Student's t p-value significance testing.
  - Bivariate sensor correlation analysis (Pearson $r$ and Spearman $\rho$).
  - Data quality diagnostics: missingness rate, communication gaps, sensor flatline detection, and DHT11 quantisation notes.
- **Hardware-Specific Digital Twin Health Index & RUL Model (`src/dashboard/live_rul.py`)**:
  - Calibrated for 0.02g benchtop rotating equipment rather than heavy plant machinery.
  - Dynamic subsystem health breakdown (Vibration, Thermal, Environmental, and Electrical).
  - Automatic graceful activation: when ACS712 current (`field6`) arrives, electrical subsystem activates automatically without code changes.
  - Statistically honest RUL forecasting: compares Linear and Exponential wear models using Residual Sum of Squares (AIC equivalent).
  - 1D median filter smoothing to protect RUL from false collapse during single-sample transient spikes (entries 5 and 40).
  - Honesty rules: returns `STABLE_NO_DEGRADATION` with `null` RUL when trend is flat; never fabricates numbers.
  - 95% confidence interval estimation, cumulative operating hours counter (ignoring offline gaps > 5 min), and rule-based prescriptive actions.
- **FastAPI Endpoints (`src/dashboard/app.py`)**:
  - `GET /api/live/hardware`: Full analysed telemetry stream, digital twin status, and RUL prediction.
  - `GET /api/live/rul`: Lightweight RUL and subsystem snapshot.
  - `GET /api/live/export.csv`: Downloadable CSV export of analysed telemetry records.
  - Idempotent prescriptive maintenance work order creation when health dips below warning threshold.
- **Live Hardware Twin Frontend (`src/dashboard/static/` and `public/`)**:
  - New "Live Hardware Twin" navigation tab with pulsing `LIVE` badge.
  - Clear `LIVE (real hardware)` vs `SIMULATED` badge distinction across all dashboard tabs.
  - Interactive SVG Digital Twin diagram featuring ESP32 DevKit, MPU6050 with vibration shake animation, DHT11, ACS712, N-MOSFET, DC motor (color-coded to Health Index), buzzer sound waves, and alert LEDs.
  - Click-to-inspect component drawer showing live readings, role, and health score.
  - Connection status badge (LIVE / OFFLINE with age) and watermark overlay when device is offline.
  - 6 KPI cards with sparklines, min/max, and trend indicators; unconfirmed fields clearly marked.
  - 4 Canvas charts: Machine Health Index over time, RUL degradation extrapolation, vibration RMS with $\pm 3\sigma$ band and anomaly markers, and dual-axis temperature & humidity.
  - Descriptive statistics table and sortable anomaly episodes table.
  - Visibility-aware polling: automatically pauses when the browser tab is hidden to save bandwidth and prevent rate limiting.
  - Strict file synchronization: identical copies maintained in `src/dashboard/static/` and `public/`.
- **Firmware & Documentation**:
  - `hardware/esp32_firmware/esp32_thingspeak.ino`: 20-second rate-limited sketch uploading to ThingSpeak with dynamic AC RMS vibration extraction.
  - `hardware/esp32_firmware/secrets.example.h`: Credentials template.
  - `.env.example`: Configuration reference for local and Vercel environments.
  - Unit tests: `tests/test_live_analytics.py` and `tests/test_live_rul.py` (total test suite expanded to 43 passing tests).

