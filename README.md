# Real Hardware Industrial IoT Federated Predictive Maintenance & Digital Twin

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Edge Hardware](https://img.shields.io/badge/Hardware-ESP32%20DevKit%20V1-orange.svg)](https://www.espressif.com/)

A real-hardware predictive maintenance and edge intelligence platform connected directly to physical IoT hardware (ESP32 DevKit V1, MPU6050 accelerometer, DHT11 temperature/humidity probe, and 12V DC motor) publishing telemetry to ThingSpeak. Features pure-NumPy unsupervised edge anomaly detection, honest Remaining Useful Life (RUL) forecasting, time-partitioned federated learning, and automated AI diagnostics insights powered by Anthropic Claude.

---

## 🌟 Key Capabilities

1. **Real Hardware Telemetry Ingestion**:
   - Live ingestion from ThingSpeak Channel `3591448` via robust serverless-safe adapter (`src/dashboard/thingspeak.py`).
   - 10-second in-memory caching to respect ThingSpeak's 15s rate limits; automatic offline detection if telemetry age exceeds 60 seconds (3x sampling interval).
   - Validates physical bounds, normalizes data types, detects gaps, and flags anomalies.

2. **Interactive Live Monitor & SVG Digital Twin**:
   - Single responsive dashboard with 8 real sensor KPI cards:
     - Field 1: Temperature (°C, DHT11)
     - Field 2: Relative Humidity (%, DHT11)
     - Field 3: Secondary Temperature (°C, tracks field 1)
     - Field 4: Pressure (hPa, atmospheric)
     - Field 5: Vibration RMS (g, MPU6050 dynamic acceleration)
     - Field 6: Motor Current (A, ACS712 reserved channel, auto-activates when data arrives)
     - Field 7: ESP32 Hardware Anomaly Flag
     - Field 8: ESP32 Buzzer/Alarm Flag
   - Interactive SVG schematic of the physical hardware testbed (ESP32, MPU6050, DHT11, N-MOSFET, DC motor, LEDs, buzzer) dynamically reflecting real-time state.
   - Five HTML5 Canvas charts: Vibration RMS with $\pm 3\sigma$ envelope, dual-axis thermal/humidity, unconfirmed channels, motor current, and composite Health Index (0-100) with warning and failure thresholds.

3. **Analysis & Prediction Engine**:
   - **Descriptive Statistics**: Mean, standard deviation, median, Median Absolute Deviation (MAD), percentiles, and hourly rate of change.
   - **Unsupervised ML Anomaly Detection**: 4-layer symmetric bottleneck Autoencoder (5 -> 4 -> 2 -> 4 -> 5) reconstruction error + scikit-learn Isolation Forest scoring calibrated on normal hardware operating states.
   - **Honest RUL Estimation**: Trend extrapolation toward physical limits (0.25 g vibration threshold) with 90% confidence intervals. When the machine operates stably with negligible degradation slope, it displays `STABLE - no degradation trend detected` (> 1000 h) rather than artificial failure numbers.
   - **AI Diagnostics Summary**: Endpoint `POST /api/analysis/insights` compiling sensor statistics, trend slopes, and health scores into an engineering assessment using the Anthropic Claude API (with a 5-minute cache and deterministic rule-based fallback).

4. **Federated Learning on Real Telemetry**:
   - Real telemetry is partitioned into **3 Virtual Time Nodes** (Session A: Early, Session B: Mid, Session C: Late) to demonstrate decentralized training on genuine physical distributions.
   - Comparative evaluation across Centralized, FedAvg, FedProx, and DP-FedAvg (Differential Privacy with gradient clipping and Gaussian noise).
   - In-memory fine-tuning button ("Retrain on Latest Data") running pure-NumPy gradient aggregation without serverless overhead.

5. **Serverless-Optimized Architecture**:
   - Zero PyTorch dependencies in the production runtime path (`requirements.txt`).
   - Offline training script (`scripts/train_federated_live.py`) exports compact weights (`autoencoder_weights.npz`) and metadata to `models/live/` (< 500 KB total).
   - Pure-NumPy inference executes in sub-millisecond time on Vercel Serverless.

---

## 📁 Repository Structure

```
federated-predictive-maintenance/
├── .env.example                         # Environment configuration template
├── feeds.csv                            # Offline fallback and test telemetry (65 samples)
├── hardware/
│   └── esp32_firmware/
│       ├── esp32_thingspeak.ino         # Arduino C++ firmware for ESP32 DevKit V1
│       └── secrets.example.h            # Wi-Fi & ThingSpeak credentials template
├── models/
│   └── live/                            # Lightweight exported ML artefacts
│       ├── autoencoder_weights.npz      # Pure-NumPy Autoencoder weights
│       ├── isolation_forest.joblib      # Scikit-learn Isolation Forest model
│       └── model_metadata.json          # Model card, scaling, and FL benchmarks
├── public/                              # Vercel-served single-source static frontend
│   ├── index.html                       # Responsive 2-page UI (Live Monitor & Analysis)
│   ├── app.js                           # Reactive dashboard controller & canvas renderers
│   └── style.css                        # Design system & dark/light theme styles
├── scripts/
│   ├── start_dashboard.py               # Local FastAPI development launcher
│   └── train_federated_live.py          # Standalone offline PyTorch federated trainer
├── src/
│   ├── config.py                        # Hardware node profiles & field mappings
│   ├── dashboard/
│   │   ├── app.py                       # FastAPI application & REST endpoints
│   │   ├── live_analytics.py            # Descriptive stats, robust MAD, EWMA, CUSUM
│   │   ├── live_insights.py             # Claude API insights & deterministic fallback
│   │   ├── live_ml.py                   # Pure-NumPy ML inference & honest RUL
│   │   ├── live_rul.py                  # Physical subsystem health & RUL calculation
│   │   ├── service.py                   # Unified telemetry service layer
│   │   ├── thingspeak.py                # ThingSpeak HTTP ingestion & caching adapter
│   │   └── static/                      # Mirrored copy of public/
│   ├── federated/                       # Client, server, FedAvg, FedProx implementations
│   └── privacy/                         # DP-SGD & Secure Aggregation engines
├── tests/                               # Comprehensive unit test suite (40 tests)
├── requirements.txt                     # Production runtime dependencies (NO torch)
└── requirements-train.txt               # Offline training dependencies (with torch)
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install runtime dependencies:
```bash
git clone https://github.com/gaurav21-05/federated-predictive-maintenance.git
cd federated-predictive-maintenance
pip install -r requirements.txt
```

*(Optional) For offline ML training, install training dependencies:*
```bash
pip install -r requirements-train.txt
```

### 2. Configure Credentials

Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Fill in your credentials:
```env
TS_CHANNEL_ID=3591448
TS_READ_KEY=UP0CR69D9NOTNYPN
LIVE_DATA_FALLBACK_CSV=./feeds.csv
# Optional: Set your Anthropic API key for AI diagnostic insights
ANTHROPIC_API_KEY=
```

### 3. Launch Web Dashboard

Run the server locally:
```bash
python scripts/start_dashboard.py --host 127.0.0.1 --port 8000
```
Open your browser at **[http://localhost:8000](http://localhost:8000)** (API docs at **[http://localhost:8000/docs](http://localhost:8000/docs)**).

### 4. Run Unit Tests

Execute the full unit test suite:
```bash
python -m unittest discover tests
```

---

## 📊 Physical Field Mapping

| Channel | Sensor / Key | Unit | Description | Status |
|---|---|---|---|---|
| `field1` | `temperature_c` | °C | DHT11 Ambient Temperature (~21–25°C) | Confirmed |
| `field2` | `humidity_pct` | % | DHT11 Relative Humidity (~51–78%) | Confirmed |
| `field3` | `temp_secondary_c` | °C | Secondary Temperature Probe (tracks field1) | Unconfirmed |
| `field4` | `pressure_hpa` | hPa | Atmospheric Pressure (~993–995 hPa) | Unconfirmed |
| `field5` | `vibration_g` | g | MPU6050 Dynamic RMS (baseline ~0.02g, spikes up to 0.6g) | Confirmed |
| `field6` | `current_a` | A | ACS712 Motor Current Shunt (reserved) | Auto-Activates |
| `field7` | `esp32_anomaly` | flag | Edge Firmware Threshold Anomaly Flag (0/1) | Confirmed |
| `field8` | `esp32_alert` | flag | Edge Firmware Buzzer / LED Alarm Trigger (0/1) | Confirmed |

---

## ⚖️ Assumptions & Limitations

- **Single Physical Device**: Telemetry originates from one physical ESP32 bench device. Federated learning is evaluated across time-based virtual sessions (Session A, B, C) explicitly documented as such.
- **Unsupervised Modeling**: Due to ~65 normal operating samples from physical hardware, supervised failure classification is avoided in favor of unsupervised reconstruction and density estimation.
- **Sampling Cadence**: ThingSpeak ~20-second telemetry updates are sufficient for thermal and vibration RMS trends, but high-frequency raw vibration waveforms (>25 Hz) are processed on-chip by the ESP32.
