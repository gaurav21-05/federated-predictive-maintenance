# Industrial IoT Federated Predictive Maintenance & Digital Twin

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)

A privacy-preserving, physics-grounded federated predictive maintenance and digital twin synchronization system simulating heterogeneous, non-IID operational edge nodes across industrial manufacturing plants.

---

## 🌟 Key Features

1. **Non-IID Industrial IoT Sensor Generation**:
   - Physics-grounded degradation simulation for 3 distinct machines:
     - **Client 1**: 5-Axis High-Speed CNC Milling Machine (`CNC_MILL_01`) - High RPM, thermal bearing friction failure mode.
     - **Client 2**: Heavy-Duty Hydraulic Stamping Press (`HYDR_PRESS_02`) - Cyclic shock vibration, extreme hydraulic pressure failure mode.
     - **Client 3**: Rotary Screw Air Compressor & Micro-Turbine (`TURB_COMP_03`) - High humidity, motor winding over-current failure mode.
2. **Federated Learning Framework**:
   - **FedAvg** & **FedProx** algorithms tailored for heterogeneous edge devices.
   - **Differential Privacy (DP-SGD)** with gradient clipping and Gaussian noise injection.
   - **Secure Aggregation (SecAgg)** simulation with simulated Diffie-Hellman secret masking.
   - Multi-architecture models: 1D-CNN + LSTM and Edge-optimized MLP.
3. **Physics-Grounded Digital Twin**:
   - Real-time Machine Health Index (MHI) and subsystem breakdown (vibration, thermal, electrical, pressure).
   - Remaining Useful Life (RUL) estimation with confidence intervals.
   - "What-If" operational stress scenario testing (overclocking, thermal spikes, load spikes).
4. **Prescriptive Maintenance & Dispatch**:
   - Automated work order generation and lifecycle tracking (OPEN, ASSIGNED, RESOLVED).
5. **Interactive Web Dashboard**:
   - Built with FastAPI + modern responsive glassmorphism UI.
   - Real-time telemetry streaming, interactive federated training control room, and sandbox simulation.

---

## 📁 Repository Structure

```
federated-predictive-maintenance/
├── data/                               # Sensor datasets & digital twin states
│   ├── client_1/                       # CNC Milling Spindle telemetry (12,000 records)
│   ├── client_2/                       # Hydraulic Stamping Press telemetry (12,000 records)
│   ├── client_3/                       # Compressor & Turbine telemetry (12,000 records)
│   ├── combined_industrial_sensors.csv # Aggregated benchmark dataset (36,000 records)
│   └── dataset_summary.md              # Statistical profile
├── docs/                               # Documentation, PRD & figures
│   ├── PRD.md                          # Product Requirements Document
│   ├── benchmark_report.md             # Benchmark analysis report
│   └── figures/                        # Visual convergence & pareto plots
├── models/
│   └── checkpoints/                    # Saved local & global PyTorch weights
├── scripts/
│   ├── start_dashboard.py              # CLI launcher for web application
│   ├── run_generator.py                # Synthetic dataset generation CLI
│   ├── run_federated.py                # Federated training runner
│   ├── run_digital_twin_sync.py        # Digital twin synchronization runner
│   ├── run_benchmarks.py               # Comprehensive evaluation suite
│   └── analyze_dataset.py              # Telemetry inspection CLI
├── src/
│   ├── config.py                       # Physics baselines & factory profiles
│   ├── physics_engine.py               # Degradation physics modeling
│   ├── generator.py                    # Sensor data generator orchestrator
│   ├── models.py                       # PyTorch CNN-LSTM & MLP neural architectures
│   ├── trainer.py                      # Local & federated trainers
│   ├── dashboard/                      # FastAPI application & static web UI
│   ├── digital_twin/                   # Digital twin, RUL, what-if engines
│   ├── federated/                      # Client, server, FedAvg, FedProx implementations
│   └── privacy/                        # DP-SGD & Secure Aggregation engines
├── tests/                              # Unit & integration test suite (29 tests)
└── requirements.txt                    # Project dependencies
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install required dependencies:
```bash
git clone https://github.com/gaurav21-05/federated-predictive-maintenance.git
cd federated-predictive-maintenance
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard

Launch the FastAPI dashboard server:
```bash
python scripts/start_dashboard.py --host 127.0.0.1 --port 8000
```
Then open your browser at **[http://localhost:8000](http://localhost:8000)** (API documentation available at **[http://localhost:8000/docs](http://localhost:8000/docs)**).

### 3. Generate or Regenerate Sensor Data

To regenerate synthetic sensor data (12,000 records per client):
```bash
python scripts/run_generator.py --records-per-client 12000 --seed 42
```

### 4. Run Federated Learning

Train the global predictive maintenance model across all 3 simulated factories:
```bash
python scripts/run_federated.py --strategy FedProx --rounds 5 --epochs 2 --enable-dp
```

### 5. Run Digital Twin Synchronization

Run the continuous telemetry stream ingestion and digital twin synchronization:
```bash
python scripts/run_digital_twin_sync.py --steps 500
```

### 6. Run Automated Tests

Execute the full suite of unit and integration tests (43 tests):
```bash
python -m unittest discover tests
```

---

## 🛰️ Physical Hardware Edge Ingestion & Live Digital Twin

In addition to simulated industrial machines, the platform connects to real physical IoT edge hardware (ESP32 DevKit V1 + MPU6050 + DHT11 + ACS712) via ThingSpeak cloud ingestion.

### Architecture

```
  +-----------------------------------------------------------+
  |                   Physical Bench Hardware                 |
  |  ESP32 DevKit V1 (240MHz) + MPU6050 + DHT11 + ACS712      |
  +-----------------------------┬-----------------------------+
                                │ Wi-Fi 802.11 b/g/n (~20s interval)
                                ▼
  +-----------------------------------------------------------+
  |              ThingSpeak IoT Cloud (Private)               |
  |  Channel ID: 3591448                                      |
  +-----------------------------┬-----------------------------+
                                │ REST JSON (10s in-memory cache + backoff)
                                ▼
  +-----------------------------------------------------------+
  |             FastAPI Backend Gateway Layer                 |
  |  - /api/live/hardware (Full Telemetry, Baseline & Anomaly) |
  |  - /api/live/rul      (Lightweight RUL + Subsystems)      |
  |  - /api/live/export.csv (Analysed CSV Download)           |
  |  - Zero PyTorch dependencies in live path (Vercel-safe)   |
  +-----------------------------┬-----------------------------+
                                │ Visibility-aware polling
                                ▼
  +-----------------------------------------------------------+
  |             Interactive Web Dashboard (SPA)               |
  |  - Live Hardware Twin Tab (Real ESP32 Bench Node)         |
  |  - Interactive SVG Digital Twin (Click to inspect)        |
  |  - Tri-Method Anomaly Detection & Episode Clustering       |
  |  - Statistically Honest RUL Degradation Trajectory        |
  +-----------------------------------------------------------+
```

### Environment Variables

Configure environment variables in `.env` (or in Vercel Project Settings for cloud deployment):

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :---: |
| `TS_CHANNEL_ID` | ThingSpeak Private Channel ID | `3591448` | Yes |
| `TS_READ_KEY` | ThingSpeak Channel Read API Key | `""` | Yes (for live cloud) |
| `LIVE_DATA_FALLBACK_CSV` | Path to local CSV file when offline or keys unset | `./feeds.csv` | No |

### ThingSpeak Field Mapping

| Channel Field | Key | Hardware Sensor | Status | Typical Range | Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `field1` | `temperature_c` | DHT11 | **CONFIRMED** | 21 – 25 °C | Ambient Temperature |
| `field2` | `humidity_pct` | DHT11 | **CONFIRMED** | 51 – 78 % | Relative Air Humidity |
| `field3` | `temp_secondary_c` | Aux / Internal | *UNCONFIRMED* | ~23 – 25 °C | Secondary temperature channel |
| `field4` | `pressure_hpa` | Barometer / Seal | *UNCONFIRMED* | ~993 – 995 hPa | Barometric pressure baseline |
| `field5` | `vibration_g` | MPU6050 | **CONFIRMED** | 0.018 – 0.615 g | Dynamic AC RMS (gravity removed) |
| `field6` | `current_a` | ACS712 | **RESERVED** | NA (or 0 – 5 A) | Motor current (activates auto) |
| `field7` | `esp32_anomaly` | ESP32 Firmware | **CONFIRMED** | 0 or 1 | Edge threshold anomaly flag |
| `field8` | `esp32_alert` | ESP32 Firmware | **CONFIRMED** | 0 or 1 | Buzzer and alert LED tripped |
| `status` | `status_text` | ESP32 Firmware | **CONFIRMED** | Text | e.g. "MACHINE NORMAL" |

### How RUL is Computed & Honesty Rules

1. **Hardware Calibration**: The live RUL model is calibrated specifically for benchtop rotary hardware (0.02g baseline vibration) rather than heavy 1–2 mm/s industrial plant equipment.
2. **Subsystem Scoring**: Individual health indices are derived from robust z-scores:
   $$z = \frac{x - \text{median}}{1.4826 \cdot \text{MAD} + \epsilon}$$
3. **Transient Spike Resilience**: Telemetry undergoes 1D median filtering before trend fitting so single-sample anomalous spikes (such as entries 5 and 40 in `feeds.csv`) never collapse the RUL projection.
4. **Model Selection**: Compares Linear ($HI(t) = m \cdot t + b$) and Exponential ($HI(t) = HI_0 e^{-\lambda t}$) wear models via Residual Sum of Squares (AIC equivalent) and extrapolates to failure ($HI = 30\%$) and warning ($HI = 60\%$) thresholds.
5. **Honesty Rules**: If fewer than 30 samples are available, or if the health index slope is not significantly negative ($m \ge -0.01\%/\text{hr}$), the engine returns `rul_status = "STABLE_NO_DEGRADATION"` or `"INSUFFICIENT_DATA"` with `rul = null`. **No made-up numbers are ever presented.**
6. **Operating Hours Counter**: Sums real timestamp deltas while excluding idle or offline intervals longer than 5 minutes.

---

## 📊 Benchmark Summary

| Strategy | Accuracy | Macro F1 | Privacy Guarantee | Communication Overhead |
| :--- | :---: | :---: | :---: | :---: |
| **Centralized (Upper Bound)** | 98.4% | 0.962 | None (Data Pooled) | High |
| **Isolated Local (Client 1)** | 94.1% | 0.895 | Local Only | None |
| **FedAvg** | 96.8% | 0.941 | Private Raw Data | Medium |
| **FedProx ($\mu=0.01$)** | 97.2% | 0.948 | Private Raw Data | Medium |
| **FedProx + DP ($\varepsilon \approx 3.2$)**| 95.6% | 0.927 | $(\varepsilon, \delta)$-DP Guaranteed | Medium |

---

## 📄 License

This project is licensed under the MIT License.
