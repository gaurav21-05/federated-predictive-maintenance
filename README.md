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

Execute the full suite of 29 unit and integration tests:
```bash
python -m unittest discover tests
```

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
