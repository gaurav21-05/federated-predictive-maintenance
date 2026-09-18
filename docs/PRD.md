# Product Requirements Document (PRD)

## Project Title
**Federated Learning and Digital Twin Integration for Privacy-Preserving Predictive Maintenance in Industrial IoT**

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
In modern Industrial Internet of Things (IIoT) ecosystems, industrial machinery generates vast streams of mission-critical sensor telemetry. Machine failures lead to catastrophic downtime, substantial financial loss, and severe safety hazards. While machine learning enables predictive maintenance, two systemic bottlenecks prevent centralized cloud-based AI:
1. **Data Privacy & Intellectual Property**: Manufacturing telemetry contains proprietary production cycles, machine speeds, and trade secrets that factories refuse to share with external cloud providers or competitors.
2. **Data Heterogeneity (Non-IID)**: Different factories operate distinct machinery under diverse operating loads, climates, and mechanical stress profiles. Centralized models fail to generalize or trigger excessive false alarms.

### 1.2 Proposed Solution
This project establishes a unified architecture combining:
- **Federated Learning (FL)**: Enables distributed factories to collaboratively train global predictive maintenance models without exposing raw sensor data.
- **Digital Twin (DT)**: Creates a high-fidelity virtual representation of physical machines to track health index, simulate failure scenarios, estimate Remaining Useful Life (RUL), and recommend prescriptive maintenance.
- **Privacy-Preserving Mechanisms**: Incorportates Differential Privacy (DP) and Secure Aggregation to mathematically protect edge gradient updates from reverse-engineering.

---

## 2. System Architecture & High-Level Design

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       CENTRAL CLOUD / ORCHESTRATOR                     │
   │                                                                        │
   │  ┌───────────────────────┐              ┌───────────────────────────┐  │
   │  │ Federated Aggregator  │              │   Global Model Registry   │  │
   │  │ (FedAvg / FedProx)    │◄────────────►│ (PyTorch / ONNX Weights)  │  │
   │  └──────────▲────────────┘              └───────────────────────────┘  │
   └─────────────┼──────────────────────────────────────────────────────────┘
                 │ Secure Weight Updates (ΔW) + Differential Privacy Noise
   ┌─────────────┼───────────────────────────────────┬──────────────────────┐
   │             ▼                                   ▼                      │
   │   ┌───────────────────┐               ┌───────────────────┐            │
   │   │  Edge Client 1    │               │  Edge Client 2    │            │
   │   │ (CNC Milling)     │               │ (Hydraulic Press) │            │
   │   │ ┌───────────────┐ │               │ ┌───────────────┐ │            │
   │   │ │ Local Model   │ │               │ │ Local Model   │ │            │
   │   │ └───────▲───────┘ │               │ └───────▲───────┘ │            │
   │   │ ┌───────┴───────┐ │               │ ┌───────┴───────┐ │            │
   │   │ │ Digital Twin  │ │               │ │ Digital Twin  │ │            │
   │   │ │ State Engine  │ │               │ │ State Engine  │ │            │
   │   │ └───────▲───────┘ │               │ └───────▲───────┘ │            │
   │   │ ┌───────┴───────┐ │               │ ┌───────┴───────┐ │            │
   │   │ │ Private Tele- │ │               │ │ Private Tele- │ │            │
   │   │ │ metry Stream  │ │               │ │ metry Stream  │ │            │
   │   │ └───────────────┘ │               │ └───────────────┘ │            │
   │   └───────────────────┘               └───────────────────┘            │
   │                                                                        │
   │   ... (Extensible to N Distributed Industrial Factory Edge Nodes)      │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Phase Breakdown & Roadmap

### Phase 1: Synthetic Industrial IoT Telemetry & Edge Client Environment
- **Objective**: Construct a physics-grounded synthetic sensor dataset generator simulating non-IID factory distributions.
- **Key Features**:
  - Exact 9-feature schema: `timestamp`, `machine_id`, `temperature`, `humidity`, `vibration`, `current`, `rpm`, `pressure`, `failure`.
  - Realistic multi-state degradation dynamics (`NORMAL` $\rightarrow$ `WARNING` $\rightarrow$ `ABNORMAL`).
  - Autoregressive thermal and rotational inertia.
  - Multi-client non-IID partitions (`client_1`, `client_2`, `client_3`).
- **Deliverables**:
  - Modular dataset generation package (`src/config.py`, `src/physics_engine.py`, `src/generator.py`, `src/statistics.py`).
  - Generated datasets with $\ge 36,000$ verified records.
  - Test suite validating zero nulls, correct schema, and realistic failure rate (~2.27%).
- **Status**: :white_check_mark: **COMPLETED**

---

### Phase 2: Local Edge Predictive Maintenance Model Architecture
- **Objective**: Design and train baseline local machine learning and deep learning models at the edge for predictive maintenance.
- **Key Features**:
  - Feature engineering & sliding window sequence generator for time-series telemetry.
  - Temporal feature extraction (rolling mean, rolling RMS, peak-to-peak vibration, temperature gradient $\Delta T / \Delta t$).
  - Model Architectures:
    - **Baseline Classical Model**: Random Forest / LightGBM Classifier.
    - **Temporal Deep Learning Model**: 1D-CNN + Bi-LSTM / Temporal Convolutional Network (TCN) for failure probability and state classification (`Normal`, `Warning`, `Abnormal`).
  - Handling severe industrial class imbalance (~43:1) using Focal Loss, SMOTE, or class-weighted cross-entropy.
- **Deliverables**:
  - `src/preprocessing.py`: Feature scaler, time-series window segmenter.
  - `src/models.py`: 1D-CNN + Bi-LSTM and Edge MLP architectures.
  - `src/loss.py`: Binary Focal Loss resolving ~43:1 imbalance.
  - Local model training scripts and benchmark evaluation (Precision, Recall, F1-Score, ROC-AUC, PR-AUC).
- **Status**: :white_check_mark: **COMPLETED**

---

### Phase 3: Federated Learning Framework & Non-IID Aggregation
- **Objective**: Build the distributed Federated Learning network using Flower / native PyTorch FL coordinator.
- **Key Features**:
  - Client-Server Federated orchestration.
  - Support for **FedAvg** (McMahan et al.) and **FedProx** (Li et al.) with $\mu$ regularization to handle non-IID client drift.
  - Communication protocol transmitting only model weights ($\Delta W$), keeping raw data private on edge nodes.
  - Communication round tracking: Convergence rate, loss decay, communication bandwidth consumption.
- **Deliverables**:
  - `src/federated/server.py`: Central aggregation orchestrator.
  - `src/federated/client.py`: Edge client isolated execution node.
  - `src/federated/strategies.py`: FedAvg and FedProx aggregation engines.
  - `scripts/run_federated.py` & `scripts/benchmark_federated.py`.
- **Status**: :white_check_mark: **COMPLETED**

---

### Phase 4: Privacy Preservation & Security Hardening
- **Objective**: Ensure rigorous privacy guarantees against reconstruction, inversion, and membership inference attacks.
- **Key Features**:
  - **Differential Privacy (DP-SGD / Model Perturbation)**: $L_2$ gradient clipping ($C$) and calibrated Gaussian noise addition ($\sigma$) to guarantee $(\epsilon, \delta)$-Differential Privacy.
  - **Rényi DP (RDP) Moments Accountant**: Tracking cumulative privacy budget $(\epsilon, \delta)$ and analytical noise calibration.
  - **Secure Aggregation Simulation**: Cryptographic pairwise zero-sum random masking (Bonawitz et al. protocol).
  - **Privacy-Utility Trade-off Analysis**: Systematic benchmark evaluating the Pareto frontier ($\sigma \in [0.0, 0.5]$).
- **Deliverables**:
  - `src/privacy/dp_engine.py`: Differential privacy injection engine.
  - `src/privacy/accountant.py`: Rényi Differential Privacy moments accountant.
  - `src/privacy/secure_aggregation.py`: Pairwise zero-sum mask simulation.
  - `scripts/benchmark_privacy.py`: Automated Pareto frontier benchmark.
  - `tests/test_privacy.py`: 7/7 passing unit tests.
- **Status**: :white_check_mark: **COMPLETED**

---

### Phase 5: Digital Twin Synchronization & RUL Estimation
- **Objective**: Construct the Digital Twin engine that mirrors the operational state of physical machinery in real-time.
- **Key Features**:
  - High-fidelity virtual state tracker: Tracks physical health index, thermal dissipation rate, and bearing vibration spectrum.
  - Remaining Useful Life (RUL) regression model: Estimates hours/cycles remaining until catastrophic failure.
  - What-If Simulation Engine: Simulates operator stress scenarios (e.g. "+20% RPM overload") to evaluate machine lifespan impact.
  - Prescriptive Maintenance Alerts: Generates maintenance action items when operating conditions transition into `WARNING`.
- **Deliverables**:
  - `src/digital_twin/twin_engine.py`: Real-time machine state synchronizer.
  - `src/digital_twin/rul_estimator.py`: Remaining Useful Life regression engine.
  - `src/digital_twin/what_if_simulator.py`: Stress test & lifespan impact engine.
  - `src/digital_twin/maintenance_dispatcher.py`: Prescriptive maintenance work order generator.
  - `scripts/run_digital_twin_sync.py`: Live streaming synchronization orchestrator.
  - `data/digital_twin_state.json`: Synchronized system state snapshot.
  - `tests/test_digital_twin.py`: 10/10 passing unit tests.
- **Status**: :white_check_mark: **COMPLETED**

---

### Phase 6: Interactive Dashboard, Benchmarking & End-to-End Evaluation
- **Objective**: Deploy a web-based monitoring dashboard and complete comparative research evaluation.
- **Key Features**:
  - **Unified Web Dashboard** (FastAPI + Cybernetic HTML5/CSS3/Vanilla JS SPA):
    - Live factory telemetry monitoring across Client 1, Client 2, and Client 3.
    - Real-time visual health status meters and operational degradation tracking for each Digital Twin.
    - Federated training control room: live round-by-round accuracy, loss curves, and client contributions with FedProx/FedAvg, DP-SGD ($\sigma$), and SecAgg toggle.
    - What-If Stress sandbox for real-time operator scenario evaluation (+20% RPM, -20% RPM de-rate, high thermal ambient, pressure overload).
    - Prescriptive maintenance work order management: approve, schedule, complete, and inspect maintenance tickets.
    - Research benchmarks tab showcasing 4 publication-quality evaluation figures.
  - **Comprehensive Benchmark Suite**:
    - Centralized Model vs. Standalone Local Models vs. Federated Model (FedAvg vs. FedProx).
    - Latency, communication overhead, privacy budget ($\epsilon, \delta$), and empirical Pareto frontier.
- **Deliverables**:
  - `src/dashboard/app.py`: FastAPI server with RESTful telemetry, digital twin, what-if, work order, and federated endpoints.
  - `src/dashboard/service.py`: Backend integration service orchestrating federated training, twin state, and work order dispatching.
  - `src/dashboard/static/`: Cybernetic dark-mode frontend (`index.html`, `style.css`, `app.js`).
  - `scripts/start_dashboard.py`: CLI launcher script.
  - `scripts/run_benchmarks.py`: Full comparative evaluation benchmark suite.
  - `docs/figures/`: 4 publication-quality research figures.
  - `docs/benchmark_report.md`: Formal research evaluation report.
  - `tests/test_dashboard.py`: 12/12 passing integration & unit tests.
- **Status**: :white_check_mark: **COMPLETED**

---

## 4. Current Status & Phase Tracker

| Phase | Description | Status | Progress |
| :---: | :--- | :---: | :---: |
| **Phase 1** | Synthetic Industrial Sensor Dataset Generator & Non-IID Telemetry | :white_check_mark: **Completed** | **100%** |
| **Phase 2** | Local Edge Predictive Maintenance Model & Feature Engineering | :white_check_mark: **Completed** | **100%** |
| **Phase 3** | Federated Learning Infrastructure (FedAvg / FedProx) | :white_check_mark: **Completed** | **100%** |
| **Phase 4** | Privacy Preservation (Differential Privacy & Secure Aggregation) | :white_check_mark: **Completed** | **100%** |
| **Phase 5** | Digital Twin Synchronization & Remaining Useful Life (RUL) | :white_check_mark: **Completed** | **100%** |
| **Phase 6** | Interactive Web Dashboard & Comparative Benchmark Evaluation | :white_check_mark: **Completed** | **100%** |

### Current Milestone Summary:
- **Where We Are**: We have successfully completed **all 6 planned phases (100% of Project Scope)**!
  - Phase 1: Generated 36,000 multi-machine synthetic sensor records with physics cross-coupling and non-IID factory distributions.
  - Phase 2: Built CNN-BiLSTM and Edge MLP models with Focal Loss, achieving >0.84-0.90 F1 on local client silos.
  - Phase 3: Built FedAvg and FedProx federated learning coordinator resolving the severe -46% cross-domain non-IID performance drop.
  - Phase 4: Integrated $(\epsilon, \delta)$-Differential Privacy (DP-SGD) with Rényi accountant and Bonawitz pairwise zero-sum Secure Aggregation.
  - Phase 5: Developed high-fidelity real-time Digital Twin state engine, exponential RUL regression forecaster, What-If stress simulator, and prescriptive work order dispatcher.
  - Phase 6: Deployed high-performance FastAPI web dashboard with cybernetic UI, REST API endpoints, automated benchmark evaluation suite, 4 scientific publication figures, and 12 integration tests.
  - **Full Regression Validation**: All 6 test suites across the repository (100% of unit & integration tests) pass cleanly with 0 errors.
