# Synthetic and Simulated Assets Deletion Inventory

This document provides a comprehensive inventory of all synthetic data, generator scripts, physics simulation models, simulated dashboard endpoints, pretrained checkpoints, and hardcoded UI placeholders slated for removal or replacement to transition the project into a **Real-Data-Only** predictive maintenance platform.

---

## 1. Synthetic Sensor Datasets
These files contain purely simulated sensor readings for three hypothetical industrial factories (CNC milling machine, hydraulic press, and micro-turbine) generated via mathematical degradation formulas:

- `data/client_1/client_1_sensor_data.csv` (12,000 synthetic rows)
- `data/client_2/client_2_sensor_data.csv` (12,000 synthetic rows)
- `data/client_3/client_3_sensor_data.csv` (12,000 synthetic rows)
- `data/combined_industrial_sensors.csv` (36,000 concatenated synthetic rows)
- `data/dataset_summary.md` (statistical summary of synthetic data distributions)
- `data/digital_twin_state.json` (cached simulated machine state ticks)
- `data/privacy_benchmark_results.json` (benchmark results derived from synthetic data)

**Action**: Remove. Real physical telemetry will be served solely via ThingSpeak Channel `3591448` and offline sample `feeds.csv`.

---

## 2. Synthetic Telemetry Generators & Physics Modeling
These modules implement synthetic data generation pipelines, degradation formulas, and simulation loops:

- `generate_dataset.py` (root synthetic dataset generator script)
- `src/generator.py` (synthetic dataset orchestrator)
- `src/physics_engine.py` (synthetic wear physics equations for CNC/press/turbine)
- `scripts/run_generator.py` (CLI launcher for dataset synthesis)
- `scripts/analyze_dataset.py` (statistical telemetry inspector for synthetic data)
- `scripts/run_digital_twin_sync.py` (simulation tick loop runner)
- `scripts/run_benchmarks.py` (synthetic evaluation benchmark runner)
- `scripts/run_federated.py` (federated runner on synthetic data; replaced by `scripts/train_federated_live.py`)

**Action**: Remove. The real system ingests physical hardware telemetry from ThingSpeak without synthetic physics simulation.

---

## 3. Pretrained Neural Network Checkpoints
These PyTorch `.pt` model weights were trained exclusively on synthetic CNC, Press, and Turbine data and are invalid for real ESP32 bench hardware:

- `models/checkpoints/client_1_best_model.pt`
- `models/checkpoints/client_2_best_model.pt`
- `models/checkpoints/client_3_best_model.pt`
- `models/checkpoints/global_federated_model.pt`

**Action**: Remove. Replaced by `models/live/` containing unsupervised machine learning models (Isolation Forest and Autoencoder weights) trained strictly on real hardware telemetry.

---

## 4. Synthetic Documentation & Figures
These documents and static plots display synthetic benchmark scores (e.g. 98.4% accuracy, 0.962 F1) that do not represent real hardware:

- `docs/benchmark_report.md` (synthetic benchmark claims)
- `docs/figures/fl_convergence.png`
- `docs/figures/non_iid_comparison.png`
- `docs/figures/privacy_utility_pareto.png`
- `docs/figures/digital_twin_rul.png`
- `public/figures/*` and `src/dashboard/static/figures/*`
- README synthetic benchmark table

**Action**: Remove synthetic figures and update documentation with honest real-data metrics, caveats, and physical bench specifications.

---

## 5. Simulated Backend Service Endpoints
The following REST endpoints and service methods in `src/dashboard/app.py` and `src/dashboard/service.py` serve synthetic simulation features:

- `GET /api/overview` (serves simulated 3-machine fleet status)
- `GET /api/telemetry/stream` (streams rows from synthetic CSVs)
- `POST /api/digital-twin/step` (advances simulation ticks)
- `POST /api/digital-twin/reset` (resets synthetic stream position)
- `POST /api/digital-twin/what-if` (operational stress sandbox on synthetic models)
- `POST /api/federated/train` (in-memory federated training on synthetic data)
- `GET /api/benchmark/summary` (serves synthetic benchmark markdown)
- `src/config.py`: `CLIENT_PROFILES` with CNC, Press, and Turbine baselines
- `src/digital_twin/what_if_simulator.py` (synthetic scenario simulation)
- `src/digital_twin/twin_engine.py` (synthetic industrial twin engine)

**Action**: Remove simulated endpoints and `service.py`. Replace with stateless live endpoints (`/api/live/hardware`, `/api/live/rul`, `/api/live/export.csv`, `/api/analysis/model`, `/api/analysis/insights`).

---

## 6. Frontend Hardcoded Placeholder Numbers & Simulated Tabs
The dashboard UI contained legacy mock values and simulated tabs:

- Hardcoded top-nav metrics: `"FLEET OPTIMAL"`, `"3 Active"`, `"98.4%"`, `"ACTIVE"`
- Tab 1: Plant Fleet & Digital Twins (`#tab-twins`) with simulated buttons ("Step +1", "Step +10", "Auto-Stream", "Reset Stream"), simulated dials and client selector (`CNC_MILL_01`, `HYDR_PRESS_02`, `TURB_COMP_03`)
- Tab 2: Federated Learning Control Room (`#tab-federated`) with mock rounds and synthetic loss charts
- Tab 3: "What-If" Stress Sandbox (`#tab-whatif`) with synthetic multiplier sliders
- Tab 4: Prescriptive Work Orders (`#tab-maintenance`) displaying simulation-generated orders
- Tab 5: Research Benchmarks (`#tab-benchmark`) displaying synthetic research figures
- Hardcoded JavaScript defaults in `app.js`: fake rounds `[0.284, 0.162, 0.098]`, fake RUL `190.0 min`, etc.

**Action**: Remove all simulated tabs. The UI will feature two clean tabs:
1. **Live Monitor**: Live telemetry, KPIs, real hardware SVG twin, and event log.
2. **Analysis & Prediction**: Statistical analysis, unsupervised ML anomaly prediction, honest federated learning comparison on real data partitions, and AI-generated insights summary.
All displayed values will be populated dynamically from the API with skeleton/loading states (no hardcoded fake values).

---

## 7. Synthetic Unit Tests
The following tests in `tests/` depend on synthetic data and modules:

- `tests/test_dataset.py` (validates synthetic 12,000 row generation)
- `tests/test_digital_twin.py` (tests synthetic step synchronization)
- `tests/test_models.py` (tests synthetic CNN-LSTM model)
- `tests/test_federated.py` (tests federated training on synthetic clients)
- `tests/test_dashboard.py` (tests simulated endpoints `/api/overview`, etc.)

**Action**: Remove tests coupled to synthetic data and replace with real-data tests (`test_live_analytics.py`, `test_live_rul.py`, `test_live_ml.py`, `test_privacy.py`, `test_federated_real.py`).

