# Industrial Sensor Dataset Statistical Profile

## 1. Overview & Class Balance

| Client ID | Machine ID | Total Records | Healthy (0) | Failure (1) | Failure Rate (%) | Imbalance Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `client_1` | `CNC_MILL_01` | 12,000 | 11,726 | 274 | 2.28% | 11726:274 (~42.8:1) |
| `client_2` | `HYDR_PRESS_02` | 12,000 | 11,730 | 270 | 2.25% | 11730:270 (~43.4:1) |
| `client_3` | `TURB_COMP_03` | 12,000 | 11,725 | 275 | 2.29% | 11725:275 (~42.6:1) |
| `combined` | `All 3 Machines` | 36,000 | 35,181 | 819 | 2.27% | 35181:819 (~43.0:1) |

## 2. Sensor Feature Distributions (Mean ± Std [Min, Max])

| Feature | Unit | Client 1 (CNC Mill) | Client 2 (Hydraulic Press) | Client 3 (Compressor/Turbine) |
| :--- | :---: | :---: | :---: | :---: |
| **Temperature** | °C | 65.0 ± 9.1 [43.74, 92.98] | 56.7 ± 14.4 [35.84, 98.0] | 60.5 ± 9.9 [39.87, 88.06] |
| **Humidity** | % RH | 45.9 ± 3.1 [39.88, 50.09] | 38.9 ± 3.1 [33.1, 42.87] | 72.9 ± 3.1 [66.51, 77.3] |
| **Vibration** | mm/s | 1.4 ± 0.8 [0.2, 5.67] | 3.0 ± 1.8 [0.5, 9.59] | 1.9 ± 0.9 [0.3, 6.8] |
| **Current** | A | 20.1 ± 5.3 [5.0, 36.92] | 37.3 ± 11.0 [10.0, 74.47] | 30.5 ± 9.2 [8.0, 60.0] |
| **Rpm** | RPM | 3648.2 ± 655.0 [0.0, 3866.6] | 979.2 ± 313.7 [0.0, 1124.5] | 2980.4 ± 541.3 [0.0, 3159.5] |
| **Pressure** | bar | 6.0 ± 0.6 [2.57, 6.31] | 187.3 ± 22.4 [91.94, 266.63] | 8.4 ± 1.0 [3.89, 8.96] |

## 3. Physical Feature Correlation with Machine Failure

Demonstrates that failure is **not random** and strongly tied to physical anomalies:

| Feature | Client 1 Correlation | Client 2 Correlation | Client 3 Correlation | Dominant Physical Failure Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Temperature** | `+0.410` | `+0.290` | `+0.386` | Thermal runaway from spindle friction or bearing overheating |
| **Humidity** | `+0.016` | `+0.005` | `+0.014` | Environmental drift causing moisture condensation & corrosion |
| **Vibration** | `+0.750` | `+0.500` | `+0.742` | Bearing spall, mechanical shock, and dynamic unbalance |
| **Current** | `+0.425` | `+0.455` | `+0.473` | Motor overload, winding resistance breakdown, and hydraulic head resistance |
| **Rpm** | `-0.193` | `-0.157` | `-0.166` | Kinematic drag, slip, and motor stall under critical mechanical resistance |
| **Pressure** | `-0.487` | `+0.425` | `-0.500` | Hydraulic line over-pressure surges or pneumatic seal blowout |