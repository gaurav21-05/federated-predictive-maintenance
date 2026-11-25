# ESP32 Edge Sensor Firmware & Telemetry Ingestion Guide

Industrial IoT Predictive Maintenance & Physics-Grounded Digital Twin Platform  
**Target Hardware:** ESP32 DevKit V1 + MPU6050 (IMU/Vibration) + DHT11 (Temperature & Humidity)

---

## 1. Overview & Architecture

The edge firmware layer supports two ingestion paradigms:
1. **ThingSpeak Cloud Ingestion (`esp32_thingspeak.ino`)**: Transmits telemetry to ThingSpeak Channel 3591448 every 20s. The FastAPI backend polls and caches data from ThingSpeak statelessly.
2. **Direct Backend Ingestion (`esp32_firmware.ino`)**: Transmits formatted JSON directly to `POST /api/telemetry/ingest`.

```
 +--------------------------------------------------------------------------+
 |                              ESP32 DevKit V1                            |
 |                                                                          |
 |  +-----------------------+    I2C Fast Mode    +-----------------------+ |
 |  |    MPU6050 (IMU)      | <-----------------> |  SDA: GPIO 21         | |
 |  |  Accel & Gyroscope    |    (400 kHz)        |  SCL: GPIO 22         | |
 |  +-----------------------+                     +-----------------------+ |
 |                                                                          |
 |  +-----------------------+    Single-Wire      +-----------------------+ |
 |  |     DHT11 Sensor      | <-----------------> |  DATA: GPIO 4         | |
 |  |  Temp & Rel. Humidity |   (10k Pull-up)     +-----------------------+ |
 |  +-----------------------+                                               |
 |                                                                          |
 |  +--------------------------------------------------------------------+  |
 |  | Firmware Engine:                                                   |  |
 |  |  1. Windowed burst sampling (100 samples @ 100 Hz)                 |  |
 |  |  2. Static gravity (DC) removal across X, Y, Z axes                |  |
 |  |  3. Dynamic AC acceleration Root-Mean-Square (RMS) in g            |  |
 |  |  4. Edge anomaly rule trip & buzzer trigger (>0.15g / >0.25g)      |  |
 |  |  5. 20s rate-limited transmission to ThingSpeak update endpoint    |  |
 |  +--------------------------------------------------------------------+  |
 +-----------------------------------┬--------------------------------------+
                                     │  HTTP GET / POST (Wi-Fi 802.11 b/g/n)
                                     ▼
                     +-------------------------------+
                     |     ThingSpeak IoT Cloud      |
                     |     Channel ID: 3591448       |
                     +---------------┬---------------+
                                     │ REST API (10s server cache)
                                     ▼
                     +-------------------------------+
                     |    FastAPI Live Gateway       |
                     |    /api/live/hardware         |
                     |    - Robust Z-Score, EWMA     |
                     |    - Honest Wear RUL Model    |
                     +-------------------------------+
```

### ThingSpeak Telemetry Channel Fields

| Field | Parameter | Sensor / Source | Status | Description |
| :--- | :--- | :--- | :---: | :--- |
| `field1` | `temperature_c` | DHT11 | **CONFIRMED** | Ambient temperature (°C) |
| `field2` | `humidity_pct` | DHT11 | **CONFIRMED** | Relative air humidity (% RH) |
| `field3` | `temp_secondary_c` | Aux / Internal | *UNCONFIRMED* | Secondary temperature channel |
| `field4` | `pressure_hpa` | Barometer / Seal | *UNCONFIRMED* | Barometric pressure baseline (~994 hPa) |
| `field5` | `vibration_g` | MPU6050 | **CONFIRMED** | Dynamic vibration RMS in g (gravity removed) |
| `field6` | `current_a` | ACS712 | **RESERVED** | Motor current in A (NA if disconnected) |
| `field7` | `esp32_anomaly` | ESP32 Firmware | **CONFIRMED** | Edge anomaly trip flag (0 or 1) |
| `field8` | `esp32_alert` | ESP32 Firmware | **CONFIRMED** | Buzzer / red LED alarm flag (0 or 1) |
| `status` | `status_text` | ESP32 Firmware | **CONFIRMED** | Machine status string |

---

## 2. Hardware Bill of Materials (BOM) & Wiring

### Bill of Materials
| Component | Specification | Description |
| :--- | :--- | :--- |
| **Microcontroller** | ESP32 DevKit V1 (ESP-WROOM-32) | 30-pin or 36-pin, 2.4 GHz Wi-Fi + Bluetooth |
| **Vibration Sensor** | MPU6050 Breakout Board (GY-521) | 3-axis accelerometer ($\pm 4g$), 3-axis gyro ($\pm 500^\circ/s$) |
| **Environment Sensor** | DHT11 Sensor Module | Temperature (0 to $50^\circ\text{C}$), Humidity (20 to $90\%$ RH) |
| **Pull-Up Resistor** | $4.7\text{ k}\Omega$ to $10\text{ k}\Omega$ | Pull-up between DHT11 DATA and 3.3V (often on module) |
| **Interconnects** | Breadboard & Jumper Wires | Female-to-Male or Male-to-Male |
| **Power Source** | Micro-USB Cable (5V / 1A+) | Stable power from host workstation or 5V USB wall adapter |

---

### Wiring Pinout Table

> [!CAUTION]
> Always connect MPU6050 and DHT11 VCC to the **3.3V (3V3)** pin of the ESP32. Connecting them to 5V (VIN) can damage the ESP32 GPIO pins, which are **not 5V tolerant**.

| Sensor Pin | ESP32 Pin | Signal / Function | Wiring Notes |
| :--- | :--- | :--- | :--- |
| **MPU6050 VCC** | **3V3** | Power (+3.3V DC) | Keep power lead short to minimize noise |
| **MPU6050 GND** | **GND** | Ground (0V) | Common ground reference |
| **MPU6050 SCL** | **GPIO 22** | I2C Clock (SCL) | Default hardware I2C clock |
| **MPU6050 SDA** | **GPIO 21** | I2C Data (SDA) | Default hardware I2C data |
| **MPU6050 AD0** | **GND** | I2C Address Select | Ground selects address `0x68`; connect to 3V3 for `0x69` |
| **MPU6050 INT** | *Unconnected* | Hardware Interrupt | Not used in polling mode |
| **DHT11 VCC** | **3V3** | Power (+3.3V DC) | Stable 3.3V rail |
| **DHT11 GND** | **GND** | Ground (0V) | Common ground |
| **DHT11 DATA** | **GPIO 4** | Single-Wire Bidirectional | Requires $4.7\text{k}\Omega - 10\text{k}\Omega$ pullup to 3V3 |
| **Status LED** | **GPIO 2** | Built-in Blue LED | Solid ON = HTTP ACK, Blinking = Reconnecting |

---

## 3. Library Installation & Toolchain Setup

### Option A: Arduino IDE Setup (Recommended for beginners)

1. **Install Arduino IDE**: Download and install Arduino IDE 2.x from [arduino.cc](https://www.arduino.cc/en/software).
2. **Add ESP32 Board Package**:
   - Open **File > Preferences**.
   - In *Additional Board Manager URLs*, add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Go to **Tools > Board > Boards Manager...**, search for `esp32` by Espressif Systems, and click **Install** (v2.0.x or v3.0.x).
3. **Install Required Libraries**:
   Open **Sketch > Include Library > Manage Libraries...** and install the following:
   - **`Adafruit MPU6050`** by Adafruit (v2.2.4 or higher)
     - *When prompted, click "Install All" to automatically install dependencies: `Adafruit Unified Sensor` and `Adafruit BusIO`.*
   - **`DHT sensor library`** by Adafruit (v1.4.4 or higher)
   - **`ArduinoJson`** by Benoit Blanchon (v6.21.x or v7.x supported)

---

### Option B: PlatformIO / VS Code Setup

If using VS Code with PlatformIO, create or configure `platformio.ini`:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
upload_speed = 921600
lib_deps =
    adafruit/Adafruit MPU6050@^2.2.6
    adafruit/Adafruit Unified Sensor@^1.1.14
    adafruit/DHT sensor library@^1.4.6
    bblanchon/ArduinoJson@^7.0.4
```

---

## 4. Configuration & Credentials Management

> [!IMPORTANT]
> To prevent leaking private Wi-Fi passwords and internal network endpoints into Git, credentials are kept in a local `config.h` file that is ignored by Git.

1. Navigate to the firmware directory:
   ```bash
   cd hardware/esp32_firmware
   ```
2. Copy the template configuration file:
   ```bash
   cp config.example.h config.h
   ```
3. Open `config.h` and edit your network parameters:
   ```cpp
   // Set your 2.4 GHz Wi-Fi credentials
   #define WIFI_SSID         "MyFactoryWiFi"
   #define WIFI_PASSWORD     "SecurePassword123"

   // Set your workstation's LAN IP address hosting the FastAPI backend:
   // (Run 'ipconfig' on Windows or 'ip a' on Linux to find your local IP)
   #define BACKEND_URL       "http://192.168.1.150:8000/api/telemetry/ingest"

   // Set the target machine identifier (maps to digital twin client):
   // CNC_MILL_01 -> client_1 | HYDR_PRESS_02 -> client_2 | TURB_COMP_03 -> client_3
   #define MACHINE_ID        "CNC_MILL_01"

   // Interval between successive telemetry dispatches (ms)
   #define TELEMETRY_INTERVAL_MS  2000UL
   ```
4. Verify that `config.h` is excluded from git staging:
   ```bash
   git status
   ```
   *(Ensure `config.h` is NOT listed under untracked files).*

---

## 5. Vibration Feature Calculation: Mathematical & Physical Principles

### Why Single-Sample Gravity Deviation is NOT Vibration RMS
In many rudimentary tutorials, authoring code calculates:
$$\text{"Vibration"} \stackrel{?}{=} \left| \sqrt{a_x^2 + a_y^2 + a_z^2} - 1.0g \right| \quad \text{(Single Sample)}$$
**This approach is physically flawed and rejected in industrial engineering for three reasons:**
1. **Gravitational Vector Tilt**: If the sensor rests on a machine inclined at an angle $\theta$, or has manufacturing zero-g bias ($\pm 50\text{ mg}$ typical on MEMS), this formula will interpret steady-state static tilt as constant mechanical vibration even when the machine is shut off ($0\text{ RPM}$).
2. **Harmonic Zero-Crossings**: Industrial vibration is an alternating dynamic oscillation. A single instantaneous sample might capture the sinusoidal wave at its zero-crossing ($\tilde{a}(t) = 0$), falsely reporting zero vibration during violent resonance.
3. **Standards Non-Compliance**: International condition monitoring standards (**ISO 10816** and **ISO 20816**) define vibration severity strictly as the **Root-Mean-Square (RMS)** of dynamic vibration over an integrated observation window $\Delta t$, not an instantaneous scalar.

---

### Implemented Calibrated Algorithm

The firmware implements the following windowed dynamic vibration pipeline:

```
Raw Accelerations (ax, ay, az) 
       │
       ▼
Burst Window Buffer (N = 100 samples @ 100 Hz, Δt = 1.0s)
       │
       ▼
Compute Static Gravity (DC Component):
  ā_i = (1/N) * Σ a_i[k]    for i ∈ {x, y, z}
       │
       ▼
Extract Dynamic AC Vibration:
  ã_i[k] = a_i[k] - ā_i     (Eliminates Earth's 1g vector and mounting tilt)
       │
       ▼
Compute Per-Axis RMS Acceleration:
  RMS_i = √[ (1/N) * Σ (ã_i[k])² ]
       │
       ▼
Total Tri-Axial Vibration Severity (g RMS):
  vibration_g = √( RMS_x² + RMS_y² + RMS_z² )
```

#### Step 1: Windowed Sampling
The microcontroller samples $N = 100$ acceleration vectors from the MPU6050 at a calibrated rate of $f_s = 100\text{ Hz}$ ($T_s = 10\text{ ms}$), providing an observation window of $\Delta t = 1.0\text{ second}$.

#### Step 2: Static Gravity (DC Vector) Removal
For each Cartesian axis $i \in \{x, y, z\}$, the static gravity component is calculated as the mean across the window:
$$\bar{a}_i = \frac{1}{N} \sum_{k=0}^{N-1} a_i[k]$$
The dynamic oscillating acceleration for each sample $k$ is obtained by subtracting the axis DC bias:
$$\tilde{a}_i[k] = a_i[k] - \bar{a}_i$$
This removes the $1.0g$ Earth gravity vector regardless of whether the ESP32 is mounted horizontally, vertically, or at an arbitrary tilt angle.

#### Step 3: Per-Axis Dynamic RMS
The dynamic acceleration Root-Mean-Square along each axis is computed as:
$$\text{RMS}_i = \sqrt{\frac{1}{N} \sum_{k=0}^{N-1} \left( \tilde{a}_i[k] \right)^2} = \sigma(a_i)$$

#### Step 4: Combined Tri-Axial Vibration Severity
The overall vibration severity metric sent to the backend is the vector magnitude of the directional RMS values:
$$\text{vibration\_g} = \sqrt{\text{RMS}_x^2 + \text{RMS}_y^2 + \text{RMS}_z^2}$$

#### Step 5: Relation to Velocity Severity ($v_{\text{RMS}}$ in $\text{mm/s}$)
In the backend Digital Twin engine (`src/dashboard/service.py`), incoming `vibration_g` is converted to physical RMS vibration velocity in $\text{mm/s}$ according to the standard industrial scale:
$$v_{\text{RMS}} = \text{vibration\_g} \times 9.81\text{ mm/s}$$
This value is evaluated against ISO 10816 machine vibration severity limits (e.g. Class I/II machines: Nominal $< 1.8\text{ mm/s}$, Alert $1.8 - 4.5\text{ mm/s}$, Critical $> 4.5\text{ mm/s}$).

---

## 6. Backend API Telemetry Contract

The firmware communicates directly with the FastAPI backend endpoint:

- **Method**: `POST`
- **Path**: `/api/telemetry/ingest`
- **Content-Type**: `application/json`

### Request JSON Payload Specification
```json
{
  "machine_id": "CNC_MILL_01",
  "temperature_c": 24.5,
  "humidity_pct": 52.0,
  "acceleration_x": 0.012,
  "acceleration_y": -0.008,
  "acceleration_z": 0.995,
  "gyro_x": 0.15,
  "gyro_y": -0.08,
  "gyro_z": 0.04,
  "vibration_g": 0.038,
  "source": "ESP32_LIVE",
  "timestamp": "2026-10-16T02:15:30Z"
}
```

### Field Validation Matrix (Pydantic Contract)
| Field Name | Type | Constraints / Bounds | Description |
| :--- | :--- | :--- | :--- |
| `machine_id` | `string` | 1–64 characters, non-empty | Unique asset tag (`CNC_MILL_01`, `HYDR_PRESS_02`, `TURB_COMP_03`) |
| `temperature_c` | `float` | $\ge -40.0$, $\le 150.0$ | Ambient / housing temperature in $^\circ\text{C}$ (DHT11) |
| `humidity_pct` | `float` | $\ge 0.0$, $\le 100.0$ | Relative humidity percentage (DHT11) |
| `acceleration_x` | `float` | $\ge -50.0$, $\le 50.0$ | Static orientation acceleration along X in $g$ |
| `acceleration_y` | `float` | $\ge -50.0$, $\le 50.0$ | Static orientation acceleration along Y in $g$ |
| `acceleration_z` | `float` | $\ge -50.0$, $\le 50.0$ | Static orientation acceleration along Z in $g$ |
| `gyro_x` | `float` | $\ge -2000.0$, $\le 2000.0$ | Angular velocity around X in $^\circ/s$ |
| `gyro_y` | `float` | $\ge -2000.0$, $\le 2000.0$ | Angular velocity around Y in $^\circ/s$ |
| `gyro_z` | `float` | $\ge -2000.0$, $\le 2000.0$ | Angular velocity around Z in $^\circ/s$ |
| `vibration_g` | `float` | $\ge 0.0$, $\le 50.0$ | Calibrated dynamic vibration RMS in $g$ |
| `source` | `enum` | Must be `"ESP32_LIVE"` | Telemetry source attribution |
| `timestamp` | `string` \| `null` | ISO-8601 string or `null` | UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) or `null` (server fallback) |

---

## 7. Flashing & Running the Firmware

1. Connect the ESP32 board to your computer via USB.
2. In Arduino IDE:
   - Select Board: **ESP32 Dev Module** (or **DOIT ESP32 DEVKIT V1**).
   - Select Port: (e.g. `COM3` on Windows, `/dev/ttyUSB0` on Linux, `/dev/cu.usbserial` on macOS).
   - Upload Speed: `921600` (or `115200` if flashing fails).
3. Open `hardware/esp32_firmware/esp32_firmware.ino`.
4. Click **Upload** ($\rightarrow$).
5. Once uploaded, open the **Serial Monitor** at **115200 baud**.

### Expected Serial Monitor Diagnostic Output:
```text
**************************************************************
*  IIoT Federated Predictive Maintenance & Digital Twin      *
*  ESP32 Edge Sensor Telemetry Firmware v1.0.0               *
**************************************************************
*  Machine ID:     CNC_MILL_01
*  Source Tag:     ESP32_LIVE
*  Ingest API:     http://192.168.1.100:8000/api/telemetry/ingest
*  Cadence:        2000 ms
*  Vib Window:     100 samples @ 10000 µs (~100 Hz)
*  Pins:           MPU6050 SDA=21, SCL=22 | DHT11 DATA=4
**************************************************************

[INIT] Probing MPU6050 on I2C (SDA=GPIO21, SCL=GPIO22, Addr=0x68)...
[INIT] MPU6050 initialized successfully (Range: ±4g / ±500°/s, DLPF: 44Hz).
[INIT] Initializing DHT11 on GPIO 4
[INIT] DHT11 ready. Initial reading: 24.2 °C, 51.0 % RH
[WIFI] Connecting to SSID: 'MyFactoryWiFi' (2.4 GHz).....
[WIFI] Connected successfully!
[WIFI] IP Address:    192.168.1.42
[WIFI] Signal (RSSI): -54 dBm
[NTP] Configuring SNTP (Servers: pool.ntp.org, time.nist.gov)...
[NTP] Time synchronized successfully: 2026-10-16T02:15:30Z (UTC)
[SYSTEM] Initialization complete. Entering autonomous monitoring loop.

==================== [CYCLE #1] ====================
Asset ID:        CNC_MILL_01 | Source: ESP32_LIVE
Timestamp:       2026-10-16T02:15:32Z
Temperature:     24.2 °C  | Humidity: 51.0 % RH
Static Tilt Acc: X=0.014g, Y=-0.009g, Z=0.998g
Angular Vel:     X=0.12°/s, Y=-0.05°/s, Z=0.02°/s
Vibration (RMS): 0.0382 g (Peak Dynamic: 0.0714 g)
Free Heap:       278144 bytes
Outgoing JSON:   {"machine_id":"CNC_MILL_01","temperature_c":24.2,"humidity_pct":51.0,"acceleration_x":0.014,"acceleration_y":-0.009,"acceleration_z":0.998,"gyro_x":0.12,"gyro_y":-0.05,"gyro_z":0.02,"vibration_g":0.038,"source":"ESP32_LIVE","timestamp":"2026-10-16T02:15:32Z"}
[HTTP SUCCESS] Status 200 OK in 62 ms
[BACKEND ACK]  Response: {"status":"success","machine_id":"CNC_MILL_01","timestamp":"2026-10-16T02:15:32Z","source":"ESP32_LIVE","storage":"in_memory","persisted":false,"target_client_id":"client_1","digital_twin":{"overall_health_index":98.6,"operational_state":"NOMINAL","rul_minutes":1425.0,"primary_fault_subsystem":"none"},"new_work_orders":[]}
Transmission Stats: 1 attempted, 1 acknowledged (Success Rate: 100.0%)
====================================================
```
