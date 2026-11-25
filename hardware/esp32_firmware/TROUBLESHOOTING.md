# ESP32 Firmware Diagnostic & Troubleshooting Guide

This guide details systematically categorized root causes, diagnostic procedures, and resolutions for physical hardware, communication protocols, network transport, and backend validation issues encountered when deploying the ESP32 edge telemetry node.

---

## Quick Reference Diagnostic Matrix

| Stage | Common Error / Symptom | Root Cause Category | Primary Fix |
| :--- | :--- | :--- | :--- |
| **I2C Bus** | `[ERROR] MPU6050 chip detection failed!` | Hardware / Wiring | Check SDA (21), SCL (22), AD0 pull-down, VCC 3.3V |
| **DHT Bus** | `[WARN] DHT11 reading failed (NaN)` | Timing / Pull-up | Verify $4.7\text{k}\Omega - 10\text{k}\Omega$ pull-up, avoid sampling $< 1.5\text{s}$ |
| **Wi-Fi** | Continuous `.....` dots, connection timeout | RF / Frequency Band | Connect to **2.4 GHz** Wi-Fi (ESP32 does NOT support 5 GHz) |
| **HTTP Transport** | `HTTP POST failed: connection refused (-1)` | Network Routing | Replace `localhost` with host PC LAN IP (`192.168.x.x`) |
| **HTTP Transport** | `HTTP POST failed: read timeout (-11)` | Firewall / Host Service | Unblock port 8000 in Windows Firewall; verify FastAPI is running |
| **Backend API** | `Server rejected upload: HTTP 422` | Schema Violation | Out-of-bounds sensor values or invalid ISO-8601 timestamp |
| **Microcontroller**| `Guru Meditation Error` / Watchdog Reset | Resource Exhaustion | Sockets unclosed (`http.end()`); excessive stack allocation |

---

## 1. MPU6050 Vibration & Inertial Sensor Troubleshooting

### Symptom: `[ERROR] MPU6050 chip detection failed!`
The firmware probes I2C address `0x68` (and optionally `0x69`), but the MPU6050 does not acknowledge its WHO_AM_I register (`0x75`).

#### Diagnostic Steps:
1. **Check Power Rails**:
   - Measure voltage between MPU6050 `VCC` and `GND` with a multimeter. It must read between **3.2V and 3.4V DC**.
   - Ensure the module is connected to the ESP32 **3V3** pin, NOT the 5V/VIN pin.
2. **Verify I2C Pin Assignment**:
   - ESP32 default hardware I2C pins are:
     - **SDA: GPIO 21**
     - **SCL: GPIO 22**
   - Confirm wires are not swapped (`SDA` to `GPIO 21`, `SCL` to `GPIO 22`).
3. **Check the AD0 Pin (I2C Address Select)**:
   - If `AD0` is floating or tied to `GND`, the I2C address is `0x68`.
   - If `AD0` is connected to `3V3`, the I2C address is `0x69`.
   - If using `0x69`, update `MPU6050_I2C_ADDR` in `config.h`:
     ```cpp
     #define MPU6050_I2C_ADDR 0x69
     ```
4. **Run an I2C Bus Scanner**:
   Flash a minimal I2C scanner sketch to verify if any device responds on the bus:
   ```cpp
   #include <Wire.h>
   void setup() {
     Serial.begin(115200);
     Wire.begin(21, 22);
   }
   void loop() {
     for (byte addr = 1; addr < 127; addr++) {
       Wire.beginTransmission(addr);
       if (Wire.endTransmission() == 0) {
         Serial.printf("Device found at 0x%02X\n", addr);
       }
     }
     delay(2000);
   }
   ```
   If no device is found:
   - Check for loose breadboard contacts or defective jumper wires.
   - Some bare breakout boards lack onboard I2C pull-ups. Add two $4.7\text{ k}\Omega$ pull-up resistors from SDA $\rightarrow$ 3.3V and SCL $\rightarrow$ 3.3V.

---

## 2. DHT11 Temperature & Humidity Sensor Troubleshooting

### Symptom: `[WARN] DHT11 reading failed (NaN)`
The single-wire handshake between the ESP32 and DHT11 failed to complete, or data failed the 8-bit checksum.

#### Diagnostic Steps:
1. **Pull-Up Resistor Check**:
   - The DHT11 uses an open-drain single-wire communication protocol.
   - If using a bare 4-pin DHT11 sensor (blue plastic grid), you **MUST** install a $4.7\text{ k}\Omega$ to $10\text{ k}\Omega$ resistor between `DATA` (GPIO 4) and `VCC` (3.3V).
   - If using a 3-pin DHT11 module with a small PCB, verify that the onboard SMD resistor (marked 103 or 472) is intact.
2. **Sampling Rate Violations**:
   - The DHT11 physical thermistor and capacitive polymer element require **at least 1.0 to 1.5 seconds** between readings.
   - The firmware handles this by polling the DHT11 only once per telemetry interval (`TELEMETRY_INTERVAL_MS = 2000UL`). Do not reduce `TELEMETRY_INTERVAL_MS` below 1500 ms when using DHT11.
3. **Cable Length**:
   - Jumper wire length between ESP32 and DHT11 should be under 1 meter (ideally under 20 cm on a breadboard) to prevent line capacitance from distorting the microsecond-level pulse timings.

---

## 3. Wi-Fi Connectivity & Signal Diagnostics

### Symptom: Continuous `.....` in Serial Monitor / Connection Timeout
The ESP32 radio fails to authenticate or associate with the Wi-Fi Access Point.

#### Diagnostic Steps:
1. **2.4 GHz vs 5.0 GHz Dual-Band Routers**:
   - **ESP32 hardware ONLY supports 2.4 GHz (802.11 b/g/n)**. It has no 5 GHz radio.
   - If your router broadcasts a single unified SSID for both 2.4 GHz and 5 GHz with band steering, split the bands in your router settings or create a dedicated 2.4 GHz IoT SSID (e.g. `Factory_IoT_2.4G`).
2. **Signal Strength (RSSI)**:
   - Look at the diagnostic output:
     - `RSSI > -65 dBm`: Excellent signal
     - `RSSI -65 to -75 dBm`: Good signal
     - `RSSI -75 to -85 dBm`: Weak signal (packet loss and re-transmissions likely)
     - `RSSI < -85 dBm`: Unacceptable (frequent disconnects)
3. **Network Isolation / Captive Portals**:
   - Ensure the Wi-Fi network does not have a web-based login (captive portal).
   - Ensure "Client Isolation" or "AP Isolation" on the router is disabled if you want the ESP32 to talk to your laptop on the same local subnet.

---

## 4. HTTP Transport & Ingestion API Troubleshooting

### Symptom: `[TRANSPORT ERROR] HTTP POST failed: connection refused (-1)`
The ESP32 sent a TCP SYN packet to port 8000, but the host operating system immediately replied with a TCP RST (connection refused).

#### Root Causes & Fixes:
1. **Using "localhost" or "127.0.0.1" in `config.h`**:
   - `127.0.0.1` on the ESP32 refers to the **ESP32 itself**, not your development PC!
   - In `config.h`, set `BACKEND_URL` to the **local IPv4 address of your computer on the Wi-Fi network**:
     ```cpp
     // INCORRECT:
     // #define BACKEND_URL "http://127.0.0.1:8000/api/telemetry/ingest"

     // CORRECT:
     #define BACKEND_URL "http://192.168.1.150:8000/api/telemetry/ingest"
     ```
   - To find your computer's IP address:
     - **Windows**: Open PowerShell and run `ipconfig`. Look for `IPv4 Address` under your active Wi-Fi adapter.
     - **macOS / Linux**: Run `ifconfig` or `ip a`. Look for `inet 192.168.x.x`.
2. **Backend is Not Running or Bound to Loopback Only**:
   - Ensure the FastAPI server is running with host set to `0.0.0.0` or your LAN IP:
     ```bash
     python -m uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000
     ```
     *(Binding to `127.0.0.1` only accepts local connections from the same machine; binding to `0.0.0.0` allows connections from edge microcontrollers on the local network).*

---

### Symptom: `[TRANSPORT ERROR] HTTP POST failed: read timeout (-11)`
The ESP32 successfully established a TCP connection, but received no response within `HTTP_TIMEOUT_MS` (5000 ms).

#### Root Causes & Fixes:
1. **Windows Defender Firewall / OS Firewall**:
   - Windows Firewall frequently blocks inbound connections on port 8000 from external devices on "Public" Wi-Fi networks.
   - Set your Wi-Fi network profile to **Private** in Windows Settings.
   - Or open PowerShell as Administrator and allow inbound traffic on port 8000:
     ```powershell
     New-NetFirewallRule -DisplayName "FastAPI IIoT Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
     ```
2. **Test Connectivity from Another Device**:
   - From your smartphone or another machine on the same Wi-Fi, open:
     `http://<YOUR_PC_IP>:8000/api/overview`
   - If your phone cannot load the page, your PC's firewall is blocking incoming connections or your router has client isolation enabled.

---

### Symptom: `[HTTP ERROR] Server rejected upload with status code: 422`
The FastAPI backend received the HTTP request, but Pydantic rejected the JSON payload due to a contract validation failure.

#### Root Causes & Fixes:
1. **Sensor Reading Out of Range**:
   - Pydantic enforces strict physical bounds in `src/dashboard/app.py`:
     - `temperature_c`: $[-40.0, 150.0]$
     - `humidity_pct`: $[0.0, 100.0]$
     - `acceleration_x/y/z`: $[-50.0, 50.0]$
     - `gyro_x/y/z`: $[-2000.0, 2000.0]$
     - `vibration_g`: $[0.0, 50.0]$
   - If a sensor disconnected and returned an unhandled floating point `NaN` or unconstrained garbage value (e.g. `1e9`), Pydantic will return HTTP 422.
   - The firmware includes automatic bounds clamping (`constrain`) and NaN detection to prevent this.
2. **Invalid Timestamp String**:
   - The backend validator requires an ISO-8601 string parseable by `datetime.fromisoformat` (e.g. `2026-10-16T02:15:30Z`), or an explicit JSON `null`.
   - If NTP has not synchronized, the firmware sets `timestamp = nullptr`, which serializes to `null` and allows FastAPI to supply server-side UTC time.
3. **Invalid `source` Tag**:
   - `source` must strictly be `"ESP32_LIVE"` or `"SIMULATED"`.
   - Verify that `SOURCE_TAG` in `config.h` is set to `"ESP32_LIVE"`.

---

## 5. Microcontroller Stability & Memory Management

### Symptom: ESP32 Restarts Unexpectedly or Panics
#### Causes & Mitigations:
1. **Unclosed HTTP Connections (Socket Exhaustion)**:
   - If `http.end()` is not called after every request, ESP32's underlying lwIP socket pool is exhausted within ~10 requests, resulting in heap exhaustion or crash.
   - **Resolution**: `esp32_firmware.ino` unconditionally invokes `http.end()` in all code paths.
2. **Power Supply Brownouts**:
   - When the ESP32 Wi-Fi radio initiates a transmission burst, current draw can spike to ~300–400 mA for a few milliseconds.
   - Poor quality USB cables, unpowered USB hubs, or high-resistance breadboard jumper wires can cause the 3.3V rail to dip below 2.8V, triggering the ESP32 hardware Brownout Detector (`Brownout detector was triggered`).
   - **Resolution**: Use a high-quality USB data cable plugged directly into your computer or a 5V 1A+ powered adapter. If needed, connect a $100\mu\text{F} - 470\mu\text{F}$ electrolytic capacitor across 3V3 and GND on the breadboard.
3. **Free Heap Monitoring**:
   - The firmware prints `Free Heap: %u bytes` on every cycle. Normal steady-state operation should maintain $> 200,000$ bytes of free heap without continuous depletion.
