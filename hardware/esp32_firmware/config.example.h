/**
 * ============================================================================
 * Configuration Template: config.example.h
 * Industrial IoT Predictive Maintenance - ESP32 Firmware
 * ============================================================================
 * 
 * INSTRUCTIONS:
 * 1. Copy or rename this file to "config.h" in the same directory:
 *      cp config.example.h config.h
 * 2. Update WIFI_SSID, WIFI_PASSWORD, and BACKEND_URL with your local network settings.
 * 3. Never commit "config.h" with sensitive credentials to version control.
 *    (config.h is included in .gitignore)
 * ============================================================================
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// 1. Wi-Fi Network Credentials
// ============================================================================
// Note: ESP32 only supports 2.4 GHz Wi-Fi networks (802.11 b/g/n). 5 GHz is NOT supported.
#define WIFI_SSID         "YOUR_WIFI_SSID"
#define WIFI_PASSWORD     "YOUR_WIFI_PASSWORD"
#define WIFI_TIMEOUT_MS   15000UL   // Maximum time (ms) to wait for initial Wi-Fi connection
#define WIFI_RETRY_DELAY  5000UL    // Backoff delay (ms) between reconnection attempts

// ============================================================================
// 2. Backend Telemetry Ingestion API Endpoint
// ============================================================================
// IMPORTANT: Do NOT use "localhost" or "127.0.0.1"!
// "localhost" refers to the ESP32 itself. Use your workstation's LAN IP address
// (e.g. run 'ipconfig' on Windows or 'ifconfig' / 'ip a' on Linux/macOS).
// Example: "http://192.168.1.150:8000/api/telemetry/ingest"
#define BACKEND_URL       "http://192.168.1.100:8000/api/telemetry/ingest"
#define HTTP_TIMEOUT_MS   5000UL    // HTTP POST request timeout in milliseconds

// ============================================================================
// 3. Asset & Edge Attribution Identification
// ============================================================================
// Machine identifier must match or map to factory clients:
// - CNC_MILL_01   -> Maps to Fleet Client 1 (Milling Machine)
// - HYDR_PRESS_02 -> Maps to Fleet Client 2 (Hydraulic Press)
// - TURB_COMP_03  -> Maps to Fleet Client 3 (Turbo Compressor)
#define MACHINE_ID        "CNC_MILL_01"

// Telemetry source attribution strictly enforced by backend Pydantic schema:
// Allowed values: "ESP32_LIVE" or "SIMULATED"
#define SOURCE_TAG        "ESP32_LIVE"

// ============================================================================
// 4. Ingestion Cadence & Anti-Flooding Controls
// ============================================================================
// Interval between successive telemetry dispatches to the backend.
// Industrial standard for condition monitoring: 2000 ms to 5000 ms.
// Avoid setting below 1000 ms to prevent HTTP socket exhaustion and backend flooding.
#define TELEMETRY_INTERVAL_MS  2000UL

// ============================================================================
// 5. Hardware Pinout Definition
// ============================================================================
// MPU6050 (I2C Bus)
#define PIN_I2C_SDA       21        // ESP32 default I2C SDA
#define PIN_I2C_SCL       22        // ESP32 default I2C SCL
#define MPU6050_I2C_ADDR  0x68      // 0x68 (AD0 -> GND) or 0x69 (AD0 -> 3V3)

// DHT11 (Single-Wire Bus)
#define PIN_DHT_DATA      4         // GPIO 4 connected to DHT11 DATA pin
#define DHT_TYPE          DHT11     // Sensor model: DHT11 (or DHT22 if upgraded)

// Status Indicator LED
#define PIN_STATUS_LED    2         // ESP32 DevKit built-in blue LED on GPIO 2

// ============================================================================
// 6. Vibration Sampling & Feature Extraction Configuration
// ============================================================================
// In accordance with ISO 10816 / ISO 20816 vibration severity principles:
// A single-sample tilt or gravity deviation is NOT vibration RMS.
// Firmware collects a burst window of N samples at fixed rate, removes the static
// gravity vector (DC bias) per axis, and computes the dynamic Root Mean Square (RMS).
#define VIBRATION_SAMPLE_COUNT        100   // Number of accelerometer samples per calculation window
#define VIBRATION_SAMPLE_INTERVAL_US  10000 // Sample interval in microseconds (10,000 µs = 100 Hz sampling)

// ============================================================================
// 7. SNTP Real-Time Clock Synchronization
// ============================================================================
// If enabled, ESP32 synchronizes UTC time via NTP and formats ISO-8601 timestamps
// e.g., "2026-10-16T02:15:30Z". If disabled or before sync, timestamp is sent as
// JSON null, allowing the FastAPI backend to apply server-side UTC time.
#define ENABLE_NTP_SYNC   true
#define NTP_SERVER_1      "pool.ntp.org"
#define NTP_SERVER_2      "time.nist.gov"
#define GMT_OFFSET_SEC    0         // UTC offset in seconds (0 for ISO-8601 'Z')
#define DAYLIGHT_OFFSET_SEC 0

// ============================================================================
// 8. Optional Industrial Auxiliary Channels
// ============================================================================
// Complementary parameters supported by the backend schema:
// Enabled by default to provide stable baseline operating point values (motor
// current, rotational speed, hydraulic pressure) for the Digital Twin engine.
#define SEND_COMPLEMENTARY_FIELDS true
#define AUX_CURRENT_A     18.5f     // Motor load current in Amperes
#define AUX_RPM           3850.0f   // Spindle rotational speed
#define AUX_PRESSURE_BAR  6.2f      // Hydraulic/pneumatic pressure

#endif // CONFIG_H
