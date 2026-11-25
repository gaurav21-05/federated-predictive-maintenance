/**
 * ============================================================================
 * ESP32 Edge Telemetry Firmware for Industrial IoT Predictive Maintenance
 * ============================================================================
 * Target Hardware:
 *   - Microcontroller: ESP32 DevKit V1 (ESP-WROOM-32)
 *   - Inertial / Vibration: MPU6050 (3-Axis Accelerometer & 3-Axis Gyroscope)
 *   - Environment: DHT11 (Digital Relative Humidity & Ambient Temperature)
 * 
 * Hardware Wiring:
 *   - MPU6050 SDA  -> GPIO 21 (I2C Data)
 *   - MPU6050 SCL  -> GPIO 22 (I2C Clock)
 *   - MPU6050 VCC  -> 3V3
 *   - MPU6050 GND  -> GND
 *   - DHT11 DATA   -> GPIO 4  (Single-Wire Digital IO, 10kΩ pullup to 3.3V)
 *   - DHT11 VCC    -> 3V3
 *   - DHT11 GND    -> GND
 *   - Status LED   -> GPIO 2  (Built-in DevKit LED)
 * 
 * Backend Telemetry Ingestion Contract:
 *   - Endpoint: POST /api/telemetry/ingest
 *   - Source: "ESP32_LIVE"
 *   - Vibration Feature: Windowed dynamic acceleration Root-Mean-Square (RMS) in g
 *     with static Earth gravity vector (DC component) removed.
 *     (Single-sample gravity deviation is physically rejected per ISO 10816 / 20816).
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <time.h>

// Third-Party Sensor and Serialization Libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <ArduinoJson.h>

// Configuration: Load user config or fall back to example template
#if __has_include("config.h")
  #include "config.h"
#else
  #warning "config.h not found. Falling back to default configuration in config.example.h. Please copy config.example.h to config.h!"
  #include "config.example.h"
#endif

// ============================================================================
// Global Sensor & Hardware Objects
// ============================================================================
Adafruit_MPU6050 mpu;
DHT dht(PIN_DHT_DATA, DHT_TYPE);

// Physical Constants
#define GRAVITY_MSS 9.80665f      // Earth standard gravitational acceleration in m/s^2
#define RAD_TO_DEG  57.2957795f   // 180 / PI for angular velocity conversion

// Runtime Telemetry & Health Tracking
struct FirmwareDiagnostics {
  uint32_t sequenceNumber;
  uint32_t packetsAttempted;
  uint32_t packetsAcked;
  uint32_t consecutiveFailures;
  bool mpuOnline;
  bool dhtOnline;
  bool ntpSynced;
  float lastValidTempC;
  float lastValidHumidityPct;
  unsigned long lastTelemetryMillis;
  unsigned long lastWifiRetryMillis;
};

FirmwareDiagnostics diagnostics = {
  .sequenceNumber = 0,
  .packetsAttempted = 0,
  .packetsAcked = 0,
  .consecutiveFailures = 0,
  .mpuOnline = false,
  .dhtOnline = false,
  .ntpSynced = false,
  .lastValidTempC = 25.0f,
  .lastValidHumidityPct = 50.0f,
  .lastTelemetryMillis = 0,
  .lastWifiRetryMillis = 0
};

// ============================================================================
// Forward Declarations
// ============================================================================
void printSystemBanner();
bool initializeMPU6050();
bool initializeDHT();
void initializeWiFi();
void ensureWiFiConnection();
void initializeSNTP();
bool getIso8601Timestamp(char* buffer, size_t maxLen);
bool sampleVibrationAndKinematics(
  float &outVibrationRmsG,
  float &outPeakG,
  float &outAccX, float &outAccY, float &outAccZ,
  float &outGyroX, float &outGyroY, float &outGyroZ
);
bool readEnvironment(float &outTempC, float &outHumidityPct);
void dispatchTelemetry();

// ============================================================================
// Setup Routine
// ============================================================================
void setup() {
  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  // Initialize high-speed UART for diagnostics
  Serial.begin(115200);
  delay(1000);

  printSystemBanner();

  // 1. Initialize I2C Bus for MPU6050
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000); // 400 kHz Fast-Mode I2C for high-speed vibration burst sampling
  diagnostics.mpuOnline = initializeMPU6050();

  // 2. Initialize DHT11 Sensor
  diagnostics.dhtOnline = initializeDHT();

  // 3. Connect to Wi-Fi
  initializeWiFi();

  // 4. Initialize SNTP for ISO-8601 UTC Timestamps
  #if ENABLE_NTP_SYNC
    if (WiFi.status() == WL_CONNECTED) {
      initializeSNTP();
    }
  #endif

  Serial.println(F("[SYSTEM] Initialization complete. Entering autonomous monitoring loop.\n"));
}

// ============================================================================
// Main Execution Loop
// ============================================================================
void loop() {
  // Maintain Wi-Fi connectivity without blocking sensor operations
  ensureWiFiConnection();

  // Enforce telemetry transmission cadence (anti-flooding control)
  unsigned long currentMillis = millis();
  if (currentMillis - diagnostics.lastTelemetryMillis >= TELEMETRY_INTERVAL_MS) {
    diagnostics.lastTelemetryMillis = currentMillis;
    dispatchTelemetry();
  }

  // Brief yield for RTOS scheduler background housekeeping
  delay(10);
}

// ============================================================================
// Hardware Initialization Functions
// ============================================================================
bool initializeMPU6050() {
  Serial.print(F("[INIT] Probing MPU6050 on I2C (SDA=GPIO"));
  Serial.print(PIN_I2C_SDA);
  Serial.print(F(", SCL=GPIO"));
  Serial.print(PIN_I2C_SCL);
  Serial.print(F(", Addr=0x"));
  Serial.print(MPU6050_I2C_ADDR, HEX);
  Serial.println(F(")..."));

  if (!mpu.begin(MPU6050_I2C_ADDR, &Wire)) {
    Serial.println(F("[ERROR] MPU6050 chip detection failed!"));
    Serial.println(F("        Check VCC (3.3V), GND, SDA (21), SCL (22), and AD0 pull-down."));
    return false;
  }

  // Industrial sensor configuration:
  // - Accelerometer ±4g range: optimal dynamic range for industrial spindle/motor vibrations.
  // - Gyroscope ±500 deg/s range: captures rapid rotational transients without saturation.
  // - Digital Low Pass Filter (DLPF): 44 Hz bandwidth to suppress high-frequency electrical switching noise.
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);

  Serial.println(F("[INIT] MPU6050 initialized successfully (Range: ±4g / ±500°/s, DLPF: 44Hz)."));
  return true;
}

bool initializeDHT() {
  Serial.print(F("[INIT] Initializing DHT11 on GPIO "));
  Serial.println(PIN_DHT_DATA);
  dht.begin();

  // Perform a test read (DHT11 requires ~1-2 seconds after power-on to stabilize)
  delay(1200);
  float testT = dht.readTemperature();
  float testH = dht.readHumidity();

  if (isnan(testT) || isnan(testH)) {
    Serial.println(F("[WARN] Initial DHT11 probe returned NaN (sensor warming up or check pullup resistor)."));
    return false;
  }

  diagnostics.lastValidTempC = testT;
  diagnostics.lastValidHumidityPct = testH;
  Serial.printf("[INIT] DHT11 ready. Initial reading: %.1f °C, %.1f %% RH\n", testT, testH);
  return true;
}

void initializeWiFi() {
  Serial.printf("[WIFI] Connecting to SSID: '%s' (2.4 GHz)...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < WIFI_TIMEOUT_MS) {
    digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    Serial.println(F("[WIFI] Connected successfully!"));
    Serial.printf("[WIFI] IP Address:    %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[WIFI] Gateway:       %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("[WIFI] MAC Address:   %s\n", WiFi.macAddress().c_str());
    Serial.printf("[WIFI] Signal (RSSI): %d dBm\n", WiFi.RSSI());
  } else {
    digitalWrite(PIN_STATUS_LED, LOW);
    Serial.println(F("[WIFI] Connection timed out. Operating in offline measurement mode; will retry periodically."));
  }
}

void ensureWiFiConnection() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  unsigned long currentMillis = millis();
  if (currentMillis - diagnostics.lastWifiRetryMillis >= WIFI_RETRY_DELAY) {
    diagnostics.lastWifiRetryMillis = currentMillis;
    Serial.println(F("[WIFI] Connection lost. Attempting background reconnect..."));
    digitalWrite(PIN_STATUS_LED, LOW);
    WiFi.disconnect();
    WiFi.reconnect();
  }
}

void initializeSNTP() {
  Serial.printf("[NTP] Configuring SNTP (Servers: %s, %s)...\n", NTP_SERVER_1, NTP_SERVER_2);
  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER_1, NTP_SERVER_2);

  // Allow up to 3 seconds for initial sync
  struct tm timeinfo;
  for (int i = 0; i < 6; i++) {
    if (getLocalTime(&timeinfo, 500)) {
      diagnostics.ntpSynced = true;
      char timeBuffer[32];
      strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
      Serial.printf("[NTP] Time synchronized successfully: %s (UTC)\n", timeBuffer);
      return;
    }
  }
  Serial.println(F("[WARN] NTP sync pending. Telemetry will emit null timestamp (server will apply UTC)."));
}

bool getIso8601Timestamp(char* buffer, size_t maxLen) {
  if (!diagnostics.ntpSynced) {
    // Check if NTP has synchronized in the background
    time_t now = time(nullptr);
    if (now > 1600000000) { // Valid timestamp past 2020
      diagnostics.ntpSynced = true;
    } else {
      return false;
    }
  }

  time_t now;
  struct tm timeinfo;
  time(&now);
  gmtime_r(&now, &timeinfo);
  strftime(buffer, maxLen, "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
  return true;
}

// ============================================================================
// Vibration Feature Extraction & Kinematics
// ============================================================================
/**
 * Computes calibrated dynamic vibration RMS following ISO 10816 / ISO 20816 principles:
 * 
 * 1. Physical Principle:
 *    A stationary accelerometer experiences Earth's static gravitational acceleration:
 *      ||a_static|| = sqrt(ax^2 + ay^2 + az^2) ≈ 1.0g
 *    A single instantaneous sample deviation | ||a|| - 1.0g | is NOT vibration RMS.
 *    It merely reflects static sensor tilt angle, mounting orientation, and DC sensor bias.
 * 
 * 2. Multi-Sample Window Acquisition:
 *    We sample N readings (e.g. N = 100) at high frequency (100 Hz, T = 10 ms).
 * 
 * 3. Static Gravity (DC Component) Removal:
 *    Compute mean acceleration vector across the window:
 *      mean_x = (1/N) * sum(ax[k]), mean_y = (1/N) * sum(ay[k]), mean_z = (1/N) * sum(az[k])
 *    The dynamic (AC) oscillatory vibration for sample k is:
 *      dyn_x[k] = ax[k] - mean_x
 *      dyn_y[k] = ay[k] - mean_y
 *      dyn_z[k] = az[k] - mean_z
 * 
 * 4. Dynamic Vibration RMS Severity (in g):
 *    RMS_x = sqrt( (1/N) * sum(dyn_x[k]^2) )
 *    RMS_y = sqrt( (1/N) * sum(dyn_y[k]^2) )
 *    RMS_z = sqrt( (1/N) * sum(dyn_z[k]^2) )
 *    Total Tri-Axial Vibration Severity:
 *      vibration_g = sqrt( RMS_x^2 + RMS_y^2 + RMS_z^2 )
 */
bool sampleVibrationAndKinematics(
  float &outVibrationRmsG,
  float &outPeakG,
  float &outAccX, float &outAccY, float &outAccZ,
  float &outGyroX, float &outGyroY, float &outGyroZ
) {
  if (!diagnostics.mpuOnline) {
    // Attempt re-init if previously offline
    diagnostics.mpuOnline = initializeMPU6050();
    if (!diagnostics.mpuOnline) {
      // Return safe clamped baseline if hardware is unavailable
      outVibrationRmsG = 0.0f;
      outPeakG = 0.0f;
      outAccX = 0.0f; outAccY = 0.0f; outAccZ = 1.0f;
      outGyroX = 0.0f; outGyroY = 0.0f; outGyroZ = 0.0f;
      return false;
    }
  }

  const int N = VIBRATION_SAMPLE_COUNT;
  float ax[N], ay[N], az[N];
  float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f;
  sensors_event_t a, g, temp;

  // High-frequency burst sampling window
  for (int i = 0; i < N; i++) {
    unsigned long sampleStart = micros();

    mpu.getEvent(&a, &g, &temp);

    // Convert Adafruit MPU6050 m/s^2 output to g-units
    ax[i] = a.acceleration.x / GRAVITY_MSS;
    ay[i] = a.acceleration.y / GRAVITY_MSS;
    az[i] = a.acceleration.z / GRAVITY_MSS;

    sumX += ax[i];
    sumY += ay[i];
    sumZ += az[i];

    // Maintain stable sampling rate (e.g. 10,000 µs = 100 Hz)
    long elapsedUs = micros() - sampleStart;
    long remainingUs = (long)VIBRATION_SAMPLE_INTERVAL_US - elapsedUs;
    if (remainingUs > 50) {
      delayMicroseconds(remainingUs);
    }
  }

  // 1. Calculate Mean (DC gravity component & mounting tilt)
  float meanX = sumX / (float)N;
  float meanY = sumY / (float)N;
  float meanZ = sumZ / (float)N;

  // 2. Calculate dynamic AC power & peak dynamic magnitude
  float varX = 0.0f, varY = 0.0f, varZ = 0.0f;
  float maxDynamicMagSq = 0.0f;

  for (int i = 0; i < N; i++) {
    float dynX = ax[i] - meanX;
    float dynY = ay[i] - meanY;
    float dynZ = az[i] - meanZ;

    varX += dynX * dynX;
    varY += dynY * dynY;
    varZ += dynZ * dynZ;

    float magSq = dynX * dynX + dynY * dynY + dynZ * dynZ;
    if (magSq > maxDynamicMagSq) {
      maxDynamicMagSq = magSq;
    }
  }

  float rmsX = sqrtf(varX / (float)N);
  float rmsY = sqrtf(varY / (float)N);
  float rmsZ = sqrtf(varZ / (float)N);

  // Total Tri-Axial Dynamic Vibration RMS (g)
  outVibrationRmsG = sqrtf(rmsX * rmsX + rmsY * rmsY + rmsZ * rmsZ);
  outPeakG = sqrtf(maxDynamicMagSq);

  // Boundary clamp against backend schema [0.0, 50.0]
  if (outVibrationRmsG < 0.0f) outVibrationRmsG = 0.0f;
  if (outVibrationRmsG > 50.0f) outVibrationRmsG = 50.0f;

  // Provide instantaneous orientation acceleration (latest sample or window mean)
  outAccX = meanX;
  outAccY = meanY;
  outAccZ = meanZ;

  // Convert angular velocities from rad/s to deg/s
  outGyroX = g.gyro.x * RAD_TO_DEG;
  outGyroY = g.gyro.y * RAD_TO_DEG;
  outGyroZ = g.gyro.z * RAD_TO_DEG;

  // Clamp kinematics to schema bounds: acc [-50.0, 50.0], gyro [-2000.0, 2000.0]
  outAccX = constrain(outAccX, -50.0f, 50.0f);
  outAccY = constrain(outAccY, -50.0f, 50.0f);
  outAccZ = constrain(outAccZ, -50.0f, 50.0f);
  outGyroX = constrain(outGyroX, -2000.0f, 2000.0f);
  outGyroY = constrain(outGyroY, -2000.0f, 2000.0f);
  outGyroZ = constrain(outGyroZ, -2000.0f, 2000.0f);

  return true;
}

// ============================================================================
// Environmental Sensing (DHT11)
// ============================================================================
bool readEnvironment(float &outTempC, float &outHumidityPct) {
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (isnan(t) || isnan(h)) {
    Serial.println(F("[WARN] DHT11 reading failed (NaN). Retaining previous valid reading."));
    outTempC = diagnostics.lastValidTempC;
    outHumidityPct = diagnostics.lastValidHumidityPct;
    return false;
  }

  // Schema sanity bounds: temp [-40.0, 150.0], humidity [0.0, 100.0]
  outTempC = constrain(t, -40.0f, 150.0f);
  outHumidityPct = constrain(h, 0.0f, 100.0f);

  diagnostics.lastValidTempC = outTempC;
  diagnostics.lastValidHumidityPct = outHumidityPct;
  diagnostics.dhtOnline = true;
  return true;
}

// ============================================================================
// Telemetry Ingestion Dispatcher
// ============================================================================
void dispatchTelemetry() {
  diagnostics.sequenceNumber++;
  diagnostics.packetsAttempted++;

  // 1. Read Sensors
  float vibRmsG = 0.0f, peakG = 0.0f;
  float ax = 0.0f, ay = 0.0f, az = 1.0f;
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
  sampleVibrationAndKinematics(vibRmsG, peakG, ax, ay, az, gx, gy, gz);

  float tempC = 25.0f, humidityPct = 50.0f;
  readEnvironment(tempC, humidityPct);

  // 2. Generate ISO-8601 UTC Timestamp (if synchronized)
  char isoTimestamp[32];
  bool hasTimestamp = getIso8601Timestamp(isoTimestamp, sizeof(isoTimestamp));

  // 3. Serialize JSON Payload (Compatible with ArduinoJson v6 and v7)
  #if ARDUINOJSON_VERSION_MAJOR >= 7
    JsonDocument doc;
  #else
    StaticJsonDocument<512> doc;
  #endif

  doc["machine_id"]     = MACHINE_ID;
  doc["temperature_c"]  = round(tempC * 10.0f) / 10.0f;
  doc["humidity_pct"]   = round(humidityPct * 10.0f) / 10.0f;
  doc["acceleration_x"] = round(ax * 1000.0f) / 1000.0f;
  doc["acceleration_y"] = round(ay * 1000.0f) / 1000.0f;
  doc["acceleration_z"] = round(az * 1000.0f) / 1000.0f;
  doc["gyro_x"]         = round(gx * 100.0f) / 100.0f;
  doc["gyro_y"]         = round(gy * 100.0f) / 100.0f;
  doc["gyro_z"]         = round(gz * 100.0f) / 100.0f;
  doc["vibration_g"]    = round(vibRmsG * 1000.0f) / 1000.0f;
  doc["source"]         = SOURCE_TAG;

  if (hasTimestamp) {
    doc["timestamp"] = isoTimestamp;
  } else {
    // Setting explicit null instructs backend FastAPI to apply server UTC
    doc["timestamp"] = nullptr;
  }

  #if SEND_COMPLEMENTARY_FIELDS
    doc["current_a"]     = AUX_CURRENT_A;
    doc["rpm"]           = AUX_RPM;
    doc["pressure_bar"]  = AUX_PRESSURE_BAR;
  #endif

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // 4. Print Local Diagnostics to Serial Monitor
  Serial.printf("\n==================== [CYCLE #%u] ====================\n", diagnostics.sequenceNumber);
  Serial.printf("Asset ID:        %s | Source: %s\n", MACHINE_ID, SOURCE_TAG);
  Serial.printf("Timestamp:       %s\n", hasTimestamp ? isoTimestamp : "NULL (Server UTC fallback)");
  Serial.printf("Temperature:     %.1f °C  | Humidity: %.1f %% RH\n", tempC, humidityPct);
  Serial.printf("Static Tilt Acc: X=%.3fg, Y=%.3fg, Z=%.3fg\n", ax, ay, az);
  Serial.printf("Angular Vel:     X=%.2f°/s, Y=%.2f°/s, Z=%.2f°/s\n", gx, gy, gz);
  Serial.printf("Vibration (RMS): %.4f g (Peak Dynamic: %.4f g)\n", vibRmsG, peakG);
  Serial.printf("Free Heap:       %u bytes\n", ESP.getFreeHeap());
  Serial.printf("Outgoing JSON:   %s\n", jsonPayload.c_str());

  // 5. Transmit via HTTP POST to Ingestion API
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("[DISPATCH ERROR] Wi-Fi is currently disconnected. Payload queued/dropped."));
    diagnostics.consecutiveFailures++;
    return;
  }

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  unsigned long requestStart = millis();
  int httpResponseCode = http.POST(jsonPayload);
  unsigned long latencyMs = millis() - requestStart;

  // Requirement 13: Never report a successful upload when the HTTP request failed.
  if (httpResponseCode >= 200 && httpResponseCode < 300) {
    diagnostics.packetsAcked++;
    diagnostics.consecutiveFailures = 0;
    digitalWrite(PIN_STATUS_LED, HIGH);

    String responseBody = http.getString();
    Serial.printf("[HTTP SUCCESS] Status %d OK in %lu ms\n", httpResponseCode, latencyMs);
    Serial.printf("[BACKEND ACK]  Response: %s\n", responseBody.c_str());
  } else if (httpResponseCode > 0) {
    // Server responded with an HTTP error code (e.g., 400, 422, 500)
    diagnostics.consecutiveFailures++;
    digitalWrite(PIN_STATUS_LED, LOW);

    String errorBody = http.getString();
    Serial.printf("[HTTP ERROR] Server rejected upload with status code: %d (in %lu ms)\n", httpResponseCode, latencyMs);
    Serial.printf("[HTTP ERROR] Server details: %s\n", errorBody.c_str());
    Serial.println(F("[AUDIT NOTICE] Failed upload marked as REJECTED. Packet NOT acknowledged."));
  } else {
    // Network / Socket level transport error (e.g., connection refused, timeout)
    diagnostics.consecutiveFailures++;
    digitalWrite(PIN_STATUS_LED, LOW);

    Serial.printf("[TRANSPORT ERROR] HTTP POST failed: %s (code %d) after %lu ms\n", 
                  http.errorToString(httpResponseCode).c_str(), httpResponseCode, latencyMs);
    Serial.printf("                  Target Endpoint: %s\n", BACKEND_URL);
    Serial.println(F("                  Verify backend is running, port is open, and IP is reachable from ESP32."));
  }

  // Always release connection resources
  http.end();

  Serial.printf("Transmission Stats: %u attempted, %u acknowledged (Success Rate: %.1f%%)\n",
                diagnostics.packetsAttempted,
                diagnostics.packetsAcked,
                (diagnostics.packetsAttempted > 0) ? 
                  ((float)diagnostics.packetsAcked / (float)diagnostics.packetsAttempted * 100.0f) : 0.0f);
  Serial.println(F("====================================================\n"));
}

// ============================================================================
// Diagnostic Banner
// ============================================================================
void printSystemBanner() {
  Serial.println(F("\n**************************************************************"));
  Serial.println(F("*  IIoT Federated Predictive Maintenance & Digital Twin      *"));
  Serial.println(F("*  ESP32 Edge Sensor Telemetry Firmware v1.0.0               *"));
  Serial.println(F("**************************************************************"));
  Serial.printf ("*  Machine ID:     %s\n", MACHINE_ID);
  Serial.printf ("*  Source Tag:     %s\n", SOURCE_TAG);
  Serial.printf ("*  Ingest API:     %s\n", BACKEND_URL);
  Serial.printf ("*  Cadence:        %lu ms\n", TELEMETRY_INTERVAL_MS);
  Serial.printf ("*  Vib Window:     %d samples @ %u µs (~%.0f Hz)\n", 
                 VIBRATION_SAMPLE_COUNT, VIBRATION_SAMPLE_INTERVAL_US, 
                 1000000.0f / (float)VIBRATION_SAMPLE_INTERVAL_US);
  Serial.printf ("*  Pins:           MPU6050 SDA=%d, SCL=%d | DHT11 DATA=%d\n", 
                 PIN_I2C_SDA, PIN_I2C_SCL, PIN_DHT_DATA);
  Serial.println(F("**************************************************************\n"));
}
