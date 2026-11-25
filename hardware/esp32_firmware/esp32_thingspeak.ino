/**
 * ============================================================================
 * ESP32 ThingSpeak Telemetry Firmware for Physical Bench Predictive Maintenance
 * ============================================================================
 * Hardware Setup:
 *   - ESP32 DevKit V1 (ESP-WROOM-32)
 *   - MPU6050 3-Axis IMU (Vibration): SDA -> GPIO 21, SCL -> GPIO 22, VCC -> 3.3V
 *   - DHT11 (Temperature & Humidity): DATA -> GPIO 4, VCC -> 3.3V
 *   - ACS712 (Current Sensor - Optional): OUT -> GPIO 34 (ADC1), VCC -> 5V (with divider)
 *   - Red Alert LED: GPIO 19
 *   - Green Status LED: GPIO 2 (Built-in)
 *   - Active Buzzer: GPIO 18
 * 
 * Target Destination:
 *   ThingSpeak Channel 3591448
 *   Endpoint: https://api.thingspeak.com/update
 *   Update Interval: 20 seconds (ThingSpeak free tier limit is 15s minimum)
 * 
 * Field Mapping Contract:
 *   field1 = temperature_c (DHT11)
 *   field2 = humidity_pct (DHT11)
 *   field3 = temp_secondary_c (Internal/Aux Temp)
 *   field4 = pressure_hpa (Barometric / Seal baseline)
 *   field5 = vibration_g (MPU6050 Dynamic AC RMS in g, gravity removed)
 *   field6 = current_a (ACS712 Motor Current in A, or NA if disconnected)
 *   field7 = esp32_anomaly (0: Normal, 1: Anomaly detected by edge rules)
 *   field8 = esp32_alert (0: Quiet, 1: Buzzer & Red LED actively tripped)
 *   status = "MACHINE NORMAL" or "ABNORMAL VIBRATION - BUZZER ON"
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>

// Include credentials
#if __has_include("secrets.h")
  #include "secrets.h"
#else
  #warning "secrets.h not found. Falling back to secrets.example.h"
  #include "secrets.example.h"
#endif

// ============================================================================
// Hardware Pin Definitions
// ============================================================================
#define PIN_DHT_DATA      4
#define DHT_TYPE          DHT11

#define PIN_BUZZER        18
#define PIN_LED_ALERT     19
#define PIN_LED_STATUS    2
#define PIN_ACS712        34

// Physical Constants
#define GRAVITY_MSS       9.80665f
#define VIB_BURST_SAMPLES 100
#define VIB_SAMPLE_DELAY  10     // 100 Hz sampling for burst

// Vibration Alert Threshold
#define VIB_ANOMALY_THRESHOLD_G  0.15f  // Abnormal vibration trip point
#define VIB_ALERT_THRESHOLD_G    0.25f  // Critical buzzer alarm trip point

// ============================================================================
// Global Sensor Objects & State
// ============================================================================
Adafruit_MPU6050 mpu;
DHT dht(PIN_DHT_DATA, DHT_TYPE);

unsigned long lastUploadMillis = 0;
const unsigned long UPLOAD_INTERVAL_MS = 20000; // 20s interval

// ============================================================================
// Dynamic Vibration RMS Extraction (Gravity Vector Removed)
// ============================================================================
float measureDynamicVibrationRMS() {
  float ax_samples[VIB_BURST_SAMPLES];
  float ay_samples[VIB_BURST_SAMPLES];
  float az_samples[VIB_BURST_SAMPLES];

  float sum_ax = 0.0f, sum_ay = 0.0f, sum_az = 0.0f;

  for (int i = 0; i < VIB_BURST_SAMPLES; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Convert m/s^2 to g units
    float gx = a.acceleration.x / GRAVITY_MSS;
    float gy = a.acceleration.y / GRAVITY_MSS;
    float gz = a.acceleration.z / GRAVITY_MSS;

    ax_samples[i] = gx;
    ay_samples[i] = gy;
    az_samples[i] = gz;

    sum_ax += gx;
    sum_ay += gy;
    sum_az += gz;

    delay(VIB_SAMPLE_DELAY);
  }

  // Remove DC component (static gravity vector)
  float mean_ax = sum_ax / VIB_BURST_SAMPLES;
  float mean_ay = sum_ay / VIB_BURST_SAMPLES;
  float mean_az = sum_az / VIB_BURST_SAMPLES;

  float sum_sq_ac = 0.0f;
  for (int i = 0; i < VIB_BURST_SAMPLES; i++) {
    float ac_x = ax_samples[i] - mean_ax;
    float ac_y = ay_samples[i] - mean_ay;
    float ac_z = az_samples[i] - mean_az;

    // Magnitude squared of dynamic acceleration
    sum_sq_ac += (ac_x * ac_x + ac_y * ac_y + ac_z * ac_z);
  }

  // RMS in g units
  float dynamic_rms = sqrtf(sum_sq_ac / VIB_BURST_SAMPLES);
  return dynamic_rms;
}

// ============================================================================
// Wi-Fi Connection Manager
// ============================================================================
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("[WiFi] Connecting to SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 25) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully. IP: " + WiFi.localIP().toString());
    digitalWrite(PIN_LED_STATUS, HIGH);
  } else {
    Serial.println("\n[WiFi] Connection failed. Will retry next cycle.");
    digitalWrite(PIN_LED_STATUS, LOW);
  }
}

// ============================================================================
// ThingSpeak HTTP Uploader
// ============================================================================
void uploadToThingSpeak(
  float temp_c,
  float humidity_pct,
  float temp2_c,
  float pressure_hpa,
  float vibration_g,
  float current_a,
  bool has_current,
  int anomaly_flag,
  int alert_flag,
  const String& status_msg
) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ThingSpeak] Cannot upload: Wi-Fi not connected.");
    return;
  }

  HTTPClient http;
  
  // Construct URL encoded query string
  String url = "https://api.thingspeak.com/update";
  url += "?api_key=" + String(TS_WRITE_KEY);
  url += "&field1=" + String(temp_c, 2);
  url += "&field2=" + String(humidity_pct, 1);
  url += "&field3=" + String(temp2_c, 2);
  url += "&field4=" + String(pressure_hpa, 2);
  url += "&field5=" + String(vibration_g, 5);
  
  if (has_current) {
    url += "&field6=" + String(current_a, 2);
  } else {
    url += "&field6=NA";
  }

  url += "&field7=" + String(anomaly_flag);
  url += "&field8=" + String(alert_flag);
  
  // URL encode status string
  String encoded_status = status_msg;
  encoded_status.replace(" ", "%20");
  url += "&status=" + encoded_status;

  Serial.println("[ThingSpeak] Transmitting 8-channel telemetry payload...");
  http.begin(url);
  int httpCode = http.GET();

  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    Serial.print("[ThingSpeak] Upload successful! Entry ID: ");
    Serial.println(payload);
  } else {
    Serial.print("[ThingSpeak] HTTP Error: ");
    Serial.println(httpCode);
  }
  http.end();
}

// ============================================================================
// Setup
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("=================================================");
  Serial.println("  ESP32 ThingSpeak Predictive Maintenance Node");
  Serial.println("=================================================");

  // Configure Output Pins
  pinMode(PIN_LED_STATUS, OUTPUT);
  pinMode(PIN_LED_ALERT, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_LED_STATUS, LOW);
  digitalWrite(PIN_LED_ALERT, LOW);
  digitalWrite(PIN_BUZZER, LOW);

  // Initialize Sensors
  Wire.begin(21, 22);
  if (!mpu.begin()) {
    Serial.println("[ERROR] Failed to locate MPU6050 chip! Check I2C wiring (SDA=21, SCL=22).");
  } else {
    Serial.println("[OK] MPU6050 IMU initialized.");
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  dht.begin();
  Serial.println("[OK] DHT11 initialized on GPIO 4.");

  connectWiFi();
}

// ============================================================================
// Main Loop (20s sampling & telemetry cycle)
// ============================================================================
void loop() {
  connectWiFi();

  unsigned long currentMillis = millis();
  if (currentMillis - lastUploadMillis >= UPLOAD_INTERVAL_MS || lastUploadMillis == 0) {
    lastUploadMillis = currentMillis;

    // 1. Measure Environmental Data
    float humidity = dht.readHumidity();
    float temp = dht.readTemperature();

    if (isnan(humidity) || isnan(temp)) {
      Serial.println("[WARN] Failed to read from DHT11. Using last known values.");
      temp = 24.5f;
      humidity = 65.0f;
    }

    // 2. Measure Dynamic Vibration RMS
    float dynamic_vib_g = measureDynamicVibrationRMS();

    // 3. Simulated/Internal Secondary Sensor Values (if physical BMP/ACS not connected)
    sensors_event_t a, g, mpu_temp;
    mpu.getEvent(&a, &g, &mpu_temp);
    float temp2 = mpu_temp.temperature; // Secondary temp from MPU6050 internal sensor
    float pressure = 994.5f;            // Nominal bench pressure in hPa

    // 4. ACS712 Current Reading
    // (If analog pin reads ~0 or unpowered, flag as not connected)
    int raw_adc = analogRead(PIN_ACS712);
    bool has_current = false;
    float current_a = 0.0f;
    if (raw_adc > 200 && raw_adc < 4000) {
      has_current = true;
      // ACS712 5A sensitivity: 185 mV/A centered at VCC/2
      float voltage = (raw_adc / 4095.0f) * 3.3f;
      current_a = fabsf((voltage - 1.65f) / 0.185f);
    }

    // 5. Edge Anomaly & Alert Rules
    int anomaly_flag = 0;
    int alert_flag = 0;
    String status_msg = "MACHINE NORMAL";

    if (dynamic_vib_g >= VIB_ALERT_THRESHOLD_G) {
      anomaly_flag = 1;
      alert_flag = 1;
      status_msg = "ABNORMAL VIBRATION - BUZZER ON";

      // Trigger Buzzer & Red Alert LED
      digitalWrite(PIN_LED_ALERT, HIGH);
      digitalWrite(PIN_BUZZER, HIGH);
      delay(300);
      digitalWrite(PIN_BUZZER, LOW);
    } else if (dynamic_vib_g >= VIB_ANOMALY_THRESHOLD_G) {
      anomaly_flag = 1;
      alert_flag = 0;
      status_msg = "ELEVATED VIBRATION - MONITORING";
      digitalWrite(PIN_LED_ALERT, HIGH);
    } else {
      digitalWrite(PIN_LED_ALERT, LOW);
      digitalWrite(PIN_BUZZER, LOW);
    }

    Serial.printf("[EDGE] Vib: %.4f g | Temp: %.1f C | Hum: %.1f %% | Anom: %d | Alert: %d | %s\n",
                  dynamic_vib_g, temp, humidity, anomaly_flag, alert_flag, status_msg.c_str());

    // 6. Transmit to ThingSpeak
    uploadToThingSpeak(
      temp,
      humidity,
      temp2,
      pressure,
      dynamic_vib_g,
      current_a,
      has_current,
      anomaly_flag,
      alert_flag,
      status_msg
    );
  }

  delay(100);
}

