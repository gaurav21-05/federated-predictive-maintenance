/**
 * ThingSpeak & Wi-Fi Credentials Template
 * 
 * Instructions:
 * 1. Copy this file to "secrets.h" in the same directory:
 *    cp secrets.example.h secrets.h
 * 2. Fill in your Wi-Fi SSID, password, ThingSpeak Channel ID, and Write API Key.
 * 3. Never commit "secrets.h" to git (it is listed in .gitignore).
 */

#ifndef SECRETS_H
#define SECRETS_H

// Wi-Fi Access Point Credentials
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"

// ThingSpeak Channel Configuration
#define TS_CHANNEL_ID   3591448
#define TS_WRITE_KEY    "YOUR_THINGSPEAK_WRITE_API_KEY"

#endif // SECRETS_H

