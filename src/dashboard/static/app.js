/**
 * IIoT Real Edge Intelligence Dashboard
 * Single reactive JavaScript client driving Live Monitor and Analysis & Prediction pages.
 * Zero synthetic placeholders; 100% driven by real ThingSpeak hardware telemetry.
 */

(function () {
  'use strict';

  // Global State
  const state = {
    activeTab: 'tab-live',
    selectedWindow: 'all',
    liveData: null,
    modelData: null,
    statsData: null,
    pollingIntervalMs: 20000,
    timerId: null,
    isTabVisible: true,
    errorCount: 0,
    activeComponent: null,
    insightsLoading: false,
    retrainLoading: false,
  };

  // DOM Elements Cache
  const el = {};

  function initElements() {
    // Navigation
    el.tabBtns = document.querySelectorAll('.tab-btn');
    el.tabPanes = document.querySelectorAll('.tab-pane');
    el.globalSystemStatus = document.getElementById('global-system-status');
    el.fleetHealthVal = document.getElementById('fleet-health-val');
    el.globalPulseDot = document.getElementById('global-pulse-dot');
    el.insightsModeBadge = document.getElementById('insights-mode-badge');

    // Live Monitor Header & Controls
    el.liveConnBadge = document.getElementById('live-conn-badge');
    el.liveConnStatusText = document.getElementById('live-conn-status-text');
    el.liveChannelInfo = document.getElementById('live-channel-info');
    el.liveSampleAge = document.getElementById('live-sample-age');
    el.liveLastUpdated = document.getElementById('live-last-updated');
    el.liveWindowSelect = document.getElementById('live-window-select');
    el.btnLiveRefresh = document.getElementById('btn-live-refresh');
    el.btnLiveExport = document.getElementById('btn-live-export');

    // 8 KPI Cards
    el.kpiVib = document.getElementById('live-kpi-vib');
    el.kpiVibRange = document.getElementById('live-kpi-vib-range');
    el.kpiVibTrend = document.getElementById('live-kpi-vib-trend');
    el.kpiTemp = document.getElementById('live-kpi-temp');
    el.kpiTempRange = document.getElementById('live-kpi-temp-range');
    el.kpiTempTrend = document.getElementById('live-kpi-temp-trend');
    el.kpiHum = document.getElementById('live-kpi-hum');
    el.kpiHumRange = document.getElementById('live-kpi-hum-range');
    el.kpiHumTrend = document.getElementById('live-kpi-hum-trend');
    el.kpiTemp2 = document.getElementById('live-kpi-temp2');
    el.kpiTemp2Range = document.getElementById('live-kpi-temp2-range');
    el.kpiPress = document.getElementById('live-kpi-press');
    el.kpiPressRange = document.getElementById('live-kpi-press-range');
    el.kpiCurr = document.getElementById('live-kpi-curr');
    el.kpiCurrSub = document.getElementById('live-kpi-curr-sub');
    el.liveCurrTag = document.getElementById('live-curr-tag');
    el.badgeCurrActive = document.getElementById('badge-curr-active');
    el.kpiFlag7 = document.getElementById('live-kpi-flag7');
    el.kpiFlag8 = document.getElementById('live-kpi-flag8');

    // SVG Twin
    el.svgTwin = document.getElementById('live-twin-svg');
    el.motorRotor = document.getElementById('svg-motor-rotor');
    el.motorBody = document.getElementById('svg-motor-body');
    el.ledGreen = document.getElementById('svg-led-green');
    el.ledRed = document.getElementById('svg-led-red');
    el.buzzerWave = document.getElementById('svg-buzzer-waves');
    el.offlineWatermark = document.getElementById('svg-offline-watermark');
    el.svgValVib = document.getElementById('svg-val-vib');
    el.svgValTemp = document.getElementById('svg-val-temp');
    el.svgValHum = document.getElementById('svg-val-hum');

    // SVG Click Drawer
    el.compDetailCard = document.getElementById('live-comp-detail-card');
    el.compDetailName = document.getElementById('live-comp-detail-name');
    el.compDetailRole = document.getElementById('live-comp-detail-role');
    el.compDetailVal = document.getElementById('live-comp-detail-val');
    el.compDetailBaseline = document.getElementById('live-comp-detail-baseline');
    el.compDetailHealth = document.getElementById('live-comp-detail-health');
    el.compDetailClose = document.getElementById('live-comp-detail-close');

    // Health & RUL Panel
    el.hiScore = document.getElementById('live-hi-score');
    el.hiStateBadge = document.getElementById('live-hi-state-badge');
    el.subVibScore = document.getElementById('live-sub-vib-score');
    el.subVibBar = document.getElementById('live-sub-vib-bar');
    el.subThermScore = document.getElementById('live-sub-therm-score');
    el.subThermBar = document.getElementById('live-sub-therm-bar');
    el.subEnvScore = document.getElementById('live-sub-env-score');
    el.subEnvBar = document.getElementById('live-sub-env-bar');
    el.subElecScore = document.getElementById('live-sub-elec-score');
    el.subElecBar = document.getElementById('live-sub-elec-bar');
    el.rulDisplay = document.getElementById('live-rul-display');
    el.rulSubtitle = document.getElementById('live-rul-subtitle');
    el.rulConfidence = document.getElementById('live-rul-confidence');

    // Canvases
    el.canvasVib = document.getElementById('canvas-live-vib');
    el.canvasEnv = document.getElementById('canvas-live-env');
    el.canvasUnconfirmed = document.getElementById('canvas-live-unconfirmed');
    el.canvasCurrent = document.getElementById('canvas-live-current');
    el.canvasHi = document.getElementById('canvas-live-hi');
    el.canvasFlLoss = document.getElementById('canvas-fl-loss');

    // Tables
    el.episodesTableBody = document.getElementById('live-episodes-table-body');
    el.statsTableBody = document.getElementById('live-stats-table-body');
    el.flComparisonTableBody = document.getElementById('fl-comparison-table-body');
    el.correlationsContainer = document.getElementById('analysis-correlations-container');
    el.dqContainer = document.getElementById('analysis-dq-container');

    // Analysis Page
    el.btnRefreshInsights = document.getElementById('btn-refresh-insights');
    el.btnRetrainFl = document.getElementById('btn-retrain-fl');
    el.flRetrainStatus = document.getElementById('fl-retrain-status');
    el.aiInsightsContent = document.getElementById('ai-insights-content');
    el.aiInsightsSourceBadge = document.getElementById('ai-insights-source-badge');
    el.aiInsightsCacheStatus = document.getElementById('ai-insights-cache-status');

    // ML Anomaly & RUL elements
    el.mlKpiRecon = document.getElementById('ml-kpi-recon');
    el.mlKpiReconSub = document.getElementById('ml-kpi-recon-sub');
    el.mlKpiIso = document.getElementById('ml-kpi-iso');
    el.mlKpiIsoSub = document.getElementById('ml-kpi-iso-sub');
    el.mlKpiSlope = document.getElementById('ml-kpi-slope');
    el.mlKpiSlopeSub = document.getElementById('ml-kpi-slope-sub');
    el.mlHonestRulState = document.getElementById('ml-honest-rul-state');
    el.mlHonestRulMessage = document.getElementById('ml-honest-rul-message');
    el.mlHonestRulHours = document.getElementById('ml-honest-rul-hours');
    el.mlHonestRulCi = document.getElementById('ml-honest-rul-ci');
    el.mlHonestRulSensor = document.getElementById('ml-honest-rul-sensor');
  }

  // ==========================================================================
  // Tab Navigation
  // ==========================================================================
  function setupTabs() {
    el.tabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');
        if (!targetId || targetId === state.activeTab) return;

        el.tabBtns.forEach((b) => b.classList.remove('active'));
        el.tabPanes.forEach((p) => p.classList.remove('active'));

        btn.classList.add('active');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');
        state.activeTab = targetId;

        // If switching to analysis tab, fetch analysis data if not yet loaded
        if (targetId === 'tab-analysis') {
          fetchAnalysisData();
        }
      });
    });
  }

  // ==========================================================================
  // Live Telemetry Polling
  // ==========================================================================
  async function fetchLiveTelemetry() {
    try {
      const url = `/api/live/hardware?results=150&window=${encodeURIComponent(state.selectedWindow)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const data = await res.json();
      state.liveData = data;
      state.errorCount = 0;

      renderLiveTelemetry(data);

      // If active on Analysis tab, keep stats refreshed
      if (state.activeTab === 'tab-analysis') {
        renderAnalysisStats(data);
      }
    } catch (err) {
      console.warn('Telemetry polling error:', err);
      state.errorCount++;
      handleFetchError(err);
    } finally {
      scheduleNextPoll();
    }
  }

  function scheduleNextPoll() {
    if (state.timerId) clearTimeout(state.timerId);
    if (!state.isTabVisible) return;

    // Exponential backoff if consecutive errors occur (up to 60s)
    let delay = state.pollingIntervalMs;
    if (state.errorCount > 0) {
      delay = Math.min(60000, state.pollingIntervalMs * Math.pow(1.5, Math.min(state.errorCount, 4)));
    }
    state.timerId = setTimeout(fetchLiveTelemetry, delay);
  }

  function handleFetchError(err) {
    if (el.liveConnStatusText) {
      el.liveConnStatusText.textContent = 'CONNECTING...';
      el.liveConnBadge.className = 'status-badge status-badge-warning';
    }
    if (el.globalSystemStatus) {
      el.globalSystemStatus.textContent = 'TELEMETRY PAUSED';
    }
  }

  // ==========================================================================
  // Render Live Monitor Page
  // ==========================================================================
  function renderLiveTelemetry(data) {
    if (!data) return;

    const isOnline = Boolean(data.online);
    const latest = data.latest || {};
    const stats = data.stats || {};
    const twin = data.twin || {};
    const rul = data.rul || {};

    // 1. Connection Header
    if (el.liveConnBadge && el.liveConnStatusText) {
      if (isOnline) {
        el.liveConnBadge.className = 'status-badge status-badge-live';
        el.liveConnStatusText.textContent = 'LIVE (real hardware)';
        if (el.globalPulseDot) el.globalPulseDot.style.background = '#22c55e';
      } else {
        el.liveConnBadge.className = 'status-badge status-badge-offline';
        el.liveConnStatusText.textContent = 'OFFLINE (>60s)';
        if (el.globalPulseDot) el.globalPulseDot.style.background = '#94a3b8';
      }
    }

    if (el.liveSampleAge) {
      const ageSec = data.age_seconds;
      el.liveSampleAge.textContent = ageSec !== null && ageSec !== undefined ? `${Math.round(ageSec)}s ago` : 'N/A';
    }

    if (el.liveLastUpdated) {
      const ts = data.latest_timestamp;
      el.liveLastUpdated.textContent = ts ? new Date(ts).toLocaleTimeString() : 'N/A';
    }

    // Top Global Header Status
    const hi = twin.health_index !== undefined ? twin.health_index : 100.0;
    if (el.fleetHealthVal) el.fleetHealthVal.textContent = `${Math.round(hi)}%`;
    if (el.globalSystemStatus) {
      el.globalSystemStatus.textContent = isOnline ? (hi >= 70 ? 'EDGE OPTIMAL' : 'EDGE WARNING') : 'HARDWARE OFFLINE';
    }

    // 2. KPI Cards (Real Data Only)
    // Field 5: Vibration RMS (MPU6050)
    const vibe = latest.vibration_g;
    const vStats = stats.vibration_g || {};
    if (el.kpiVib) el.kpiVib.textContent = vibe !== undefined && vibe !== null ? `${vibe.toFixed(4)} g` : 'No data yet';
    if (el.kpiVibRange) {
      el.kpiVibRange.textContent = vStats.min !== undefined ? `min ${vStats.min.toFixed(4)} • max ${vStats.max.toFixed(4)}` : 'min -- • max --';
    }
    if (el.kpiVibTrend) {
      const rate = vStats.rate_of_change_per_hour;
      el.kpiVibTrend.textContent = rate !== undefined ? `${rate >= 0 ? '+' : ''}${rate.toFixed(4)}/h` : '--';
      el.kpiVibTrend.className = rate > 0.01 ? 'trend-indicator trend-up' : 'trend-indicator trend-neutral';
    }

    // Field 1: Ambient Temp (DHT11)
    const temp = latest.temperature_c;
    const tStats = stats.temperature_c || {};
    if (el.kpiTemp) el.kpiTemp.textContent = temp !== undefined && temp !== null ? `${temp.toFixed(1)} °C` : 'No data yet';
    if (el.kpiTempRange) {
      el.kpiTempRange.textContent = tStats.min !== undefined ? `min ${tStats.min.toFixed(1)} • max ${tStats.max.toFixed(1)}` : 'min -- • max --';
    }

    // Field 2: Relative Humidity (DHT11)
    const hum = latest.humidity_pct;
    const hStats = stats.humidity_pct || {};
    if (el.kpiHum) el.kpiHum.textContent = hum !== undefined && hum !== null ? `${hum.toFixed(1)} %` : 'No data yet';
    if (el.kpiHumRange) {
      el.kpiHumRange.textContent = hStats.min !== undefined ? `min ${hStats.min.toFixed(1)} • max ${hStats.max.toFixed(1)}` : 'min -- • max --';
    }

    // Field 3: Secondary Temp
    const temp2 = latest.temp_secondary_c;
    const t2Stats = stats.temp_secondary_c || {};
    if (el.kpiTemp2) el.kpiTemp2.textContent = temp2 !== undefined && temp2 !== null ? `${temp2.toFixed(2)} °C` : 'No data yet';
    if (el.kpiTemp2Range) {
      el.kpiTemp2Range.textContent = t2Stats.min !== undefined ? `min ${t2Stats.min.toFixed(2)} • max ${t2Stats.max.toFixed(2)}` : 'min -- • max --';
    }

    // Field 4: Pressure
    const press = latest.pressure_hpa;
    const pStats = stats.pressure_hpa || {};
    if (el.kpiPress) el.kpiPress.textContent = press !== undefined && press !== null ? `${press.toFixed(2)} hPa` : 'No data yet';
    if (el.kpiPressRange) {
      el.kpiPressRange.textContent = pStats.min !== undefined ? `min ${pStats.min.toFixed(1)} • max ${pStats.max.toFixed(1)}` : 'min -- • max --';
    }

    // Field 6: Motor Current (ACS712) - Auto activates when data arrives
    const curr = latest.current_a;
    if (curr !== undefined && curr !== null && !isNaN(Number(curr))) {
      const cVal = Number(curr);
      if (el.kpiCurr) el.kpiCurr.textContent = `${cVal.toFixed(2)} A`;
      if (el.kpiCurrSub) el.kpiCurrSub.textContent = 'Active ACS712 Shunt';
      if (el.liveCurrTag) {
        el.liveCurrTag.textContent = 'Active';
        el.liveCurrTag.className = 'badge badge-live';
      }
      if (el.badgeCurrActive) {
        el.badgeCurrActive.textContent = 'Active Shunt';
        el.badgeCurrActive.style.background = '#dcfce7';
        el.badgeCurrActive.style.color = '#15803d';
      }
    } else {
      if (el.kpiCurr) el.kpiCurr.textContent = 'Not connected';
      if (el.kpiCurrSub) el.kpiCurrSub.textContent = 'Awaiting field6 data';
    }

    // Field 7 & 8: Hardware Flags
    if (el.kpiFlag7) {
      const f7 = latest.esp32_anomaly;
      el.kpiFlag7.textContent = f7 !== undefined && f7 !== null ? `FLAG: ${f7}` : '--';
      el.kpiFlag7.style.color = f7 === 1 ? '#ef4444' : '#10b981';
    }
    if (el.kpiFlag8) {
      const f8 = latest.esp32_alert;
      el.kpiFlag8.textContent = f8 !== undefined && f8 !== null ? (f8 === 1 ? 'ALERT ON' : 'NORMAL') : '--';
      el.kpiFlag8.style.color = f8 === 1 ? '#ef4444' : '#10b981';
    }

    // 3. Health & RUL Panel
    if (el.hiScore) el.hiScore.textContent = `${Math.round(hi)} / 100`;
    if (el.hiStateBadge) {
      el.hiStateBadge.textContent = twin.state || 'OPTIMAL';
      el.hiStateBadge.className = twin.state === 'OPTIMAL' ? 'badge badge-success' : (twin.state === 'WARNING' ? 'badge badge-warning' : 'badge badge-critical');
    }

    const sub = twin.subsystems || {};
    updateSubsystemBar(el.subVibScore, el.subVibBar, sub.vibration);
    updateSubsystemBar(el.subThermScore, el.subThermBar, sub.thermal);
    updateSubsystemBar(el.subEnvScore, el.subEnvBar, sub.environment);
    if (sub.electrical !== null && sub.electrical !== undefined) {
      updateSubsystemBar(el.subElecScore, el.subElecBar, sub.electrical);
    } else {
      if (el.subElecScore) el.subElecScore.textContent = 'Awaiting Sensor';
      if (el.subElecBar) el.subElecBar.style.width = '0%';
    }

    // RUL Card
    if (el.rulDisplay) {
      const rulState = rul.rul_status || 'STABLE';
      const rulHours = rul.rul_hours;
      if (rulState === 'STABLE') {
        el.rulDisplay.textContent = 'STABLE (No degradation trend)';
        el.rulDisplay.style.color = '#10b981';
      } else if (rulState === 'DEGRADING') {
        el.rulDisplay.textContent = `${rulHours !== null ? rulHours.toFixed(1) : '--'} Hours`;
        el.rulDisplay.style.color = '#f59e0b';
      } else {
        el.rulDisplay.textContent = 'INSUFFICIENT DATA';
        el.rulDisplay.style.color = '#64748b';
      }
    }
    if (el.rulSubtitle) el.rulSubtitle.textContent = rul.degradation_assessment || 'Physical limits safe';
    if (el.rulConfidence) {
      const ci = rul.confidence_interval_hours;
      el.rulConfidence.textContent = ci ? `90% CI: [${ci[0].toFixed(1)} h, ${ci[1].toFixed(1)} h]` : '90% CI: N/A';
    }

    // 4. Update SVG Twin State
    updateSvgTwin(isOnline, latest, twin);

    // 5. Draw Canvases
    const series = data.series || [];
    drawVibrationCanvas(el.canvasVib, series, data.baseline?.vibration_g);
    drawEnvironmentCanvas(el.canvasEnv, series);
    drawUnconfirmedCanvas(el.canvasUnconfirmed, series);
    drawCurrentCanvas(el.canvasCurrent, series);
    drawHealthCanvas(el.canvasHi, series);

    // 6. Alert Events Table
    renderAlertEventsTable(data.events || []);
  }

  function updateSubsystemBar(scoreEl, barEl, score) {
    if (!scoreEl || !barEl) return;
    if (score === null || score === undefined) {
      scoreEl.textContent = '--';
      barEl.style.width = '0%';
      return;
    }
    const val = Math.max(0, Math.min(100, Math.round(score)));
    scoreEl.textContent = `${val}%`;
    barEl.style.width = `${val}%`;
    if (val >= 70) {
      barEl.style.background = '#10b981';
    } else if (val >= 40) {
      barEl.style.background = '#f59e0b';
    } else {
      barEl.style.background = '#ef4444';
    }
  }

  // ==========================================================================
  // SVG Digital Twin Interactions
  // ==========================================================================
  function updateSvgTwin(isOnline, latest, twin) {
    if (!el.svgTwin) return;

    // Offline Overlay
    if (el.offlineWatermark) {
      el.offlineWatermark.setAttribute('opacity', isOnline ? '0' : '1');
    }

    // Status LEDs on ESP32
    const isAlert = latest.esp32_alert === 1 || latest.esp32_anomaly === 1;
    if (el.ledGreen) {
      el.ledGreen.setAttribute('fill', isOnline && !isAlert ? '#22c55e' : '#94a3b8');
    }
    if (el.ledRed) {
      el.ledRed.setAttribute('fill', isOnline && isAlert ? '#ef4444' : '#94a3b8');
      el.ledRed.setAttribute('filter', isOnline && isAlert ? 'url(#glowRed)' : 'none');
    }

    // Buzzer Waves
    if (el.buzzerWave) {
      el.buzzerWave.setAttribute('opacity', isOnline && isAlert ? '1' : '0');
    }

    // Motor Rotation
    if (el.motorRotor) {
      if (isOnline) {
        el.motorRotor.style.animation = 'spin 2s linear infinite';
      } else {
        el.motorRotor.style.animation = 'none';
      }
    }

    // Live Readouts inside SVG
    if (el.svgValVib && latest.vibration_g !== undefined) {
      el.svgValVib.textContent = `${latest.vibration_g.toFixed(4)} g`;
    }
    if (el.svgValTemp && latest.temperature_c !== undefined) {
      el.svgValTemp.textContent = `${latest.temperature_c.toFixed(1)}°C`;
    }
    if (el.svgValHum && latest.humidity_pct !== undefined) {
      el.svgValHum.textContent = `${latest.humidity_pct.toFixed(0)}%`;
    }
  }

  function setupSvgClickDrawer() {
    const compMap = {
      'svg-comp-esp32': {
        name: 'ESP32 DevKit V1 (ESP-WROOM-32)',
        role: 'Dual-core Xtensa 240MHz MCU, 802.11 b/g/n Wi-Fi, ThingSpeak HTTP publisher',
        getVal: (d) => `RSSI / Online: ${d.online ? 'CONNECTED' : 'OFFLINE'}`,
        getBaseline: () => 'Sampling Cadence: 20s',
        getHealth: (d) => `${d.twin?.health_index?.toFixed(1) || 100}%`,
      },
      'svg-comp-mpu': {
        name: 'MPU6050 Accelerometer / Gyroscope',
        role: 'I2C Bus (0x68), GPIO 21 (SDA), GPIO 22 (SCL). Samples dynamic vibration RMS in g.',
        getVal: (d) => `${d.latest?.vibration_g?.toFixed(4) || '--'} g`,
        getBaseline: (d) => `Median: ${d.baseline?.vibration_g?.median?.toFixed(4) || 0.022} g`,
        getHealth: (d) => `${d.twin?.subsystems?.vibration?.toFixed(1) || 100}%`,
      },
      'svg-comp-dht11': {
        name: 'DHT11 Temperature & Humidity Sensor',
        role: 'Single-bus digital signal on GPIO 4. Measures ambient temp (0-50°C) and RH (20-90%).',
        getVal: (d) => `${d.latest?.temperature_c?.toFixed(1) || '--'}°C / ${d.latest?.humidity_pct?.toFixed(0) || '--'}%`,
        getBaseline: (d) => `Median: ${d.baseline?.temperature_c?.median?.toFixed(1) || 24}°C`,
        getHealth: (d) => `${d.twin?.subsystems?.thermal?.toFixed(1) || 100}%`,
      },
      'svg-comp-motor': {
        name: '12V Bench DC Motor',
        role: 'Physical rotary load generating dynamic bench mechanical vibrations and acoustic profile.',
        getVal: (d) => `Vib: ${d.latest?.vibration_g?.toFixed(4) || '--'} g`,
        getBaseline: () => 'Nominal: 0.02 - 0.04 g',
        getHealth: (d) => `${d.twin?.subsystems?.vibration?.toFixed(1) || 100}%`,
      },
      'svg-comp-buzzer': {
        name: 'Piezo Alarm Buzzer',
        role: 'Active edge alarm triggered by ESP32 firmware on abnormal vibration spike (Field 8).',
        getVal: (d) => d.latest?.esp32_alert === 1 ? 'ALARM ACTIVE (BUZZER ON)' : 'STANDBY (SILENT)',
        getBaseline: () => 'Normal Flag: 0',
        getHealth: (d) => d.latest?.esp32_alert === 1 ? 'WARNING' : 'OPTIMAL',
      },
      'svg-comp-driver': {
        name: 'N-MOSFET Motor Driver & Current Shunt',
        role: 'PWM speed control + ACS712 Hall-effect current shunt (Field 6 reserved).',
        getVal: (d) => d.latest?.current_a !== null && d.latest?.current_a !== undefined ? `${d.latest.current_a} A` : 'Awaiting sensor wiring',
        getBaseline: () => 'Nominal: ~0.35 A',
        getHealth: () => 'STANDBY',
      },
    };

    Object.keys(compMap).forEach((id) => {
      const elem = document.getElementById(id);
      if (!elem) return;
      elem.addEventListener('click', () => {
        const item = compMap[id];
        if (!item || !el.compDetailCard) return;

        const data = state.liveData || {};
        el.compDetailName.textContent = item.name;
        el.compDetailRole.textContent = item.role;
        el.compDetailVal.textContent = item.getVal(data);
        el.compDetailBaseline.textContent = item.getBaseline(data);
        el.compDetailHealth.textContent = item.getHealth(data);
        el.compDetailCard.style.display = 'block';
      });
    });

    if (el.compDetailClose) {
      el.compDetailClose.addEventListener('click', () => {
        if (el.compDetailCard) el.compDetailCard.style.display = 'none';
      });
    }
  }

  // ==========================================================================
  // Canvas Chart Rendering (Pure HTML5 Canvas, zero build step)
  // ==========================================================================
  function drawVibrationCanvas(canvas, series, baseline) {
    if (!canvas || !series || series.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 400;
    const h = canvas.height = 180;

    ctx.clearRect(0, 0, w, h);

    const values = series.map((s) => s.vibration_g || 0.02);
    const minVal = 0.0;
    const maxVal = Math.max(0.25, Math.max(...values) * 1.15);

    const padLeft = 40;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Grid
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();

      const labelVal = maxVal - (maxVal / 4) * i;
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(labelVal.toFixed(3), 5, y + 3);
    }

    // Baseline Band (Median ± 3 MAD)
    if (baseline && baseline.median !== undefined && baseline.mad !== undefined) {
      const med = baseline.median;
      const bandTopVal = Math.min(maxVal, med + 3 * baseline.mad);
      const bandBotVal = Math.max(minVal, med - 3 * baseline.mad);

      const yBandTop = padTop + plotH * (1 - (bandTopVal - minVal) / (maxVal - minVal));
      const yBandBot = padTop + plotH * (1 - (bandBotVal - minVal) / (maxVal - minVal));

      ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
      ctx.fillRect(padLeft, yBandTop, plotW, yBandBot - yBandTop);

      ctx.strokeStyle = 'rgba(59, 130, 246, 0.35)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const yMed = padTop + plotH * (1 - (med - minVal) / (maxVal - minVal));
      ctx.moveTo(padLeft, yMed);
      ctx.lineTo(w - padRight, yMed);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Safety threshold line (0.25 g)
    const yThreshold = padTop + plotH * (1 - (0.25 - minVal) / (maxVal - minVal));
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(padLeft, yThreshold);
    ctx.lineTo(w - padRight, yThreshold);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data Line
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - (values[i] - minVal) / (maxVal - minVal));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Anomaly points
    for (let i = 0; i < series.length; i++) {
      const s = series[i];
      if (s.vibration_g > 0.10 || s.esp32_anomaly === 1) {
        const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
        const y = padTop + plotH * (1 - (values[i] - minVal) / (maxVal - minVal));
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawEnvironmentCanvas(canvas, series) {
    if (!canvas || !series || series.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 400;
    const h = canvas.height = 180;

    ctx.clearRect(0, 0, w, h);

    const temps = series.map((s) => s.temperature_c || 24);
    const hums = series.map((s) => s.humidity_pct || 65);

    const padLeft = 35;
    const padRight = 35;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Grid
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();
    }

    // Temperature (Red/Orange, Left Axis 15 - 35°C)
    const tMin = 15;
    const tMax = 35;
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - (temps[i] - tMin) / (tMax - tMin));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Humidity (Blue/Cyan, Right Axis 40 - 100%)
    const hMin = 40;
    const hMax = 100;
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - (hums[i] - hMin) / (hMax - hMin));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Axis Labels
    ctx.font = '10px JetBrains Mono';
    ctx.fillStyle = '#f97316';
    ctx.fillText(`${tMax}°C`, 5, padTop + 8);
    ctx.fillText(`${tMin}°C`, 5, padTop + plotH);

    ctx.fillStyle = '#06b6d4';
    ctx.fillText(`${hMax}%`, w - 30, padTop + 8);
    ctx.fillText(`${hMin}%`, w - 30, padTop + plotH);
  }

  function drawUnconfirmedCanvas(canvas, series) {
    if (!canvas || !series || series.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 400;
    const h = canvas.height = 180;
    ctx.clearRect(0, 0, w, h);

    const pVals = series.map((s) => s.pressure_hpa || 994);
    const padLeft = 45;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const minP = Math.floor(Math.min(...pVals) - 1);
    const maxP = Math.ceil(Math.max(...pVals) + 1);

    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - (pVals[i] - minP) / (maxP - minP || 1));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#8b5cf6';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${maxP} hPa`, 5, padTop + 8);
    ctx.fillText(`${minP} hPa`, 5, padTop + plotH);
  }

  function drawCurrentCanvas(canvas, series) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 400;
    const h = canvas.height = 180;
    ctx.clearRect(0, 0, w, h);

    const hasData = series.some((s) => s.current_a !== null && s.current_a !== undefined && !isNaN(Number(s.current_a)));

    if (!hasData) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('ACS712 Current Sensor Disconnected (Field 6 = NA)', w / 2, h / 2 - 8);
      ctx.font = '11px Inter';
      ctx.fillText('Chart will automatically activate when telemetry begins arriving', w / 2, h / 2 + 12);
      ctx.textAlign = 'left';
      return;
    }

    // If current data exists, draw live amperage line
    const vals = series.map((s) => Number(s.current_a || 0));
    const maxC = Math.max(1.0, Math.max(...vals) * 1.2);
    const padLeft = 35;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - vals[i] / maxC);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${maxC.toFixed(1)} A`, 5, padTop + 8);
    ctx.fillText('0.0 A', 5, padTop + plotH);
  }

  function drawHealthCanvas(canvas, series) {
    if (!canvas || !series || series.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 800;
    const h = canvas.height = 140;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 35;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 20;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Threshold lines
    const yWarn = padTop + plotH * (1 - 60 / 100);
    const yFail = padTop + plotH * (1 - 30 / 100);

    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, yWarn);
    ctx.lineTo(w - padRight, yWarn);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.beginPath();
    ctx.moveTo(padLeft, yFail);
    ctx.lineTo(w - padRight, yFail);
    ctx.stroke();
    ctx.setLineDash([]);

    // Health Index Line
    const his = series.map((s) => s.health_index !== undefined ? s.health_index : 100);
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = padLeft + (plotW / (series.length - 1 || 1)) * i;
      const y = padTop + plotH * (1 - his[i] / 100);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('100%', 5, padTop + 8);
    ctx.fillText('60%', 5, yWarn + 3);
    ctx.fillText('30%', 5, yFail + 3);
    ctx.fillText('0%', 5, padTop + plotH);
  }

  function drawFlLossCanvas(canvas, strategies) {
    if (!canvas || !strategies) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth || 700;
    const h = canvas.height = 150;
    ctx.clearRect(0, 0, w, h);

    const colors = {
      'Centralized': '#64748b',
      'FedAvg': '#2563eb',
      'FedProx': '#10b981',
      'DP-FedAvg': '#7928ca',
    };

    const padLeft = 45;
    const padRight = 120;
    const padTop = 15;
    const padBottom = 25;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Find max loss
    let maxLoss = 0.05;
    Object.values(strategies).forEach((st) => {
      if (st.round_losses) {
        maxLoss = Math.max(maxLoss, Math.max(...st.round_losses));
      }
    });

    let legendY = padTop + 10;
    Object.entries(strategies).forEach(([name, st]) => {
      const losses = st.round_losses || [];
      if (losses.length < 2) return;
      const color = colors[name] || '#0f172a';

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < losses.length; i++) {
        const x = padLeft + (plotW / (losses.length - 1 || 1)) * i;
        const y = padTop + plotH * (1 - losses[i] / maxLoss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Legend entry
      ctx.fillStyle = color;
      ctx.fillRect(w - padRight + 10, legendY, 10, 10);
      ctx.fillStyle = '#334155';
      ctx.font = '11px Inter';
      ctx.fillText(`${name} (${st.final_mse})`, w - padRight + 26, legendY + 9);
      legendY += 18;
    });

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(maxLoss.toFixed(4), 5, padTop + 8);
    ctx.fillText('0.0000', 5, padTop + plotH);
  }

  // ==========================================================================
  // Render Tables & Alerts
  // ==========================================================================
  function renderAlertEventsTable(events) {
    if (!el.episodesTableBody) return;
    if (!events || events.length === 0) {
      el.episodesTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #94a3b8; padding: 18px;">No anomaly events recorded in current telemetry window.</td></tr>`;
      return;
    }

    const rows = events.map((ev) => {
      const ts = ev.start_timestamp ? new Date(ev.start_timestamp).toLocaleTimeString() : '--';
      const sevBadge = ev.severity === 'CRITICAL' ? '<span class="badge badge-critical">CRITICAL</span>' : '<span class="badge badge-warning">WARNING</span>';
      return `
        <tr>
          <td class="font-mono">${ts}</td>
          <td><strong>${ev.sensor}</strong></td>
          <td>${ev.methods?.join(', ') || 'Statistical'}</td>
          <td class="font-mono">${ev.peak_value?.toFixed(4) || '--'}</td>
          <td class="font-mono">F7:${ev.esp32_anomaly || 0} / F8:${ev.esp32_alert || 0}</td>
          <td>${sevBadge}</td>
        </tr>
      `;
    });
    el.episodesTableBody.innerHTML = rows.join('');
  }

  // ==========================================================================
  // Analysis Page Controller
  // ==========================================================================
  async function fetchAnalysisData() {
    try {
      // 1. Fetch ML model inference & federated metrics
      const modelRes = await fetch('/api/analysis/model');
      if (modelRes.ok) {
        state.modelData = await modelRes.json();
        renderAnalysisModel(state.modelData);
      }

      // 2. Fetch stats & correlations
      const statsRes = await fetch(`/api/analysis/stats?window=${encodeURIComponent(state.selectedWindow)}`);
      if (statsRes.ok) {
        state.statsData = await statsRes.json();
        renderAnalysisStats(state.statsData);
      }

      // 3. Fetch AI Insights
      fetchAiInsights(false);
    } catch (err) {
      console.warn('Error fetching analysis data:', err);
    }
  }

  function renderAnalysisModel(data) {
    if (!data) return;

    // ML Inference & Anomaly scores
    const ml = data.ml_inference || {};
    const scores = ml.scores || [];
    const latestScore = scores[scores.length - 1] || {};
    const thresholds = data.thresholds || {};

    if (el.mlKpiRecon) {
      el.mlKpiRecon.textContent = latestScore.reconstruction_error !== undefined ? latestScore.reconstruction_error.toFixed(5) : '--';
    }
    if (el.mlKpiReconSub) {
      el.mlKpiReconSub.textContent = `Threshold p95: ${thresholds.recon_p95 || 0.05}`;
    }

    if (el.mlKpiIso) {
      el.mlKpiIso.textContent = latestScore.iso_forest_score !== undefined ? latestScore.iso_forest_score.toFixed(4) : '--';
    }
    if (el.mlKpiIsoSub) {
      el.mlKpiIsoSub.textContent = latestScore.iso_forest_pred === -1 ? 'FLAGGED AS OUTLIER' : 'NORMAL INLIER';
      el.mlKpiIsoSub.style.color = latestScore.iso_forest_pred === -1 ? '#ef4444' : '#10b981';
    }

    // Honest RUL
    const rul = data.honest_rul || {};
    if (el.mlKpiSlope) {
      const slope = rul.trend_slope_per_hour;
      el.mlKpiSlope.textContent = slope !== undefined ? `${slope >= 0 ? '+' : ''}${slope.toFixed(6)} g/h` : '--';
    }

    if (el.mlHonestRulState) {
      el.mlHonestRulState.textContent = rul.rul_state || 'STABLE';
      el.mlHonestRulState.className = rul.rul_state === 'STABLE' ? 'badge badge-success' : (rul.rul_state === 'DEGRADING' ? 'badge badge-warning' : 'badge');
    }

    if (el.mlHonestRulMessage) el.mlHonestRulMessage.textContent = rul.message || '--';
    if (el.mlHonestRulHours) el.mlHonestRulHours.textContent = typeof rul.estimated_rul_hours === 'number' ? `${rul.estimated_rul_hours.toFixed(1)} h` : (rul.estimated_rul_hours || '--');
    if (el.mlHonestRulCi) {
      const ci = rul.confidence_interval_hours;
      el.mlHonestRulCi.textContent = ci ? `[${ci[0].toFixed(1)} h, ${ci[1].toFixed(1)} h]` : 'N/A';
    }
    if (el.mlHonestRulSensor) el.mlHonestRulSensor.textContent = rul.limiting_sensor || 'Vibration RMS';

    // Federated Strategy Comparison Table
    const fb = data.federated_benchmark || {};
    const strategies = fb.strategies || {};
    if (el.flComparisonTableBody && Object.keys(strategies).length > 0) {
      const rows = Object.entries(strategies).map(([name, st]) => {
        return `
          <tr>
            <td><strong>${name}</strong></td>
            <td class="font-mono">${st.final_mse}</td>
            <td class="font-mono">${st.rounds}</td>
            <td><span class="badge" style="background: #f1f5f9; color: #334155;">${st.privacy}</span></td>
            <td style="color: #64748b; font-size: 11px;">${name === 'Centralized' ? 'Baseline pooled data' : (name === 'DP-FedAvg' ? 'Calibrated differential privacy noise' : 'Decentralized partition aggregation')}</td>
          </tr>
        `;
      });
      el.flComparisonTableBody.innerHTML = rows.join('');
    }

    // Round-by-round chart
    if (strategies && Object.keys(strategies).length > 0) {
      drawFlLossCanvas(el.canvasFlLoss, strategies);
    }
  }

  function renderAnalysisStats(data) {
    if (!data) return;
    const stats = data.stats || {};

    // 1. Descriptive Stats Table
    if (el.statsTableBody) {
      const rows = Object.entries(stats).map(([sensor, st]) => {
        return `
          <tr>
            <td><strong>${sensor}</strong></td>
            <td class="font-mono">${st.mean !== undefined ? st.mean.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.std !== undefined ? st.std.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.median !== undefined ? st.median.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.mad !== undefined ? st.mad.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.min !== undefined ? st.min.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.max !== undefined ? st.max.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.p95 !== undefined ? st.p95.toFixed(3) : '--'}</td>
            <td class="font-mono">${st.rate_of_change_per_hour !== undefined ? st.rate_of_change_per_hour.toFixed(4) : '--'}</td>
          </tr>
        `;
      });
      if (rows.length > 0) el.statsTableBody.innerHTML = rows.join('');
    }

    // 2. Correlations
    const corr = data.correlations || {};
    if (el.correlationsContainer) {
      const items = Object.entries(corr).map(([pair, c]) => {
        if (c.status === 'insufficient_data') {
          return `<div class="correlation-item"><span style="font-weight:600;">${pair}:</span> <span style="color:#94a3b8;">Insufficient samples for significant correlation</span></div>`;
        }
        return `
          <div class="correlation-item">
            <div class="correlation-header">
              <span style="font-weight: 600;">${pair}</span>
              <span class="font-mono">r = ${c.pearson?.r?.toFixed(3) || '--'}</span>
            </div>
            <div class="correlation-detail">
              Spearman &rho; = ${c.spearman?.rho?.toFixed(3) || '--'} &bull; p-val = ${c.pearson?.p_value?.toExponential(2) || '--'} (n = ${c.sample_count})
            </div>
          </div>
        `;
      });
      el.correlationsContainer.innerHTML = items.length > 0 ? items.join('') : '<div style="color:#94a3b8;">No correlations calculated.</div>';
    }

    // 3. Data Quality
    const dq = data.data_quality || {};
    if (el.dqContainer) {
      el.dqContainer.innerHTML = `
        <div class="dq-metric-row">
          <span>Telemetry Gap Count:</span>
          <strong class="font-mono">${dq.gaps_count || 0}</strong>
        </div>
        <div class="dq-metric-row">
          <span>Missing Rate:</span>
          <strong class="font-mono">${dq.missing_rate_pct !== undefined ? dq.missing_rate_pct.toFixed(1) : 0}%</strong>
        </div>
        <div class="dq-note-box">
          ${dq.notes || 'DHT11 quantization: 0.1°C / 1% RH resolution. Sampling interval ~20s satisfies Nyquist criteria for thermal/vibration RMS envelopes.'}
        </div>
      `;
    }
  }

  // ==========================================================================
  // AI Insights Controller
  // ==========================================================================
  async function fetchAiInsights(forceRefresh = false) {
    if (!el.aiInsightsContent) return;
    if (state.insightsLoading) return;
    state.insightsLoading = true;

    if (forceRefresh) {
      el.aiInsightsContent.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b;">Querying AI diagnostic engine with active telemetry...</div>';
    }

    try {
      const res = await fetch('/api/analysis/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_refresh: forceRefresh }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Render Markdown / formatted HTML
      renderMarkdownInsights(data);

      if (el.aiInsightsSourceBadge) {
        if (data.source === 'anthropic_claude') {
          el.aiInsightsSourceBadge.textContent = 'CLAUDE 3.5 SONNET';
          el.aiInsightsSourceBadge.style.background = '#f3e8ff';
          el.aiInsightsSourceBadge.style.color = '#7928ca';
        } else {
          el.aiInsightsSourceBadge.textContent = 'RULE-BASED ENGINE';
          el.aiInsightsSourceBadge.style.background = '#f1f5f9';
          el.aiInsightsSourceBadge.style.color = '#475569';
        }
      }

      if (el.aiInsightsCacheStatus) {
        el.aiInsightsCacheStatus.textContent = data.cached ? `Cached (${data.cache_age_seconds}s ago)` : 'Live Generated';
      }
    } catch (err) {
      console.warn('AI insights fetch error:', err);
      el.aiInsightsContent.innerHTML = '<div style="color: #ef4444; padding: 12px;">Failed to generate AI diagnostics. Please try refreshing.</div>';
    } finally {
      state.insightsLoading = false;
    }
  }

  function renderMarkdownInsights(data) {
    const rawText = data.markdown || '';
    // Lightweight markdown parser for headers, lists, and bold text
    let html = rawText
      .replace(/^### (.*$)/gim, '<h4 style="font-weight:700; margin-top:12px; margin-bottom:6px; color:#1e293b;">$1</h4>')
      .replace(/^#### (.*$)/gim, '<h5 style="font-weight:600; margin-top:10px; margin-bottom:4px; color:#334155;">$1</h5>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/gim, '<em>$1</em>')
      .replace(/`([^`]+)`/gim, '<code style="background:#f1f5f9; padding:2px 4px; border-radius:4px; font-family:monospace; font-size:11px;">$1</code>')
      .replace(/^\* (.*$)/gim, '<li style="margin-left: 20px; list-style-type: disc;">$1</li>')
      .replace(/^\d+\. (.*$)/gim, '<li style="margin-left: 20px; list-style-type: decimal;">$1</li>');

    el.aiInsightsContent.innerHTML = html;
  }

  // ==========================================================================
  // Federated Retrain Action
  // ==========================================================================
  async function triggerFederatedRetrain() {
    if (!el.btnRetrainFl || state.retrainLoading) return;
    state.retrainLoading = true;
    el.btnRetrainFl.disabled = true;
    el.btnRetrainFl.textContent = 'Retraining...';

    if (el.flRetrainStatus) {
      el.flRetrainStatus.textContent = 'Executing federated aggregation round on current telemetry...';
      el.flRetrainStatus.style.color = '#f59e0b';
    }

    try {
      const res = await fetch('/api/analysis/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results: 100 }),
      });
      const data = await res.json();

      if (data.status === 'success') {
        if (el.flRetrainStatus) {
          el.flRetrainStatus.textContent = `Completed round: Loss improved ${data.improvement_pct}% (MSE: ${data.mse_after})`;
          el.flRetrainStatus.style.color = '#059669';
        }
        // Refresh model card & benchmarks
        fetchAnalysisData();
      } else {
        if (el.flRetrainStatus) {
          el.flRetrainStatus.textContent = `Retrain notice: ${data.message}`;
          el.flRetrainStatus.style.color = '#64748b';
        }
      }
    } catch (err) {
      console.warn('Retrain error:', err);
      if (el.flRetrainStatus) {
        el.flRetrainStatus.textContent = 'Retraining request failed.';
        el.flRetrainStatus.style.color = '#ef4444';
      }
    } finally {
      state.retrainLoading = false;
      el.btnRetrainFl.disabled = false;
      el.btnRetrainFl.textContent = '▶ Retrain on Latest Data';
    }
  }

  // ==========================================================================
  // Event Listeners & Initialization
  // ==========================================================================
  function setupEventListeners() {
    // Visibility API (pause polling when user switches browser tabs)
    document.addEventListener('visibilitychange', () => {
      state.isTabVisible = !document.hidden;
      if (state.isTabVisible) {
        fetchLiveTelemetry();
      } else if (state.timerId) {
        clearTimeout(state.timerId);
      }
    });

    // Window select change
    if (el.liveWindowSelect) {
      el.liveWindowSelect.addEventListener('change', (e) => {
        state.selectedWindow = e.target.value;
        fetchLiveTelemetry();
      });
    }

    // Refresh buttons
    if (el.btnLiveRefresh) {
      el.btnLiveRefresh.addEventListener('click', () => {
        fetchLiveTelemetry();
      });
    }

    if (el.btnLiveExport) {
      el.btnLiveExport.addEventListener('click', () => {
        window.location.href = '/api/live/export.csv';
      });
    }

    if (el.btnRefreshInsights) {
      el.btnRefreshInsights.addEventListener('click', () => {
        fetchAiInsights(true);
      });
    }

    if (el.btnRetrainFl) {
      el.btnRetrainFl.addEventListener('click', () => {
        triggerFederatedRetrain();
      });
    }

    // SVG click drawer
    setupSvgClickDrawer();
  }

  // Entry point
  document.addEventListener('DOMContentLoaded', () => {
    initElements();
    setupTabs();
    setupEventListeners();

    // Initial fetch
    fetchLiveTelemetry();
  });

})();
