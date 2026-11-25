/**
 * Live Hardware Digital Twin Controller
 * Connects real physical ESP32 telemetry via ThingSpeak to the interactive dashboard.
 * Implements 20s visibility-aware polling, interactive SVG twin, dynamic canvas charts,
 * statistical diagnostics tables, and RUL degradation forecasting.
 */

(function () {
  'use strict';

  // State Management
  const state = {
    selectedWindow: 'all',
    data: null,
    pollingIntervalMs: 20000,
    timerId: null,
    isTabVisible: true,
    errorCount: 0,
    activeComponent: null,
  };

  // DOM Elements
  const el = {};

  function initElements() {
    el.tabBtn = document.getElementById('btn-tab-live');
    el.tabPane = document.getElementById('tab-live');
    el.connBadge = document.getElementById('live-conn-badge');
    el.connStatusText = document.getElementById('live-conn-status-text');
    el.channelInfo = document.getElementById('live-channel-info');
    el.lastUpdated = document.getElementById('live-last-updated');
    el.windowSelect = document.getElementById('live-window-select');
    el.btnRefresh = document.getElementById('btn-live-refresh');
    el.btnExport = document.getElementById('btn-live-export');

    // KPI Readouts
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
    el.kpiPress = document.getElementById('live-kpi-press');
    el.kpiCurr = document.getElementById('live-kpi-curr');

    // Twin & RUL
    el.hiScore = document.getElementById('live-hi-score');
    el.hiStateBadge = document.getElementById('live-hi-state-badge');
    el.rulDisplay = document.getElementById('live-rul-display');
    el.rulSubtitle = document.getElementById('live-rul-subtitle');
    el.rulConfidence = document.getElementById('live-rul-confidence');
    el.rulMethod = document.getElementById('live-rul-method');
    el.rulOperatingHours = document.getElementById('live-operating-hours');
    el.rulTimeToWarn = document.getElementById('live-time-to-warn');
    el.rulFailProb = document.getElementById('live-fail-prob');
    el.rulFailProbBar = document.getElementById('live-fail-prob-bar');
    el.prescriptiveAction = document.getElementById('live-prescriptive-action');

    // Subsystem Bars
    el.subVibScore = document.getElementById('live-sub-vib-score');
    el.subVibBar = document.getElementById('live-sub-vib-bar');
    el.subThermScore = document.getElementById('live-sub-therm-score');
    el.subThermBar = document.getElementById('live-sub-therm-bar');
    el.subEnvScore = document.getElementById('live-sub-env-score');
    el.subEnvBar = document.getElementById('live-sub-env-bar');
    el.subElecScore = document.getElementById('live-sub-elec-score');
    el.subElecBar = document.getElementById('live-sub-elec-bar');

    // SVG Twin
    el.svgTwin = document.getElementById('live-twin-svg');
    el.motorRotor = document.getElementById('svg-motor-rotor');
    el.motorBody = document.getElementById('svg-motor-body');
    el.sensorMpu = document.getElementById('svg-comp-mpu');
    el.ledGreen = document.getElementById('svg-led-green');
    el.ledRed = document.getElementById('svg-led-red');
    el.buzzerWave = document.getElementById('svg-buzzer-waves');
    el.offlineWatermark = document.getElementById('svg-offline-watermark');

    // Component Inspection Drawer
    el.compDetailCard = document.getElementById('live-comp-detail-card');
    el.compDetailName = document.getElementById('live-comp-detail-name');
    el.compDetailVal = document.getElementById('live-comp-detail-val');
    el.compDetailBaseline = document.getElementById('live-comp-detail-baseline');
    el.compDetailHealth = document.getElementById('live-comp-detail-health');
    el.compDetailRole = document.getElementById('live-comp-detail-role');
    el.compDetailClose = document.getElementById('live-comp-detail-close');

    // Canvases
    el.canvasHi = document.getElementById('canvas-live-hi');
    el.canvasRul = document.getElementById('canvas-live-rul');
    el.canvasVib = document.getElementById('canvas-live-vib');
    el.canvasEnv = document.getElementById('canvas-live-env');

    // Diagnostic Tables & Stats
    el.statsTableBody = document.getElementById('live-stats-table-body');
    el.episodesTableBody = document.getElementById('live-episodes-table-body');
    el.reconTotalAnom = document.getElementById('live-recon-total-anom');
    el.reconEsp32Flag = document.getElementById('live-recon-esp32-flags');
    el.reconAgreement = document.getElementById('live-recon-agreement');
    el.correlationsList = document.getElementById('live-correlations-list');
    el.dqScore = document.getElementById('live-dq-score');
    el.dqGaps = document.getElementById('live-dq-gaps');
    el.dqMissing = document.getElementById('live-dq-missing');
    el.dqAcsStatus = document.getElementById('live-dq-acs-status');
  }

  // ==========================================================================
  // Data Fetching & Polling
  // ==========================================================================
  async function fetchLiveData() {
    try {
      const url = `/api/live/hardware?results=150&window=${state.selectedWindow}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      state.data = data;
      state.errorCount = 0;
      updateUI();
    } catch (err) {
      console.warn('[Live Hardware] Telemetry fetch warning:', err.message);
      state.errorCount++;
      // Exponential backoff up to 60s
      const backoff = Math.min(60000, state.pollingIntervalMs * Math.pow(1.5, state.errorCount));
      scheduleNextPoll(backoff);
      renderConnectionStatus(false, null, true);
    }
  }

  function scheduleNextPoll(delayMs = state.pollingIntervalMs) {
    if (state.timerId) {
      clearTimeout(state.timerId);
    }
    if (!state.isTabVisible) return;

    state.timerId = setTimeout(async () => {
      await fetchLiveData();
      scheduleNextPoll();
    }, delayMs);
  }

  // ==========================================================================
  // UI Rendering Core
  // ==========================================================================
  function updateUI() {
    if (!state.data) return;
    const d = state.data;

    renderConnectionStatus(d.online, d.age_seconds, false);
    renderHeaderInfo(d);
    renderKPIs(d);
    renderTwinState(d);
    renderRulPanel(d);
    renderSvgTwin(d);
    renderCharts(d);
    renderDiagnostics(d);
  }

  function renderConnectionStatus(online, ageSeconds, isError = false) {
    if (!el.connBadge || !el.connStatusText) return;

    if (isError) {
      el.connBadge.className = 'status-badge status-badge-warning';
      el.connStatusText.textContent = 'CONNECTING / RETRYING';
      return;
    }

    if (online) {
      el.connBadge.className = 'status-badge status-badge-live';
      const ageStr = ageSeconds !== null ? `${Math.round(ageSeconds)}s ago` : 'now';
      el.connStatusText.textContent = `LIVE (${ageStr})`;
    } else {
      el.connBadge.className = 'status-badge status-badge-offline';
      let ageText = 'unknown';
      if (ageSeconds !== null) {
        if (ageSeconds < 3600) {
          ageText = `${Math.round(ageSeconds / 60)}m ago`;
        } else if (ageSeconds < 86400) {
          ageText = `${(ageSeconds / 3600).toFixed(1)}h ago`;
        } else {
          ageText = `${(ageSeconds / 86400).toFixed(1)}d ago`;
        }
      }
      el.connStatusText.textContent = `OFFLINE (${ageText})`;
    }
  }

  function renderHeaderInfo(d) {
    if (el.channelInfo) {
      el.channelInfo.textContent = `Channel ${d.channel_id} • ${d.channel_info?.name || 'ESP32 Bench'}`;
    }
    if (el.lastUpdated && d.latest_timestamp) {
      const date = new Date(d.latest_timestamp);
      el.lastUpdated.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
  }

  function renderKPIs(d) {
    const latest = d.latest || {};
    const stats = d.stats || {};

    // 1. Vibration
    if (el.kpiVib) {
      el.kpiVib.textContent = latest.vibration_g !== undefined && latest.vibration_g !== null ? `${latest.vibration_g.toFixed(4)} g` : '--';
    }
    if (el.kpiVibRange && stats.vibration_g) {
      el.kpiVibRange.textContent = `min ${stats.vibration_g.min?.toFixed(3) || '--'} • max ${stats.vibration_g.max?.toFixed(3) || '--'}`;
    }
    if (el.kpiVibTrend && stats.vibration_g) {
      const roc = stats.vibration_g.rate_of_change_per_hour || 0;
      el.kpiVibTrend.textContent = `${roc >= 0 ? '+' : ''}${roc.toFixed(4)}/hr`;
      el.kpiVibTrend.className = `trend-indicator ${roc > 0.005 ? 'trend-up-danger' : 'trend-stable'}`;
    }

    // 2. Temperature
    if (el.kpiTemp) {
      el.kpiTemp.textContent = latest.temperature_c !== undefined && latest.temperature_c !== null ? `${latest.temperature_c.toFixed(1)} °C` : '--';
    }
    if (el.kpiTempRange && stats.temperature_c) {
      el.kpiTempRange.textContent = `min ${stats.temperature_c.min?.toFixed(1) || '--'} • max ${stats.temperature_c.max?.toFixed(1) || '--'}`;
    }
    if (el.kpiTempTrend && stats.temperature_c) {
      const roc = stats.temperature_c.rate_of_change_per_hour || 0;
      el.kpiTempTrend.textContent = `${roc >= 0 ? '+' : ''}${roc.toFixed(2)}/hr`;
    }

    // 3. Humidity
    if (el.kpiHum) {
      el.kpiHum.textContent = latest.humidity_pct !== undefined && latest.humidity_pct !== null ? `${latest.humidity_pct.toFixed(1)} %` : '--';
    }
    if (el.kpiHumRange && stats.humidity_pct) {
      el.kpiHumRange.textContent = `min ${stats.humidity_pct.min?.toFixed(0) || '--'}% • max ${stats.humidity_pct.max?.toFixed(0) || '--'}%`;
    }

    // 4. Field 3 (Unconfirmed Temp 2)
    if (el.kpiTemp2) {
      el.kpiTemp2.textContent = latest.temp_secondary_c !== undefined && latest.temp_secondary_c !== null ? `${latest.temp_secondary_c.toFixed(2)} °C` : '--';
    }

    // 5. Field 4 (Unconfirmed Pressure)
    if (el.kpiPress) {
      el.kpiPress.textContent = latest.pressure_hpa !== undefined && latest.pressure_hpa !== null ? `${latest.pressure_hpa.toFixed(1)} hPa` : '--';
    }

    // 6. Field 6 (ACS712 Current) - Graceful NA handling
    if (el.kpiCurr) {
      if (latest.current_a !== undefined && latest.current_a !== null) {
        el.kpiCurr.textContent = `${latest.current_a.toFixed(2)} A`;
        el.kpiCurr.className = 'kpi-value text-normal';
      } else {
        el.kpiCurr.textContent = 'Not connected';
        el.kpiCurr.className = 'kpi-value text-muted-badge';
      }
    }
  }

  function renderTwinState(d) {
    const twin = d.twin || {};
    const hi = twin.health_index || 100.0;
    const subs = twin.subsystems || {};

    if (el.hiScore) el.hiScore.textContent = `${hi.toFixed(1)}%`;
    if (el.hiStateBadge) {
      el.hiStateBadge.textContent = twin.state || 'OPTIMAL';
      el.hiStateBadge.className = `status-pill ${
        twin.state === 'OPTIMAL' ? 'status-pill-success' : twin.state === 'WARNING' ? 'status-pill-warning' : 'status-pill-critical'
      }`;
    }

    // Subsystems
    _updateSubBar(el.subVibScore, el.subVibBar, subs.vibration);
    _updateSubBar(el.subThermScore, el.subThermBar, subs.thermal);
    _updateSubBar(el.subEnvScore, el.subEnvBar, subs.environment);
    _updateSubBar(el.subElecScore, el.subElecBar, subs.electrical, true);
  }

  function _updateSubBar(valEl, barEl, score, isElectrical = false) {
    if (!valEl || !barEl) return;
    if (score === null || score === undefined) {
      valEl.textContent = isElectrical ? 'Reserved (N/A)' : '--';
      barEl.style.width = '0%';
      barEl.className = 'subsystem-fill-bar bg-muted';
      return;
    }
    valEl.textContent = `${score.toFixed(0)}%`;
    barEl.style.width = `${Math.min(100, Math.max(0, score))}%`;
    barEl.className = `subsystem-fill-bar ${
      score >= 80 ? 'bg-success' : score >= 50 ? 'bg-warning' : 'bg-critical'
    }`;
  }

  function renderRulPanel(d) {
    const rul = d.rul || {};
    const status = rul.rul_status;

    if (el.rulDisplay) {
      if (status === 'STABLE_NO_DEGRADATION') {
        el.rulDisplay.textContent = 'STABLE';
        el.rulDisplay.className = 'rul-highlight text-success';
        if (el.rulSubtitle) el.rulSubtitle.textContent = 'Nominal health baseline with no significant wear degradation detected.';
      } else if (status === 'DEGRADING') {
        el.rulDisplay.textContent = `${rul.rul_hours?.toFixed(1) || '--'} hrs`;
        el.rulDisplay.className = 'rul-highlight text-warning';
        if (el.rulSubtitle) {
          el.rulSubtitle.textContent = `95% CI: [${rul.ci_lower_hours || 0} - ${rul.ci_upper_hours || 0} hrs] • Time to warn: ${rul.time_to_warning_hours || 0} hrs`;
        }
      } else {
        el.rulDisplay.textContent = 'INSUFFICIENT DATA';
        el.rulDisplay.className = 'rul-highlight text-muted';
        if (el.rulSubtitle) el.rulSubtitle.textContent = `Need at least ${rul.warning_threshold || 30} operational telemetry points (currently ${rul.samples_used || 0}).`;
      }
    }

    if (el.rulConfidence) {
      el.rulConfidence.textContent = rul.confidence || 'HIGH';
      el.rulConfidence.className = `badge badge-${(rul.confidence || 'HIGH').toLowerCase()}`;
    }
    if (el.rulMethod) {
      el.rulMethod.textContent = (rul.method || 'model').replace(/_/g, ' ');
    }
    if (el.rulOperatingHours) {
      el.rulOperatingHours.textContent = `${rul.operating_hours || 0} hrs`;
    }
    if (el.rulTimeToWarn) {
      el.rulTimeToWarn.textContent = rul.time_to_warning_hours !== null && rul.time_to_warning_hours !== undefined ? `${rul.time_to_warning_hours} hrs` : '--';
    }
    if (el.rulFailProb) {
      const probPct = Math.round((rul.failure_probability || 0.01) * 100);
      el.rulFailProb.textContent = `${probPct}%`;
      if (el.rulFailProbBar) el.rulFailProbBar.style.width = `${probPct}%`;
    }
    if (el.prescriptiveAction) {
      el.prescriptiveAction.textContent = rul.recommended_action || 'Routine baseline monitoring active.';
    }
  }

  function renderSvgTwin(d) {
    const isOnline = d.online;
    const latest = d.latest || {};
    const twin = d.twin || {};
    const hi = twin.health_index || 100.0;
    const isAlert = latest.esp32_alert === 1;

    // Offline Watermark
    if (el.offlineWatermark) {
      el.offlineWatermark.style.display = isOnline ? 'none' : 'block';
    }

    // Motor Color based on Health Index
    if (el.motorBody) {
      if (!isOnline) {
        el.motorBody.setAttribute('fill', '#94a3b8');
      } else if (hi >= 80) {
        el.motorBody.setAttribute('fill', '#0070f3'); // Clean Vercel Blue
      } else if (hi >= 50) {
        el.motorBody.setAttribute('fill', '#f5a623'); // Warning Amber
      } else {
        el.motorBody.setAttribute('fill', '#ee0000'); // Critical Red
      }
    }

    // Motor Rotor Animation (Spin if online)
    if (el.motorRotor) {
      el.motorRotor.style.animationPlayState = isOnline ? 'running' : 'paused';
    }

    // MPU6050 Dynamic Vibration Shake
    if (el.sensorMpu) {
      const vib = latest.vibration_g || 0.02;
      if (vib > 0.15) {
        el.sensorMpu.classList.add('svg-shake-severe');
        el.sensorMpu.classList.remove('svg-shake-mild');
      } else if (vib > 0.05) {
        el.sensorMpu.classList.add('svg-shake-mild');
        el.sensorMpu.classList.remove('svg-shake-severe');
      } else {
        el.sensorMpu.classList.remove('svg-shake-mild', 'svg-shake-severe');
      }
    }

    // LEDs & Buzzer Waves
    if (el.ledGreen) {
      el.ledGreen.setAttribute('fill', isOnline ? '#00dfd8' : '#64748b');
    }
    if (el.ledRed) {
      el.ledRed.setAttribute('fill', isAlert ? '#ee0000' : '#475569');
      el.ledRed.classList.toggle('svg-pulse-glow', isAlert);
    }
    if (el.buzzerWave) {
      el.buzzerWave.style.display = isAlert ? 'block' : 'none';
    }
  }

  // ==========================================================================
  // Canvas Chart Renderers
  // ==========================================================================
  function renderCharts(d) {
    const series = d.series || [];
    if (series.length === 0) return;

    renderHiChart(series);
    renderRulExtrapolationChart(series, d.rul);
    renderVibrationChart(series, d.baseline?.vibration_g, d.anomalies);
    renderEnvChart(series);
  }

  function _setupCanvas(canvas) {
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Subtle hairline grid
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let x = 40; x < width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 15; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    return { ctx, width, height };
  }

  // 1. Health Index Time Series Chart
  function renderHiChart(series) {
    const c = _setupCanvas(el.canvasHi);
    if (!c) return;
    const { ctx, width, height } = c;
    const padL = 40, padR = 20, padT = 20, padB = 25;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Warning (60%) and Failure (30%) lines
    const y60 = padT + plotH * (1 - 60 / 100);
    const y30 = padT + plotH * (1 - 30 / 100);

    ctx.strokeStyle = '#f5a623';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, y60);
    ctx.lineTo(padL + plotW, y60);
    ctx.stroke();

    ctx.strokeStyle = '#ee0000';
    ctx.beginPath();
    ctx.moveTo(padL, y30);
    ctx.lineTo(padL + plotW, y30);
    ctx.stroke();
    ctx.setLineDash([]);

    // HI Line
    ctx.strokeStyle = '#0070f3';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    const n = series.length;
    series.forEach((s, i) => {
      const x = padL + (i / Math.max(1, n - 1)) * plotW;
      const val = Math.max(0, Math.min(100, s.health_index || 100));
      const y = padT + plotH * (1 - val / 100);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Geist, sans-serif';
    ctx.fillText('100%', 10, padT + 8);
    ctx.fillText('60% (warn)', 10, y60 + 3);
    ctx.fillText('30% (crit)', 10, y30 + 3);
  }

  // 2. RUL Extrapolation Projection Chart
  function renderRulExtrapolationChart(series, rul) {
    const c = _setupCanvas(el.canvasRul);
    if (!c) return;
    const { ctx, width, height } = c;
    const padL = 45, padR = 30, padT = 20, padB = 25;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const n = series.length;
    const halfW = plotW * 0.55;

    // Historical Section
    ctx.strokeStyle = '#0070f3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((s, i) => {
      const x = padL + (i / Math.max(1, n - 1)) * halfW;
      const val = Math.max(0, Math.min(100, s.health_index || 100));
      const y = padT + plotH * (1 - val / 100);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Current point
    const lastHI = series[series.length - 1].health_index || 100;
    const curX = padL + halfW;
    const curY = padT + plotH * (1 - lastHI / 100);

    ctx.fillStyle = '#0070f3';
    ctx.beginPath();
    ctx.arc(curX, curY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Projected Future
    const endX = padL + plotW;
    let targetHI = lastHI;
    if (rul && rul.rul_status === 'DEGRADING') {
      targetHI = rul.failure_threshold || 30.0;
    }
    const targetY = padT + plotH * (1 - targetHI / 100);

    // Confidence Band
    ctx.fillStyle = 'rgba(0, 112, 243, 0.08)';
    ctx.beginPath();
    ctx.moveTo(curX, curY);
    ctx.lineTo(endX, Math.max(padT, targetY - 18));
    ctx.lineTo(endX, Math.min(padT + plotH, targetY + 18));
    ctx.closePath();
    ctx.fill();

    // Projection Line
    ctx.strokeStyle = rul && rul.rul_status === 'DEGRADING' ? '#f5a623' : '#0070f3';
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(curX, curY);
    ctx.lineTo(endX, targetY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Failure Threshold line
    const yFail = padT + plotH * (1 - 30 / 100);
    ctx.strokeStyle = '#ee0000';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, yFail);
    ctx.lineTo(padL + plotW, yFail);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px Geist, sans-serif';
    ctx.fillText('History', padL + halfW * 0.4, padT + 12);
    ctx.fillText('Forecast', padL + halfW + (plotW - halfW) * 0.3, padT + 12);
  }

  // 3. Vibration Chart with Baseline & Anomaly Markers
  function renderVibrationChart(series, vibBaseline, anomalies = []) {
    const c = _setupCanvas(el.canvasVib);
    if (!c) return;
    const { ctx, width, height } = c;
    const padL = 45, padR = 20, padT = 20, padB = 25;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const vals = series.map((s) => s.vibration_g || 0);
    const maxVal = Math.max(0.1, ...vals) * 1.15;

    // Baseline +/- 3 sigma shaded band
    if (vibBaseline) {
      const yMedian = padT + plotH * (1 - vibBaseline.median / maxVal);
      const yUpper3 = padT + plotH * (1 - vibBaseline.upper_3sigma / maxVal);

      ctx.fillStyle = 'rgba(79, 70, 229, 0.06)';
      ctx.fillRect(padL, yUpper3, plotW, Math.max(2, yMedian - yUpper3));

      ctx.strokeStyle = '#6366f1';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, yUpper3);
      ctx.lineTo(padL + plotW, yUpper3);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Telemetry Line
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const n = series.length;
    series.forEach((s, i) => {
      const x = padL + (i / Math.max(1, n - 1)) * plotW;
      const y = padT + plotH * (1 - (s.vibration_g || 0) / maxVal);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Anomaly Markers (Known Spikes)
    const anomEntries = new Set((anomalies || []).map((a) => a.entry_id));
    series.forEach((s, i) => {
      if (anomEntries.has(s.entry_id)) {
        const x = padL + (i / Math.max(1, n - 1)) * plotW;
        const y = padT + plotH * (1 - (s.vibration_g || 0) / maxVal);

        ctx.fillStyle = '#ee0000';
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '10px Geist, sans-serif';
    ctx.fillText(`${maxVal.toFixed(3)}g`, 5, padT + 8);
    ctx.fillText('0g', 20, padT + plotH);
  }

  // 4. Dual Temperature & Humidity Chart
  function renderEnvChart(series) {
    const c = _setupCanvas(el.canvasEnv);
    if (!c) return;
    const { ctx, width, height } = c;
    const padL = 40, padR = 40, padT = 20, padB = 25;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const n = series.length;
    // Temp: 15°C to 40°C
    const tempMin = 15, tempMax = 40;
    // Humidity: 20% to 100%
    const humMin = 20, humMax = 100;

    // Draw Humidity (Cyan Line)
    ctx.strokeStyle = '#00dfd8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    series.forEach((s, i) => {
      const x = padL + (i / Math.max(1, n - 1)) * plotW;
      const h = s.humidity_pct !== null && s.humidity_pct !== undefined ? s.humidity_pct : 60;
      const y = padT + plotH * (1 - (h - humMin) / (humMax - humMin));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Temp (Pink/Red Line)
    ctx.strokeStyle = '#ff0080';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    series.forEach((s, i) => {
      const x = padL + (i / Math.max(1, n - 1)) * plotW;
      const t = s.temperature_c !== null && s.temperature_c !== undefined ? s.temperature_c : 24;
      const y = padT + plotH * (1 - (t - tempMin) / (tempMax - tempMin));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dual-axis text
    ctx.fillStyle = '#ff0080';
    ctx.font = '10px Geist, sans-serif';
    ctx.fillText('Temp (°C)', padL, padT - 5);
    ctx.fillStyle = '#00dfd8';
    ctx.fillText('Humidity (%)', padL + plotW - 60, padT - 5);
  }

  // ==========================================================================
  // Diagnostic Tables & Reconciliation
  // ==========================================================================
  function renderDiagnostics(d) {
    const stats = d.stats || {};
    const episodes = d.events || [];
    const recon = d.reconciliation || {};
    const corrs = d.correlations || {};
    const dq = d.data_quality || {};

    // 1. Stats Table
    if (el.statsTableBody) {
      el.statsTableBody.innerHTML = '';
      const rows = [
        { label: 'Vibration RMS (MPU6050)', key: 'vibration_g', unit: 'g' },
        { label: 'Ambient Temperature (DHT11)', key: 'temperature_c', unit: '°C' },
        { label: 'Relative Humidity (DHT11)', key: 'humidity_pct', unit: '%' },
        { label: 'Secondary Temp (Unconfirmed)', key: 'temp_secondary_c', unit: '°C' },
        { label: 'Pressure (Unconfirmed)', key: 'pressure_hpa', unit: 'hPa' },
        { label: 'Motor Current (ACS712)', key: 'current_a', unit: 'A' },
      ];

      rows.forEach((r) => {
        const s = stats[r.key];
        const tr = document.createElement('tr');
        if (!s || !s.connected) {
          tr.innerHTML = `
            <td><strong>${r.label}</strong></td>
            <td colspan="7" class="text-muted">Not connected (Reserved)</td>
          `;
        } else {
          tr.innerHTML = `
            <td><strong>${r.label}</strong></td>
            <td>${s.mean?.toFixed(3)} ${r.unit}</td>
            <td>${s.std?.toFixed(3)}</td>
            <td>${s.min?.toFixed(3)}</td>
            <td>${s.max?.toFixed(3)}</td>
            <td>${s.p50?.toFixed(3)}</td>
            <td>${s.p95?.toFixed(3)}</td>
            <td>${s.rate_of_change_per_hour >= 0 ? '+' : ''}${s.rate_of_change_per_hour?.toFixed(4)}/hr</td>
          `;
        }
        el.statsTableBody.appendChild(tr);
      });
    }

    // 2. Anomaly Episodes Table
    if (el.episodesTableBody) {
      el.episodesTableBody.innerHTML = '';
      if (episodes.length === 0) {
        el.episodesTableBody.innerHTML = '<tr><td colspan="6" class="text-muted text-center">No anomaly episodes recorded in this window.</td></tr>';
      } else {
        episodes.forEach((ep) => {
          const tr = document.createElement('tr');
          const startDate = new Date(ep.start_time);
          tr.innerHTML = `
            <td><code>${ep.episode_id}</code></td>
            <td>${startDate.toLocaleTimeString()}</td>
            <td>${ep.duration_seconds}s (${ep.sample_count} ticks)</td>
            <td><strong>${ep.peak_vibration_g} g</strong></td>
            <td><span class="badge badge-${ep.primary_severity.toLowerCase()}">${ep.primary_severity}</span></td>
            <td>${ep.esp32_alerted ? '<span class="text-critical font-bold">YES (Alert On)</span>' : 'None'}</td>
          `;
          el.episodesTableBody.appendChild(tr);
        });
      }
    }

    // 3. Reconciliation
    if (el.reconTotalAnom) el.reconTotalAnom.textContent = recon.engine_flagged_count || 0;
    if (el.reconEsp32Flag) el.reconEsp32Flag.textContent = recon.esp32_flagged_count || 0;
    if (el.reconAgreement) el.reconAgreement.textContent = `${recon.agreement_rate_pct || 100}%`;

    // 4. Correlations
    if (el.correlationsList) {
      el.correlationsList.innerHTML = '';
      Object.keys(corrs).forEach((k) => {
        const item = corrs[k];
        const div = document.createElement('div');
        div.className = 'correlation-item';
        div.innerHTML = `
          <div class="correlation-header">
            <span><strong>${item.var1}</strong> vs <strong>${item.var2}</strong></span>
            <span class="font-mono">${item.pearson_r !== undefined ? `r = ${item.pearson_r.toFixed(3)}` : 'N/A'}</span>
          </div>
          <div class="correlation-detail text-muted">${item.interpretation || 'Insufficient data'}</div>
        `;
        el.correlationsList.appendChild(div);
      });
    }

    // 5. Data Quality Report
    if (el.dqScore) el.dqScore.textContent = `${d.data_quality?.quality_score || 95}%`;
    if (el.dqGaps) el.dqGaps.textContent = `${dq.gap_count || 0} gaps (${dq.total_gap_duration_seconds || 0}s total)`;
    if (el.dqMissing) {
      const vibMiss = dq.missing_rates_pct?.vibration_g || 0;
      el.dqMissing.textContent = `${vibMiss}% vibration missing`;
    }
    if (el.dqAcsStatus) el.dqAcsStatus.textContent = dq.acs712_status || 'DISCONNECTED_RESERVED';
  }

  // ==========================================================================
  // Interactive Component Inspection Drawer
  // ==========================================================================
  function showComponentDetails(name, role, val, baseline, health) {
    if (!el.compDetailCard) return;
    state.activeComponent = name;
    el.compDetailName.textContent = name;
    el.compDetailRole.textContent = role;
    el.compDetailVal.textContent = val;
    el.compDetailBaseline.textContent = baseline;
    el.compDetailHealth.textContent = health;
    el.compDetailCard.style.display = 'block';
  }

  function hideComponentDetails() {
    if (el.compDetailCard) el.compDetailCard.style.display = 'none';
    state.activeComponent = null;
  }

  function setupComponentClicks() {
    const components = [
      { id: 'svg-comp-esp32', name: 'ESP32 Microcontroller', role: 'Wi-Fi Edge Compute & SNTP UTC Ingestion', getVal: () => 'Online (240 MHz)' },
      { id: 'svg-comp-mpu', name: 'MPU6050 Accelerometer/IMU', role: '3-Axis Vibration RMS & Spindle Dynamics', getVal: () => `${state.data?.latest?.vibration_g?.toFixed(4) || '--'} g` },
      { id: 'svg-comp-dht', name: 'DHT11 Sensor Module', role: 'Ambient Temp & Relative Humidity Monitoring', getVal: () => `${state.data?.latest?.temperature_c || '--'} °C / ${state.data?.latest?.humidity_pct || '--'}%` },
      { id: 'svg-comp-acs', name: 'ACS712 Current Sensor', role: 'Motor Current Ingestion (Reserved)', getVal: () => state.data?.latest?.current_a ? `${state.data.latest.current_a} A` : 'Not connected' },
      { id: 'svg-comp-motor', name: '12V DC Industrial Bench Motor', role: 'Rotating Equipment Degradation Target', getVal: () => `Health: ${state.data?.twin?.health_index?.toFixed(1) || 100}%` },
      { id: 'svg-comp-mosfet', name: 'N-MOSFET Driver Module', role: 'PWM Speed Control & Power Gate', getVal: () => '12V Rail Active' },
    ];

    components.forEach((c) => {
      const domEl = document.getElementById(c.id);
      if (domEl) {
        domEl.style.cursor = 'pointer';
        domEl.addEventListener('click', () => {
          showComponentDetails(c.name, c.role, c.getVal(), 'Baseline: Normal', `Health Score: ${state.data?.twin?.health_index?.toFixed(0) || 100}%`);
        });
      }
    });

    if (el.compDetailClose) {
      el.compDetailClose.addEventListener('click', hideComponentDetails);
    }
  }

  // ==========================================================================
  // Initialization & Event Listeners
  // ==========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    initElements();
    setupComponentClicks();

    // Window selector
    if (el.windowSelect) {
      el.windowSelect.addEventListener('change', (e) => {
        state.selectedWindow = e.target.value;
        fetchLiveData();
      });
    }

    // Refresh button
    if (el.btnRefresh) {
      el.btnRefresh.addEventListener('click', () => {
        fetchLiveData();
      });
    }

    // Export button
    if (el.btnExport) {
      el.btnExport.addEventListener('click', () => {
        window.open('/api/live/export.csv', '_blank');
      });
    }

    // Tab visibility handling (pause polling on background tab)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        state.isTabVisible = false;
        if (state.timerId) clearTimeout(state.timerId);
      } else {
        state.isTabVisible = true;
        fetchLiveData();
        scheduleNextPoll();
      }
    });

    // Start initial fetch & polling
    fetchLiveData();
    scheduleNextPoll();
  });
})();
