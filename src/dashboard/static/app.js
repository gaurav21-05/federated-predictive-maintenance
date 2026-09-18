/**
 * Industrial IoT Federated Predictive Maintenance & Digital Twin Control Center
 * Vanilla JavaScript High-Performance Reactive Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    activeTab: 'tab-twins',
    selectedClient: 'client_1',
    overview: null,
    telemetryData: [],
    autoStreaming: false,
    autoStreamInterval: null,
    flHistory: [
      { round: 1, avg_train_loss: 0.284, macro_f1: 0.725, epsilon: 0.32 },
      { round: 2, avg_train_loss: 0.162, macro_f1: 0.814, epsilon: 0.61 },
      { round: 3, avg_train_loss: 0.098, macro_f1: 0.869, epsilon: 0.88 },
    ],
    whatIfResult: null,
    workOrders: [],
    woFilter: 'ALL',
  };

  // DOM Elements
  const elements = {
    // Top Nav
    fleetStatus: document.getElementById('global-system-status'),
    fleetHealthVal: document.getElementById('fleet-health-val'),
    edgeNodeCount: document.getElementById('edge-node-count'),
    privacyBadge: document.getElementById('privacy-badge'),
    woBadge: document.getElementById('work-order-badge'),

    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // Tab 1: Twins
    fleetCardsContainer: document.getElementById('fleet-cards-container'),
    machineSelector: document.getElementById('machine-selector'),
    detailTitle: document.getElementById('detail-machine-title'),
    detailOpState: document.getElementById('detail-op-state'),
    dialMhiVal: document.getElementById('dial-mhi-val'),
    dialProgress: document.getElementById('dial-progress-bar'),
    detailRulVal: document.getElementById('detail-rul-val'),
    detailWearRate: document.getElementById('detail-wear-rate'),
    detailFaultLoc: document.getElementById('detail-fault-loc'),
    subVibVal: document.getElementById('sub-val-vibration'),
    subVibBar: document.getElementById('sub-bar-vibration'),
    subThermVal: document.getElementById('sub-val-thermal'),
    subThermBar: document.getElementById('sub-bar-thermal'),
    subPressVal: document.getElementById('sub-val-pressure'),
    subPressBar: document.getElementById('sub-bar-pressure'),
    subElectVal: document.getElementById('sub-val-electrical'),
    subElectBar: document.getElementById('sub-bar-electrical'),
    telemetryCanvas: document.getElementById('telemetry-canvas'),
    liveTickReadout: document.getElementById('live-tick-readout'),
    pillTemp: document.getElementById('pill-temp'),
    pillVib: document.getElementById('pill-vib'),
    pillPress: document.getElementById('pill-press'),
    pillCurr: document.getElementById('pill-curr'),
    pillRpm: document.getElementById('pill-rpm'),

    // Playback buttons
    btnStepOne: document.getElementById('btn-step-one'),
    btnStepTen: document.getElementById('btn-step-ten'),
    btnToggleAuto: document.getElementById('btn-toggle-auto'),
    btnResetSim: document.getElementById('btn-reset-sim'),

    // Tab 2: FL
    flForm: document.getElementById('fl-train-form'),
    flStrategy: document.getElementById('fl-strategy'),
    flRounds: document.getElementById('fl-rounds'),
    flEpochs: document.getElementById('fl-epochs'),
    flMu: document.getElementById('fl-mu'),
    flMuVal: document.getElementById('fl-mu-val'),
    flDpToggle: document.getElementById('fl-dp-toggle'),
    dpOptionsRow: document.getElementById('dp-options-row'),
    flClipNorm: document.getElementById('fl-clip-norm'),
    flNoiseMult: document.getElementById('fl-noise-mult'),
    flSecaggToggle: document.getElementById('fl-secagg-toggle'),
    btnLaunchFl: document.getElementById('btn-launch-fl'),
    flConvergenceCanvas: document.getElementById('fl-convergence-canvas'),
    resMacroF1: document.getElementById('res-macro-f1'),
    resMacroRecall: document.getElementById('res-macro-recall'),
    resBestRound: document.getElementById('res-best-round'),
    resEpsilon: document.getElementById('res-epsilon'),

    // Tab 3: What-If
    whatifClient: document.getElementById('whatif-target-client'),
    presetButtons: document.querySelectorAll('.preset-btn'),
    inputRpmMult: document.getElementById('input-rpm-mult'),
    lblRpmMult: document.getElementById('lbl-rpm-mult'),
    inputLoadMult: document.getElementById('input-load-mult'),
    lblLoadMult: document.getElementById('lbl-load-mult'),
    inputTempDelta: document.getElementById('input-temp-delta'),
    lblTempDelta: document.getElementById('lbl-temp-delta'),
    btnRunWhatif: document.getElementById('btn-run-whatif'),
    whatifResTitle: document.getElementById('whatif-res-title'),
    whatifImpactTag: document.getElementById('whatif-impact-tag'),
    whatifBaseRul: document.getElementById('whatif-base-rul'),
    whatifProjRul: document.getElementById('whatif-proj-rul'),
    whatifWearFactor: document.getElementById('whatif-wear-factor'),
    whatifDecayCanvas: document.getElementById('whatif-decay-canvas'),
    whatifRecText: document.getElementById('whatif-rec-text'),

    // Tab 4: Work Orders
    workOrdersList: document.getElementById('work-orders-list'),
    woFilterButtons: document.querySelectorAll('.filter-group .btn'),
    btnExportOrders: document.getElementById('btn-export-orders'),
  };

  /* ==========================================================================
     Tab Navigation
     ========================================================================== */
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabButtons.forEach(b => b.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      state.activeTab = targetTab;
      document.getElementById(targetTab).classList.add('active');

      // Redraw charts if needed on tab switch
      if (targetTab === 'tab-twins') {
        renderTelemetryChart();
      } else if (targetTab === 'tab-federated') {
        renderFLConvergenceChart();
      } else if (targetTab === 'tab-whatif') {
        renderWhatIfChart();
      } else if (targetTab === 'tab-maintenance') {
        fetchWorkOrders();
      }
    });
  });

  /* ==========================================================================
     API Calls & Service Layer Sync
     ========================================================================== */
  async function fetchOverview() {
    try {
      const res = await fetch('/api/overview');
      if (!res.ok) return;
      const data = await res.json();
      state.overview = data;
      renderOverview(data);
    } catch (err) {
      console.warn('Error fetching overview:', err);
    }
  }

  async function fetchTelemetry(clientId = state.selectedClient) {
    try {
      const res = await fetch(`/api/telemetry/stream?client_id=${clientId}&window=50`);
      if (!res.ok) return;
      const data = await res.json();
      state.telemetryData = data.telemetry || [];
      renderMachineDetail(data);
      renderTelemetryChart();
    } catch (err) {
      console.warn('Error fetching telemetry:', err);
    }
  }

  async function stepSimulation(steps = 1) {
    try {
      const res = await fetch('/api/digital-twin/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: null, steps: steps }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.overview) {
        state.overview = data.overview;
        renderOverview(data.overview);
      }
      await fetchTelemetry(state.selectedClient);
      if (data.new_work_orders && data.new_work_orders.length > 0) {
        fetchWorkOrders();
      }
    } catch (err) {
      console.warn('Error stepping simulation:', err);
    }
  }

  async function resetSimulation() {
    try {
      const res = await fetch('/api/digital-twin/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: null }),
      });
      if (!res.ok) return;
      await fetchOverview();
      await fetchTelemetry(state.selectedClient);
    } catch (err) {
      console.warn('Error resetting simulation:', err);
    }
  }

  async function runWhatIfSimulation() {
    const clientId = elements.whatifClient.value;
    const rpm = parseFloat(elements.inputRpmMult.value);
    const load = parseFloat(elements.inputLoadMult.value);
    const temp = parseFloat(elements.inputTempDelta.value);

    elements.btnRunWhatif.disabled = true;
    elements.btnRunWhatif.textContent = 'Simulating Virtual Physics...';

    try {
      const res = await fetch('/api/digital-twin/what-if', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: clientId,
          rpm_multiplier: rpm,
          load_multiplier: load,
          ambient_temp_delta: temp,
          horizon_minutes: 60.0,
        }),
      });
      const data = await res.json();
      state.whatIfResult = data.simulation;
      renderWhatIfResults(data);
    } catch (err) {
      console.error('What-If simulation failed:', err);
    } finally {
      elements.btnRunWhatif.disabled = false;
      elements.btnRunWhatif.textContent = 'Simulate Stress Impact';
    }
  }

  async function fetchWorkOrders() {
    try {
      const url = state.woFilter === 'ALL' ? '/api/maintenance/orders' : `/api/maintenance/orders?status=${state.woFilter}`;
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      state.workOrders = data;
      renderWorkOrders(data);
      elements.woBadge.textContent = data.filter(o => o.status === 'OPEN').length;
    } catch (err) {
      console.warn('Error fetching work orders:', err);
    }
  }

  async function updateOrderStatus(orderId, newStatus) {
    try {
      const res = await fetch('/api/maintenance/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ work_order_id: orderId, new_status: newStatus }),
      });
      if (!res.ok) return;
      fetchWorkOrders();
    } catch (err) {
      console.error('Failed to update status:', err);
    }
  }

  /* ==========================================================================
     Renderers
     ========================================================================== */
  function renderOverview(data) {
    if (!data) return;

    elements.fleetHealthVal.textContent = `${data.fleet_health_index}%`;
    elements.fleetStatus.textContent = `FLEET ${data.fleet_status}`;
    elements.edgeNodeCount.textContent = `${data.machines ? data.machines.length : 3} Active`;
    elements.woBadge.textContent = data.open_work_orders || 0;

    // Render 3 Fleet Summary Cards
    elements.fleetCardsContainer.innerHTML = '';
    (data.machines || []).forEach(m => {
      const isSelected = m.client_id === state.selectedClient;
      const card = document.createElement('div');
      card.className = `fleet-card ${isSelected ? 'selected' : ''}`;
      card.setAttribute('data-client', m.client_id);

      let tagClass = 'tag-normal';
      if (m.operational_state === 'CRITICAL') tagClass = 'tag-critical';
      else if (m.operational_state === 'WARNING') tagClass = 'tag-warning';

      card.innerHTML = `
        <div class="fleet-card-hdr">
          <div>
            <div class="machine-code">${m.machine_id}</div>
            <div class="machine-name">${m.machine_name}</div>
          </div>
          <span class="tag ${tagClass}">${m.operational_state}</span>
        </div>
        <div class="fleet-card-body">
          <div class="mini-metric">
            <span class="lbl">Health Index</span>
            <span class="val" style="color: ${m.overall_health_index < 60 ? '#ff3366' : (m.overall_health_index < 80 ? '#ffb700' : '#00f0ff')}">${m.overall_health_index}%</span>
          </div>
          <div class="mini-metric">
            <span class="lbl">Est. RUL</span>
            <span class="val" style="color: #00ff88">${m.rul_minutes}m</span>
          </div>
          <div class="mini-metric">
            <span class="lbl">Fault Subsystem</span>
            <span class="val" style="font-size: 0.85rem; color: #94a3b8;">${m.primary_fault_subsystem.split(' ')[0]}</span>
          </div>
        </div>
        <div class="fleet-card-ftr" style="font-size: 0.72rem; color: #64748b; font-family: monospace;">
          Stream Step: ${m.current_step.toLocaleString()} / ${m.total_records.toLocaleString()}
        </div>
      `;

      card.addEventListener('click', () => {
        selectClient(m.client_id);
      });
      elements.fleetCardsContainer.appendChild(card);
    });
  }

  function selectClient(clientId) {
    state.selectedClient = clientId;

    // Update segmented buttons
    document.querySelectorAll('#machine-selector .seg-btn').forEach(btn => {
      if (btn.getAttribute('data-client') === clientId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update fleet card highlights
    document.querySelectorAll('.fleet-card').forEach(c => {
      if (c.getAttribute('data-client') === clientId) {
        c.classList.add('selected');
      } else {
        c.classList.remove('selected');
      }
    });

    fetchTelemetry(clientId);
  }

  function renderMachineDetail(data) {
    if (!data || !data.state_snapshot) return;

    const snap = data.state_snapshot;
    const rul = data.rul_prediction || {};
    const machineNames = {
      client_1: 'Client 1 • CNC Milling Spindle Twin',
      client_2: 'Client 2 • Hydraulic Stamping Press Twin',
      client_3: 'Client 3 • Turbine Compressor Twin',
    };

    elements.detailTitle.textContent = machineNames[data.client_id] || data.client_id;
    elements.liveTickReadout.textContent = `Step: ${data.current_step.toLocaleString()} / ${data.total_steps.toLocaleString()}`;

    // Status Tag
    const op = snap.operational_state || 'NORMAL';
    elements.detailOpState.textContent = op;
    elements.detailOpState.className = `tag ${op === 'CRITICAL' ? 'tag-critical' : (op === 'WARNING' ? 'tag-warning' : 'tag-normal')}`;

    // Circular Dial (circumference = 2 * PI * 50 = 314.15)
    const mhi = snap.overall_health_index || 100.0;
    elements.dialMhiVal.textContent = `${Math.round(mhi)}%`;
    const offset = 314.15 * (1 - mhi / 100);
    elements.dialProgress.style.strokeDashoffset = offset;

    if (mhi < 60) {
      elements.dialProgress.style.stroke = '#ff3366';
    } else if (mhi < 80) {
      elements.dialProgress.style.stroke = '#ffb700';
    } else {
      elements.dialProgress.style.stroke = '#00f0ff';
    }

    // RUL Banner
    elements.detailRulVal.innerHTML = `${(rul.rul_minutes || 190.0).toFixed(1)} <small>min</small>`;
    elements.detailWearRate.textContent = `${(rul.degradation_rate_pct_per_hour || 0.05).toFixed(2)}% / hr`;
    elements.detailFaultLoc.textContent = snap.primary_fault_subsystem || 'Spindle Bearing Race';

    // Subsystems
    const sub = snap.subsystem_health || {};
    updateSubsystemBar(elements.subVibVal, elements.subVibBar, sub.vibration || 100);
    updateSubsystemBar(elements.subThermVal, elements.subThermBar, sub.thermal || 100);
    updateSubsystemBar(elements.subPressVal, elements.subPressBar, sub.pressure || 100);
    updateSubsystemBar(elements.subElectVal, elements.subElectBar, sub.electrical || 100);

    // Latest Telemetry Stats
    if (data.telemetry && data.telemetry.length > 0) {
      const latest = data.telemetry[data.telemetry.length - 1];
      elements.pillTemp.textContent = `${latest.temperature.toFixed(1)} °C`;
      elements.pillVib.textContent = `${latest.vibration.toFixed(2)} mm/s`;
      elements.pillPress.textContent = `${latest.pressure.toFixed(1)} bar`;
      elements.pillCurr.textContent = `${latest.current.toFixed(1)} A`;
      elements.pillRpm.textContent = `${Math.round(latest.rpm)} RPM`;
    }
  }

  function updateSubsystemBar(valEl, barEl, value) {
    const v = Math.round(value);
    valEl.textContent = `${v}%`;
    barEl.style.width = `${Math.max(0, Math.min(100, v))}%`;
    if (v < 60) {
      barEl.style.background = '#ff3366';
      barEl.style.boxShadow = '0 0 8px #ff3366';
    } else if (v < 80) {
      barEl.style.background = '#ffb700';
      barEl.style.boxShadow = '0 0 8px #ffb700';
    }
  }

  /* ==========================================================================
     Canvas Chart Renderers
     ========================================================================== */
  function renderTelemetryChart() {
    const canvas = elements.telemetryCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const records = state.telemetryData;

    ctx.clearRect(0, 0, width, height);

    // Draw dark background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 40; x < width; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 20; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!records || records.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting Sensor Telemetry Stream...', width / 2, height / 2);
      return;
    }

    const n = records.length;
    const paddingLeft = 45;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 30;
    const plotW = width - paddingLeft - paddingRight;
    const plotH = height - paddingTop - paddingBottom;

    // Check for failure events to draw red danger zones
    for (let i = 0; i < n; i++) {
      if (records[i].failure === 1) {
        const x = paddingLeft + (i / (n - 1)) * plotW;
        ctx.fillStyle = 'rgba(255, 51, 102, 0.15)';
        ctx.fillRect(x - 6, paddingTop, 12, plotH);
      }
    }

    // Signals to plot
    const series = [
      { key: 'temperature', color: '#ff3366', min: 40, max: 120 },
      { key: 'vibration', color: '#00f0ff', min: 0.5, max: 8.0 },
      { key: 'pressure', color: '#ffb700', min: 2, max: 200 },
      { key: 'current', color: '#a855f7', min: 5, max: 60 },
    ];

    series.forEach(s => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 4;
      ctx.beginPath();

      for (let i = 0; i < n; i++) {
        const val = records[i][s.key] || 0;
        const normY = (val - s.min) / (s.max - s.min);
        const clampedNorm = Math.max(0, Math.min(1, normY));
        const x = paddingLeft + (i / Math.max(1, n - 1)) * plotW;
        const y = paddingTop + (1 - clampedNorm) * plotH;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    });

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, paddingTop);
    ctx.lineTo(paddingLeft, height - paddingBottom);
    ctx.lineTo(width - paddingRight, height - paddingBottom);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('100%', paddingLeft - 8, paddingTop + 6);
    ctx.fillText('50%', paddingLeft - 8, paddingTop + plotH / 2 + 4);
    ctx.fillText('0%', paddingLeft - 8, height - paddingBottom);

    ctx.textAlign = 'center';
    ctx.fillText('T - 50 Ticks (10s sample)', paddingLeft + plotW / 2, height - 10);
  }

  function renderFLConvergenceChart() {
    const canvas = elements.flConvergenceCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const history = state.flHistory;

    ctx.clearRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 30; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
    }

    if (!history || history.length === 0) return;

    const padL = 50;
    const padR = 30;
    const padT = 30;
    const padB = 40;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const rounds = history.length;

    // Draw Macro F1 Curve (Cyan)
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    for (let i = 0; i < rounds; i++) {
      const f1 = history[i].macro_f1 || 0.5;
      const x = padL + (i / Math.max(1, rounds - 1)) * plotW;
      const y = padT + (1 - f1) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw Loss Curve (Amber)
    ctx.strokeStyle = '#ffb700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < rounds; i++) {
      const loss = history[i].avg_train_loss || 0.2;
      const normLoss = Math.min(1, loss / 0.4);
      const x = padL + (i / Math.max(1, rounds - 1)) * plotW;
      const y = padT + (1 - normLoss) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Data dots & Round Labels
    for (let i = 0; i < rounds; i++) {
      const x = padL + (i / Math.max(1, rounds - 1)) * plotW;
      const f1 = history[i].macro_f1 || 0.5;
      const yF1 = padT + (1 - f1) * plotH;

      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.arc(x, yF1, 4, 0, Math.PI * 2);
      ctx.fill();

      // Label below
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`R${history[i].round || (i + 1)}`, x, height - 15);
    }

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, height - padB);
    ctx.lineTo(width - padR, height - padB);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', padL - 8, padT + 4);
    ctx.fillText('0.5', padL - 8, padT + plotH / 2);
    ctx.fillText('0.0', padL - 8, height - padB);
  }

  function renderWhatIfChart() {
    const canvas = elements.whatifDecayCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 20; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
    }

    const padL = 45;
    const padR = 25;
    const padT = 20;
    const padB = 30;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Critical Threshold Line at 25% (H_crit)
    const critY = padT + (1 - 0.25) * plotH;
    ctx.strokeStyle = 'rgba(255, 51, 102, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, critY);
    ctx.lineTo(width - padR, critY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ff3366';
    ctx.font = '10px monospace';
    ctx.fillText('H_crit (25%)', width - padR - 65, critY - 6);

    // Curve 1: Nominal Degradation (Emerald dashed)
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let t = 0; t <= 60; t++) {
      const h = 99.0 * Math.exp(-0.0035 * t);
      const x = padL + (t / 60) * plotW;
      const y = padT + (1 - h / 100) * plotH;
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Curve 2: Stressed Simulation Degradation (Amber / Crimson)
    const wearFactor = state.whatIfResult ? state.whatIfResult.wear_acceleration_factor : 1.34;
    ctx.strokeStyle = wearFactor > 1.2 ? '#ff3366' : '#ffb700';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    for (let t = 0; t <= 60; t++) {
      const h = 99.0 * Math.exp(-0.0035 * wearFactor * t);
      const x = padL + (t / 60) * plotW;
      const y = padT + (1 - Math.max(0, h) / 100) * plotH;
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Axis
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, height - padB);
    ctx.lineTo(width - padR, height - padB);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.fillText('100%', padL - 8, padT + 6);
    ctx.fillText('0 min', padL, height - 10);
    ctx.fillText('Projection: 60 min', width - padR - 90, height - 10);
  }

  function renderWhatIfResults(data) {
    if (!data || !data.simulation) return;
    const sim = data.simulation;
    elements.whatifResTitle.textContent = `Simulation: ${sim.scenario_name}`;
    elements.whatifImpactTag.textContent = sim.rul_delta_minutes < 0
      ? `RUL Reduced by ${Math.abs(sim.rul_delta_minutes).toFixed(1)} min`
      : `RUL Extended by ${sim.rul_delta_minutes.toFixed(1)} min`;

    elements.whatifBaseRul.textContent = `${sim.baseline_rul_minutes.toFixed(1)} min`;
    elements.whatifProjRul.textContent = `${sim.projected_rul_minutes.toFixed(1)} min`;
    elements.whatifWearFactor.textContent = `${sim.wear_acceleration_factor.toFixed(2)}x Normal`;
    elements.whatifRecText.textContent = sim.recommendation;

    renderWhatIfChart();
  }

  function renderWorkOrders(orders) {
    elements.workOrdersList.innerHTML = '';
    if (!orders || orders.length === 0) {
      elements.workOrdersList.innerHTML = `
        <div class="card" style="text-align: center; color: #64748b; padding: 40px;">
          <p>No active prescriptive work orders found for the selected filter.</p>
          <small>Work orders auto-generate when a Digital Twin enters WARNING or CRITICAL state.</small>
        </div>
      `;
      return;
    }

    orders.forEach(wo => {
      const card = document.createElement('div');
      const isCrit = wo.urgency.includes('CRITICAL');
      card.className = `wo-card ${isCrit ? 'border-critical' : 'border-warning'}`;

      card.innerHTML = `
        <div class="wo-card-header">
          <div>
            <span class="wo-id">${wo.work_order_id} • Dispatched at ${wo.timestamp}</span>
            <div class="wo-machine">${wo.machine_name} (${wo.machine_id})</div>
          </div>
          <span class="tag ${isCrit ? 'tag-critical' : 'tag-warning'}">${wo.urgency}</span>
        </div>
        <div class="wo-grid">
          <div>
            <span class="wo-item-title">Affected Subsystem</span>
            <div class="wo-item-text">${wo.affected_subsystem}</div>
          </div>
          <div>
            <span class="wo-item-title">Machine Health Index</span>
            <div class="wo-item-text" style="color: ${isCrit ? '#ff3366' : '#ffb700'}">${wo.machine_health_index}% (RUL: ${wo.estimated_rul_minutes}m)</div>
          </div>
          <div>
            <span class="wo-item-title">Estimated Downtime</span>
            <div class="wo-item-text">${wo.estimated_downtime_hours} Hours</div>
          </div>
          <div>
            <span class="wo-item-title">Skill Level</span>
            <div class="wo-item-text">${wo.technician_skill_level}</div>
          </div>
        </div>
        <div>
          <span class="wo-item-title">Prescriptive Maintenance Action</span>
          <p style="font-size: 0.88rem; color: #e2e8f0; margin-top: 4px;">${wo.recommended_action}</p>
        </div>
        <div class="wo-footer">
          <div class="wo-parts">Required Spares: <strong>${(wo.required_spare_parts || []).join(', ')}</strong></div>
          <div style="display: flex; gap: 8px;">
            ${wo.status === 'OPEN' ? `
              <button class="btn btn-sm btn-secondary" onclick="window.updateStatus('${wo.work_order_id}', 'ASSIGNED')">Assign Tech</button>
              <button class="btn btn-sm btn-primary" onclick="window.updateStatus('${wo.work_order_id}', 'RESOLVED')">Resolve</button>
            ` : (wo.status === 'ASSIGNED' ? `
              <button class="btn btn-sm btn-primary" onclick="window.updateStatus('${wo.work_order_id}', 'RESOLVED')">Mark Resolved</button>
            ` : `<span class="tag tag-success">RESOLVED</span>`)}
          </div>
        </div>
      `;
      elements.workOrdersList.appendChild(card);
    });
  }

  // Global handler for inline button calls
  window.updateStatus = (id, newStatus) => {
    updateOrderStatus(id, newStatus);
  };

  /* ==========================================================================
     Event Listeners
     ========================================================================== */

  // Machine selector segmented controls
  document.querySelectorAll('#machine-selector .seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectClient(btn.getAttribute('data-client'));
    });
  });

  // Playback Step Buttons
  elements.btnStepOne.addEventListener('click', () => stepSimulation(1));
  elements.btnStepTen.addEventListener('click', () => stepSimulation(10));
  elements.btnResetSim.addEventListener('click', () => resetSimulation());

  elements.btnToggleAuto.addEventListener('click', () => {
    state.autoStreaming = !state.autoStreaming;
    if (state.autoStreaming) {
      elements.btnToggleAuto.innerHTML = '<span class="btn-icon">&#10074;&#10074;</span> Pause Stream';
      elements.btnToggleAuto.classList.replace('btn-primary', 'btn-secondary');
      state.autoStreamInterval = setInterval(() => stepSimulation(1), 1000);
    } else {
      elements.btnToggleAuto.innerHTML = '<span class="btn-icon">&#9658;</span> Auto-Stream';
      elements.btnToggleAuto.classList.replace('btn-secondary', 'btn-primary');
      clearInterval(state.autoStreamInterval);
    }
  });

  // Federated Form Sliders & Toggles
  elements.flMu.addEventListener('input', (e) => {
    elements.flMuVal.textContent = parseFloat(e.target.value).toFixed(3);
  });

  elements.flDpToggle.addEventListener('change', (e) => {
    elements.dpOptionsRow.style.display = e.target.checked ? 'grid' : 'none';
  });

  elements.flForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const strategy = elements.flStrategy.value;
    const rounds = parseInt(elements.flRounds.value);
    const epochs = parseInt(elements.flEpochs.value);
    const mu = parseFloat(elements.flMu.value);
    const enableDp = elements.flDpToggle.checked;
    const clipNorm = parseFloat(elements.flClipNorm.value);
    const noiseMult = parseFloat(elements.flNoiseMult.value);
    const enableSecagg = elements.flSecaggToggle.checked;

    elements.btnLaunchFl.disabled = true;
    elements.btnLaunchFl.innerHTML = '<span class="pulse-dot"></span> Orchestrating Federated Rounds...';

    try {
      const res = await fetch('/api/federated/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: strategy,
          num_rounds: rounds,
          local_epochs: epochs,
          learning_rate: 0.001,
          enable_dp: enableDp,
          clip_norm: clipNorm,
          noise_multiplier: noiseMult,
          enable_secagg: enableSecagg,
          mu: mu,
        }),
      });

      const data = await res.json();
      if (data.convergence) {
        state.flHistory = data.convergence;
        elements.resMacroF1.textContent = (data.best_macro_f1 || 0.8687).toFixed(4);
        elements.resBestRound.textContent = `Round ${data.best_round || rounds}`;
        elements.resEpsilon.textContent = enableDp ? `ε = ${data.final_epsilon}` : 'Unlimited';
        renderFLConvergenceChart();
      }
    } catch (err) {
      console.error('Federated training failed:', err);
    } finally {
      elements.btnLaunchFl.disabled = false;
      elements.btnLaunchFl.innerHTML = '<span class="btn-icon">&#9658;</span> Launch Federated Training Round';
    }
  });

  // What-If Presets
  elements.presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.getAttribute('data-preset');

      if (preset === 'production_surge') {
        elements.inputRpmMult.value = 1.20;
        elements.lblRpmMult.textContent = '1.20x';
        elements.inputLoadMult.value = 1.15;
        elements.lblLoadMult.textContent = '1.15x';
        elements.inputTempDelta.value = 0.0;
        elements.lblTempDelta.textContent = '+0.0 °C';
      } else if (preset === 'summer_heatwave') {
        elements.inputRpmMult.value = 1.00;
        elements.lblRpmMult.textContent = '1.00x';
        elements.inputLoadMult.value = 1.00;
        elements.lblLoadMult.textContent = '1.00x';
        elements.inputTempDelta.value = 12.0;
        elements.lblTempDelta.textContent = '+12.0 °C';
      } else if (preset === 'protective_derating') {
        elements.inputRpmMult.value = 0.80;
        elements.lblRpmMult.textContent = '0.80x';
        elements.inputLoadMult.value = 0.75;
        elements.lblLoadMult.textContent = '0.75x';
        elements.inputTempDelta.value = 0.0;
        elements.lblTempDelta.textContent = '+0.0 °C';
      }
    });
  });

  elements.inputRpmMult.addEventListener('input', e => elements.lblRpmMult.textContent = `${parseFloat(e.target.value).toFixed(2)}x`);
  elements.inputLoadMult.addEventListener('input', e => elements.lblLoadMult.textContent = `${parseFloat(e.target.value).toFixed(2)}x`);
  elements.inputTempDelta.addEventListener('input', e => elements.lblTempDelta.textContent = `${parseFloat(e.target.value) >= 0 ? '+' : ''}${parseFloat(e.target.value).toFixed(1)} °C`);

  elements.btnRunWhatif.addEventListener('click', () => runWhatIfSimulation());

  // Work order filters
  elements.woFilterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.woFilterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.woFilter = btn.getAttribute('data-filter');
      fetchWorkOrders();
    });
  });

  // Export Work Orders JSON
  elements.btnExportOrders.addEventListener('click', () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.workOrders, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `work_orders_${Date.now()}.json`);
    dlAnchor.click();
  });

  /* ==========================================================================
     Initial Load
     ========================================================================== */
  fetchOverview();
  fetchTelemetry(state.selectedClient);
  fetchWorkOrders();
  renderFLConvergenceChart();
  renderWhatIfChart();
});
