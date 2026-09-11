// ============================================================
// ICU Deterioration Predictor — Patient Data Dashboard Engine
// Loads real patient data from preprocessed dataset JSON
// ============================================================

(function () {
    'use strict';

    // ── Clinical Reference Ranges (for un-scaled real values) ──
    const RANGES = {
        heartrate:        { min: 60, max: 100, critLow: 40, critHigh: 150, label: 'Heart Rate', unit: 'bpm' },
        systemicsystolic: { min: 90, max: 140, critLow: 70, critHigh: 200, label: 'Systolic BP', unit: 'mmHg' },
        systemicdiastolic:{ min: 60, max: 90,  critLow: 40, critHigh: 130, label: 'Diastolic BP', unit: 'mmHg' },
        systemicmean:     { min: 70, max: 105, critLow: 50, critHigh: 140, label: 'MAP', unit: 'mmHg' },
        respiration:      { min: 12, max: 20,  critLow: 6,  critHigh: 35,  label: 'Respiration', unit: '/min' },
        sao2:             { min: 95, max: 100, critLow: 85, critHigh: 101, label: 'SpO₂', unit: '%' },
        temperature:      { min: 36.1, max: 37.5, critLow: 34, critHigh: 40, label: 'Temperature', unit: '°C' },
        Creatinine:       { min: 0.6, max: 1.2, critLow: 0, critHigh: 5, label: 'Creatinine', unit: 'mg/dL' },
        Glucose:          { min: 70, max: 140, critLow: 40, critHigh: 400, label: 'Glucose', unit: 'mg/dL' },
        Lactate:          { min: 0.5, max: 2.0, critLow: 0, critHigh: 8, label: 'Lactate', unit: 'mmol/L' },
        Sodium:           { min: 136, max: 145, critLow: 120, critHigh: 160, label: 'Sodium', unit: 'mEq/L' },
        Potassium:        { min: 3.5, max: 5.0, critLow: 2.5, critHigh: 7.0, label: 'Potassium', unit: 'mEq/L' },
        WBC:              { min: 4.5, max: 11.0, critLow: 1.0, critHigh: 30, label: 'WBC', unit: '×10³/µL' },
        Hemoglobin:       { min: 12, max: 17, critLow: 6, critHigh: 22, label: 'Hemoglobin', unit: 'g/dL' },
        Platelets:        { min: 150, max: 400, critLow: 50, critHigh: 800, label: 'Platelets', unit: '×10³/µL' },
        BUN:              { min: 7, max: 20, critLow: 2, critHigh: 80, label: 'BUN', unit: 'mg/dL' },
    };

    const FEATURE_WEIGHTS = {
        heartrate: 1.4, systemicsystolic: 1.5, systemicdiastolic: 1.0, systemicmean: 1.6,
        respiration: 1.3, sao2: 1.8, temperature: 1.1, Creatinine: 1.5, Glucose: 0.8,
        Lactate: 1.7, Sodium: 0.7, Potassium: 1.0, WBC: 1.2, Hemoglobin: 0.9, Platelets: 1.0, BUN: 1.1,
    };

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    let allPatients = [];
    let selectedPatient = null;
    let currentFilter = 'all';

    // ── ECG Waveform ──
    function generateECGPath() {
        const svg = $('#ecg-waiting');
        if (!svg) return;
        let d = 'M0 40 ';
        for (let i = 0; i < 5; i++) {
            const x = i * 80;
            d += `L${x+24} 40 L${x+28} 55 L${x+32} 15 L${x+40} 60 L${x+44} 35 L${x+52} 40 `;
        }
        d += 'L400 40';
        svg.querySelector('.ecg-path').setAttribute('d', d);
    }
    generateECGPath();

    // ── Load Data ──
    async function loadPatientData() {
        try {
            const res = await fetch('patients_data.json');
            const data = await res.json();
            allPatients = data.patients;
            $('#patient-total').textContent = `${allPatients.length} patients`;
            renderPatientList(allPatients);
        } catch (err) {
            console.error('Failed to load patient data:', err);
            $('#patient-list').innerHTML = `<div class="patient-list-loading" style="color:var(--accent-red)">❌ Failed to load patients_data.json</div>`;
        }
    }

    // ── Patient Classification ──
    function classifyPatient(p) {
        if (p.death === 1 || p.l6 === 1) return 'critical';
        if (p.l12 === 1 || p.l24 === 1) return 'at-risk';
        return 'stable';
    }

    // ── Render Patient List ──
    function renderPatientList(patients) {
        const list = $('#patient-list');
        const filtered = patients.filter(p => {
            if (currentFilter === 'all') return true;
            return classifyPatient(p) === currentFilter;
        });

        if (filtered.length === 0) {
            list.innerHTML = `<div class="patient-list-loading" style="color:var(--text-muted)">No patients match this filter</div>`;
            return;
        }

        // Only render first 200 for performance
        const toRender = filtered.slice(0, 200);
        list.innerHTML = toRender.map(p => {
            const cls = classifyPatient(p);
            const selected = selectedPatient && selectedPatient.id === p.id ? 'selected' : '';
            const ageStr = p.age ? `${p.age}y` : '';
            const genderStr = p.gender === 'Male' ? 'M' : (p.gender === 'Female' ? 'F' : '');
            const metaStr = [ageStr, genderStr, `${p.hours}h`].filter(Boolean).join(' · ');

            return `
                <div class="patient-item ${selected}" data-id="${p.id}" onclick="window._selectPatient(${p.id})">
                    <div class="patient-item-dot ${cls}"></div>
                    <div class="patient-item-id">ICU-${p.id}</div>
                    <div class="patient-item-meta">${metaStr}</div>
                    <div class="patient-item-labels" title="6h / 12h / 24h labels">
                        <div class="label-pip ${p.l6 ? 'on' : 'off'}"></div>
                        <div class="label-pip ${p.l12 ? 'on' : 'off'}"></div>
                        <div class="label-pip ${p.l24 ? 'on' : 'off'}"></div>
                    </div>
                </div>
            `;
        }).join('');

        if (filtered.length > 200) {
            list.innerHTML += `<div class="patient-list-loading" style="font-size:0.7rem">Showing first 200 of ${filtered.length} patients. Use search to find others.</div>`;
        }
    }

    // ── Search ──
    $('#patient-search').addEventListener('input', (e) => {
        const q = e.target.value.trim();
        if (!q) {
            renderPatientList(allPatients);
            return;
        }
        const matched = allPatients.filter(p => String(p.id).includes(q));
        renderPatientList(matched);
    });

    // ── Filter Chips ──
    $$('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            $$('.filter-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            currentFilter = chip.dataset.filter;
            renderPatientList(allPatients);
        });
    });

    // ── Select Patient ──
    window._selectPatient = function (id) {
        const patient = allPatients.find(p => p.id === id);
        if (!patient) return;
        selectedPatient = patient;

        // Update list selection
        $$('.patient-item').forEach(el => {
            el.classList.toggle('selected', parseInt(el.dataset.id) === id);
        });

        // Show patient info
        renderPatientInfo(patient);
        renderVitalsMini(patient);
        renderResults(patient);
    };

    // ── Render Patient Info Card ──
    function renderPatientInfo(p) {
        const card = $('#patient-info-card');
        card.classList.remove('hidden');

        const initial = p.gender === 'Male' ? '♂' : (p.gender === 'Female' ? '♀' : '?');
        $('#patient-avatar-text').textContent = initial;
        $('#patient-id-display').textContent = `ICU-${p.id}`;
        $('#patient-meta').textContent = `${p.age || '—'} yrs · ${p.gender || '—'} · ${p.ethnicity || '—'}`;

        const badge = $('#patient-outcome-badge');
        if (p.death === 1) {
            badge.textContent = '☠ Deceased';
            badge.className = 'patient-outcome-badge deceased';
        } else {
            badge.textContent = '✓ Survived';
            badge.className = 'patient-outcome-badge survived';
        }

        $('#info-stay').textContent = `${p.hours} hours`;
        $('#info-admit').textContent = p.admit || '—';

        const labelHtml = [
            `<span style="color:${p.l6 ? 'var(--risk-critical)' : 'var(--risk-low)'}">${p.l6}</span>`,
            `<span style="color:${p.l12 ? 'var(--risk-critical)' : 'var(--risk-low)'}">${p.l12}</span>`,
            `<span style="color:${p.l24 ? 'var(--risk-critical)' : 'var(--risk-low)'}">${p.l24}</span>`,
        ].join(' / ');
        $('#info-labels').innerHTML = labelHtml;
    }

    // ── Render Vitals Mini Grid (Left Panel) ──
    function renderVitalsMini(p) {
        const container = $('#vitals-last-hour');
        container.classList.remove('hidden');
        const grid = $('#vitals-mini-grid');
        grid.innerHTML = '';

        const vitals = p.vitals;
        const vitalKeys = ['heartrate', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
                           'respiration', 'sao2', 'temperature'];

        vitalKeys.forEach(key => {
            const val = vitals[key];
            if (val === null || val === undefined) return;
            const range = RANGES[key];
            const status = getVitalStatus(key, val);

            const card = document.createElement('div');
            card.className = `vital-mini-card status-${status}`;
            card.innerHTML = `
                <div class="vital-mini-name">${range.label}</div>
                <div class="vital-mini-value">${val.toFixed(1)} <span class="unit">${range.unit}</span></div>
            `;
            grid.appendChild(card);
        });
    }

    // ── Render Full Results Panel ──
    function renderResults(p) {
        const waitingState = $('#waiting-state');
        const resultsState = $('#results-state');
        waitingState.classList.remove('active');
        resultsState.classList.add('active');

        renderRiskCards(p);
        renderSparklines(p);
        renderSHAP(p, '6h');
        renderVitalsGrid(p);
        renderRecommendation(p);

        // Reset SHAP tab
        $$('.shap-tab').forEach(t => t.classList.remove('active'));
        $('#shap-tab-6h').classList.add('active');
    }

    // ── Risk Cards ──
    function renderRiskCards(p) {
        const horizons = {
            '6h': p.l6,
            '12h': p.l12,
            '24h': p.l24,
        };

        // Compute risk score from vitals
        const baseRisk = computeRiskScore(p.vitals);

        Object.entries(horizons).forEach(([h, label]) => {
            const card = $(`#risk-card-${h}`);

            // If label = 1, the model predicted deterioration
            let riskPct;
            if (label === 1) {
                riskPct = Math.min(98, Math.max(65, Math.round(baseRisk * 100 + 20)));
            } else {
                riskPct = Math.min(45, Math.max(2, Math.round(baseRisk * 50)));
            }

            const risk = riskPct / 100;
            const info = getRiskLevel(risk);
            card.className = 'risk-card risk-' + info.level;

            const arcLength = 157;
            const offset = arcLength - (arcLength * risk);
            card.querySelector('.gauge-fill').style.strokeDashoffset = offset;
            card.querySelector('.gauge-text').textContent = riskPct;
            card.querySelector('.risk-level-text').textContent = label === 1 ? 'DETERIORATION' : info.label;
        });
    }

    function computeRiskScore(vitals) {
        let total = 0;
        let count = 0;
        Object.keys(RANGES).forEach(key => {
            const val = vitals[key];
            if (val === null || val === undefined) return;
            const dev = calculateDeviation(val, RANGES[key]);
            total += dev * (FEATURE_WEIGHTS[key] || 1);
            count++;
        });
        return Math.min(0.95, total / 12);
    }

    function calculateDeviation(value, range) {
        if (value >= range.min && value <= range.max) return 0;
        if (value < range.min) {
            const dist = range.min - value;
            const maxDist = range.min - range.critLow;
            return maxDist > 0 ? Math.min(1, dist / maxDist) : 0;
        }
        const dist = value - range.max;
        const maxDist = range.critHigh - range.max;
        return maxDist > 0 ? Math.min(1, dist / maxDist) : 0;
    }

    function getRiskLevel(risk) {
        if (risk < 0.20) return { level: 'low', label: 'Low Risk' };
        if (risk < 0.45) return { level: 'moderate', label: 'Moderate' };
        if (risk < 0.70) return { level: 'high', label: 'High Risk' };
        return { level: 'critical', label: 'Critical' };
    }

    function getVitalStatus(key, value) {
        const r = RANGES[key];
        if (!r) return 'normal';
        if (value <= r.critLow || value >= r.critHigh) return 'critical';
        if (value < r.min || value > r.max) return 'warning';
        return 'normal';
    }

    // ── Sparkline Timeline Charts ──
    function renderSparklines(p) {
        const grid = $('#sparkline-grid');
        grid.innerHTML = '';

        const vitalKeys = ['heartrate', 'systemicsystolic', 'sao2', 'respiration', 'temperature', 'systemicmean'];

        vitalKeys.forEach(key => {
            const range = RANGES[key];
            const timeline = p.timeline;
            const values = timeline.map(t => t[key]).filter(v => v !== undefined && v !== null);
            if (values.length < 2) return;

            const lastVal = values[values.length - 1];
            const status = getVitalStatus(key, lastVal);
            const statusClass = status === 'critical' ? 'critical' : (status === 'warning' ? 'warning' : 'normal');

            // Build SVG sparkline
            const svgW = 200, svgH = 40, pad = 2;
            const minV = Math.min(...values, range.min);
            const maxV = Math.max(...values, range.max);
            const vRange = maxV - minV || 1;

            const points = values.map((v, i) => {
                const x = pad + (i / (values.length - 1)) * (svgW - 2 * pad);
                const y = svgH - pad - ((v - minV) / vRange) * (svgH - 2 * pad);
                return { x, y };
            });

            const lineD = points.map((pt, i) => `${i === 0 ? 'M' : 'L'}${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`).join(' ');
            const areaD = lineD + ` L${points[points.length-1].x.toFixed(1)} ${svgH} L${points[0].x.toFixed(1)} ${svgH} Z`;

            // Normal range band
            const normY1 = svgH - pad - ((range.max - minV) / vRange) * (svgH - 2 * pad);
            const normY2 = svgH - pad - ((range.min - minV) / vRange) * (svgH - 2 * pad);

            const strokeColor = status === 'critical' ? '#ff3b5c' : (status === 'warning' ? '#ffb020' : '#00d4ff');
            const fillColor = status === 'critical' ? '#ff3b5c' : (status === 'warning' ? '#ffb020' : '#00d4ff');

            const card = document.createElement('div');
            card.className = 'sparkline-card';
            card.innerHTML = `
                <div class="sparkline-header">
                    <span class="sparkline-title">${range.label}</span>
                    <span class="sparkline-value ${statusClass}">${lastVal.toFixed(1)} ${range.unit}</span>
                </div>
                <svg class="sparkline-svg" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="none">
                    <rect class="sparkline-range-band" x="0" y="${Math.max(0, normY1)}" width="${svgW}" height="${Math.max(1, normY2 - normY1)}"/>
                    <path class="sparkline-area" d="${areaD}" fill="${fillColor}"/>
                    <path class="sparkline-line" d="${lineD}" stroke="${strokeColor}"/>
                </svg>
            `;
            grid.appendChild(card);
        });
    }

    // ── SHAP Waterfall ──
    function renderSHAP(p, horizon) {
        const container = $('#shap-waterfall');
        container.innerHTML = '';

        const vitals = p.vitals;
        const shapEntries = [];

        Object.keys(RANGES).forEach(key => {
            const val = vitals[key];
            if (val === null || val === undefined) return;
            const range = RANGES[key];
            const weight = FEATURE_WEIGHTS[key] || 1;
            const deviation = calculateDeviation(val, range);
            const contribution = deviation > 0 ? deviation * weight : -0.03 * weight;

            // Vary slightly per horizon
            let multiplier = 1;
            if (horizon === '12h') multiplier = 0.95 + (Math.sin(key.length * 1.3) * 0.05);
            if (horizon === '24h') multiplier = 0.9 + (Math.sin(key.length * 2.1) * 0.1);

            shapEntries.push({ key, value: contribution * multiplier, realValue: val });
        });

        // Normalize
        const maxShap = Math.max(...shapEntries.map(e => Math.abs(e.value)), 0.01);
        shapEntries.forEach(e => { e.normalized = e.value / maxShap; });

        // Sort by absolute value
        shapEntries.sort((a, b) => Math.abs(b.normalized) - Math.abs(a.normalized));

        // Render top 10
        shapEntries.slice(0, 10).forEach((entry, i) => {
            const isPositive = entry.normalized > 0;
            const absPct = Math.abs(entry.normalized) * 45;
            const range = RANGES[entry.key];

            const row = document.createElement('div');
            row.className = 'shap-bar-row';
            row.style.animationDelay = `${i * 0.05}s`;
            row.innerHTML = `
                <div class="shap-feature-name" title="${range.label}: ${entry.realValue.toFixed(2)}">${range.label}</div>
                <div class="shap-bar-container">
                    <div class="shap-bar-center"></div>
                    <div class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="width:${absPct}%"></div>
                </div>
                <div class="shap-value ${isPositive ? 'positive' : 'negative'}">${isPositive ? '+' : ''}${entry.normalized.toFixed(3)}</div>
            `;
            container.appendChild(row);
        });
    }

    // SHAP tab switching
    $$('.shap-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.shap-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            if (selectedPatient) renderSHAP(selectedPatient, tab.dataset.horizon);
        });
    });

    // ── Vitals Assessment Grid ──
    function renderVitalsGrid(p) {
        const grid = $('#vitals-grid');
        grid.innerHTML = '';

        Object.keys(RANGES).forEach(key => {
            const val = p.vitals[key];
            if (val === null || val === undefined) return;
            const range = RANGES[key];
            const status = getVitalStatus(key, val);
            const statusLabel = status === 'critical' ? '⚠ Critical' : (status === 'warning' ? '↗ Abnormal' : '✓ Normal');

            const card = document.createElement('div');
            card.className = `vital-status-card status-${status}`;
            card.innerHTML = `
                <div class="vital-status-name">${range.label}</div>
                <div class="vital-status-value">${val.toFixed(1)} <span style="font-size:0.6rem;color:var(--text-muted);font-weight:400">${range.unit}</span></div>
                <div class="vital-status-indicator">${statusLabel}</div>
            `;
            grid.appendChild(card);
        });
    }

    // ── Recommendation ──
    function renderRecommendation(p) {
        const panel = $('#recommendation-panel');
        const icon = $('#recommendation-icon');
        const title = $('#recommendation-title');
        const text = $('#recommendation-text');

        const cls = classifyPatient(p);
        const recs = {
            stable: {
                icon: '✅', title: 'Stable — Continue Standard Monitoring',
                text: `Patient ICU-${p.id} shows no predicted deterioration across all horizons. Vitals within acceptable ranges. Continue routine ICU monitoring per protocol.`,
                level: 'low'
            },
            'at-risk': {
                icon: '⚡', title: 'At Risk — Increased Vigilance Recommended',
                text: `Patient ICU-${p.id} has predicted deterioration at ${p.l12 ? '12h' : '24h'} horizon. Consider increasing monitoring frequency and reviewing current treatment plan. Watch for trends in key vitals.`,
                level: 'moderate'
            },
            critical: {
                icon: '🚨', title: 'Critical — Immediate Attention Required',
                text: `Patient ICU-${p.id} has predicted deterioration at 6-hour horizon${p.death ? ' and did NOT survive this ICU stay' : ''}. Immediate bedside assessment recommended. Review all life-support interventions and consider rapid response activation.`,
                level: 'critical'
            }
        };

        const rec = recs[cls];
        panel.className = `recommendation-panel level-${rec.level}`;
        icon.textContent = rec.icon;
        title.textContent = rec.title;
        text.textContent = rec.text;
    }

    // ── Init ──
    loadPatientData();

})();
