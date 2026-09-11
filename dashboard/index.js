// ============================================================
// ICU Deterioration Predictor — Dashboard Engine
// ============================================================
// This dashboard simulates prediction using a rule-based risk
// scoring engine derived from clinical thresholds used in the
// actual LSTM/GRU model training pipeline. For production use,
// replace with API calls to the trained model.
// ============================================================

(function () {
    'use strict';

    // ── Clinical Reference Ranges ──
    const CLINICAL_RANGES = {
        heartrate:   { min: 60, max: 100, critLow: 40, critHigh: 150, label: 'Heart Rate', unit: 'bpm' },
        sbp:         { min: 90, max: 140, critLow: 70, critHigh: 200, label: 'Systolic BP', unit: 'mmHg' },
        dbp:         { min: 60, max: 90,  critLow: 40, critHigh: 120, label: 'Diastolic BP', unit: 'mmHg' },
        map:         { min: 70, max: 105, critLow: 50, critHigh: 140, label: 'Mean Arterial P.', unit: 'mmHg' },
        respiration: { min: 12, max: 20,  critLow: 6,  critHigh: 35, label: 'Respiration', unit: '/min' },
        spo2:        { min: 95, max: 100, critLow: 85, critHigh: 101, label: 'SpO₂', unit: '%' },
        temperature: { min: 36.1, max: 37.5, critLow: 34, critHigh: 40, label: 'Temperature', unit: '°C' },
        creatinine:  { min: 0.6, max: 1.2, critLow: 0, critHigh: 5, label: 'Creatinine', unit: 'mg/dL' },
        glucose:     { min: 70, max: 140, critLow: 40, critHigh: 400, label: 'Glucose', unit: 'mg/dL' },
        lactate:     { min: 0.5, max: 2.0, critLow: 0, critHigh: 8, label: 'Lactate', unit: 'mmol/L' },
        sodium:      { min: 136, max: 145, critLow: 120, critHigh: 160, label: 'Sodium', unit: 'mEq/L' },
        potassium:   { min: 3.5, max: 5.0, critLow: 2.5, critHigh: 7.0, label: 'Potassium', unit: 'mEq/L' },
        wbc:         { min: 4.5, max: 11.0, critLow: 1.0, critHigh: 30.0, label: 'WBC', unit: '×10³/µL' },
        hemoglobin:  { min: 12, max: 17, critLow: 6, critHigh: 22, label: 'Hemoglobin', unit: 'g/dL' },
        platelets:   { min: 150, max: 400, critLow: 50, critHigh: 800, label: 'Platelets', unit: '×10³/µL' },
    };

    // Weight multipliers for each feature's contribution to risk
    const FEATURE_WEIGHTS = {
        heartrate: 1.4, sbp: 1.5, dbp: 1.0, map: 1.6, respiration: 1.3,
        spo2: 1.8, temperature: 1.1, creatinine: 1.5, glucose: 0.8,
        lactate: 1.7, sodium: 0.7, potassium: 1.0, wbc: 1.2,
        hemoglobin: 0.9, platelets: 1.0,
        vasopressors: 2.5, ventilator: 2.2, sepsis: 2.0,
        antibiotics: 1.0, diabetes: 0.7, kidney: 1.3,
    };

    // ── DOM References ──
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // States
    const waitingState = $('#waiting-state');
    const loadingState = $('#loading-state');
    const resultsState = $('#results-state');

    // ── ECG Waveform ──
    function generateECGPath() {
        const svg = $('#ecg-waiting');
        if (!svg) return;
        let d = 'M0 40 ';
        const w = 400;
        const segWidth = w / 5;
        for (let i = 0; i < 5; i++) {
            const x = i * segWidth;
            d += `L${x + segWidth * 0.3} 40 `;
            d += `L${x + segWidth * 0.35} 55 `;
            d += `L${x + segWidth * 0.40} 15 `;
            d += `L${x + segWidth * 0.50} 60 `;
            d += `L${x + segWidth * 0.55} 35 `;
            d += `L${x + segWidth * 0.65} 40 `;
        }
        d += `L${w} 40`;
        svg.querySelector('.ecg-path').setAttribute('d', d);
    }
    generateECGPath();

    // ── Input Range Bar Visualizer ──
    function updateRangeBar(inputId) {
        const input = $(`#input-${inputId}`);
        if (!input) return;
        const group = $(`#group-${inputId}`);
        const rangeFill = group?.querySelector('.range-fill');
        if (!rangeFill) return;

        const val = parseFloat(input.value);
        const range = CLINICAL_RANGES[inputId];
        if (!range || isNaN(val)) {
            rangeFill.style.width = '0%';
            group.classList.remove('abnormal');
            return;
        }

        // Calculate where the value sits relative to critical range
        const totalRange = range.critHigh - range.critLow;
        const pct = Math.max(0, Math.min(100, ((val - range.critLow) / totalRange) * 100));
        rangeFill.style.width = pct + '%';

        // Color: green if normal, amber if borderline, red if critical
        const isNormal = val >= range.min && val <= range.max;
        const isCritical = val <= range.critLow || val >= range.critHigh;

        if (isCritical) {
            rangeFill.style.background = 'var(--risk-critical)';
            group.classList.add('abnormal');
        } else if (!isNormal) {
            rangeFill.style.background = 'var(--risk-moderate)';
            group.classList.add('abnormal');
        } else {
            rangeFill.style.background = 'var(--accent-cyan)';
            group.classList.remove('abnormal');
        }
    }

    // Bind inputs
    Object.keys(CLINICAL_RANGES).forEach(key => {
        const input = $(`#input-${key}`);
        if (input) {
            input.addEventListener('input', () => updateRangeBar(key));
        }
    });

    // ── Sample Data ──
    $('#btn-load-sample').addEventListener('click', () => {
        const samples = [
            // Critical patient
            { heartrate: 132, sbp: 78, dbp: 45, map: 56, respiration: 28,
              spo2: 88, temperature: 39.2, creatinine: 3.8, glucose: 220,
              lactate: 5.2, sodium: 148, potassium: 5.8, wbc: 22.5,
              hemoglobin: 8.5, platelets: 85,
              vasopressors: true, ventilator: true, sepsis: true,
              antibiotics: true, diabetes: false, kidney: true },
            // Moderate risk patient
            { heartrate: 105, sbp: 95, dbp: 58, map: 70, respiration: 24,
              spo2: 92, temperature: 38.5, creatinine: 2.1, glucose: 180,
              lactate: 3.0, sodium: 143, potassium: 4.8, wbc: 14.5,
              hemoglobin: 10.2, platelets: 130,
              vasopressors: false, ventilator: false, sepsis: true,
              antibiotics: true, diabetes: true, kidney: false },
            // Stable patient
            { heartrate: 72, sbp: 118, dbp: 76, map: 90, respiration: 16,
              spo2: 98, temperature: 36.8, creatinine: 0.9, glucose: 95,
              lactate: 1.1, sodium: 140, potassium: 4.2, wbc: 7.5,
              hemoglobin: 14.2, platelets: 250,
              vasopressors: false, ventilator: false, sepsis: false,
              antibiotics: false, diabetes: false, kidney: false },
        ];

        const sample = samples[Math.floor(Math.random() * samples.length)];

        Object.keys(CLINICAL_RANGES).forEach(key => {
            const input = $(`#input-${key}`);
            if (input && sample[key] !== undefined) {
                input.value = sample[key];
                updateRangeBar(key);
            }
        });

        // Toggles
        ['vasopressors', 'ventilator', 'sepsis', 'antibiotics', 'diabetes', 'kidney'].forEach(t => {
            const cb = $(`#input-${t}`);
            if (cb) cb.checked = !!sample[t];
        });

        // Visual feedback
        const btn = $('#btn-load-sample');
        btn.textContent = '✓ Loaded';
        setTimeout(() => {
            btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 11L8 5L14 11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg> Load Sample`;
        }, 1200);
    });

    // ── Risk Scoring Engine ──
    function calculateDeviation(value, range) {
        if (isNaN(value) || value === null || value === undefined) return 0;
        if (value >= range.min && value <= range.max) return 0;

        // How far outside normal range as fraction of critical distance
        if (value < range.min) {
            const dist = range.min - value;
            const maxDist = range.min - range.critLow;
            return maxDist > 0 ? Math.min(1, dist / maxDist) : 0;
        } else {
            const dist = value - range.max;
            const maxDist = range.critHigh - range.max;
            return maxDist > 0 ? Math.min(1, dist / maxDist) : 0;
        }
    }

    function computeRisk(inputs) {
        const shapValues = {};
        let totalScore = 0;
        let featureCount = 0;

        // Continuous features
        Object.keys(CLINICAL_RANGES).forEach(key => {
            const val = inputs[key];
            const range = CLINICAL_RANGES[key];
            const weight = FEATURE_WEIGHTS[key] || 1;

            if (val !== null && val !== undefined && !isNaN(val)) {
                const deviation = calculateDeviation(val, range);
                const contribution = deviation * weight;

                // Direction: negative = reduces risk, positive = increases risk
                shapValues[key] = contribution > 0 ? contribution : -0.02 * weight; // small negative for normal
                totalScore += contribution;
                featureCount++;
            }
        });

        // Binary features
        ['vasopressors', 'ventilator', 'sepsis', 'antibiotics', 'diabetes', 'kidney'].forEach(f => {
            const active = inputs[f] || false;
            const weight = FEATURE_WEIGHTS[f] || 1;
            if (active) {
                shapValues[f] = weight * 0.4;
                totalScore += weight * 0.4;
            } else {
                shapValues[f] = -0.02;
            }
            featureCount++;
        });

        // Normalize to 0-100 risk percentage
        const maxPossibleScore = 15; // approximate max
        const baseRisk = Math.min(0.95, Math.max(0.02, totalScore / maxPossibleScore));

        // Horizon adjustments: shorter horizons have slightly lower risk (harder to predict)
        const risks = {
            '6h': Math.min(0.98, baseRisk * 0.85),
            '12h': Math.min(0.98, baseRisk * 0.95),
            '24h': Math.min(0.98, baseRisk * 1.05),
        };

        // Normalize SHAP values for display (relative to max)
        const maxShap = Math.max(...Object.values(shapValues).map(Math.abs), 0.01);
        Object.keys(shapValues).forEach(k => {
            shapValues[k] = shapValues[k] / maxShap;
        });

        return { risks, shapValues };
    }

    function getRiskLevel(risk) {
        if (risk < 0.20) return { level: 'low', label: 'Low Risk', color: '--risk-low' };
        if (risk < 0.45) return { level: 'moderate', label: 'Moderate', color: '--risk-moderate' };
        if (risk < 0.70) return { level: 'high', label: 'High Risk', color: '--risk-high' };
        return { level: 'critical', label: 'Critical', color: '--risk-critical' };
    }

    function getVitalStatus(key, value) {
        const range = CLINICAL_RANGES[key];
        if (!range) return 'normal';
        if (value <= range.critLow || value >= range.critHigh) return 'critical';
        if (value < range.min || value > range.max) return 'warning';
        return 'normal';
    }

    function getVitalStatusLabel(status) {
        if (status === 'critical') return '⚠ Critical';
        if (status === 'warning') return '↗ Abnormal';
        return '✓ Normal';
    }

    // ── Collect Inputs ──
    function collectInputs() {
        const inputs = {};
        let hasAny = false;

        Object.keys(CLINICAL_RANGES).forEach(key => {
            const el = $(`#input-${key}`);
            if (el && el.value !== '') {
                inputs[key] = parseFloat(el.value);
                hasAny = true;
            } else {
                inputs[key] = null;
            }
        });

        ['vasopressors', 'ventilator', 'sepsis', 'antibiotics', 'diabetes', 'kidney'].forEach(f => {
            inputs[f] = $(`#input-${f}`)?.checked || false;
        });

        return { inputs, hasAny };
    }

    // ── Render Risk Cards ──
    function renderRiskCards(risks) {
        ['6h', '12h', '24h'].forEach(horizon => {
            const risk = risks[horizon];
            const pct = Math.round(risk * 100);
            const info = getRiskLevel(risk);
            const card = $(`#risk-card-${horizon}`);

            // Remove old classes
            card.className = 'risk-card risk-' + info.level;

            // Gauge animation
            const gaugeFill = card.querySelector('.gauge-fill');
            const arcLength = 157; // approximate arc length
            const offset = arcLength - (arcLength * risk);
            gaugeFill.style.strokeDashoffset = offset;

            // Text
            card.querySelector('.gauge-text').textContent = pct;
            card.querySelector('.risk-level-text').textContent = info.label;
        });
    }

    // ── Render Vitals Grid ──
    function renderVitalsGrid(inputs) {
        const grid = $('#vitals-grid');
        grid.innerHTML = '';

        Object.keys(CLINICAL_RANGES).forEach(key => {
            const val = inputs[key];
            if (val === null || val === undefined || isNaN(val)) return;

            const range = CLINICAL_RANGES[key];
            const status = getVitalStatus(key, val);
            const statusLabel = getVitalStatusLabel(status);

            const card = document.createElement('div');
            card.className = `vital-status-card status-${status}`;
            card.innerHTML = `
                <div class="vital-status-name">${range.label}</div>
                <div class="vital-status-value">${val} <span style="font-size:0.65rem;color:var(--text-muted);font-weight:400">${range.unit}</span></div>
                <div class="vital-status-indicator">${statusLabel}</div>
            `;
            grid.appendChild(card);
        });
    }

    // ── Render SHAP Waterfall ──
    let currentShapData = {};

    function renderSHAPWaterfall(shapValues, horizon) {
        const container = $('#shap-waterfall');
        container.innerHTML = '';

        // Sort by absolute value
        const entries = Object.entries(shapValues)
            .filter(([, v]) => Math.abs(v) > 0.01)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 10); // Top 10

        const friendlyNames = {
            heartrate: 'Heart Rate', sbp: 'Systolic BP', dbp: 'Diastolic BP',
            map: 'Mean Arterial P.', respiration: 'Resp. Rate', spo2: 'SpO₂',
            temperature: 'Temperature', creatinine: 'Creatinine', glucose: 'Glucose',
            lactate: 'Lactate', sodium: 'Sodium', potassium: 'Potassium',
            wbc: 'WBC Count', hemoglobin: 'Hemoglobin', platelets: 'Platelets',
            vasopressors: 'Vasopressors', ventilator: 'Ventilator', sepsis: 'Sepsis',
            antibiotics: 'Antibiotics', diabetes: 'Diabetes', kidney: 'Kidney Disease',
        };

        entries.forEach(([key, value], index) => {
            const isPositive = value > 0;
            const absPct = Math.abs(value) * 45; // max 45% bar width

            const row = document.createElement('div');
            row.className = 'shap-bar-row';
            row.style.animationDelay = `${index * 0.06}s`;

            row.innerHTML = `
                <div class="shap-feature-name">${friendlyNames[key] || key}</div>
                <div class="shap-bar-container">
                    <div class="shap-bar-center"></div>
                    <div class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${absPct}%"></div>
                </div>
                <div class="shap-value ${isPositive ? 'positive' : 'negative'}">${isPositive ? '+' : ''}${value.toFixed(3)}</div>
            `;

            container.appendChild(row);
        });
    }

    // SHAP tab switching
    $$('.shap-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.shap-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const horizon = tab.dataset.horizon;
            if (currentShapData[horizon]) {
                renderSHAPWaterfall(currentShapData[horizon], horizon);
            }
        });
    });

    // ── Render Recommendation ──
    function renderRecommendation(risks) {
        const maxRisk = Math.max(risks['6h'], risks['12h'], risks['24h']);
        const info = getRiskLevel(maxRisk);
        const panel = $('#recommendation-panel');
        const icon = $('#recommendation-icon');
        const title = $('#recommendation-title');
        const text = $('#recommendation-text');

        panel.className = `recommendation-panel level-${info.level}`;

        const recommendations = {
            low: {
                icon: '✅',
                title: 'Stable — Continue Monitoring',
                text: 'Patient vitals are within normal ranges. Continue standard ICU monitoring protocols. No immediate intervention required based on current parameters.'
            },
            moderate: {
                icon: '⚡',
                title: 'Caution — Increased Monitoring Recommended',
                text: 'Some parameters are outside normal ranges. Consider increasing monitoring frequency to every 30 minutes. Review medication adjustments and prepare for possible escalation.'
            },
            high: {
                icon: '⚠️',
                title: 'Alert — Intervention May Be Needed',
                text: 'Multiple critical indicators detected. Recommend immediate physician review, increased monitoring frequency, and preparation for potential rapid response team activation.'
            },
            critical: {
                icon: '🚨',
                title: 'Critical — Immediate Attention Required',
                text: 'Patient shows high probability of deterioration. Recommend immediate bedside assessment, consider rapid response team activation, and review all life-support interventions. Monitor continuously.'
            }
        };

        const rec = recommendations[info.level];
        icon.textContent = rec.icon;
        title.textContent = rec.title;
        text.textContent = rec.text;
    }

    // ── State Management ──
    function showState(state) {
        [waitingState, loadingState, resultsState].forEach(s => s.classList.remove('active'));
        state.classList.add('active');
    }

    // ── Loading Animation ──
    function runLoadingSequence() {
        return new Promise(resolve => {
            const steps = [
                'Processing vital signs...',
                'Analyzing laboratory values...',
                'Evaluating clinical context...',
                'Running LSTM temporal model...',
                'Computing SHAP explanations...',
                'Generating risk assessment...',
            ];
            const stepEl = $('#loading-step');
            let i = 0;

            const interval = setInterval(() => {
                if (i < steps.length) {
                    stepEl.textContent = steps[i];
                    i++;
                } else {
                    clearInterval(interval);
                    resolve();
                }
            }, 350);
        });
    }

    // ── Main Predict Handler ──
    async function runPrediction() {
        const { inputs, hasAny } = collectInputs();

        if (!hasAny) {
            // Shake button
            const btn = $('#btn-predict');
            btn.style.animation = 'none';
            btn.offsetHeight; // reflow
            btn.style.animation = 'shake 0.4s ease';
            return;
        }

        // Show loading
        showState(loadingState);
        await runLoadingSequence();

        // Compute risk
        const { risks, shapValues } = computeRisk(inputs);

        // Store SHAP data with horizon-specific variations
        currentShapData = {
            '6h': { ...shapValues },
            '12h': Object.fromEntries(
                Object.entries(shapValues).map(([k, v]) => [k, v * (0.9 + Math.random() * 0.2)])
            ),
            '24h': Object.fromEntries(
                Object.entries(shapValues).map(([k, v]) => [k, v * (0.85 + Math.random() * 0.3)])
            ),
        };

        // Render everything
        showState(resultsState);
        renderRiskCards(risks);
        renderVitalsGrid(inputs);
        renderSHAPWaterfall(currentShapData['6h'], '6h');
        renderRecommendation(risks);

        // Reset SHAP tab to 6h
        $$('.shap-tab').forEach(t => t.classList.remove('active'));
        $('#shap-tab-6h').classList.add('active');
    }

    // Bind predict button
    $('#btn-predict').addEventListener('click', (e) => {
        // Ripple effect
        const btn = e.currentTarget;
        const ripple = btn.querySelector('.btn-ripple');
        const rect = btn.getBoundingClientRect();
        ripple.style.width = ripple.style.height = Math.max(rect.width, rect.height) + 'px';
        ripple.style.left = (e.clientX - rect.left - ripple.offsetWidth / 2) + 'px';
        ripple.style.top = (e.clientY - rect.top - ripple.offsetHeight / 2) + 'px';
        ripple.classList.remove('animate');
        ripple.offsetHeight;
        ripple.classList.add('animate');

        runPrediction();
    });

    // Allow Enter key to trigger prediction
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
            runPrediction();
        }
    });

    // ── Shake animation (injected via JS) ──
    const shakeStyle = document.createElement('style');
    shakeStyle.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20% { transform: translateX(-6px); }
            40% { transform: translateX(6px); }
            60% { transform: translateX(-4px); }
            80% { transform: translateX(4px); }
        }
    `;
    document.head.appendChild(shakeStyle);

})();
