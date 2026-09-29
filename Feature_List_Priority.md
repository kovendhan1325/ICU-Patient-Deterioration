# ICU Patient Deterioration Prediction — Feature Documentation

**Project:** ICU Patient Deterioration Prediction System  
**Dataset:** `eicu_hourly_labeled_6h_12h_24h.csv`  
**Source:** eICU Collaborative Research Database  
**Total Records:** ~94,049 hourly observations  
**Date:** September 2026

---

## Overview

This document lists all features used in the ICU Patient Deterioration Prediction project, ordered by **priority/importance** based on the model's SHAP-derived feature weights and clinical relevance for predicting patient deterioration at 6-hour, 12-hour, and 24-hour horizons.

---

## Feature Categories

The features are organized into three main categories:

| Category | Count | Description |
|----------|-------|-------------|
| Vital Signs (Time-Series) | 7 | Continuous physiological measurements recorded hourly |
| Laboratory Values (Time-Series) | 9 | Blood test results updated periodically |
| Demographic & Administrative | 5 | Static patient information |
| Missingness Indicators | 16 | Binary flags indicating imputed values |
| Target Labels & Metadata | 8 | Prediction targets and data splits (NOT used as inputs) |

---

## Features Listed by Priority (High → Low)

### 🔴 Priority 1 — Critical Predictors (Highest Importance)

These features have the **strongest influence** on deterioration prediction based on SHAP analysis and clinical significance.

| # | Feature Name | Type | Unit | Normal Range | Weight | Description |
|---|-------------|------|------|-------------|--------|-------------|
| 1 | **`sao2`** | Vital Sign | % | 95–100 | **1.8** | Peripheral oxygen saturation (SpO₂). The single most important predictor — low SpO₂ strongly signals respiratory failure and imminent deterioration. |
| 2 | **`Lactate`** | Lab Value | mmol/L | 0.5–2.0 | **1.7** | Blood lactate level. Elevated lactate indicates tissue hypoperfusion, sepsis, or shock — a hallmark of clinical deterioration. |
| 3 | **`systemicmean`** | Vital Sign | mmHg | 70–105 | **1.6** | Mean Arterial Pressure (MAP). Critical indicator of organ perfusion. MAP < 65 mmHg is a key criterion for septic shock. |
| 4 | **`systemicsystolic`** | Vital Sign | mmHg | 90–140 | **1.5** | Systolic Blood Pressure. Hypotension (< 90 mmHg) is a red flag for hemodynamic instability. |
| 5 | **`Creatinine`** | Lab Value | mg/dL | 0.6–1.2 | **1.5** | Serum creatinine. Rising creatinine indicates acute kidney injury (AKI), a common organ failure in ICU deterioration. |

---

### 🟠 Priority 2 — Important Predictors (High Importance)

These features significantly contribute to the model's predictions.

| # | Feature Name | Type | Unit | Normal Range | Weight | Description |
|---|-------------|------|------|-------------|--------|-------------|
| 6 | **`heartrate`** | Vital Sign | bpm | 60–100 | **1.4** | Heart rate. Tachycardia (> 100 bpm) or bradycardia (< 60 bpm) signals cardiovascular stress or compensatory mechanisms. |
| 7 | **`respiration`** | Vital Sign | /min | 12–20 | **1.3** | Respiratory rate. Tachypnea (> 20/min) is one of the earliest signs of clinical deterioration and a qSOFA criterion. |
| 8 | **`WBC`** | Lab Value | ×10³/µL | 4.5–11.0 | **1.2** | White Blood Cell count. Leukocytosis or leukopenia indicates infection, sepsis, or immune response — key deterioration triggers. |

---

### 🟡 Priority 3 — Moderate Predictors

These features provide supporting information for prediction accuracy.

| # | Feature Name | Type | Unit | Normal Range | Weight | Description |
|---|-------------|------|------|-------------|--------|-------------|
| 9 | **`temperature`** | Vital Sign | °C | 36.1–37.5 | **1.1** | Body temperature. Fever (> 38.3°C) or hypothermia (< 36°C) are SIRS criteria and indicate infection/sepsis. |
| 10 | **`BUN`** | Lab Value | mg/dL | 7–20 | **1.1** | Blood Urea Nitrogen. Elevated BUN reflects renal dysfunction and dehydration — early indicators of organ failure. |
| 11 | **`systemicdiastolic`** | Vital Sign | mmHg | 60–90 | **1.0** | Diastolic Blood Pressure. Complements systolic BP in assessing cardiovascular stability. |
| 12 | **`Potassium`** | Lab Value | mEq/L | 3.5–5.0 | **1.0** | Serum potassium. Both hypo- and hyperkalemia cause dangerous cardiac arrhythmias in ICU patients. |
| 13 | **`Platelets`** | Lab Value | ×10³/µL | 150–400 | **1.0** | Platelet count. Thrombocytopenia is a component of DIC and sepsis-related coagulopathy. |

---

### 🟢 Priority 4 — Supporting Predictors (Lower Importance)

These features add supplementary context to the prediction model.

| # | Feature Name | Type | Unit | Normal Range | Weight | Description |
|---|-------------|------|------|-------------|--------|-------------|
| 14 | **`Hemoglobin`** | Lab Value | g/dL | 12–17 | **0.9** | Blood hemoglobin. Anemia (low Hgb) affects oxygen delivery and can exacerbate deterioration. |
| 15 | **`Glucose`** | Lab Value | mg/dL | 70–140 | **0.8** | Blood glucose. Both hypoglycemia and hyperglycemia are associated with poor ICU outcomes. |
| 16 | **`Sodium`** | Lab Value | mEq/L | 136–145 | **0.7** | Serum sodium. Dysnatremia affects neurological function and fluid balance management. |

---

### 🔵 Priority 5 — Demographic & Contextual Features

Static features providing patient background context.

| # | Feature Name | Type | Values/Range | Description |
|---|-------------|------|-------------|-------------|
| 17 | **`age`** | Demographic | Numeric (years) | Patient age. Older patients have higher baseline risk. Encoded as `age_numeric`. |
| 18 | **`gender`** | Demographic | Male / Female | Patient biological sex. Encoded as `gender_encoded` (binary). |
| 19 | **`ethnicity`** | Demographic | Caucasian, African American, etc. | Patient ethnicity. Contextual demographic information. |
| 20 | **`unitadmitsource`** | Administrative | ICU, ER, Floor, etc. | Source of ICU admission — indicates severity at entry. |
| 21 | **`unitstaytype`** | Administrative | admit, readmit, stepdown/other | Type of ICU stay — readmissions may indicate higher risk. |

---

### ⚪ Priority 6 — Missingness Indicator Features

Binary flags (0 or 1) indicating whether each corresponding feature value was **imputed** (missing in the original record). These help the model learn patterns in data availability itself — missingness can be clinically informative (e.g., a test not ordered may indicate stability).

| # | Feature Name | Original Feature |
|---|-------------|-----------------|
| 22 | `heartrate_missing` | heartrate |
| 23 | `systemicsystolic_missing` | systemicsystolic |
| 24 | `systemicdiastolic_missing` | systemicdiastolic |
| 25 | `systemicmean_missing` | systemicmean |
| 26 | `respiration_missing` | respiration |
| 27 | `sao2_missing` | sao2 |
| 28 | `temperature_missing` | temperature |
| 29 | `WBC_missing` | WBC |
| 30 | `Hemoglobin_missing` | Hemoglobin |
| 31 | `Platelets_missing` | Platelets |
| 32 | `Creatinine_missing` | Creatinine |
| 33 | `BUN_missing` | BUN |
| 34 | `Sodium_missing` | Sodium |
| 35 | `Potassium_missing` | Potassium |
| 36 | `Glucose_missing` | Glucose |
| 37 | `Lactate_missing` | Lactate |

---

## Target Variables (NOT Used as Input Features)

These columns are **prediction targets** and metadata — they are **excluded** from model input to prevent data leakage.

| Column | Description |
|--------|-------------|
| `label_6h` | Binary label — will the patient deteriorate within the next **6 hours**? |
| `label_12h` | Binary label — will the patient deteriorate within the next **12 hours**? |
| `label_24h` | Binary label — will the patient deteriorate within the next **24 hours**? |
| `icu_death` | Binary — did the patient die during the ICU stay? |
| `hours_to_event` | Hours until deterioration event |
| `discharge_hour` | Hour of ICU discharge |
| `unitdischargeoffset` | Discharge offset in minutes |
| `split` | Data split assignment: `train` / `val` / `test` |

---

## Feature Engineering in the Model Pipeline

In the model pipeline (Steps 6–9), additional derived features are created from the 24-hour time-series for **classical ML models** (Logistic Regression, Random Forest, XGBoost):

| Derived Feature | Description |
|----------------|-------------|
| **Last Value** | Most recent hourly reading for each feature |
| **Mean** | Average value across the 24-hour observation window |
| **Standard Deviation** | Variability/instability measure over 24 hours |
| **Linear Trend** | Slope of the feature over time — captures worsening or improving trajectory |

> For **deep learning models** (LSTM, GRU), raw 24-hour sequences of all features are used directly as input without summary feature engineering.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total columns in dataset | 48 |
| Input features (used by model) | 37 |
| Excluded columns (targets + metadata) | 8 |
| Identifier columns | 3 |
| Vital sign features | 7 |
| Lab value features | 9 |
| Demographic features | 5 |
| Missingness indicators | 16 |

---

*Document generated for the ICU Patient Deterioration Prediction Project.*
