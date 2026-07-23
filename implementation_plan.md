# eICU Preprocessing Pipeline — Implementation Plan

Complete data preprocessing pipeline for **"Adaptive ICU Patient Deterioration Prediction using Temporal Deep Learning and Explainable AI"**.

## Current State

| Item | Status |
|---|---|
| Data files (`.csv.gz`) | ✅ All 6 in `required files/` |
| Python 3.14.6 | ✅ Installed |
| pandas, numpy, sklearn | ❌ Not installed — need `pip install` |
| respiratoryCharting.csv.gz | ✅ Present in `required files/` |

## User Review Required

> [!IMPORTANT]
> **Python packages need to be installed.** I will run `pip install pandas numpy scikit-learn` before executing the pipeline. Approve if OK.

> [!WARNING]
> **The eICU demo dataset is small** (~2,500 patients). The pipeline will work identically on the full dataset but expect limited sequence counts from the demo.

## Open Questions

> [!IMPORTANT]
> 1. **Sequence length**: Your spec says `Sequence Length = 24`. Should this be **24 hours** (each timestep = 1 hour)? I'll default to this.
> 2. **Train/Val/Test split**: I'll use **70/15/15** by `patientunitstayid` (no patient leakage). OK?
> 3. **Scaler choice**: I'll use **StandardScaler** for vitals/labs (Gaussian-like) and keep binary flags unscaled. OK?

---

## Proposed Changes

### Project Structure (All inside `database preproccesssing/`)

```
preprocessing_pipeline/
├── config.py                  # All paths, column lists, thresholds
├── step01_load_data.py        # Load .csv.gz, select columns
├── step02_clean_data.py       # Dedup, type conversion, impossible values
├── step03_impute_missing.py   # Median/forward-fill/backward-fill + report
├── step04_outlier_removal.py  # Physiological range clipping + report
├── step05_feature_engineer.py # Lab pivot, merge tables, derived features
├── step06_encode_scale.py     # One-hot/label encode, StandardScaler
├── step07_target_sequence.py  # Target labels, LSTM sequence generation
├── run_pipeline.py            # Master orchestrator — runs all steps
├── reports/                   # Auto-generated reports
│   ├── missing_value_report.csv
│   ├── outlier_summary.csv
│   ├── feature_list.txt
│   └── data_dictionary.md
└── output/                    # Final outputs
    ├── processed_icu_dataset.csv
    ├── processed_lstm_dataset.npy
    ├── X_train.npy / X_val.npy / X_test.npy
    ├── y_train.npy / y_val.npy / y_test.npy
    └── scaler.pkl
```

---

### [NEW] `config.py`
Central configuration: file paths, column selections, lab names, medication keywords, outlier thresholds, sequence length, split ratios.

---

### [NEW] `step01_load_data.py`
- Load all 6 `.csv.gz` files via `pd.read_csv(..., compression='gzip')`
- Select only required columns per table
- Filter `lab.csv` to 9 lab tests, `medication.csv` to 6 drug categories (keyword matching), `respiratoryCharting.csv` to FiO₂/PEEP/Ventilator Mode
- Filter `diagnosis.csv` to 8 conditions via `diagnosisstring LIKE` matching
- Save intermediate cleaned tables as pickle files

---

### [NEW] `step02_clean_data.py`
- Remove duplicate rows across all tables
- Remove duplicate `patientunitstayid` in `patient` (keep first)
- Convert `age` (handles `> 89` string → 90), convert numeric types
- Remove rows with all-null vital/lab values
- Generate cleaning summary

---

### [NEW] `step03_impute_missing.py`
- Generate **before** missing-value report (counts + percentages per column)
- Numerical: median imputation → forward fill → backward fill (per patient)
- Categorical: most-frequent imputation
- Generate **after** missing-value report
- Save both reports to `reports/missing_value_report.csv`

---

### [NEW] `step04_outlier_removal.py`
- Apply physiological range clipping:

| Feature | Min | Max |
|---|---|---|
| Heart Rate | 30 | 220 |
| Systolic BP | 40 | 300 |
| Diastolic BP | 20 | 200 |
| Mean BP | 25 | 250 |
| Respiration | 4 | 60 |
| SpO₂ | 50 | 100 |
| Temperature | 30 | 45 |
| Lab values | 0 | varies |

- Values outside range → set to NaN → re-impute with forward fill
- Generate `reports/outlier_summary.csv` (count of clipped values per feature)

---

### [NEW] `step05_feature_engineer.py`
- **Lab pivot**: Long → wide format (`labname` becomes columns: Glucose, Creatinine, etc.)
- **Merge all tables** on `patientunitstayid` and aligned time offsets (hourly bins)
- **Derived features**:
  - BMI = weight / (height/100)²
  - Pulse Pressure = systolic − diastolic
  - MAP (verify vs `systemicmean`)
  - Shock Index = heartrate / systolic
  - Rolling mean/std (window=6h) for vitals and labs
  - Rate of change (diff) for HR, BP, SpO₂, respiration
  - Trend features (slope over past 6 observations)

---

### [NEW] `step06_encode_scale.py`
- **One-Hot Encode**: gender, ethnicity
- **Binary flags**: 8 diagnosis categories, 6 medication categories, ventilator status
- **Label Encode**: any remaining categorical if needed
- **StandardScaler**: fit on train split, transform all splits
- Save scaler as `output/scaler.pkl`

---

### [NEW] `step07_target_sequence.py`
- **Target generation**:
  - `target_6h = 1` if deterioration (death/discharge to expired) occurs within 6h of current timestep
  - `target_12h`, `target_24h` similarly
  - Based on `unitdischargestatus == 'Expired'` and `unitdischargeoffset`
- **Sequence generation**:
  - Sliding window of length 24 (hours) per patient
  - Shape: `(N_sequences, 24, F)` where F ≈ 40–55 features
- **Train/Val/Test split** (70/15/15 by patient, no leakage)
- Save: `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`
- Save flat: `processed_icu_dataset.csv`, `processed_lstm_dataset.npy`

---

### [NEW] `run_pipeline.py`
- Orchestrates all 7 steps in sequence
- Prints progress, timing, and final dataset statistics
- Generates `reports/data_dictionary.md` and `reports/feature_list.txt`

---

## Verification Plan

### Automated Tests
```bash
pip install pandas numpy scikit-learn
python run_pipeline.py
```

### Manual Verification
- Check `reports/missing_value_report.csv` for before/after imputation
- Check `reports/outlier_summary.csv` for clipped counts
- Verify final shapes printed by `run_pipeline.py` match `(N, 24, F)` format
- Inspect `processed_icu_dataset.csv` for sanity (no nulls, correct columns)
- Verify `output/` contains all expected `.npy` and `.pkl` files
