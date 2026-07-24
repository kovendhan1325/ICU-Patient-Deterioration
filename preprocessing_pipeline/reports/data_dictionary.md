# Data Dictionary

## Project
**Adaptive ICU Patient Deterioration Prediction using Temporal Deep Learning and Explainable AI**

## Dataset
eICU Collaborative Research Database v2.0.1 (Demo)

---

## Dataset Shapes

| Split | Sequences | Shape |
|---|---|---|
| Train | 64,121 | (64121, 24, 37) |
| Val | 15,389 | (15389, 24, 37) |
| Test | 13,667 | (13667, 24, 37) |

- **Sequence Length**: 24 hours
- **Number of Features**: 37
- **Prediction Horizons**: [6, 12, 24]
- **Patients (Train/Val/Test)**: 1654/354/356

---

## Feature Descriptions

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `heartrate` | Continuous | Heart rate (beats per minute) |
| 2 | `systemicsystolic` | Continuous | Systolic blood pressure (mmHg) |
| 3 | `systemicdiastolic` | Continuous | Diastolic blood pressure (mmHg) |
| 4 | `systemicmean` | Continuous | Mean arterial pressure from monitor (mmHg) |
| 5 | `respiration` | Continuous | Respiratory rate (breaths per minute) |
| 6 | `temperature` | Continuous | Body temperature (Celsius) |
| 7 | `Bilirubin` | Continuous | Feature: Bilirubin |
| 8 | `Creatinine` | Continuous | Feature: Creatinine |
| 9 | `Glucose` | Continuous | Feature: Glucose |
| 10 | `Hemoglobin` | Continuous | Feature: Hemoglobin |
| 11 | `Lactate` | Continuous | Feature: Lactate |
| 12 | `Platelets` | Continuous | Feature: Platelets |
| 13 | `Potassium` | Continuous | Feature: Potassium |
| 14 | `Sodium` | Continuous | Feature: Sodium |
| 15 | `WBC` | Continuous | Feature: WBC |
| 16 | `resp_fio2` | Continuous | Respiratory charting: fio2 |
| 17 | `resp_peep` | Continuous | Respiratory charting: peep |
| 18 | `age` | Continuous | Patient age (years, capped at 90) |
| 19 | `diag_Diabetes` | Binary | Binary flag: 1 if patient diagnosed with Diabetes |
| 20 | `diag_Hypertension` | Binary | Binary flag: 1 if patient diagnosed with Hypertension |
| 21 | `diag_Heart_Disease` | Binary | Binary flag: 1 if patient diagnosed with Heart Disease |
| 22 | `diag_Kidney_Disease` | Binary | Binary flag: 1 if patient diagnosed with Kidney Disease |
| 23 | `diag_Sepsis` | Binary | Binary flag: 1 if patient diagnosed with Sepsis |
| 24 | `diag_Pneumonia` | Binary | Binary flag: 1 if patient diagnosed with Pneumonia |
| 25 | `diag_COPD` | Binary | Binary flag: 1 if patient diagnosed with COPD |
| 26 | `diag_Liver_Disease` | Binary | Binary flag: 1 if patient diagnosed with Liver Disease |
| 27 | `med_Antibiotics` | Binary | Binary flag: 1 if patient received Antibiotics |
| 28 | `med_Vasopressors` | Binary | Binary flag: 1 if patient received Vasopressors |
| 29 | `med_Sedatives` | Binary | Binary flag: 1 if patient received Sedatives |
| 30 | `med_Insulin` | Binary | Binary flag: 1 if patient received Insulin |
| 31 | `med_Anticoagulants` | Binary | Binary flag: 1 if patient received Anticoagulants |
| 32 | `med_Steroids` | Binary | Binary flag: 1 if patient received Steroids |
| 33 | `bmi` | Continuous | Body Mass Index = weight / (height/100)^2 |
| 34 | `pulse_pressure` | Continuous | Systolic - Diastolic blood pressure |
| 35 | `map_calculated` | Continuous | Calculated Mean Arterial Pressure |
| 36 | `shock_index` | Continuous | Heart Rate / Systolic Blood Pressure |
| 37 | `gender_Male` | Binary | One-hot encoded gender: Male |

---

## Target Variables

| Target | Description |
|---|---|
| `target_6h` | 1 if patient deteriorates (dies) within 6 hours of current timestep |
| `target_12h` | 1 if patient deteriorates (dies) within 12 hours of current timestep |
| `target_24h` | 1 if patient deteriorates (dies) within 24 hours of current timestep |

## Output Files

| File | Description |
|---|---|
| `processed_icu_dataset.csv` | Flat preprocessed dataset (all features + targets) |
| `processed_lstm_dataset.npy` | Combined LSTM data dictionary (pickle) |
| `X_train.npy` | Training input sequences |
| `X_val.npy` | Validation input sequences |
| `X_test.npy` | Test input sequences |
| `y_train.npy` | Training labels |
| `y_val.npy` | Validation labels |
| `y_test.npy` | Test labels |
| `scaler.pkl` | Fitted StandardScaler for inference |