# Data Dictionary

## Project
**Adaptive ICU Patient Deterioration Prediction using Temporal Deep Learning and Explainable AI**

## Dataset
eICU Collaborative Research Database v2.0.1 (Demo)

---

## Dataset Shapes

| Split | Sequences | Shape |
|---|---|---|
| Train | 64,121 | (64121, 24, 69) |
| Val | 15,389 | (15389, 24, 69) |
| Test | 13,667 | (13667, 24, 69) |

- **Sequence Length**: 24 hours
- **Number of Features**: 69
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
| 16 | `resp_fio2_(%)` | Continuous | Respiratory charting: fio2 (%) |
| 17 | `resp_fio2` | Continuous | Respiratory charting: fio2 |
| 18 | `resp_peep` | Continuous | Respiratory charting: peep |
| 19 | `resp_peep/cpap` | Continuous | Respiratory charting: peep/cpap |
| 20 | `resp_ps_above_peep` | Continuous | Respiratory charting: ps above peep |
| 21 | `resp_set_fraction_of_inspired_oxygen_(fio2)` | Continuous | Respiratory charting: set fraction of inspired oxygen (fio2) |
| 22 | `resp_unable_to_obtain_peepi_and_vtrap` | Continuous | Respiratory charting: unable to obtain peepi and vtrap |
| 23 | `age` | Continuous | Patient age (years, capped at 90) |
| 24 | `admissionheight` | Continuous | Height at admission (cm) |
| 25 | `admissionweight` | Continuous | Weight at admission (kg) |
| 26 | `diag_Diabetes` | Binary | Binary flag: 1 if patient diagnosed with Diabetes |
| 27 | `diag_Hypertension` | Binary | Binary flag: 1 if patient diagnosed with Hypertension |
| 28 | `diag_Heart_Disease` | Binary | Binary flag: 1 if patient diagnosed with Heart Disease |
| 29 | `diag_Kidney_Disease` | Binary | Binary flag: 1 if patient diagnosed with Kidney Disease |
| 30 | `diag_Sepsis` | Binary | Binary flag: 1 if patient diagnosed with Sepsis |
| 31 | `diag_Pneumonia` | Binary | Binary flag: 1 if patient diagnosed with Pneumonia |
| 32 | `diag_COPD` | Binary | Binary flag: 1 if patient diagnosed with COPD |
| 33 | `diag_Liver_Disease` | Binary | Binary flag: 1 if patient diagnosed with Liver Disease |
| 34 | `med_Antibiotics` | Binary | Binary flag: 1 if patient received Antibiotics |
| 35 | `med_Vasopressors` | Binary | Binary flag: 1 if patient received Vasopressors |
| 36 | `med_Sedatives` | Binary | Binary flag: 1 if patient received Sedatives |
| 37 | `med_Insulin` | Binary | Binary flag: 1 if patient received Insulin |
| 38 | `med_Anticoagulants` | Binary | Binary flag: 1 if patient received Anticoagulants |
| 39 | `med_Steroids` | Binary | Binary flag: 1 if patient received Steroids |
| 40 | `bmi` | Continuous | Body Mass Index = weight / (height/100)^2 |
| 41 | `pulse_pressure` | Continuous | Systolic - Diastolic blood pressure |
| 42 | `map_calculated` | Continuous | Calculated Mean Arterial Pressure |
| 43 | `shock_index` | Continuous | Heart Rate / Systolic Blood Pressure |
| 44 | `heartrate_rolling_mean` | Continuous | 6-hour rolling mean of heartrate |
| 45 | `heartrate_rolling_std` | Continuous | 6-hour rolling standard deviation of heartrate |
| 46 | `heartrate_rate_of_change` | Continuous | Hour-over-hour change in heartrate |
| 47 | `systemicsystolic_rolling_mean` | Continuous | 6-hour rolling mean of systemicsystolic |
| 48 | `systemicsystolic_rolling_std` | Continuous | 6-hour rolling standard deviation of systemicsystolic |
| 49 | `systemicsystolic_rate_of_change` | Continuous | Hour-over-hour change in systemicsystolic |
| 50 | `systemicdiastolic_rolling_mean` | Continuous | 6-hour rolling mean of systemicdiastolic |
| 51 | `systemicdiastolic_rolling_std` | Continuous | 6-hour rolling standard deviation of systemicdiastolic |
| 52 | `systemicdiastolic_rate_of_change` | Continuous | Hour-over-hour change in systemicdiastolic |
| 53 | `systemicmean_rolling_mean` | Continuous | 6-hour rolling mean of systemicmean |
| 54 | `systemicmean_rolling_std` | Continuous | 6-hour rolling standard deviation of systemicmean |
| 55 | `systemicmean_rate_of_change` | Continuous | Hour-over-hour change in systemicmean |
| 56 | `respiration_rolling_mean` | Continuous | 6-hour rolling mean of respiration |
| 57 | `respiration_rolling_std` | Continuous | 6-hour rolling standard deviation of respiration |
| 58 | `respiration_rate_of_change` | Continuous | Hour-over-hour change in respiration |
| 59 | `temperature_rolling_mean` | Continuous | 6-hour rolling mean of temperature |
| 60 | `temperature_rolling_std` | Continuous | 6-hour rolling standard deviation of temperature |
| 61 | `temperature_rate_of_change` | Continuous | Hour-over-hour change in temperature |
| 62 | `gender_Female` | Binary | One-hot encoded gender: Female |
| 63 | `gender_Male` | Binary | One-hot encoded gender: Male |
| 64 | `ethnicity_African American` | Binary | One-hot encoded ethnicity: African American |
| 65 | `ethnicity_Asian` | Binary | One-hot encoded ethnicity: Asian |
| 66 | `ethnicity_Caucasian` | Binary | One-hot encoded ethnicity: Caucasian |
| 67 | `ethnicity_Hispanic` | Binary | One-hot encoded ethnicity: Hispanic |
| 68 | `ethnicity_Native American` | Binary | One-hot encoded ethnicity: Native American |
| 69 | `ethnicity_Other/Unknown` | Binary | One-hot encoded ethnicity: Other/Unknown |

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