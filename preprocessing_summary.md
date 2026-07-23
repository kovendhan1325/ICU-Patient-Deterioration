# Data Preprocessing Summary & Required Outputs

## 1. Confirm Minimum Required Tables
Yes, the 6 tables listed (`patient.csv`, `vitalPeriodic.csv`, `lab.csv`, `diagnosis.csv`, `medication.csv`, `respiratoryCharting.csv`) represent the **minimum required tables** for an ICU patient deterioration prediction model. They cover all essential domains:
- **Demographics & Outcomes**: `patient.csv`
- **Physiological signs (High frequency)**: `vitalPeriodic.csv`
- **Clinical tests (Low frequency)**: `lab.csv`
- **Conditions/Comorbidities**: `diagnosis.csv`
- **Interventions**: `medication.csv`, `respiratoryCharting.csv`

## 2. Missing Important Columns
The selected columns capture the core required features, but the following optional columns could be considered for future improvements depending on exact modeling goals:
- `patient.csv`: `hospitaladmitoffset` (to track time spent in hospital prior to ICU admission).
- `vitalAperiodic.csv`: Non-periodic vitals like non-invasive blood pressures (NIBP) can sometimes capture sudden deteriorations between periodic charting.
- `infusionDrug.csv`: For continuous infusions (e.g., precise vasopressor dosing) rather than just start times in `medication.csv`.

*For the current LSTM pipeline, the selected columns are perfectly sufficient and well-rounded.*

## 3. SQL SELECT Statements to Extract the Required Columns

```sql
-- 1. patient
SELECT 
    patientunitstayid, patienthealthsystemstayid, gender, age, 
    admissionheight, admissionweight, ethnicity, unitadmitsource, 
    unitvisitnumber, hospitaladmitoffset, unitdischargeoffset, 
    unitdischargestatus, hospitaldischargestatus
FROM patient;

-- 2. vitalPeriodic
SELECT 
    patientunitstayid, observationoffset, heartrate, 
    systemicsystolic, systemicdiastolic, systemicmean, 
    respiration, spo2, temperature
FROM vitalPeriodic;

-- 3. lab
SELECT 
    patientunitstayid, labresultoffset, labname, labresult
FROM lab
WHERE LOWER(labname) IN (
    'creatinine', 'glucose', 'lactate', 'total bilirubin', 
    'sodium', 'potassium', 'hgb', 'wbc x 1000', 'platelets x 1000'
);

-- 4. diagnosis
SELECT 
    patientunitstayid, diagnosisstring
FROM diagnosis;

-- 5. medication
SELECT 
    patientunitstayid, drugname, drugstartoffset
FROM medication;

-- 6. respiratoryCharting
SELECT 
    patientunitstayid, respchartoffset, respchartvaluelabel, respchartvalue
FROM respiratoryCharting
WHERE LOWER(respchartvaluelabel) IN ('fio2', 'peep', 'ventilator mode');
```

## 4. How Tables Should be Joined
The core unifying key across all tables is `patientunitstayid`, which uniquely identifies a single ICU stay.

Because temporal tables (`vitalPeriodic`, `lab`, `medication`, `respiratoryCharting`) are recorded at different timestamps (offsets), you **cannot do a direct flat JOIN** on just `patientunitstayid` without causing massive row duplication.

**The correct joining approach (as implemented in the Python pipeline):**
1. **Temporal Alignment (Binning)**: Convert all offsets (`observationoffset`, `labresultoffset`, etc.) into standardized hourly time-bins (e.g., `hour = offset // 60`).
2. **Aggregation**: Aggregate the values within each hour (e.g., take the mean of vitals, the median of labs, or flag if a medication was started).
3. **Pivoting**: Convert long-format tables (like `lab`, `respiratoryCharting`) into wide-format where each test/label becomes its own column.
4. **Left Join**: Start with the high-frequency table (hourly `vitals`) and perform a `LEFT JOIN` on `(patientunitstayid, hour)` with the pivoted lab and respiratory tables.
5. **Static Join**: Perform a `LEFT JOIN` on `patientunitstayid` to add the static `patient` demographics, `diagnosis` flags, and overarching `medication` flags.

## 5. Cleaned Dataset Schema for LSTM Training

The final dataset for the LSTM is a 3D tensor with shape:
`(Number of Sequences, Sequence Length, Number of Features)` -> `(N, 24, F)`

### Tensor Shape
- **Sequence Length**: 24 hours (a rolling 24-hour window per patient)
- **Features (F)**: ~50-60 features (Vitals, Labs, Demographics, Interventions, Target Flags, Rolling Stats)
- **Target (y)**: Binary (0 or 1) indicating if deterioration occurs in the next `H` hours (6, 12, or 24).

### Schema of the Flattened Dataset (`processed_icu_dataset.csv`)

| Column Category | Example Columns | Data Type |
|-----------------|----------------|-----------|
| **Identifiers** | `patientunitstayid`, `hour` | Integer |
| **Vitals (Binned)** | `heartrate`, `spo2`, `temperature` | Float (Scaled) |
| **Labs (Pivoted)** | `Glucose`, `Lactate`, `Creatinine` | Float (Scaled) |
| **Respiratory** | `resp_fio2`, `resp_peep` | Float (Scaled) |
| **Demographics** | `age`, `admissionheight`, `admissionweight` | Float (Scaled) |
| **Categorical (OHE)**| `gender_Female`, `ethnicity_Caucasian` | Binary (0/1) |
| **Clinical Flags** | `diag_Sepsis`, `med_Vasopressors` | Binary (0/1) |
| **Derived Features**| `bmi`, `shock_index`, `pulse_pressure` | Float (Scaled) |
| **Rolling Stats** | `heartrate_rolling_mean`, `spo2_rate_of_change`| Float (Scaled) |
| **Target Labels** | `target_6h`, `target_12h`, `target_24h` | Binary (0/1) |

All numeric features are processed using `StandardScaler`, and missing temporal gaps are resolved via Forward Fill -> Backward Fill -> Median Imputation.
