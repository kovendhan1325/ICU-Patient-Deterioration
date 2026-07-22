I am building a Final Year Project titled:

"Adaptive ICU Patient Deterioration Prediction using Temporal Deep Learning and Explainable AI"

I am using the eICU Collaborative Research Database (eICU-CRD v2.0.1).

I only need the columns required for my project. Ignore all other tables and columns.

The project predicts ICU patient deterioration within the next 6, 12, and 24 hours using LSTM/GRU and Explainable AI (SHAP).

Extract ONLY the following tables and columns.

======================================================
1. patient.csv
======================================================

Required Columns:

patientunitstayid
patienthealthsystemstayid
gender
age
admissionheight
admissionweight
ethnicity
unitadmitsource
unitvisitnumber
hospitaladmitoffset
unitdischargeoffset
unitdischargestatus
hospitaldischargestatus

======================================================
2. vitalPeriodic.csv
======================================================

Required Columns:

patientunitstayid
observationoffset
heartrate
systemicsystolic
systemicdiastolic
systemicmean
respiration
spo2
temperature

======================================================
3. lab.csv
======================================================

Required Columns:

patientunitstayid
labresultoffset
labname
labresult

Only keep these laboratory tests:

Creatinine
Glucose
Lactate
Bilirubin
Sodium
Potassium
Hemoglobin
WBC
Platelets

Remove every other laboratory test.

======================================================
4. diagnosis.csv
======================================================

Required Columns:

patientunitstayid
diagnosisstring

From diagnosisstring identify:

Diabetes
Hypertension
Heart Disease
Kidney Disease
Sepsis
Pneumonia
COPD
Liver Disease

======================================================
5. medication.csv
======================================================

Required Columns:

patientunitstayid
drugname
drugstartoffset

Keep only medications related to:

Antibiotics
Vasopressors
Sedatives
Insulin
Anticoagulants
Steroids

Ignore all other drugs.

======================================================
6. respiratoryCharting.csv (Optional)
======================================================

Required Columns:

patientunitstayid
respchartoffset
respchartvaluelabel
respchartvalue

Keep only:

FiO₂
PEEP
Ventilator Mode

======================================================
7. Target Variable
======================================================

Use one or more of the following outcomes:

hospitaldischargestatus
unitdischargestatus
ICU mortality
Hospital mortality

Generate a binary target:

0 = No Deterioration

1 = Deterioration

======================================================
Output Required
======================================================

1. Confirm that these are the minimum required tables.
2. Confirm whether any important column is missing.
3. Provide SQL SELECT statements to extract only these columns.
4. Explain how these tables should be joined using patientunitstayid.
5. Generate the final cleaned dataset schema for LSTM training.

Do not include unnecessary columns or tables.