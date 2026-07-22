"""
==============================================================================
CONFIG.PY - Central Configuration for eICU Preprocessing Pipeline
==============================================================================
Project: Adaptive ICU Patient Deterioration Prediction using
         Temporal Deep Learning and Explainable AI
Dataset: eICU Collaborative Research Database v2.0.1
==============================================================================
"""

import os

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "required files")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Create directories
for d in [INTERMEDIATE_DIR, OUTPUT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# SOURCE FILES
# ============================================================================

SOURCE_FILES = {
    "patient": os.path.join(DATA_DIR, "patient.csv.gz"),
    "vitalPeriodic": os.path.join(DATA_DIR, "vitalPeriodic.csv.gz"),
    "lab": os.path.join(DATA_DIR, "lab.csv.gz"),
    "diagnosis": os.path.join(DATA_DIR, "diagnosis.csv.gz"),
    "medication": os.path.join(DATA_DIR, "medication.csv.gz"),
    "respiratoryCharting": os.path.join(DATA_DIR, "respiratoryCharting.csv.gz"),
}

# ============================================================================
# COLUMN SELECTIONS
# ============================================================================

PATIENT_COLS = [
    "patientunitstayid", "patienthealthsystemstayid",
    "gender", "age", "admissionheight", "admissionweight",
    "ethnicity", "unitadmitsource", "unitvisitnumber",
    "hospitaladmitoffset", "unitdischargeoffset",
    "unitdischargestatus", "hospitaldischargestatus",
]

VITAL_COLS = [
    "patientunitstayid", "observationoffset",
    "heartrate", "systemicsystolic", "systemicdiastolic",
    "systemicmean", "respiration", "spo2", "temperature",
]

LAB_COLS = [
    "patientunitstayid", "labresultoffset", "labname", "labresult",
]

DIAGNOSIS_COLS = [
    "patientunitstayid", "diagnosisstring",
]

MEDICATION_COLS = [
    "patientunitstayid", "drugname", "drugstartoffset",
]

RESPIRATORY_COLS = [
    "patientunitstayid", "respchartoffset",
    "respchartvaluelabel", "respchartvalue",
]

# ============================================================================
# LAB TESTS TO KEEP (case-insensitive matching)
# ============================================================================

LAB_TESTS = [
    "creatinine", "glucose", "lactate", "sodium", "potassium",
    "Hgb", "WBC x 1000", "platelets x 1000", "total bilirubin",
]

# Alternative name mappings for standardization
LAB_NAME_MAP = {
    "creatinine": "Creatinine",
    "glucose": "Glucose",
    "lactate": "Lactate",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "hgb": "Hemoglobin",
    "wbc x 1000": "WBC",
    "platelets x 1000": "Platelets",
    "total bilirubin": "Bilirubin",
}

# ============================================================================
# DIAGNOSIS KEYWORDS (case-insensitive substring matching)
# ============================================================================

DIAGNOSIS_KEYWORDS = {
    "Diabetes": ["diabetes", "diabetic", "dm "],
    "Hypertension": ["hypertension", "hypertensive", "htn"],
    "Heart_Disease": ["heart failure", "cardiac", "coronary", "cardiomyopathy",
                      "myocardial", "arrhythmia", "atrial fibrillation"],
    "Kidney_Disease": ["kidney", "renal", "nephro"],
    "Sepsis": ["sepsis", "septic"],
    "Pneumonia": ["pneumonia"],
    "COPD": ["copd", "chronic obstructive", "emphysema"],
    "Liver_Disease": ["liver", "hepatic", "cirrhosis", "hepatitis"],
}

# ============================================================================
# MEDICATION KEYWORDS (case-insensitive substring matching)
# ============================================================================

MEDICATION_KEYWORDS = {
    "Antibiotics": [
        "vancomycin", "piperacillin", "tazobactam", "meropenem",
        "ceftriaxone", "cefepime", "azithromycin", "levofloxacin",
        "ciprofloxacin", "metronidazole", "amoxicillin", "ampicillin",
        "clindamycin", "doxycycline", "gentamicin", "tobramycin",
        "linezolid", "cefazolin", "trimethoprim", "sulfamethoxazole",
    ],
    "Vasopressors": [
        "norepinephrine", "epinephrine", "vasopressin", "dopamine",
        "phenylephrine", "dobutamine", "milrinone", "levophed",
    ],
    "Sedatives": [
        "propofol", "midazolam", "dexmedetomidine", "lorazepam",
        "diazepam", "ketamine", "fentanyl", "precedex",
    ],
    "Insulin": [
        "insulin",
    ],
    "Anticoagulants": [
        "heparin", "warfarin", "enoxaparin", "lovenox",
        "apixaban", "rivaroxaban",
    ],
    "Steroids": [
        "hydrocortisone", "dexamethasone", "methylprednisolone",
        "prednisone", "prednisolone", "solumedrol",
    ],
}

# ============================================================================
# RESPIRATORY CHARTING LABELS TO KEEP
# ============================================================================

RESPIRATORY_LABELS = ["FiO2", "PEEP", "Ventilator Mode"]

# ============================================================================
# OUTLIER THRESHOLDS (physiological ranges)
# ============================================================================

OUTLIER_THRESHOLDS = {
    "heartrate":        (30, 220),
    "systemicsystolic": (40, 300),
    "systemicdiastolic":(20, 200),
    "systemicmean":     (25, 250),
    "respiration":      (4, 60),
    "spo2":             (50, 100),
    "temperature":      (30, 45),
    # Lab thresholds (reasonable clinical ranges)
    "Creatinine":       (0, 30),
    "Glucose":          (10, 1500),
    "Lactate":          (0, 30),
    "Sodium":           (100, 180),
    "Potassium":        (1.0, 10.0),
    "Hemoglobin":       (1, 25),
    "WBC":              (0, 100),
    "Platelets":        (0, 1500),
    "Bilirubin":        (0, 60),
}

# ============================================================================
# TIME-SERIES / SEQUENCE PARAMETERS
# ============================================================================

# Hourly binning: offsets in eICU are in minutes
HOUR_BIN_MINUTES = 60

# Sequence length for LSTM (in hours)
SEQUENCE_LENGTH = 24

# Prediction horizons (in hours)
PREDICTION_HORIZONS = [6, 12, 24]

# ============================================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================================
# FEATURE ENGINEERING WINDOWS
# ============================================================================

ROLLING_WINDOW = 6  # hours for rolling statistics
TREND_WINDOW = 6    # hours for trend calculation

