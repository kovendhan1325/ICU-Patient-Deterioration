"""
==============================================================================
RUN_PIPELINE.PY - Master Orchestrator
==============================================================================
Runs all 7 preprocessing steps in sequence and generates final reports.

Usage:
    python run_pipeline.py

Output:
    - output/processed_icu_dataset.csv
    - output/processed_lstm_dataset.npy
    - output/X_train.npy, X_val.npy, X_test.npy
    - output/y_train.npy, y_val.npy, y_test.npy
    - output/scaler.pkl
    - reports/missing_value_report.csv
    - reports/outlier_summary.csv
    - reports/feature_list.txt
    - reports/data_dictionary.md
==============================================================================
"""

import time
import os
import sys
import numpy as np

# Ensure we can import from the pipeline directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, REPORTS_DIR, SEQUENCE_LENGTH, PREDICTION_HORIZONS
from step01_load_data import run_step01
from step02_clean_data import run_step02
from step03_impute_missing import run_step03
from step04_outlier_removal import run_step04
from step05_feature_engineer import run_step05
from step06_encode_scale import run_step06
from step07_target_sequence import run_step07


def generate_data_dictionary(info):
    """Generate a comprehensive data dictionary as Markdown."""
    feature_cols = info["feature_cols"]
    merged_df = info["merged_df"]

    doc = []
    doc.append("# Data Dictionary")
    doc.append("")
    doc.append("## Project")
    doc.append("**Adaptive ICU Patient Deterioration Prediction using Temporal Deep Learning and Explainable AI**")
    doc.append("")
    doc.append("## Dataset")
    doc.append("eICU Collaborative Research Database v2.0.1 (Demo)")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Dataset Shapes")
    doc.append("")
    doc.append(f"| Split | Sequences | Shape |")
    doc.append(f"|---|---|---|")
    doc.append(f"| Train | {info['X_train_shape'][0]:,} | {info['X_train_shape']} |")
    doc.append(f"| Val | {info['X_val_shape'][0]:,} | {info['X_val_shape']} |")
    doc.append(f"| Test | {info['X_test_shape'][0]:,} | {info['X_test_shape']} |")
    doc.append("")
    doc.append(f"- **Sequence Length**: {SEQUENCE_LENGTH} hours")
    doc.append(f"- **Number of Features**: {len(feature_cols)}")
    doc.append(f"- **Prediction Horizons**: {PREDICTION_HORIZONS}")
    doc.append(f"- **Patients (Train/Val/Test)**: {info['n_patients_train']}/{info['n_patients_val']}/{info['n_patients_test']}")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Feature Descriptions")
    doc.append("")
    doc.append("| # | Feature | Type | Description |")
    doc.append("|---|---|---|---|")

    for i, col in enumerate(feature_cols):
        dtype = str(merged_df[col].dtype) if col in merged_df.columns else "float32"

        # Auto-generate description based on column name
        if col.startswith("diag_"):
            desc = f"Binary flag: 1 if patient diagnosed with {col.replace('diag_', '').replace('_', ' ')}"
            ftype = "Binary"
        elif col.startswith("med_"):
            desc = f"Binary flag: 1 if patient received {col.replace('med_', '').replace('_', ' ')}"
            ftype = "Binary"
        elif col.startswith("gender_"):
            desc = f"One-hot encoded gender: {col.replace('gender_', '')}"
            ftype = "Binary"
        elif col.startswith("ethnicity_"):
            desc = f"One-hot encoded ethnicity: {col.replace('ethnicity_', '')}"
            ftype = "Binary"
        elif col.startswith("resp_"):
            desc = f"Respiratory charting: {col.replace('resp_', '').replace('_', ' ')}"
            ftype = "Continuous"
        elif col.endswith("_rolling_mean"):
            base = col.replace("_rolling_mean", "")
            desc = f"6-hour rolling mean of {base}"
            ftype = "Continuous"
        elif col.endswith("_rolling_std"):
            base = col.replace("_rolling_std", "")
            desc = f"6-hour rolling standard deviation of {base}"
            ftype = "Continuous"
        elif col.endswith("_rate_of_change"):
            base = col.replace("_rate_of_change", "")
            desc = f"Hour-over-hour change in {base}"
            ftype = "Continuous"
        elif col == "bmi":
            desc = "Body Mass Index = weight / (height/100)^2"
            ftype = "Continuous"
        elif col == "pulse_pressure":
            desc = "Systolic - Diastolic blood pressure"
            ftype = "Continuous"
        elif col == "map_calculated":
            desc = "Calculated Mean Arterial Pressure"
            ftype = "Continuous"
        elif col == "shock_index":
            desc = "Heart Rate / Systolic Blood Pressure"
            ftype = "Continuous"
        elif col == "heartrate":
            desc = "Heart rate (beats per minute)"
            ftype = "Continuous"
        elif col == "systemicsystolic":
            desc = "Systolic blood pressure (mmHg)"
            ftype = "Continuous"
        elif col == "systemicdiastolic":
            desc = "Diastolic blood pressure (mmHg)"
            ftype = "Continuous"
        elif col == "systemicmean":
            desc = "Mean arterial pressure from monitor (mmHg)"
            ftype = "Continuous"
        elif col == "respiration":
            desc = "Respiratory rate (breaths per minute)"
            ftype = "Continuous"
        elif col == "spo2":
            desc = "Peripheral oxygen saturation (%)"
            ftype = "Continuous"
        elif col == "temperature":
            desc = "Body temperature (Celsius)"
            ftype = "Continuous"
        elif col == "age":
            desc = "Patient age (years, capped at 90)"
            ftype = "Continuous"
        elif col == "admissionheight":
            desc = "Height at admission (cm)"
            ftype = "Continuous"
        elif col == "admissionweight":
            desc = "Weight at admission (kg)"
            ftype = "Continuous"
        else:
            desc = f"Feature: {col}"
            ftype = "Continuous"

        doc.append(f"| {i+1} | `{col}` | {ftype} | {desc} |")

    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## Target Variables")
    doc.append("")
    doc.append("| Target | Description |")
    doc.append("|---|---|")
    doc.append("| `target_6h` | 1 if patient deteriorates (dies) within 6 hours of current timestep |")
    doc.append("| `target_12h` | 1 if patient deteriorates (dies) within 12 hours of current timestep |")
    doc.append("| `target_24h` | 1 if patient deteriorates (dies) within 24 hours of current timestep |")
    doc.append("")
    doc.append("## Output Files")
    doc.append("")
    doc.append("| File | Description |")
    doc.append("|---|---|")
    doc.append("| `processed_icu_dataset.csv` | Flat preprocessed dataset (all features + targets) |")
    doc.append("| `processed_lstm_dataset.npy` | Combined LSTM data dictionary (pickle) |")
    doc.append("| `X_train.npy` | Training input sequences |")
    doc.append("| `X_val.npy` | Validation input sequences |")
    doc.append("| `X_test.npy` | Test input sequences |")
    doc.append("| `y_train.npy` | Training labels |")
    doc.append("| `y_val.npy` | Validation labels |")
    doc.append("| `y_test.npy` | Test labels |")
    doc.append("| `scaler.pkl` | Fitted StandardScaler for inference |")

    return "\n".join(doc)


def main():
    """Run the complete preprocessing pipeline."""
    print("+" + "=" * 68 + "+")
    print("|  eICU PREPROCESSING PIPELINE                                       |")
    print("|  Adaptive ICU Patient Deterioration Prediction                      |")
    print("|  Temporal Deep Learning + Explainable AI                            |")
    print("+" + "=" * 68 + "+")

    total_start = time.time()

    # Step 1: Load
    t = time.time()
    data = run_step01()
    print(f"  * Step 01 took {time.time() - t:.1f}s")

    # Step 2: Clean
    t = time.time()
    data = run_step02(data)
    print(f"  * Step 02 took {time.time() - t:.1f}s")

    # Step 3: Impute
    t = time.time()
    data = run_step03(data)
    print(f"  * Step 03 took {time.time() - t:.1f}s")

    # Step 4: Outliers
    t = time.time()
    data = run_step04(data)
    print(f"  * Step 04 took {time.time() - t:.1f}s")

    # Step 5: Feature Engineering
    t = time.time()
    merged = run_step05(data)
    print(f"  * Step 05 took {time.time() - t:.1f}s")

    # Step 6: Encode & Scale
    t = time.time()
    merged = run_step06(merged)
    print(f"  * Step 06 took {time.time() - t:.1f}s")

    # Step 7: Target & Sequences
    t = time.time()
    info = run_step07(merged)
    print(f"  * Step 07 took {time.time() - t:.1f}s")

    # Generate Data Dictionary
    print("\n" + "=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)
    doc = generate_data_dictionary(info)
    doc_path = os.path.join(REPORTS_DIR, "data_dictionary.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  * Data dictionary saved to {doc_path}")

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "+" + "=" * 68 + "+")
    print("|  PIPELINE COMPLETE                                                 |")
    print("+" + "=" * 68 + "+")
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"\n  Output files:")
    for f_name in os.listdir(OUTPUT_DIR):
        f_path = os.path.join(OUTPUT_DIR, f_name)
        size_mb = os.path.getsize(f_path) / (1024 * 1024)
        print(f"    {f_name:40s} {size_mb:.2f} MB")

    print(f"\n  Reports:")
    for f_name in os.listdir(REPORTS_DIR):
        print(f"    {f_name}")

    print(f"\n  LSTM Input Shape:  {info['X_train_shape']}")
    print(f"  Features:          {len(info['feature_cols'])}")
    print(f"  Sequence Length:   {SEQUENCE_LENGTH}")
    print(f"  Prediction Target: target_24h")
    print(f"\n  * Ready for LSTM/GRU training!")


if __name__ == "__main__":
    main()

