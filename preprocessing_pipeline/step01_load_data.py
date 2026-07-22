"""
==============================================================================
STEP 01 - Load Required Tables & Select Columns
==============================================================================
Loads all 6 .csv.gz files, selects only the required columns, and applies
initial filtering (lab tests, medications, diagnoses, respiratory labels).
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import (
    SOURCE_FILES, INTERMEDIATE_DIR,
    PATIENT_COLS, VITAL_COLS, LAB_COLS, DIAGNOSIS_COLS,
    MEDICATION_COLS, RESPIRATORY_COLS,
    LAB_TESTS, LAB_NAME_MAP,
    DIAGNOSIS_KEYWORDS, MEDICATION_KEYWORDS, RESPIRATORY_LABELS,
)


def load_patient():
    """Load patient.csv.gz and select required columns."""
    print("  Loading patient.csv.gz ...")
    df = pd.read_csv(SOURCE_FILES["patient"], compression="gzip")
    # Keep only columns that exist
    cols = [c for c in PATIENT_COLS if c in df.columns]
    df = df[cols]
    print(f"    -> {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_vitals():
    """Load vitalPeriodic.csv.gz and select required columns."""
    print("  Loading vitalPeriodic.csv.gz ...")
    df = pd.read_csv(SOURCE_FILES["vitalPeriodic"], compression="gzip")
    cols = [c for c in VITAL_COLS if c in df.columns]
    df = df[cols]
    print(f"    -> {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_labs():
    """Load lab.csv.gz, select columns, and filter to required tests."""
    print("  Loading lab.csv.gz ...")
    df = pd.read_csv(SOURCE_FILES["lab"], compression="gzip")
    cols = [c for c in LAB_COLS if c in df.columns]
    df = df[cols]

    # Filter to required lab tests (case-insensitive)
    lab_tests_lower = [t.lower() for t in LAB_TESTS]
    df["labname_lower"] = df["labname"].str.lower().str.strip()
    df = df[df["labname_lower"].isin(lab_tests_lower)].copy()
    
    # Standardize lab names
    df["labname"] = df["labname_lower"].map(LAB_NAME_MAP).fillna(df["labname"])
    df.drop(columns=["labname_lower"], inplace=True)

    # Ensure labresult is numeric
    df["labresult"] = pd.to_numeric(df["labresult"], errors="coerce")

    print(f"    -> {df.shape[0]:,} rows after filtering to {len(LAB_TESTS)} lab tests")
    print(f"    -> Lab tests found: {df['labname'].unique().tolist()}")
    return df


def load_diagnosis():
    """Load diagnosis.csv.gz, select columns, and filter to required conditions."""
    print("  Loading diagnosis.csv.gz ...")
    df = pd.read_csv(SOURCE_FILES["diagnosis"], compression="gzip")
    cols = [c for c in DIAGNOSIS_COLS if c in df.columns]
    df = df[cols]

    # Build a combined regex pattern for all diagnosis keywords
    all_keywords = []
    for keywords in DIAGNOSIS_KEYWORDS.values():
        all_keywords.extend(keywords)

    pattern = "|".join(all_keywords)
    df = df[df["diagnosisstring"].str.contains(pattern, case=False, na=False)].copy()

    print(f"    -> {df.shape[0]:,} rows after filtering to relevant diagnoses")
    return df


def load_medication():
    """Load medication.csv.gz, select columns, and filter to required drug categories."""
    print("  Loading medication.csv.gz ...")
    df = pd.read_csv(SOURCE_FILES["medication"], compression="gzip")
    cols = [c for c in MEDICATION_COLS if c in df.columns]
    df = df[cols]

    # Build a combined regex pattern for all medication keywords
    all_keywords = []
    for keywords in MEDICATION_KEYWORDS.values():
        all_keywords.extend(keywords)

    pattern = "|".join(all_keywords)
    df = df[df["drugname"].str.contains(pattern, case=False, na=False)].copy()

    print(f"    -> {df.shape[0]:,} rows after filtering to relevant medications")
    return df


def load_respiratory():
    """Load respiratoryCharting.csv.gz, select columns, and filter to required labels."""
    filepath = SOURCE_FILES.get("respiratoryCharting")
    if filepath is None or not os.path.exists(filepath):
        print("  ! respiratoryCharting.csv.gz not found - skipping.")
        return None

    print("  Loading respiratoryCharting.csv.gz ...")
    df = pd.read_csv(filepath, compression="gzip")
    cols = [c for c in RESPIRATORY_COLS if c in df.columns]
    df = df[cols]

    # Filter to required labels (case-insensitive partial match)
    resp_pattern = "|".join(RESPIRATORY_LABELS)
    df = df[df["respchartvaluelabel"].str.contains(resp_pattern, case=False, na=False)].copy()

    print(f"    -> {df.shape[0]:,} rows after filtering to FiO2/PEEP/Ventilator Mode")
    return df


def run_step01():
    """Execute Step 01: Load all data and save intermediates."""
    print("\n" + "=" * 70)
    print("STEP 01 - Loading Required Tables & Selecting Columns")
    print("=" * 70)

    data = {}
    data["patient"] = load_patient()
    data["vitals"] = load_vitals()
    data["labs"] = load_labs()
    data["diagnosis"] = load_diagnosis()
    data["medication"] = load_medication()
    data["respiratory"] = load_respiratory()

    # Save intermediates
    output_path = os.path.join(INTERMEDIATE_DIR, "step01_loaded.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  * Step 01 complete. Saved to {output_path}")
    return data


if __name__ == "__main__":
    run_step01()

