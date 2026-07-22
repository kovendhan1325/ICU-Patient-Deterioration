"""
==============================================================================
STEP 02 - Data Cleaning
==============================================================================
- Remove duplicate rows
- Remove duplicate patients (keep first stay)
- Convert data types (age, numeric columns)
- Remove rows with all-null vital/lab values
- Handle inconsistent values
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import INTERMEDIATE_DIR


def clean_patient(df):
    """Clean patient table."""
    print("  Cleaning patient ...")
    initial = len(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate patientunitstayid (keep first)
    df = df.drop_duplicates(subset=["patientunitstayid"], keep="first")

    # Handle age: eICU stores "> 89" as string for patients older than 89
    df["age"] = df["age"].replace("> 89", "90")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Convert height and weight to numeric
    df["admissionheight"] = pd.to_numeric(df["admissionheight"], errors="coerce")
    df["admissionweight"] = pd.to_numeric(df["admissionweight"], errors="coerce")

    # Remove impossible height/weight
    df.loc[df["admissionheight"] < 50, "admissionheight"] = np.nan   # < 50 cm
    df.loc[df["admissionheight"] > 250, "admissionheight"] = np.nan  # > 250 cm
    df.loc[df["admissionweight"] < 20, "admissionweight"] = np.nan   # < 20 kg
    df.loc[df["admissionweight"] > 400, "admissionweight"] = np.nan  # > 400 kg

    # Convert offsets to numeric
    for col in ["hospitaladmitoffset", "unitdischargeoffset"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def clean_vitals(df):
    """Clean vitals table."""
    print("  Cleaning vitals ...")
    initial = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Convert all vital columns to numeric
    vital_cols = ["heartrate", "systemicsystolic", "systemicdiastolic",
                  "systemicmean", "respiration", "spo2", "temperature"]
    for col in vital_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert offset to numeric
    df["observationoffset"] = pd.to_numeric(df["observationoffset"], errors="coerce")

    # Remove rows where ALL vital signs are null (useless rows)
    existing_vitals = [c for c in vital_cols if c in df.columns]
    df = df.dropna(subset=existing_vitals, how="all")

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def clean_labs(df):
    """Clean labs table."""
    print("  Cleaning labs ...")
    initial = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Ensure labresult is numeric (should already be from step01)
    df["labresult"] = pd.to_numeric(df["labresult"], errors="coerce")

    # Remove rows with null lab results
    df = df.dropna(subset=["labresult"])

    # Remove negative lab values (impossible)
    df = df[df["labresult"] >= 0]

    # Convert offset to numeric
    df["labresultoffset"] = pd.to_numeric(df["labresultoffset"], errors="coerce")

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def clean_diagnosis(df):
    """Clean diagnosis table."""
    print("  Cleaning diagnosis ...")
    initial = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove rows with null diagnosis strings
    df = df.dropna(subset=["diagnosisstring"])

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def clean_medication(df):
    """Clean medication table."""
    print("  Cleaning medication ...")
    initial = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove rows with null drug names
    df = df.dropna(subset=["drugname"])

    # Convert offset to numeric
    if "drugstartoffset" in df.columns:
        df["drugstartoffset"] = pd.to_numeric(df["drugstartoffset"], errors="coerce")

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def clean_respiratory(df):
    """Clean respiratory charting table."""
    if df is None:
        return None

    print("  Cleaning respiratory charting ...")
    initial = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Convert respchartvalue to numeric where possible
    df["respchartvalue"] = pd.to_numeric(df["respchartvalue"], errors="coerce")

    # Convert offset to numeric
    if "respchartoffset" in df.columns:
        df["respchartoffset"] = pd.to_numeric(df["respchartoffset"], errors="coerce")

    final = len(df)
    print(f"    -> {initial:,} -> {final:,} rows ({initial - final:,} removed)")
    return df


def run_step02(data=None):
    """Execute Step 02: Data Cleaning."""
    print("\n" + "=" * 70)
    print("STEP 02 - Data Cleaning")
    print("=" * 70)

    # Load from intermediate if not provided
    if data is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step01_loaded.pkl"), "rb") as f:
            data = pickle.load(f)

    data["patient"] = clean_patient(data["patient"])
    data["vitals"] = clean_vitals(data["vitals"])
    data["labs"] = clean_labs(data["labs"])
    data["diagnosis"] = clean_diagnosis(data["diagnosis"])
    data["medication"] = clean_medication(data["medication"])
    data["respiratory"] = clean_respiratory(data.get("respiratory"))

    # Save intermediate
    output_path = os.path.join(INTERMEDIATE_DIR, "step02_cleaned.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  * Step 02 complete. Saved to {output_path}")
    return data


if __name__ == "__main__":
    run_step02()

