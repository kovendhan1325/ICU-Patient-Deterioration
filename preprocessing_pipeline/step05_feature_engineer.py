"""
==============================================================================
STEP 05 - Feature Engineering
==============================================================================
- Pivot labs from long to wide format
- Create hourly time bins for all temporal data
- Merge all tables on patientunitstayid + hour
- Create derived features: BMI, Pulse Pressure, MAP, Shock Index
- Rolling statistics: mean, std, rate of change, trends
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import (
    INTERMEDIATE_DIR, HOUR_BIN_MINUTES, ROLLING_WINDOW,
    DIAGNOSIS_KEYWORDS, MEDICATION_KEYWORDS,
)


def create_hour_bin(offset_col):
    """Convert minute-based offset to hourly bin."""
    return (offset_col // HOUR_BIN_MINUTES).astype(int)


def pivot_labs(labs_df):
    """Convert labs from long format to wide format (one column per lab test)."""
    print("  Pivoting labs from long to wide format ...")

    labs_df["hour"] = create_hour_bin(labs_df["labresultoffset"])

    # Aggregate: take the mean if multiple results in same hour
    labs_wide = labs_df.pivot_table(
        index=["patientunitstayid", "hour"],
        columns="labname",
        values="labresult",
        aggfunc="mean",
    ).reset_index()

    # Flatten MultiIndex columns
    labs_wide.columns.name = None

    print(f"    -> Labs pivoted: {labs_wide.shape[0]:,} rows x {labs_wide.shape[1]} cols")
    return labs_wide


def bin_vitals(vitals_df):
    """Bin vitals into hourly intervals with aggregation."""
    print("  Binning vitals into hourly intervals ...")
    vitals_df["hour"] = create_hour_bin(vitals_df["observationoffset"])

    vital_cols = ["heartrate", "systemicsystolic", "systemicdiastolic",
                  "systemicmean", "respiration", "spo2", "temperature"]
    existing = [c for c in vital_cols if c in vitals_df.columns]

    vitals_hourly = vitals_df.groupby(["patientunitstayid", "hour"])[existing].mean().reset_index()
    print(f"    -> Vitals binned: {vitals_hourly.shape[0]:,} rows x {vitals_hourly.shape[1]} cols")
    return vitals_hourly


def create_diagnosis_flags(diagnosis_df, patient_ids):
    """Create binary diagnosis flags for each patient."""
    print("  Creating diagnosis binary flags ...")
    diag_flags = pd.DataFrame({"patientunitstayid": patient_ids})

    for condition, keywords in DIAGNOSIS_KEYWORDS.items():
        pattern = "|".join(keywords)
        has_condition = diagnosis_df[
            diagnosis_df["diagnosisstring"].str.contains(pattern, case=False, na=False)
        ]["patientunitstayid"].unique()
        diag_flags[f"diag_{condition}"] = diag_flags["patientunitstayid"].isin(has_condition).astype(int)

    print(f"    -> {len(DIAGNOSIS_KEYWORDS)} diagnosis flags created")
    return diag_flags


def create_medication_flags(medication_df, patient_ids):
    """Create binary medication flags for each patient per hour."""
    print("  Creating medication binary flags ...")

    if "drugstartoffset" not in medication_df.columns:
        # Fallback: static per patient
        med_flags = pd.DataFrame({"patientunitstayid": patient_ids})
        for category, keywords in MEDICATION_KEYWORDS.items():
            pattern = "|".join(keywords)
            has_med = medication_df[
                medication_df["drugname"].str.contains(pattern, case=False, na=False)
            ]["patientunitstayid"].unique()
            med_flags[f"med_{category}"] = med_flags["patientunitstayid"].isin(has_med).astype(int)
        return med_flags

    medication_df["hour"] = create_hour_bin(medication_df["drugstartoffset"])

    # Create per-patient flags (static - any medication during the stay)
    med_flags = pd.DataFrame({"patientunitstayid": patient_ids})
    for category, keywords in MEDICATION_KEYWORDS.items():
        pattern = "|".join(keywords)
        has_med = medication_df[
            medication_df["drugname"].str.contains(pattern, case=False, na=False)
        ]["patientunitstayid"].unique()
        med_flags[f"med_{category}"] = med_flags["patientunitstayid"].isin(has_med).astype(int)

    print(f"    -> {len(MEDICATION_KEYWORDS)} medication flags created")
    return med_flags


def create_respiratory_features(resp_df):
    """Create respiratory features (FiO2, PEEP as hourly values)."""
    if resp_df is None or len(resp_df) == 0:
        print("  ! No respiratory data - skipping.")
        return None

    print("  Processing respiratory charting ...")
    resp_df["hour"] = create_hour_bin(resp_df["respchartoffset"])

    # Pivot: one column per respiratory label
    resp_wide = resp_df.pivot_table(
        index=["patientunitstayid", "hour"],
        columns="respchartvaluelabel",
        values="respchartvalue",
        aggfunc="mean",
    ).reset_index()
    resp_wide.columns.name = None

    # Rename columns
    rename = {}
    for col in resp_wide.columns:
        if col not in ["patientunitstayid", "hour"]:
            rename[col] = f"resp_{col.replace(' ', '_').lower()}"
    resp_wide = resp_wide.rename(columns=rename)

    # Filter to keep only specific features for the 37-feature set
    keep_cols = ["patientunitstayid", "hour", "resp_fio2", "resp_peep"]
    existing_keep_cols = [c for c in keep_cols if c in resp_wide.columns]
    resp_wide = resp_wide[existing_keep_cols]

    print(f"    -> Respiratory features: {resp_wide.shape[0]:,} rows x {len(existing_keep_cols)-2} features")
    return resp_wide


def add_derived_features(merged_df):
    """Add derived clinical features."""
    print("  Adding derived features ...")
    count = 0

    # BMI = weight(kg) / (height(cm)/100)^2
    if "admissionheight" in merged_df.columns and "admissionweight" in merged_df.columns:
        height_m = merged_df["admissionheight"] / 100.0
        merged_df["bmi"] = merged_df["admissionweight"] / (height_m ** 2)
        merged_df.loc[merged_df["bmi"] > 80, "bmi"] = np.nan  # sanity check
        merged_df.loc[merged_df["bmi"] < 10, "bmi"] = np.nan
        merged_df["bmi"] = merged_df["bmi"].fillna(merged_df["bmi"].median())
        count += 1
        
        # Drop raw height and weight to reduce feature space
        merged_df.drop(columns=["admissionheight", "admissionweight"], inplace=True)

    # Pulse Pressure = systolic - diastolic
    if "systemicsystolic" in merged_df.columns and "systemicdiastolic" in merged_df.columns:
        merged_df["pulse_pressure"] = merged_df["systemicsystolic"] - merged_df["systemicdiastolic"]
        count += 1

    # Mean Arterial Pressure (verify vs systemicmean)
    if "systemicsystolic" in merged_df.columns and "systemicdiastolic" in merged_df.columns:
        merged_df["map_calculated"] = (
            merged_df["systemicdiastolic"] +
            (merged_df["systemicsystolic"] - merged_df["systemicdiastolic"]) / 3
        )
        count += 1

    # Shock Index = HR / Systolic BP
    if "heartrate" in merged_df.columns and "systemicsystolic" in merged_df.columns:
        merged_df["shock_index"] = merged_df["heartrate"] / merged_df["systemicsystolic"].replace(0, np.nan)
        merged_df["shock_index"] = merged_df["shock_index"].fillna(merged_df["shock_index"].median())
        count += 1

    print(f"    -> {count} derived features created")
    return merged_df


def add_rolling_features(merged_df):
    """Add rolling statistics and trend features per patient."""
    print(f"  Adding rolling statistics (window={ROLLING_WINDOW}h) ...")

    rolling_cols = ["heartrate", "systemicsystolic", "systemicdiastolic",
                    "systemicmean", "respiration", "spo2", "temperature"]
    existing = [c for c in rolling_cols if c in merged_df.columns]

    merged_df = merged_df.sort_values(["patientunitstayid", "hour"])

    new_features = 0
    for col in existing:
        grouped = merged_df.groupby("patientunitstayid")[col]

        # Rolling mean
        merged_df[f"{col}_rolling_mean"] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        new_features += 1

        # Rolling std
        merged_df[f"{col}_rolling_std"] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std()
        )
        merged_df[f"{col}_rolling_std"] = merged_df[f"{col}_rolling_std"].fillna(0)
        new_features += 1

        # Rate of change (difference from previous hour)
        merged_df[f"{col}_rate_of_change"] = grouped.transform(
            lambda x: x.diff()
        )
        merged_df[f"{col}_rate_of_change"] = merged_df[f"{col}_rate_of_change"].fillna(0)
        new_features += 1

    print(f"    -> {new_features} rolling/trend features created")
    return merged_df


def run_step05(data=None):
    """Execute Step 05: Feature Engineering."""
    print("\n" + "=" * 70)
    print("STEP 05 - Feature Engineering (Pivot, Merge, Derived Features)")
    print("=" * 70)

    if data is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step04_no_outliers.pkl"), "rb") as f:
            data = pickle.load(f)

    patient_df = data["patient"]
    patient_ids = patient_df["patientunitstayid"].unique()

    # 1. Bin vitals hourly
    vitals_hourly = bin_vitals(data["vitals"])

    # 2. Pivot labs to wide
    labs_wide = pivot_labs(data["labs"])

    # 3. Diagnosis flags
    diag_flags = create_diagnosis_flags(data["diagnosis"], patient_ids)

    # 4. Medication flags
    med_flags = create_medication_flags(data["medication"], patient_ids)

    # 5. Respiratory features
    resp_features = create_respiratory_features(data.get("respiratory"))

    # 6. MERGE - start with vitals (hourly time grid)
    print("\n  Merging all tables ...")
    merged = vitals_hourly.copy()

    # Merge labs (on patient + hour)
    merged = merged.merge(labs_wide, on=["patientunitstayid", "hour"], how="left")

    # Merge respiratory (on patient + hour)
    if resp_features is not None:
        merged = merged.merge(resp_features, on=["patientunitstayid", "hour"], how="left")

    # Forward fill labs and respiratory per patient (labs are infrequent)
    lab_and_resp_cols = [c for c in merged.columns if c not in vitals_hourly.columns]
    if lab_and_resp_cols:
        merged[lab_and_resp_cols] = merged.groupby("patientunitstayid")[lab_and_resp_cols].transform(
            lambda x: x.ffill().bfill()
        )
        # Fill any remaining with global median
        for col in lab_and_resp_cols:
            if merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].median())

    # Merge patient demographics (static - broadcast to all hours)
    static_cols = ["patientunitstayid", "gender", "age", "admissionheight",
                   "admissionweight", "ethnicity", "unitdischargeoffset",
                   "unitdischargestatus", "hospitaldischargestatus"]
    static_existing = [c for c in static_cols if c in patient_df.columns]
    merged = merged.merge(patient_df[static_existing], on="patientunitstayid", how="left")

    # Merge diagnosis flags (static)
    merged = merged.merge(diag_flags, on="patientunitstayid", how="left")

    # Merge medication flags (static)
    merged = merged.merge(med_flags, on="patientunitstayid", how="left")

    # Fill any remaining NaN flags with 0
    flag_cols = [c for c in merged.columns if c.startswith("diag_") or c.startswith("med_")]
    merged[flag_cols] = merged[flag_cols].fillna(0)

    print(f"    -> Merged dataset: {merged.shape[0]:,} rows x {merged.shape[1]} columns")

    # 7. Derived features
    merged = add_derived_features(merged)

    # 8. Rolling features (Removed to reduce feature count for LSTM)
    # merged = add_rolling_features(merged)

    print(f"\n    -> Final merged shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")

    # Save intermediate
    output_path = os.path.join(INTERMEDIATE_DIR, "step05_merged.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"\n  * Step 05 complete. Saved to {output_path}")
    return merged


if __name__ == "__main__":
    run_step05()

