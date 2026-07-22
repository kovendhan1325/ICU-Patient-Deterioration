"""
==============================================================================
STEP 06 - Encode Categorical Features & Scale Numerical Features
==============================================================================
- One-Hot Encode: gender, ethnicity
- Binary flags already created in step05 (diagnosis, medication)
- StandardScaler for all continuous numerical features
- Save scaler object for inference
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from config import INTERMEDIATE_DIR, OUTPUT_DIR


# Columns that should NOT be scaled (IDs, targets, binary flags, time)
EXCLUDE_FROM_SCALING = {
    "patientunitstayid", "hour",
    "unitdischargeoffset", "unitdischargestatus", "hospitaldischargestatus",
    "gender", "ethnicity",
}


def encode_categorical(df):
    """One-Hot Encode categorical features."""
    print("  Encoding categorical features ...")
    encoded_count = 0

    # Gender
    if "gender" in df.columns:
        gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=int)
        df = pd.concat([df, gender_dummies], axis=1)
        df.drop(columns=["gender"], inplace=True)
        encoded_count += len(gender_dummies.columns)
        print(f"    -> Gender: {gender_dummies.columns.tolist()}")

    # Ethnicity
    if "ethnicity" in df.columns:
        ethnicity_dummies = pd.get_dummies(df["ethnicity"], prefix="ethnicity", dtype=int)
        df = pd.concat([df, ethnicity_dummies], axis=1)
        df.drop(columns=["ethnicity"], inplace=True)
        encoded_count += len(ethnicity_dummies.columns)
        print(f"    -> Ethnicity: {len(ethnicity_dummies.columns)} categories")

    print(f"    -> {encoded_count} new binary columns from encoding")
    return df


def scale_features(df):
    """StandardScaler on all continuous numerical features."""
    print("  Scaling numerical features with StandardScaler ...")

    # Identify columns to scale
    exclude = EXCLUDE_FROM_SCALING.copy()
    # Also exclude binary/flag columns
    for col in df.columns:
        if col.startswith("diag_") or col.startswith("med_") or col.startswith("gender_") or col.startswith("ethnicity_"):
            exclude.add(col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    print(f"    -> {len(cols_to_scale)} features will be scaled")

    # Fit scaler
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].values)

    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "columns": cols_to_scale}, f)
    print(f"    -> Scaler saved to {scaler_path}")

    return df, scaler, cols_to_scale


def run_step06(merged_df=None):
    """Execute Step 06: Encoding & Scaling."""
    print("\n" + "=" * 70)
    print("STEP 06 - Encode Categorical & Scale Numerical Features")
    print("=" * 70)

    if merged_df is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step05_merged.pkl"), "rb") as f:
            merged_df = pickle.load(f)

    # Encode
    merged_df = encode_categorical(merged_df)

    # Scale
    merged_df, scaler, scaled_cols = scale_features(merged_df)

    # Fill any remaining NaN with 0
    nan_count = merged_df.isna().sum().sum()
    if nan_count > 0:
        print(f"  ! Filling {nan_count:,} remaining NaN values with 0")
        merged_df = merged_df.fillna(0)

    print(f"\n    -> Final shape: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")

    # Save intermediate
    output_path = os.path.join(INTERMEDIATE_DIR, "step06_encoded_scaled.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(merged_df, f)

    print(f"\n  * Step 06 complete. Saved to {output_path}")
    return merged_df


if __name__ == "__main__":
    run_step06()

