"""
==============================================================================
STEP 07 - Target Variable Generation & LSTM Sequence Creation
==============================================================================
- Generate binary target labels for 6h, 12h, and 24h deterioration
- Split data by patient (70/15/15) - no data leakage
- Generate LSTM sequences: (N, T, F)
- Save X_train/X_val/X_test and y_train/y_val/y_test as .npy files
- Save flat processed dataset as CSV
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import (
    INTERMEDIATE_DIR, OUTPUT_DIR,
    SEQUENCE_LENGTH, PREDICTION_HORIZONS,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)


def generate_target_labels(df):
    """
    Generate binary deterioration labels for each timestep.

    Target = 1 if patient's unitdischargestatus == 'Expired' AND
    the current hour is within [horizon] hours of the discharge (death) time.

    Deterioration is defined as:
    - unitdischargestatus == 'Expired' (ICU mortality)
    - OR hospitaldischargestatus == 'Expired' (Hospital mortality)
    """
    print("  Generating target labels ...")

    # Determine deterioration: patient died during ICU or hospital stay
    df["is_deteriorated"] = (
        (df["unitdischargestatus"].str.lower() == "expired") |
        (df["hospitaldischargestatus"].str.lower() == "expired")
    ).astype(int)

    # Calculate hours until discharge for each row
    # unitdischargeoffset is in minutes from ICU admission
    # hour is in hours from ICU admission
    df["hours_until_discharge"] = (df["unitdischargeoffset"] / 60) - df["hour"]

    # Generate target for each prediction horizon
    for horizon in PREDICTION_HORIZONS:
        col_name = f"target_{horizon}h"
        df[col_name] = (
            (df["is_deteriorated"] == 1) &
            (df["hours_until_discharge"] >= 0) &
            (df["hours_until_discharge"] <= horizon)
        ).astype(int)

    # Print target distribution
    for horizon in PREDICTION_HORIZONS:
        col = f"target_{horizon}h"
        pos = df[col].sum()
        total = len(df)
        print(f"    -> {col}: {pos:,} positive / {total:,} total ({pos/max(1,total)*100:.2f}%)")

    return df


def get_feature_columns(df):
    """Get the list of feature columns (excluding IDs, targets, meta)."""
    exclude = {
        "patientunitstayid", "hour",
        "unitdischargeoffset", "unitdischargestatus", "hospitaldischargestatus",
        "is_deteriorated", "hours_until_discharge",
    }
    for h in PREDICTION_HORIZONS:
        exclude.add(f"target_{h}h")

    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.bool_]]
    return feature_cols


def split_patients(df, random_seed=RANDOM_SEED):
    """Split patient IDs into train/val/test sets."""
    print("  Splitting patients into train/val/test ...")
    patient_ids = df["patientunitstayid"].unique()
    np.random.seed(random_seed)
    np.random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    print(f"    -> Train: {len(train_ids)} patients")
    print(f"    -> Val:   {len(val_ids)} patients")
    print(f"    -> Test:  {len(test_ids)} patients")

    return train_ids, val_ids, test_ids


def create_sequences(df, patient_ids, feature_cols, target_col, seq_len=SEQUENCE_LENGTH):
    """
    Create sequences for LSTM training.
    For each patient, create sliding windows of seq_len hours.
    Returns X of shape (N_sequences, seq_len, n_features) and y of shape (N_sequences,).
    """
    X_sequences = []
    y_sequences = []

    for pid in patient_ids:
        patient_data = df[df["patientunitstayid"] == pid].sort_values("hour")

        features = patient_data[feature_cols].values
        targets = patient_data[target_col].values

        # Create sliding window sequences
        if len(features) >= seq_len:
            for i in range(len(features) - seq_len + 1):
                X_sequences.append(features[i:i + seq_len])
                # Target is the label at the LAST timestep of the sequence
                y_sequences.append(targets[i + seq_len - 1])
        elif len(features) > 0:
            # Pad shorter sequences with zeros at the beginning
            padded = np.zeros((seq_len, features.shape[1]))
            padded[-len(features):] = features
            X_sequences.append(padded)
            y_sequences.append(targets[-1])

    if len(X_sequences) == 0:
        return np.array([]).reshape(0, seq_len, len(feature_cols)), np.array([])

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_sequences, dtype=np.int32)

    return X, y


def run_step07(merged_df=None):
    """Execute Step 07: Target Labels & Sequence Generation."""
    print("\n" + "=" * 70)
    print("STEP 07 - Target Variable & LSTM Sequence Generation")
    print("=" * 70)

    if merged_df is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step06_encoded_scaled.pkl"), "rb") as f:
            merged_df = pickle.load(f)

    # 1. Generate target labels
    merged_df = generate_target_labels(merged_df)

    # 2. Get feature columns
    feature_cols = get_feature_columns(merged_df)
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"    {i+1:3d}. {col}")

    # 3. Save flat processed dataset (CSV)
    csv_path = os.path.join(OUTPUT_DIR, "processed_icu_dataset.csv")
    merged_df.to_csv(csv_path, index=False)
    print(f"\n  * Flat dataset saved to {csv_path}")
    print(f"    Shape: {merged_df.shape}")

    # 4. Split patients
    train_ids, val_ids, test_ids = split_patients(merged_df)

    # 5. Generate sequences for the primary target (24h)
    # We'll use target_24h as the primary, but save all target arrays
    primary_target = "target_24h"

    print(f"\n  Generating LSTM sequences (seq_len={SEQUENCE_LENGTH}) ...")

    X_train, y_train = create_sequences(merged_df, train_ids, feature_cols, primary_target)
    X_val, y_val = create_sequences(merged_df, val_ids, feature_cols, primary_target)
    X_test, y_test = create_sequences(merged_df, test_ids, feature_cols, primary_target)

    print(f"    -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    -> X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"    -> X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # Class distribution
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        if len(y) > 0:
            pos = y.sum()
            total = len(y)
            print(f"    -> {name} class distribution: {pos}/{total} positive ({pos/max(1,total)*100:.1f}%)")

    # 6. Save sequences
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Also save combined as processed_lstm_dataset.npy
    lstm_data = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_cols,
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": primary_target,
    }
    np.save(os.path.join(OUTPUT_DIR, "processed_lstm_dataset.npy"), lstm_data, allow_pickle=True)

    print(f"\n  * All sequences saved to {OUTPUT_DIR}")

    # 7. Save feature list
    feature_path = os.path.join(os.path.dirname(OUTPUT_DIR), "reports", "feature_list.txt")
    with open(feature_path, "w") as f:
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"Sequence Length: {SEQUENCE_LENGTH}\n")
        f.write(f"LSTM Input Shape: (N, {SEQUENCE_LENGTH}, {len(feature_cols)})\n\n")
        for i, col in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {col}\n")
    print(f"  * Feature list saved to {feature_path}")

    # Return info dict
    return {
        "merged_df": merged_df,
        "feature_cols": feature_cols,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
        "X_test_shape": X_test.shape,
        "n_patients_train": len(train_ids),
        "n_patients_val": len(val_ids),
        "n_patients_test": len(test_ids),
    }


if __name__ == "__main__":
    run_step07()

