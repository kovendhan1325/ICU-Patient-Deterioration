"""
step07b_save_all_labels.py
--------------------------
One-time helper: generates y_train/val/test for ALL three targets
(6h, 12h, 24h) using EXACTLY the same patient split and sliding-window
logic as step07, then saves them as .npy files alongside the existing
X_train / X_val / X_test arrays.

Run once before step10_v3_multi_target.py.
"""

import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (INTERMEDIATE_DIR, OUTPUT_DIR, RANDOM_SEED,
                    SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO)

# -- replicate step07's create_sequences exactly --
def create_sequences(df, patient_ids, feature_cols, target_col,
                     seq_len=SEQUENCE_LENGTH):
    X_seqs, y_seqs = [], []
    for pid in patient_ids:
        sub      = df[df["patientunitstayid"] == pid].sort_values("hour")
        features = sub[feature_cols].values
        targets  = sub[target_col].values
        if len(features) >= seq_len:
            for i in range(len(features) - seq_len + 1):
                X_seqs.append(features[i:i + seq_len])
                y_seqs.append(targets[i + seq_len - 1])
        elif len(features) > 0:
            padded = np.zeros((seq_len, features.shape[1]))
            padded[-len(features):] = features
            X_seqs.append(padded)
            y_seqs.append(targets[-1])
    if not X_seqs:
        return np.zeros((0, seq_len, len(feature_cols))), np.zeros(0)
    return (np.array(X_seqs, dtype=np.float32),
            np.array(y_seqs, dtype=np.float32))


def run():
    print("=" * 60)
    print("step07b - Saving 12h and 24h label arrays")
    print("=" * 60)

    # Load the step06 pickle (same source step07 used)
    pkl_path = os.path.join(INTERMEDIATE_DIR, "step06_encoded_scaled.pkl")
    print(f"\n  Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        merged_df = pickle.load(f)

    # Generate target labels on the full dataframe (same as step07)
    merged_df["is_deteriorated"] = (
        (merged_df["unitdischargestatus"].str.lower() == "expired") |
        (merged_df["hospitaldischargestatus"].str.lower() == "expired")
    ).astype(int)
    merged_df["hours_until_discharge"] = (
        merged_df["unitdischargeoffset"] / 60 - merged_df["hour"]
    )
    for h in [6, 12, 24]:
        col = f"target_{h}h"
        merged_df[col] = (
            (merged_df["is_deteriorated"] == 1) &
            (merged_df["hours_until_discharge"] >= 0) &
            (merged_df["hours_until_discharge"] <= h)
        ).astype(int)
        pos = merged_df[col].sum()
        print(f"  {col}: {pos:,} positives / {len(merged_df):,} rows "
              f"({pos / len(merged_df) * 100:.2f}%)")

    # Feature columns (exactly as step07)
    exclude = {
        "patientunitstayid", "hour",
        "unitdischargeoffset", "unitdischargestatus", "hospitaldischargestatus",
        "is_deteriorated", "hours_until_discharge",
        "target_6h", "target_12h", "target_24h",
    }
    feature_cols = [c for c in merged_df.columns
                    if c not in exclude and
                    merged_df[c].dtype in [
                        "float64", "float32", "int64", "int32", "uint8", "bool"
                    ]]
    print(f"\n  Feature columns: {len(feature_cols)}")

    # EXACT same patient split as step07
    patient_ids = merged_df["patientunitstayid"].unique()
    np.random.seed(RANDOM_SEED)          # <-- matches step07 exactly
    np.random.shuffle(patient_ids)

    n       = len(patient_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train_ids = patient_ids[:n_train]
    val_ids   = patient_ids[n_train:n_train + n_val]
    test_ids  = patient_ids[n_train + n_val:]
    print(f"\n  Split: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test patients")

    # Cross-check sequence counts with existing X files
    X_tr_rows = len(np.load(os.path.join(OUTPUT_DIR, "y_train.npy")))  # use 6h as ref
    X_vl_rows = len(np.load(os.path.join(OUTPUT_DIR, "y_val.npy")))
    X_ts_rows = len(np.load(os.path.join(OUTPUT_DIR, "y_test.npy")))
    print(f"\n  Reference sequence counts (from existing 6h labels):")
    print(f"    train={X_tr_rows:,}  val={X_vl_rows:,}  test={X_ts_rows:,}")

    # Generate and save labels for 12h and 24h
    for horizon in [12, 24]:
        target_col = f"target_{horizon}h"
        print(f"\n  === Generating sequences for {target_col} ===")

        _, y_train = create_sequences(merged_df, train_ids, feature_cols, target_col)
        _, y_val   = create_sequences(merged_df, val_ids,   feature_cols, target_col)
        _, y_test  = create_sequences(merged_df, test_ids,  feature_cols, target_col)

        # Verify counts match existing X shapes
        for y, ref, name in [(y_train, X_tr_rows, "train"),
                              (y_val,   X_vl_rows, "val"),
                              (y_test,  X_ts_rows, "test")]:
            if len(y) != ref:
                print(f"    [WARN] {name}: {len(y):,} derived but ref={ref:,}  "
                      f"(diff={abs(len(y)-ref)})")
            pos = int(y[:ref].sum()); total = ref
            print(f"    {name}: {total:,} seqs  pos={pos:,}  "
                  f"({pos/total*100:.2f}%)")

        # Save (truncate to match reference length)
        np.save(os.path.join(OUTPUT_DIR, f"y_train_{horizon}h.npy"),
                y_train[:X_tr_rows].astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"y_val_{horizon}h.npy"),
                y_val[:X_vl_rows].astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"y_test_{horizon}h.npy"),
                y_test[:X_ts_rows].astype(np.float32))

        print(f"    Saved: y_{{train,val,test}}_{horizon}h.npy")

    print("\n  Done. You can now run step10_v3_multi_target.py")


if __name__ == "__main__":
    run()
