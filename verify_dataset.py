import pandas as pd
import numpy as np
import os

def check_dataset(csv_path, X_train_path, X_val_path, X_test_path):
    print("Loading Dataset...")
    df = pd.read_csv(csv_path)
    
    print("\n--- 1. Missing Values ---")
    missing = df.isna().sum().sum()
    # unitdischargestatus might have NaNs, so let's check only the actual features
    
    print("\n--- 3. 37 Feature Columns + Target ---")
    meta_cols = {
        "patientunitstayid", "hour",
        "unitdischargeoffset", "unitdischargestatus", "hospitaldischargestatus",
        "is_deteriorated", "hours_until_discharge",
    }
    targets = ['target_6h', 'target_12h', 'target_24h']
    
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in targets]
    
    missing = df[feature_cols].isna().sum().sum()
    print(f"Missing Values in Features: {missing} {'[OK]' if missing == 0 else '[FAIL]'}")
    
    print("\n--- 2. Data Types ---")
    dtypes = df[feature_cols].dtypes
    non_numeric = dtypes[~dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))]
    if len(non_numeric) == 0:
        print("All feature columns are numeric [OK]")
    else:
        print(f"Non-numeric columns found: {non_numeric} [FAIL]")

    print(f"Total Columns in CSV: {len(df.columns)}")
    print(f"Feature Columns: {len(feature_cols)} {'[OK]' if len(feature_cols) == 37 else '[FAIL]'}")
    if len(feature_cols) != 37:
        print(f"Features: {feature_cols}")
        
    print(f"Target Columns Present: {all(t in df.columns for t in targets)} [OK]")

    print("\n--- 4. Duplicate Rows ---")
    duplicates = df.duplicated(subset=['patientunitstayid', 'hour']).sum()
    print(f"Duplicate Sequence Rows: {duplicates} {'[OK]' if duplicates == 0 else '[FAIL]'}")
    
    print("\n--- 5. Normalization/Scaling ---")
    binary_cols = [c for c in feature_cols if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    continuous_cols = [c for c in feature_cols if c not in binary_cols]
    
    if len(continuous_cols) > 0:
        means = df[continuous_cols].mean().abs().max()
        stds = df[continuous_cols].std().max()
        print(f"Max Absolute Mean of Continuous Features: {means:.4f}")
        print(f"Max Std of Continuous Features: {stds:.4f}")
        print("Features appear scaled/normalized [OK]")
    else:
        print("No continuous features found? [FAIL]")
        
    print("\n--- 6. Train/Validation/Test Split ---")
    try:
        X_train = np.load(X_train_path)
        X_val = np.load(X_val_path)
        X_test = np.load(X_test_path)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        seq_len_ok = X_train.shape[1] == 24 and X_val.shape[1] == 24 and X_test.shape[1] == 24
        feats_ok = X_train.shape[2] == 37 and X_val.shape[2] == 37 and X_test.shape[2] == 37
        print(f"Sequence length is 24: {'[OK]' if seq_len_ok else '[FAIL]'}")
        print(f"Feature dimension is 37: {'[OK]' if feats_ok else '[FAIL]'}")
    except Exception as e:
        print(f"Error loading splits: {e} [FAIL]")

if __name__ == '__main__':
    base_dir = r"c:\Users\LAPTOP\Downloads\database preproccesssing\preprocessing_pipeline\output"
    csv_path = os.path.join(base_dir, "processed_icu_dataset.csv")
    X_train_path = os.path.join(base_dir, "X_train.npy")
    X_val_path = os.path.join(base_dir, "X_val.npy")
    X_test_path = os.path.join(base_dir, "X_test.npy")
    check_dataset(csv_path, X_train_path, X_val_path, X_test_path)
