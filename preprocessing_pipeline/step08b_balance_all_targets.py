"""
==============================================================================
STEP 08b - Balance All Targets (6h, 12h, 24h)
==============================================================================
Balances each target's training set so the model can actually learn.

Strategy: RandomOverSampler
  - Duplicates real minority (positive) examples rather than creating
    synthetic sequences (SMOTE on temporal data creates unrealistic patterns)
  - Target minority class ratio: 33% (1 positive for every 2 negatives)
  - Applied ONLY to training set; val/test remain untouched (real distribution)

Outputs per target (saved to output/):
  X_train_bal_{horizon}h.npy  - balanced training sequences (N_bal, 24, 30)
  y_train_bal_{horizon}h.npy  - balanced training labels   (N_bal,)

Val/test labels already exist (step07b):
  y_val_{horizon}h.npy / y_test_{horizon}h.npy

For 6h:
  X_train_bal_6h.npy (uses existing y_train.npy)
  y_train_bal_6h.npy

==============================================================================
"""

import os, sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, REPORTS_DIR, RANDOM_SEED

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    raise ImportError("Run: pip install imbalanced-learn")


# Target minority ratio after balancing
SAMPLING_STRATEGY = 0.33   # 1 positive for every 2 negatives (~33% minority)


def balance_target(horizon, X_train, y_train):
    """
    Balance the training set for a given target horizon.
    Flattens 3D sequences -> 2D for ROS, then reshapes back.
    horizon: string like '6h', '12h', '24h'
    """
    N, T, F = X_train.shape
    print(f"\n  --- target_{horizon} ---")
    before = Counter(y_train.astype(int))
    print(f"  Before: {dict(before)}  ratio={before[0]//max(1,before[1])}:1  "
          f"({before[1]/N*100:.2f}% positive)")

    # Flatten: (N, T, F) -> (N, T*F)
    X_flat = X_train.reshape(N, T * F)

    ros = RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY,
                            random_state=RANDOM_SEED)
    X_res, y_res = ros.fit_resample(X_flat, y_train.astype(int))

    # Reshape back: (N_bal, T*F) -> (N_bal, T, F)
    X_bal = X_res.reshape(-1, T, F).astype(np.float32)
    y_bal = y_res.astype(np.float32)

    after = Counter(y_bal.astype(int))
    print(f"  After : {dict(after)}  ratio={after[0]//max(1,after[1])}:1  "
          f"({after[1]/len(y_bal)*100:.2f}% positive)")
    print(f"  Sequences: {N:,} -> {len(y_bal):,}  (+{len(y_bal)-N:,} duplicated)")

    return X_bal, y_bal


def run():
    print("=" * 65)
    print("STEP 08b - Balancing All Targets (6h / 12h / 24h)")
    print("=" * 65)

    # Load base X arrays (feature-selected, 30 features)
    print("\n  Loading X arrays ...")
    X_train_raw = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))      # (N, 24, 55)
    feature_mask = np.load(os.path.join(OUTPUT_DIR, "feature_mask.npy"))
    X_train = X_train_raw[:, :, feature_mask]                           # (N, 24, 30)
    print(f"  X_train shape: {X_train.shape}")

    report_lines = [
        "STEP 08b - Dataset Balancing Report",
        "Targets: 6h, 12h, 24h  |  Method: RandomOverSampler",
        "=" * 60, ""
    ]

    config = [
        ("6h",  "y_train.npy",      None),   # 6h uses original y_train
        ("12h", "y_train_12h.npy",  None),
        ("24h", "y_train_24h.npy",  None),
    ]

    for horizon, y_file, _ in config:
        # horizon is already the correct string (e.g. "6h", "12h", "24h")
        y_path = os.path.join(OUTPUT_DIR, y_file)
        if not os.path.exists(y_path):
            print(f"\n  [SKIP] {y_file} not found — run step07b first")
            continue

        y_train = np.load(y_path).astype(np.float32)

        # Ensure X and y are aligned
        n = min(len(X_train), len(y_train))
        X_tr = X_train[:n]
        y_tr = y_train[:n]

        X_bal, y_bal = balance_target(horizon, X_tr, y_tr)

        # Save (horizon already includes 'h', e.g. '6h', '12h', '24h')
        xp = os.path.join(OUTPUT_DIR, f"X_train_bal_{horizon}.npy")
        yp = os.path.join(OUTPUT_DIR, f"y_train_bal_{horizon}.npy")
        np.save(xp, X_bal)
        np.save(yp, y_bal)
        print(f"  Saved: X_train_bal_{horizon}.npy  {X_bal.shape}")
        print(f"  Saved: y_train_bal_{horizon}.npy  {y_bal.shape}")

        c = Counter(y_bal.astype(int))
        report_lines += [
            f"Target: {horizon}",
            f"  Before  : pos={int(y_tr.sum()):,}  neg={int((y_tr==0).sum()):,}",
            f"  After   : pos={c[1]:,}  neg={c[0]:,}",
            f"  X shape : {X_bal.shape}",
            f"  y shape : {y_bal.shape}", ""
        ]

    # Save report
    rp = os.path.join(REPORTS_DIR, "balancing_all_targets_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {rp}")
    print("\n  Done! You can now run step10_v3_multi_target.py")


if __name__ == "__main__":
    run()
