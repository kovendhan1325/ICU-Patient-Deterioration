"""
==============================================================================
STEP 08 - Dataset Balancing & Feature Selection
==============================================================================
Objectives:
  1. Feature Selection
     - Remove near-zero-variance features (VarianceThreshold)
     - Keep top-K features by mutual information with target (SelectKBest)

  2. Class Balancing (Training set ONLY — no val/test leakage)
     - Method: SMOTETomek (hybrid: over-samples minority + removes
       noisy borderline majority samples via Tomek Links)
     - Fallback: SMOTE + RandomUnderSampler if SMOTETomek not available

Strategy for 3-D LSTM sequences:
  - Flatten  (N, T, F)  →  (N, T*F)  for resampling
  - Apply SMOTE on flattened 2-D representation
  - Reshape  (N_new, T*F)  →  (N_new, T, F_selected)

Outputs (saved to output/):
  - X_train_balanced.npy   : balanced + feature-selected training sequences
  - y_train_balanced.npy   : balanced training labels
  - X_val_selected.npy     : feature-selected validation sequences
  - X_test_selected.npy    : feature-selected test sequences
  - feature_mask.npy       : boolean mask of selected features (length = F)
  - feature_indices.npy    : integer indices of selected features

Reports:
  - reports/balancing_report.txt
==============================================================================
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dependency check — give a clear error if imbalanced-learn is missing
# ---------------------------------------------------------------------------
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False
    print(
        "\n  [!] imbalanced-learn is NOT installed.\n"
        "      Install it with:  pip install imbalanced-learn\n"
    )
    sys.exit(1)

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

# ---------------------------------------------------------------------------
# Import paths from config
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, REPORTS_DIR, RANDOM_SEED

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Number of top features to keep after mutual-info selection.
# Set to None to keep all variance-filtered features.
TOP_K_FEATURES = 30

# Variance threshold: drop features with variance < this value
# (applied on flattened, standardised sequences)
VARIANCE_THRESHOLD = 0.01

# SMOTE: desired ratio of minority to majority AFTER resampling
# 0.5 → for every 2 majority samples there is 1 minority sample
# 1.0 → perfect 50/50 balance
SMOTE_SAMPLING_RATIO = 0.5

# Tomek under-sampling ratio (keep half the majority after Tomek cleaning)
TOMEK_SAMPLING_RATIO = 0.8

# SMOTE k-neighbors (lower if dataset is very small)
SMOTE_K_NEIGHBORS = 5


# ===========================================================================
# HELPERS
# ===========================================================================

def load_arrays():
    """Load X/y splits produced by Step 07."""
    print("  Loading sequences from output/ ...")
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    print(f"    -> X_train : {X_train.shape}  |  positives: {y_train.sum()} / {len(y_train)}"
          f"  ({y_train.mean()*100:.1f}%)")
    print(f"    -> X_val   : {X_val.shape}")
    print(f"    -> X_test  : {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def print_dist(label, y):
    pos   = int(y.sum())
    total = len(y)
    neg   = total - pos
    ratio = pos / max(1, total) * 100
    print(f"    {label:30s}  pos={pos:6,}  neg={neg:6,}  total={total:6,}  ({ratio:.1f}%)")


# ===========================================================================
# FEATURE SELECTION
# ===========================================================================

def select_features(X_train_2d, y_train, n_features_orig):
    """
    Two-stage feature selection on the FLATTENED training set.

    X_train_2d has shape (N, T*F). We treat each time-step × feature
    combination as a separate column for selection purposes, then aggregate
    the per-timestep selection back to a per-feature mask.

    Returns
    -------
    feature_mask : np.ndarray of bool, shape (F,)
        True for every original feature index that is kept.
    selected_cols_2d : np.ndarray of int
        Column indices into the FLATTENED array that are selected.
    """
    print(f"\n  Feature Selection  (original features: {n_features_orig}) ...")
    T = X_train_2d.shape[1] // n_features_orig   # sequence length

    # ------------------------------------------------------------------
    # Stage 1: Variance threshold
    # ------------------------------------------------------------------
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        vt.fit(X_train_2d)
        var_mask = vt.get_support()          # bool mask over T*F columns
    except Exception:
        var_mask = np.ones(X_train_2d.shape[1], dtype=bool)

    X_var = X_train_2d[:, var_mask]
    print(f"    -> After VarianceThreshold : {var_mask.sum()} / {len(var_mask)} columns kept")

    # Convert the 2-D column mask to a per-feature (F-dimensional) mask
    # by OR-ing across all time steps.
    var_mask_2d = var_mask.reshape(T, n_features_orig)           # (T, F)
    feature_var_mask = var_mask_2d.any(axis=0)                   # (F,)

    # ------------------------------------------------------------------
    # Stage 2: Mutual information — select top K *features* (not columns)
    # ------------------------------------------------------------------
    # Average the flattened columns back to per-feature scores
    if X_var.shape[1] == 0:
        print("    [!] No features survived VarianceThreshold — keeping all.")
        return np.ones(n_features_orig, dtype=bool)

    # Compute MI on variance-filtered, flattened data
    try:
        mi_scores_2d = mutual_info_classif(
            X_var, y_train, discrete_features=False,
            random_state=RANDOM_SEED
        )
    except Exception as e:
        print(f"    [!] mutual_info_classif failed ({e}), skipping MI step.")
        mi_scores_2d = np.ones(X_var.shape[1])

    # Map 2-D column MI scores back to original F features
    # Rebuild a full-length scores array (unselected cols get score 0)
    full_2d_mi = np.zeros(X_train_2d.shape[1])
    full_2d_mi[var_mask] = mi_scores_2d
    # Reshape to (T, F) and take max across time steps per feature
    mi_per_feature = full_2d_mi.reshape(T, n_features_orig).max(axis=0)  # (F,)

    k = min(TOP_K_FEATURES if TOP_K_FEATURES else n_features_orig,
            int(feature_var_mask.sum()))
    top_k_idx = np.argsort(mi_per_feature)[::-1][:k]

    feature_mask = np.zeros(n_features_orig, dtype=bool)
    feature_mask[top_k_idx] = True

    print(f"    -> After SelectKBest (k={k})  : {feature_mask.sum()} features kept")
    return feature_mask


def apply_feature_mask_3d(X_3d, feature_mask):
    """Apply a per-feature boolean mask to a 3-D array (N, T, F)."""
    return X_3d[:, :, feature_mask]


# ===========================================================================
# CLASS BALANCING
# ===========================================================================

def balance_training_data(X_train_flat, y_train, n_seq_len, n_features_sel):
    """
    Apply SMOTETomek to the FLATTENED training sequences.

    Parameters
    ----------
    X_train_flat : (N, T*F_sel)  float32 array
    y_train      : (N,)          int array
    n_seq_len    : int           T (sequence length)
    n_features_sel: int          F_sel (number of selected features)

    Returns
    -------
    X_balanced : (N_new, T, F_sel)
    y_balanced : (N_new,)
    """
    n_minority = int(y_train.sum())
    n_majority = int((y_train == 0).sum())

    print(f"\n  Class Balancing ...")
    print(f"    Before  ->  minority: {n_minority:,}  majority: {n_majority:,}")

    if n_minority == 0:
        print("    [!] No positive samples found — skipping balancing.")
        X_3d = X_train_flat.reshape(-1, n_seq_len, n_features_sel)
        return X_3d, y_train

    # Adjust k_neighbors if dataset is very small
    k_nn = min(SMOTE_K_NEIGHBORS, n_minority - 1)
    if k_nn < 1:
        k_nn = 1

    # Desired minority ratio for SMOTE
    desired_ratio = min(SMOTE_SAMPLING_RATIO, 1.0)

    smote = SMOTE(
        sampling_strategy=desired_ratio,
        k_neighbors=k_nn,
        random_state=RANDOM_SEED,
    )

    # Try SMOTETomek first; fall back to SMOTE + RandomUnderSampler
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.under_sampling import TomekLinks
        sampler = SMOTETomek(
            smote=smote,
            random_state=RANDOM_SEED,
        )
        X_res, y_res = sampler.fit_resample(X_train_flat, y_train)
        method_used = "SMOTETomek"
    except Exception as e:
        print(f"    [!] SMOTETomek failed ({e}), falling back to SMOTE + RandomUnderSampler")
        rus = RandomUnderSampler(
            sampling_strategy=TOMEK_SAMPLING_RATIO,
            random_state=RANDOM_SEED,
        )
        X_smote, y_smote = smote.fit_resample(X_train_flat, y_train)
        X_res, y_res = rus.fit_resample(X_smote, y_smote)
        method_used = "SMOTE + RandomUnderSampler"

    n_min_after = int(y_res.sum())
    n_maj_after = int((y_res == 0).sum())
    print(f"    Method  ->  {method_used}")
    print(f"    After   ->  minority: {n_min_after:,}  majority: {n_maj_after:,}"
          f"  total: {len(y_res):,}  ({n_min_after/max(1,len(y_res))*100:.1f}% positive)")

    # Reshape back to 3-D
    X_balanced = X_res.reshape(-1, n_seq_len, n_features_sel).astype(np.float32)
    y_balanced = y_res.astype(np.int32)

    return X_balanced, y_balanced


# ===========================================================================
# REPORT
# ===========================================================================

def write_report(
    original_shapes, y_train_orig, y_train_bal,
    feature_mask, feature_indices,
    balanced_shapes
):
    """Write a plain-text balancing report."""
    lines = [
        "=" * 70,
        "STEP 08 - BALANCING & FEATURE SELECTION REPORT",
        "=" * 70,
        "",
        "CLASS DISTRIBUTION (Training Set)",
        "-" * 40,
        f"  Before  :  positives = {int(y_train_orig.sum()):,}  "
        f"/ {len(y_train_orig):,}  "
        f"({y_train_orig.mean()*100:.2f}%)",
        f"  After   :  positives = {int(y_train_bal.sum()):,}  "
        f"/ {len(y_train_bal):,}  "
        f"({y_train_bal.mean()*100:.2f}%)",
        "",
        "FEATURE SELECTION",
        "-" * 40,
        f"  Original features  : {len(feature_mask)}",
        f"  Selected features  : {feature_mask.sum()}",
        f"  Removed features   : {(~feature_mask).sum()}",
        f"  Selected indices   : {feature_indices.tolist()}",
        "",
        "OUTPUT SHAPES",
        "-" * 40,
    ]
    for name, shape in balanced_shapes.items():
        lines.append(f"  {name:30s} : {shape}")

    lines += [
        "",
        "=" * 70,
        "NOTE: Val and Test sets were NOT resampled (no leakage).",
        "      Feature selection mask applied to all three splits.",
        "=" * 70,
    ]

    report_path = os.path.join(REPORTS_DIR, "balancing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  * Report saved to {report_path}")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def run_step08():
    """Execute Step 08: Feature Selection + Class Balancing."""
    print("\n" + "=" * 70)
    print("STEP 08 - Dataset Balancing & Feature Selection")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays()

    N, T, F = X_train.shape
    print(f"\n  Input shape: N={N}, T={T} (seq_len), F={F} (features)")

    print("\n  Class distribution (original):")
    print_dist("Train", y_train)
    print_dist("Val  ", y_val)
    print_dist("Test ", y_test)

    # ------------------------------------------------------------------
    # 2. Feature selection (on flattened training data)
    # ------------------------------------------------------------------
    X_train_flat = X_train.reshape(N, T * F)      # (N, T*F)

    feature_mask = select_features(X_train_flat, y_train, F)
    feature_indices = np.where(feature_mask)[0]
    F_sel = int(feature_mask.sum())

    print(f"\n  Selected {F_sel} features out of {F}")

    # Apply mask to all splits
    X_train_sel = apply_feature_mask_3d(X_train, feature_mask)  # (N, T, F_sel)
    X_val_sel   = apply_feature_mask_3d(X_val,   feature_mask)
    X_test_sel  = apply_feature_mask_3d(X_test,  feature_mask)

    # Flatten selected train for SMOTE
    X_train_sel_flat = X_train_sel.reshape(N, T * F_sel)

    # ------------------------------------------------------------------
    # 3. Balance training data (SMOTE / SMOTETomek)
    # ------------------------------------------------------------------
    y_train_orig = y_train.copy()
    X_train_bal, y_train_bal = balance_training_data(
        X_train_sel_flat, y_train, T, F_sel
    )

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    print("\n  Saving balanced datasets ...")

    paths = {
        "X_train_balanced.npy" : (X_train_bal, "X_train_balanced"),
        "y_train_balanced.npy" : (y_train_bal, "y_train_balanced"),
        "X_val_selected.npy"   : (X_val_sel,   "X_val_selected"),
        "y_val.npy"            : (y_val,        "y_val (unchanged)"),
        "X_test_selected.npy"  : (X_test_sel,   "X_test_selected"),
        "y_test.npy"           : (y_test,        "y_test (unchanged)"),
        "feature_mask.npy"     : (feature_mask,  "feature_mask (bool)"),
        "feature_indices.npy"  : (feature_indices, "feature_indices (int)"),
    }

    balanced_shapes = {}
    for fname, (arr, label) in paths.items():
        out_path = os.path.join(OUTPUT_DIR, fname)
        np.save(out_path, arr)
        balanced_shapes[label] = arr.shape
        print(f"    -> {fname:35s}  shape={arr.shape}")

    # ------------------------------------------------------------------
    # 5. Write report
    # ------------------------------------------------------------------
    write_report(
        original_shapes={"X_train": X_train.shape},
        y_train_orig=y_train_orig,
        y_train_bal=y_train_bal,
        feature_mask=feature_mask,
        feature_indices=feature_indices,
        balanced_shapes=balanced_shapes,
    )

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print("\n  Class distribution (after balancing):")
    print_dist("Train (balanced)", y_train_bal)
    print_dist("Val   (original)", y_val)
    print_dist("Test  (original)", y_test)

    print(f"\n  * Step 08 complete.")
    print(f"    Use X_train_balanced / y_train_balanced for model training.")
    print(f"    Use X_val_selected   / X_test_selected for evaluation.")

    return {
        "X_train_balanced": X_train_bal,
        "y_train_balanced": y_train_bal,
        "X_val_selected":   X_val_sel,
        "y_val":            y_val,
        "X_test_selected":  X_test_sel,
        "y_test":           y_test,
        "feature_mask":     feature_mask,
        "feature_indices":  feature_indices,
        "n_features_selected": F_sel,
        "seq_len":          T,
    }


if __name__ == "__main__":
    run_step08()
