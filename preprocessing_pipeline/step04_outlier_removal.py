"""
==============================================================================
STEP 04 - Outlier Detection & Removal
==============================================================================
- Apply physiological range clipping to vitals and labs
- Values outside range -> set to NaN -> re-impute with forward fill
- Generate outlier summary report
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import INTERMEDIATE_DIR, REPORTS_DIR, OUTLIER_THRESHOLDS


def clip_outliers(df, columns_to_check, report_rows):
    """
    For each column in columns_to_check, if it's in OUTLIER_THRESHOLDS,
    set out-of-range values to NaN and record counts.
    """
    for col in columns_to_check:
        if col not in df.columns or col not in OUTLIER_THRESHOLDS:
            continue

        lo, hi = OUTLIER_THRESHOLDS[col]
        mask = (df[col] < lo) | (df[col] > hi)
        count = mask.sum()

        if count > 0:
            df.loc[mask, col] = np.nan

        report_rows.append({
            "feature": col,
            "threshold_low": lo,
            "threshold_high": hi,
            "outliers_detected": int(count),
            "total_values": int(df[col].notna().sum() + count),
            "outlier_pct": round(count / max(1, len(df)) * 100, 3),
        })

    return df


def run_step04(data=None):
    """Execute Step 04: Outlier Detection & Removal."""
    print("\n" + "=" * 70)
    print("STEP 04 - Outlier Detection & Removal")
    print("=" * 70)

    if data is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step03_imputed.pkl"), "rb") as f:
            data = pickle.load(f)

    report_rows = []

    # --- Vitals outliers ---
    print("  Detecting outliers in vitals ...")
    vital_cols = ["heartrate", "systemicsystolic", "systemicdiastolic",
                  "systemicmean", "respiration", "spo2", "temperature"]
    existing_vitals = [c for c in vital_cols if c in data["vitals"].columns]
    data["vitals"] = clip_outliers(data["vitals"], existing_vitals, report_rows)

    # Re-impute NaNs created by outlier removal (forward fill per patient)
    data["vitals"] = data["vitals"].sort_values(["patientunitstayid", "observationoffset"])
    data["vitals"][existing_vitals] = data["vitals"].groupby("patientunitstayid")[existing_vitals].transform(
        lambda x: x.ffill().bfill()
    )
    # Remaining: global median
    for col in existing_vitals:
        median_val = data["vitals"][col].median()
        data["vitals"][col] = data["vitals"][col].fillna(median_val)

    # --- Labs outliers ---
    print("  Detecting outliers in labs ...")
    # Labs are in long format, so we need to check per lab test
    lab_report_rows = []
    for lab_name, (lo, hi) in OUTLIER_THRESHOLDS.items():
        mask_lab = data["labs"]["labname"] == lab_name
        if mask_lab.sum() == 0:
            continue
        sub = data["labs"].loc[mask_lab, "labresult"]
        outlier_mask = (sub < lo) | (sub > hi)
        count = outlier_mask.sum()
        if count > 0:
            data["labs"].loc[mask_lab & ((data["labs"]["labresult"] < lo) | (data["labs"]["labresult"] > hi)), "labresult"] = np.nan

        report_rows.append({
            "feature": f"lab_{lab_name}",
            "threshold_low": lo,
            "threshold_high": hi,
            "outliers_detected": int(count),
            "total_values": int(mask_lab.sum()),
            "outlier_pct": round(count / max(1, mask_lab.sum()) * 100, 3),
        })

    # Re-impute lab NaNs
    data["labs"]["labresult"] = data["labs"].groupby(
        ["patientunitstayid", "labname"]
    )["labresult"].transform(lambda x: x.ffill().bfill())
    medians = data["labs"].groupby("labname")["labresult"].transform("median")
    data["labs"]["labresult"] = data["labs"]["labresult"].fillna(medians)

    # Generate report
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(REPORTS_DIR, "outlier_summary.csv")
    report_df.to_csv(report_path, index=False)
    print(f"  * Outlier report saved to {report_path}")

    total_outliers = report_df["outliers_detected"].sum()
    print(f"    Total outliers detected and removed: {total_outliers:,}")

    # Save intermediate
    output_path = os.path.join(INTERMEDIATE_DIR, "step04_no_outliers.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  * Step 04 complete. Saved to {output_path}")
    return data


if __name__ == "__main__":
    run_step04()

