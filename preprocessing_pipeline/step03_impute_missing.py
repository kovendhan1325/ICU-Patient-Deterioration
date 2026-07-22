"""
==============================================================================
STEP 03 - Missing Value Handling
==============================================================================
- Generate before-imputation missing value report
- Numerical: median -> forward fill -> backward fill (per patient)
- Categorical: most frequent value
- Generate after-imputation missing value report
- Save both reports
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from config import INTERMEDIATE_DIR, REPORTS_DIR


def missing_report(data, label=""):
    """Generate a missing value report across all dataframes."""
    rows = []
    for name, df in data.items():
        if df is None:
            continue
        for col in df.columns:
            total = len(df)
            missing = df[col].isna().sum()
            pct = (missing / total * 100) if total > 0 else 0
            rows.append({
                "table": name,
                "column": col,
                "total_rows": total,
                "missing_count": missing,
                "missing_pct": round(pct, 2),
                "report_phase": label,
            })
    return pd.DataFrame(rows)


def impute_vitals(df):
    """Impute missing values in vitals table."""
    print("  Imputing vitals ...")
    vital_cols = ["heartrate", "systemicsystolic", "systemicdiastolic",
                  "systemicmean", "respiration", "spo2", "temperature"]
    existing = [c for c in vital_cols if c in df.columns]

    # Per-patient forward fill, then backward fill
    df = df.sort_values(["patientunitstayid", "observationoffset"])
    df[existing] = df.groupby("patientunitstayid")[existing].transform(
        lambda x: x.ffill().bfill()
    )

    # Remaining nulls: global median
    for col in existing:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df


def impute_labs(df):
    """Impute missing values in labs table."""
    print("  Imputing labs ...")
    # Lab results should not be null after step02 cleaning
    # But forward fill per patient per lab test for any remaining
    df = df.sort_values(["patientunitstayid", "labresultoffset"])
    df["labresult"] = df.groupby(["patientunitstayid", "labname"])["labresult"].transform(
        lambda x: x.ffill().bfill()
    )
    # Remaining: global median per lab test
    medians = df.groupby("labname")["labresult"].transform("median")
    df["labresult"] = df["labresult"].fillna(medians)
    return df


def impute_patient(df):
    """Impute missing values in patient table."""
    print("  Imputing patient ...")
    # Numeric columns: median
    numeric_cols = ["age", "admissionheight", "admissionweight"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: most frequent
    cat_cols = ["gender", "ethnicity", "unitadmitsource"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    return df


def run_step03(data=None):
    """Execute Step 03: Missing Value Handling."""
    print("\n" + "=" * 70)
    print("STEP 03 - Missing Value Handling")
    print("=" * 70)

    if data is None:
        with open(os.path.join(INTERMEDIATE_DIR, "step02_cleaned.pkl"), "rb") as f:
            data = pickle.load(f)

    # Generate BEFORE report
    print("  Generating missing value report (BEFORE) ...")
    report_before = missing_report(data, label="BEFORE")

    # Impute
    data["patient"] = impute_patient(data["patient"])
    data["vitals"] = impute_vitals(data["vitals"])
    data["labs"] = impute_labs(data["labs"])

    # Generate AFTER report
    print("  Generating missing value report (AFTER) ...")
    report_after = missing_report(data, label="AFTER")

    # Combine and save report
    full_report = pd.concat([report_before, report_after], ignore_index=True)
    report_path = os.path.join(REPORTS_DIR, "missing_value_report.csv")
    full_report.to_csv(report_path, index=False)
    print(f"  * Missing value report saved to {report_path}")

    # Print summary
    before_total = report_before["missing_count"].sum()
    after_total = report_after["missing_count"].sum()
    print(f"    Total missing BEFORE: {before_total:,}")
    print(f"    Total missing AFTER:  {after_total:,}")
    print(f"    Resolved: {before_total - after_total:,}")

    # Save intermediate
    output_path = os.path.join(INTERMEDIATE_DIR, "step03_imputed.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  * Step 03 complete. Saved to {output_path}")
    return data


if __name__ == "__main__":
    run_step03()

