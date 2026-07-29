# ICU Patient Deterioration Prediction - Baseline Models Final Report

## Logistic Regression Performance
- **ROC-AUC Score**: 0.6346
- **PR-AUC (Average Precision)**: 0.0718

## XGBoost Performance
- **ROC-AUC Score**: 0.6234
- **PR-AUC (Average Precision)**: 0.0788

## Patient Risk Stratification (Using XGBoost)
The model's predictions on the validation set (15,389 patients) were grouped into the following risk levels:

| Risk Level | Patient Count |
| :--- | :--- |
| **Low Risk** | 14,157 |
| **Moderate Risk** | 852 |
| **High Risk** | 341 |
| **Very High Risk** | 39 |
