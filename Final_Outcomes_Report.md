# ICU Patient Deterioration Prediction - Tuned Baseline Models Final Report

## Tuned Logistic Regression Performance
- **ROC-AUC Score**: 0.6364
- **PR-AUC (Average Precision)**: 0.0726

## Tuned XGBoost Performance
- **ROC-AUC Score**: 0.6276
- **PR-AUC (Average Precision)**: 0.0872

## Patient Risk Stratification (Using Tuned XGBoost)
The tuned model's predictions on the validation set (15,389 patients) were grouped into the following risk levels:

| Risk Level | Patient Count |
| :--- | :--- |
| **Low Risk** | 14,744 |
| **Moderate Risk** | 422 |
| **High Risk** | 193 |
| **Very High Risk** | 30 |
