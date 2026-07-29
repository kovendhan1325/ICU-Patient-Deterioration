# Notebook 4: Train Tuned Logistic Regression and XGBoost
This notebook performs rigorous hyperparameter tuning using Cross-Validation to find the optimal settings for Logistic Regression and XGBoost models.



```python
import os
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    data_dir = '/content/drive/MyDrive/ICU-Patient-Deterioration/preprocessing_pipeline/output'
    print('Running in Google Colab.')
except ImportError:
    data_dir = 'preprocessing_pipeline/output'
    print('Running Locally.')

```

    Running Locally.
    


```python
import numpy as np
import os
print('Loading data...')
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

```

    Loading data...
    


```python
N_train, seq_len, n_features = X_train.shape
N_val = X_val.shape[0]
X_train_flat = X_train.reshape(N_train, seq_len * n_features)
X_val_flat = X_val.reshape(N_val, seq_len * n_features)
print(f'Flattened X_train shape: {X_train_flat.shape}')

```

    Flattened X_train shape: (64121, 888)
    

### Hyperparameter Tuning: Logistic Regression
Using GridSearchCV with 3-fold cross-validation.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score

# We limit the grid search space to avoid extremely long training times.
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2']
}

lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
grid_search_lr = GridSearchCV(
    estimator=lr_base,
    param_grid=param_grid_lr,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2
)

print('Starting Grid Search for Logistic Regression...')
grid_search_lr.fit(X_train_flat, y_train)
print(f'Best LR Parameters: {grid_search_lr.best_params_}')
best_lr_model = grid_search_lr.best_estimator_

```

    Starting Grid Search for Logistic Regression...
    Fitting 3 folds for each of 4 candidates, totalling 12 fits
    

    C:\Users\LAPTOP\AppData\Roaming\Python\Python314\site-packages\sklearn\linear_model\_logistic.py:1403: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
      warnings.warn(
    C:\Users\LAPTOP\AppData\Roaming\Python\Python314\site-packages\sklearn\linear_model\_logistic.py:1457: FutureWarning: 'n_jobs' has no effect since 1.8 and will be removed in 1.10. You provided 'n_jobs=-1', please leave it unspecified.
      warnings.warn(msg, category=FutureWarning)
    

    Best LR Parameters: {'C': 0.01, 'penalty': 'l2'}
    


```python
y_val_prob_lr = best_lr_model.predict_proba(X_val_flat)[:, 1]
print('--- Tuned Logistic Regression Validation Performance ---')
print(f'ROC-AUC Score: {roc_auc_score(y_val, y_val_prob_lr):.4f}')
print(f'PR-AUC (Average Precision) Score: {average_precision_score(y_val, y_val_prob_lr):.4f}')

```

    --- Tuned Logistic Regression Validation Performance ---
    ROC-AUC Score: 0.6364
    PR-AUC (Average Precision) Score: 0.0726
    

### Hyperparameter Tuning: XGBoost
Using RandomizedSearchCV with 3-fold cross-validation.


```python
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

neg_count = len(y_train) - np.sum(y_train)
pos_count = np.sum(y_train)
scale_pos_weight = neg_count / pos_count

param_dist_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

random_search_xgb = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist_xgb,
    n_iter=10, # Number of parameter settings that are sampled
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print('Starting Randomized Search for XGBoost...')
random_search_xgb.fit(X_train_flat, y_train)
print(f'Best XGBoost Parameters: {random_search_xgb.best_params_}')
best_xgb_model = random_search_xgb.best_estimator_

```

    Starting Randomized Search for XGBoost...
    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    

    Best XGBoost Parameters: {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
    


```python
y_val_prob_xgb = best_xgb_model.predict_proba(X_val_flat)[:, 1]
print('--- Tuned XGBoost Validation Performance ---')
print(f'ROC-AUC Score: {roc_auc_score(y_val, y_val_prob_xgb):.4f}')
print(f'PR-AUC (Average Precision) Score: {average_precision_score(y_val, y_val_prob_xgb):.4f}')

```

    --- Tuned XGBoost Validation Performance ---
    ROC-AUC Score: 0.6276
    PR-AUC (Average Precision) Score: 0.0872
    

### Patient Risk Stratification (Using Tuned XGBoost)



```python
import pandas as pd

results_df = pd.DataFrame({
    'True_Label': y_val,
    'Deterioration_Probability': y_val_prob_xgb
})

def categorize_risk(prob):
    if prob < 0.25:
        return 'Low Risk'
    elif prob < 0.50:
        return 'Moderate Risk'
    elif prob < 0.75:
        return 'High Risk'
    else:
        return 'Very High Risk'

results_df['Risk_Level'] = results_df['Deterioration_Probability'].apply(categorize_risk)

print('Sample of Patient Predictions:')
display(results_df[['True_Label', 'Deterioration_Probability', 'Risk_Level']].head(10))

print('\nRisk Level Distribution in Validation Set (Tuned XGBoost):')
print(results_df['Risk_Level'].value_counts())

```

    Sample of Patient Predictions:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True_Label</th>
      <th>Deterioration_Probability</th>
      <th>Risk_Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.006396</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.566168</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.444877</td>
      <td>Moderate Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.000543</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.000513</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.000298</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0.000305</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0.000166</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0.000297</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0.000205</td>
      <td>Low Risk</td>
    </tr>
  </tbody>
</table>
</div>


    
    Risk Level Distribution in Validation Set (Tuned XGBoost):
    Risk_Level
    Low Risk          14744
    Moderate Risk       422
    High Risk           193
    Very High Risk       30
    Name: count, dtype: int64
    
