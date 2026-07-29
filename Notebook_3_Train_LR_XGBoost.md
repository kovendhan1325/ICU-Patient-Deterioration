# Notebook 3: Train Logistic Regression and XGBoost
This notebook trains baseline Logistic Regression and XGBoost models on the ICU dataset.
Like Random Forest, these models don't naturally handle 3D sequence data `(N, 24, 37)`, so we will flatten the time dimension to `(N, 24*37)`.



```python
import os
try:
    from google.colab import drive
    # Only mount if the drive isn't already mounted to avoid errors
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    data_dir = '/content/drive/MyDrive/ICU-Patient-Deterioration/preprocessing_pipeline/output'
    print('Running in Google Colab. Data directory set to Google Drive.')
except ImportError:
    # If not in Colab (running locally)
    data_dir = 'preprocessing_pipeline/output'
    print('Running Locally. Data directory set to local folder.')

```

    Running Locally. Data directory set to local folder.
    


```python
import numpy as np
import os

# Load Data
print('Loading data...')
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

print(f'Original X_train shape: {X_train.shape}')

```

    Loading data...
    

    Original X_train shape: (64121, 24, 37)
    

### Flatten 3D Sequences for Baseline Models
We reshape `(N, Sequence_Length, Features)` into `(N, Sequence_Length * Features)`.



```python
N_train, seq_len, n_features = X_train.shape
N_val = X_val.shape[0]

X_train_flat = X_train.reshape(N_train, seq_len * n_features)
X_val_flat = X_val.reshape(N_val, seq_len * n_features)

print(f'Flattened X_train shape: {X_train_flat.shape}')
print(f'Flattened X_val shape: {X_val_flat.shape}')

```

    Flattened X_train shape: (64121, 888)
    Flattened X_val shape: (15389, 888)
    

### Train Logistic Regression



```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Initialize and train Logistic Regression model
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
print('Training Logistic Regression...')
lr_model.fit(X_train_flat, y_train)
print('Training complete!')
```

    Training Logistic Regression...
    

    C:\Users\LAPTOP\AppData\Roaming\Python\Python314\site-packages\sklearn\linear_model\_logistic.py:1457: FutureWarning: 'n_jobs' has no effect since 1.8 and will be removed in 1.10. You provided 'n_jobs=-1', please leave it unspecified.
      warnings.warn(msg, category=FutureWarning)
    

    Training complete!
    


```python
# Logistic Regression Predictions & Evaluation
y_val_prob_lr = lr_model.predict_proba(X_val_flat)[:, 1]
print('--- Logistic Regression Validation Performance ---')
print(f'ROC-AUC Score: {roc_auc_score(y_val, y_val_prob_lr):.4f}')
print(f'PR-AUC (Average Precision) Score: {average_precision_score(y_val, y_val_prob_lr):.4f}')
```

    --- Logistic Regression Validation Performance ---
    ROC-AUC Score: 0.6346
    PR-AUC (Average Precision) Score: 0.0718
    

### Train XGBoost



```python
import xgboost as xgb

# Calculate scale_pos_weight to handle class imbalance for XGBoost
neg_count = len(y_train) - np.sum(y_train)
pos_count = np.sum(y_train)
scale_pos_weight = neg_count / pos_count

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
print('Training XGBoost...')
xgb_model.fit(X_train_flat, y_train)
print('Training complete!')
```

    Training XGBoost...
    

    Training complete!
    


```python
# XGBoost Predictions & Evaluation
y_val_prob_xgb = xgb_model.predict_proba(X_val_flat)[:, 1]
print('--- XGBoost Validation Performance ---')
print(f'ROC-AUC Score: {roc_auc_score(y_val, y_val_prob_xgb):.4f}')
print(f'PR-AUC (Average Precision) Score: {average_precision_score(y_val, y_val_prob_xgb):.4f}')
```

    --- XGBoost Validation Performance ---
    ROC-AUC Score: 0.6234
    PR-AUC (Average Precision) Score: 0.0788
    

### Patient Risk Stratification (Using XGBoost)
Categorizing the probabilities into low, moderate, high, and very high risk levels using the best performing model (typically XGBoost).


```python
import pandas as pd

# Create a DataFrame to view the final predictions and risk levels
results_df = pd.DataFrame({
    'True_Label': y_val,
    'Deterioration_Probability': y_val_prob_xgb
})

# Define risk categories based on probability thresholds
# You can adjust these thresholds based on clinical requirements
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
display(results_df[['True_Label', 'Deterioration_Probability', 'Risk_Level']].head(20))

print('\nRisk Level Distribution in Validation Set (XGBoost):')
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
      <td>0.012766</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.548737</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.445250</td>
      <td>Moderate Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.007095</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.006768</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.005225</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0.007393</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0.007818</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0.007955</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0.005360</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0.006378</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0.005903</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0.006128</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0.014742</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0.005882</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0.010171</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0.004640</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0.004961</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0.009793</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0.012824</td>
      <td>Low Risk</td>
    </tr>
  </tbody>
</table>
</div>


    
    Risk Level Distribution in Validation Set (XGBoost):
    Risk_Level
    Low Risk          14157
    Moderate Risk       852
    High Risk           341
    Very High Risk       39
    Name: count, dtype: int64
    
