
# UPDATE THIS PATH to wherever you uploaded your 'preprocessing_pipeline/output' folder in Google Drive
data_dir = 'preprocessing_pipeline/output'

import numpy as np
import os

# Load Data
print('Loading data...')
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

print(f'Original X_train shape: {X_train.shape}')

N_train, seq_len, n_features = X_train.shape
N_val = X_val.shape[0]

X_train_flat = X_train.reshape(N_train, seq_len * n_features)
X_val_flat = X_val.reshape(N_val, seq_len * n_features)

print(f'Flattened X_train shape: {X_train_flat.shape}')
print(f'Flattened X_val shape: {X_val_flat.shape}')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Initialize model (using class_weight to handle imbalanced deterioration classes)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)

print('Training Random Forest...')
rf_model.fit(X_train_flat, y_train)
print('Training complete!')

# Predictions
y_val_pred = rf_model.predict(X_val_flat)
y_val_prob = rf_model.predict_proba(X_val_flat)[:, 1]

print('--- Validation Set Performance ---')
print(classification_report(y_val, y_val_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_val, y_val_prob):.4f}')
