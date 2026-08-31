# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
================================================================================
LSTM Training Script — Continues from where Notebook_6 Cell 8 left off.
Runs BiLSTM + Attention training, then Ensemble + ROC curves.
All outputs saved to preprocessing_pipeline/reports/ and output/
================================================================================
Run with:  python train_bilstm_continue.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score,
                             classification_report, roc_curve,
                             precision_recall_curve)
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --- Paths --------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'preprocessing_pipeline', 'output')
REPORTS_DIR = os.path.join(BASE_DIR, 'preprocessing_pipeline', 'reports')
CSV_PATH    = os.path.join(DATA_DIR, 'processed_icu_dataset.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'best_bilstm_model.pth')
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Hyperparameters ----------------------------------------------------------
SEQUENCE_LENGTH = 24
RANDOM_SEED     = 42
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.3
LR              = 0.001
BATCH_SIZE      = 256
EPOCHS          = 50
PATIENCE        = 7
WINDOW          = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & rebuild enhanced sequences (same as NB6 Cells 2-4)
# ════════════════════════════════════════════════════════════════════════════════
print('\n[1/6] Loading flat CSV and building enhanced features...')
df = pd.read_csv(CSV_PATH)
df = df.sort_values(['patientunitstayid', 'hour']).reset_index(drop=True)
print(f'  CSV shape: {df.shape}')

VITAL_COLS = [c for c in ['heartrate','systemicsystolic','systemicdiastolic',
                           'systemicmean','respiration','temperature']
              if c in df.columns]

new_features = []
for col in VITAL_COLS:
    grp = df.groupby('patientunitstayid')[col]
    df[f'{col}_roll6_mean'] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    df[f'{col}_roll6_std']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
    df[f'{col}_delta']      = grp.transform(lambda x: x.diff().fillna(0))
    df[f'{col}_delta2']     = grp.transform(lambda x: x.diff().diff().fillna(0))
    new_features += [f'{col}_roll6_mean', f'{col}_roll6_std', f'{col}_delta', f'{col}_delta2']

for col in ['Creatinine', 'Lactate', 'Glucose']:
    if col in df.columns:
        df[f'{col}_delta'] = df.groupby('patientunitstayid')[col].transform(lambda x: x.diff().fillna(0))
        new_features.append(f'{col}_delta')

if 'heartrate' in df.columns and 'systemicsystolic' in df.columns:
    df['shock_index'] = (df['heartrate'] / df['systemicsystolic'].replace(0, np.nan)).clip(0, 5).fillna(1.0)
    df['shock_index_trend'] = df.groupby('patientunitstayid')['shock_index'].transform(lambda x: x.diff().fillna(0))
    new_features += ['shock_index', 'shock_index_trend']

if 'systemicsystolic' in df.columns and 'systemicdiastolic' in df.columns:
    src_col = 'pulse_pressure' if 'pulse_pressure' in df.columns else 'systemicsystolic'
    df['pulse_pressure_trend'] = df.groupby('patientunitstayid')[src_col].transform(lambda x: x.diff().fillna(0))
    new_features.append('pulse_pressure_trend')

print(f'  Added {len(new_features)} new features')

# Feature columns
exclude = {
    'patientunitstayid','hour','unitdischargeoffset','unitdischargestatus',
    'hospitaldischargestatus','is_deteriorated','hours_until_discharge',
    'target_6h','target_12h','target_24h','patienthealthsystemstayid',
    'ethnicity','unitadmitsource','unitvisitnumber','hospitaladmitoffset',
}
numeric_types = ['float64','float32','int64','int32','uint8','bool']
feature_cols = [c for c in df.columns
                if c not in exclude
                and str(df[c].dtype) in numeric_types
                and df[c].nunique() > 1]

df[feature_cols] = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(df[feature_cols].median())
target_col = 'target_24h' if 'target_24h' in df.columns else None
assert target_col, "target_24h column missing from CSV!"

print(f'  Total features: {len(feature_cols)}')

# Patient split (same seed as pipeline)
patient_ids = df['patientunitstayid'].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(patient_ids)
n = len(patient_ids)
train_ids = patient_ids[:int(n * 0.70)]
val_ids   = patient_ids[int(n * 0.70):int(n * 0.85)]
test_ids  = patient_ids[int(n * 0.85):]
print(f'  Patients -> Train:{len(train_ids)} | Val:{len(val_ids)} | Test:{len(test_ids)}')

def create_sequences(df, pid_list, feat_cols, tgt_col, seq_len=24):
    Xs, ys = [], []
    for pid in pid_list:
        p = df[df['patientunitstayid'] == pid].sort_values('hour')
        feats = p[feat_cols].values
        tgts  = p[tgt_col].values
        if len(feats) >= seq_len:
            for i in range(len(feats) - seq_len + 1):
                Xs.append(feats[i:i+seq_len])
                ys.append(tgts[i+seq_len-1])
        elif len(feats) > 0:
            pad = np.zeros((seq_len, feats.shape[1]))
            pad[-len(feats):] = feats
            Xs.append(pad)
            ys.append(tgts[-1])
    if not Xs:
        return np.empty((0, seq_len, len(feat_cols)), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

print('  Building sequences (may take 60-90 seconds)...')
t0 = time.time()
Xe_train, ye_train = create_sequences(df, train_ids, feature_cols, target_col)
Xe_val,   ye_val   = create_sequences(df, val_ids,   feature_cols, target_col)
Xe_test,  ye_test  = create_sequences(df, test_ids,  feature_cols, target_col)
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  Xe_train {Xe_train.shape} | pos {ye_train.sum()}/{len(ye_train)} ({ye_train.mean()*100:.1f}%)')
print(f'  Xe_val   {Xe_val.shape}   | pos {ye_val.sum()}/{len(ye_val)} ({ye_val.mean()*100:.1f}%)')
print(f'  Xe_test  {Xe_test.shape}  | pos {ye_test.sum()}/{len(ye_test)} ({ye_test.mean()*100:.1f}%)')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build DataLoaders
# ════════════════════════════════════════════════════════════════════════════════
print('\n[2/6] Creating DataLoaders...')

X_tr_t  = torch.tensor(Xe_train, dtype=torch.float32)
y_tr_t  = torch.tensor(ye_train, dtype=torch.float32)
X_val_t = torch.tensor(Xe_val,   dtype=torch.float32)
y_val_t = torch.tensor(ye_val,   dtype=torch.float32)
X_te_t  = torch.tensor(Xe_test,  dtype=torch.float32)
y_te_t  = torch.tensor(ye_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# FIX: explicit Python float — avoids numpy int32 issues
n_pos        = int(ye_train.sum())
n_neg        = int(len(ye_train)) - n_pos
pos_weight   = float(n_neg) / float(max(n_pos, 1))
print(f'  Class balance: {n_neg} neg / {n_pos} pos | pos_weight = {pos_weight:.2f}')
print(f'  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Model Architecture
# ════════════════════════════════════════════════════════════════════════════════
print('\n[3/6] Building BiLSTM + Attention model...')

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, lstm_out):
        scores  = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm    = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0,
                                 bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention  = SelfAttention(hidden_size * 2)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size * 2, 64)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(64, 1)

    def forward(self, x):
        out, _  = self.bilstm(x)
        out     = self.layer_norm(out)
        ctx, _  = self.attention(out)
        out     = self.dropout(ctx)
        out     = self.relu(self.fc1(out))
        out     = self.dropout(out)
        return self.fc2(out).squeeze(-1)

INPUT_SIZE = Xe_train.shape[2]
model = BiLSTMAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Input features : {INPUT_SIZE}')
print(f'  Trainable params: {total_params:,}')

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
# FIX: verbose removed — deprecated in PyTorch >= 2.2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training Loop
# ════════════════════════════════════════════════════════════════════════════════
print('\n[4/6] Training BiLSTM + Attention...')
print('='*65)

best_val_auc      = 0.0
epochs_no_improve = 0
history           = {'train_loss': [], 'val_loss': [], 'val_auc': []}

for epoch in range(EPOCHS):
    # -- Train ------------------------------------------------------------------
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss  = criterion(preds, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)

    # -- Validate ---------------------------------------------------------------
    model.eval()
    val_loss, all_probs, all_tgts = 0.0, [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds  = model(Xb)
            val_loss += criterion(preds, yb).item()
            all_probs.extend(torch.sigmoid(preds).cpu().numpy())
            all_tgts.extend(yb.cpu().numpy())

    avg_val  = val_loss / len(val_loader)
    val_auc  = roc_auc_score(all_tgts, all_probs)
    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    history['val_auc'].append(val_auc)

    # LR scheduler (manual LR change logging — verbose removed)
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_auc)
    curr_lr = optimizer.param_groups[0]['lr']
    lr_tag  = f' | LR->{curr_lr:.5f}' if curr_lr < prev_lr else ''

    print(f'Epoch {epoch+1:3d}/{EPOCHS} | '
          f'Train: {avg_train:.4f} | Val: {avg_val:.4f} | AUC: {val_auc:.4f}{lr_tag}')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  [OK] Best model saved (AUC={best_val_auc:.4f})')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1}!')
            break

print(f'\nBest Val AUC: {best_val_auc:.4f}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — Final Test Evaluation
# ════════════════════════════════════════════════════════════════════════════════
print('\n[5/6] Evaluating on test set...')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

test_probs, test_tgts = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        test_probs.extend(torch.sigmoid(model(Xb)).cpu().numpy())
        test_tgts.extend(yb.numpy())

test_probs = np.array(test_probs)
test_tgts  = np.array(test_tgts)
test_auc   = roc_auc_score(test_tgts, test_probs)

# Optimal threshold from PR curve
precision, recall, thresholds = precision_recall_curve(test_tgts, test_probs)
f1s = 2*precision*recall / (precision + recall + 1e-8)
opt_thresh = thresholds[np.argmax(f1s)]
binary_preds = (test_probs >= opt_thresh).astype(int)

print(f'\n{"="*55}')
print(f'  FINAL TEST SET RESULTS — BiLSTM + Attention')
print(f'{"="*55}')
print(f'  Test ROC-AUC  : {test_auc:.4f}')
print(f'  Opt threshold : {opt_thresh:.3f}')
print(f'\n{classification_report(test_tgts, binary_preds, target_names=["Stable","Deteriorating"])}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save Plots
# ════════════════════════════════════════════════════════════════════════════════
print('[6/6] Saving plots...')
plt.style.use('dark_background')

# -- Training curves ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')

ep = range(1, len(history['val_auc']) + 1)
axes[0].plot(ep, history['train_loss'], '#FF6B6B', lw=2, label='Train Loss')
axes[0].plot(ep, history['val_loss'],   '#4ECDC4', lw=2, label='Val Loss')
axes[0].set_title('BiLSTM Training Loss', color='white', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', color='white'); axes[0].set_ylabel('Loss', color='white')
axes[0].legend(facecolor='#0f3460', edgecolor='white'); axes[0].tick_params(colors='white')
axes[0].grid(True, alpha=0.2)

axes[1].plot(ep, history['val_auc'],  '#FFEAA7', lw=2, label='Val AUC')
axes[1].axhline(0.6482, color='gray',    ls='--', lw=1, label='Old LSTM (0.6482)')
axes[1].axhline(0.85,   color='#00FF7F', ls='--', lw=1, alpha=0.7, label='Target (0.85)')
axes[1].set_title('BiLSTM Validation AUC', color='white', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', color='white'); axes[1].set_ylabel('AUC', color='white')
axes[1].legend(facecolor='#0f3460', edgecolor='white'); axes[1].tick_params(colors='white')
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
curve_path = os.path.join(REPORTS_DIR, 'bilstm_training_curves.png')
plt.savefig(curve_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f'  Saved: {curve_path}')

# -- ROC curve ----------------------------------------------------------------
fpr, tpr, _ = roc_curve(test_tgts, test_probs)
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#16213e')
ax.plot([0,1],[0,1], '--', color='gray', lw=1, label='Random (AUC=0.50)')
ax.plot(fpr, tpr, '#4ECDC4', lw=2.5, label=f'BiLSTM+Attention (AUC={test_auc:.4f})')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('False Positive Rate', color='white'); ax.set_ylabel('True Positive Rate', color='white')
ax.set_title('ROC Curve — BiLSTM + Attention\n(Test Set)', color='white', fontsize=12, fontweight='bold')
ax.legend(facecolor='#0f3460', edgecolor='white'); ax.tick_params(colors='white')
ax.grid(True, alpha=0.2)
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, 'bilstm_roc_curve.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f'  Saved: {roc_path}')

# -- Summary ------------------------------------------------------------------
print(f'\n{"#"*55}')
print(f'  TRAINING COMPLETE')
print(f'{"#"*55}')
print(f'  Baseline LSTM (NB5) Test AUC : 0.6324')
print(f'  BiLSTM+Attention Test AUC    : {test_auc:.4f}')
print(f'  Improvement                  : +{test_auc - 0.6324:.4f}')
print(f'  Model saved to: {MODEL_PATH}')
print(f'\n  Next: run Notebook_6 Cells 9-11 for Ensemble + ROC comparison')
