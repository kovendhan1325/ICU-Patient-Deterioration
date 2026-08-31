# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
================================================================================
UPGRADED LSTM Training Script — Targets >0.85 ROC-AUC
Architecture: Conv1D + BiLSTM + Multi-Head Self-Attention
Improvements over baseline (0.6324 AUC):
  1. Focal Loss        — crushes overconfident wrong predictions on minority class
  2. WeightedRandSampler — every batch is ~50% positive, model can't ignore them
  3. Conv1D front-end  — captures local vital-sign spikes before BiLSTM sees them
  4. CosineAnnealingWarmRestarts — escapes local minima via warm restarts
  5. Heavier L2 weight decay — prevents overfitting on oversampled minority class
================================================================================
Run with:  python train_bilstm_continue.py
"""

import os, time, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, classification_report,
                             roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'preprocessing_pipeline', 'output')
REPORTS_DIR = os.path.join(BASE_DIR, 'preprocessing_pipeline', 'reports')
CSV_PATH    = os.path.join(DATA_DIR, 'processed_icu_dataset.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'best_bilstm_model.pth')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 24
RANDOM_SEED     = 42
HIDDEN_SIZE     = 192        # Wider hidden state vs 128 baseline
NUM_LAYERS      = 2
DROPOUT         = 0.4        # Slightly higher dropout to control overfitting
LR              = 3e-4       # Lower LR → more stable convergence
BATCH_SIZE      = 256
EPOCHS          = 80         # More epochs (warm restarts need room)
PATIENCE        = 12         # Longer patience (warm restarts temporarily raise loss)
WEIGHT_DECAY    = 1e-4       # Heavier L2 regularization
WINDOW          = 6

# Focal Loss hyperparameters
FOCAL_ALPHA     = 0.75       # Weight for minority class (>0.5 = penalise FN more)
FOCAL_GAMMA     = 2.0        # Focusing factor (2.0 is the standard from the paper)

# Cosine annealing warm restarts
COSINE_T0       = 10         # First restart after 10 epochs
COSINE_T_MULT   = 2          # Each restart cycle doubles in length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('=' * 65)
print('  UPGRADED ConvBiLSTM — Targeting >0.85 ROC-AUC')
print('  Improvements: Focal Loss + WeightedSampler + Conv1D + Cosine LR')
print('=' * 65)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & rebuild enhanced sequences
# ════════════════════════════════════════════════════════════════════════════════
print('\n[1/6] Loading flat CSV and building enhanced features...')
df = pd.read_csv(CSV_PATH)
df = df.sort_values(['patientunitstayid', 'hour']).reset_index(drop=True)
print(f'  CSV shape: {df.shape}')

VITAL_COLS = [c for c in ['heartrate', 'systemicsystolic', 'systemicdiastolic',
                           'systemicmean', 'respiration', 'temperature']
              if c in df.columns]

LAB_COLS = [c for c in ['Creatinine', 'Lactate', 'Glucose', 'WBC',
                         'Hemoglobin', 'Platelets', 'Sodium', 'Potassium']
            if c in df.columns]

new_features = []

# Rolling vitals: mean, std, first-order delta, second-order delta
for col in VITAL_COLS:
    grp = df.groupby('patientunitstayid')[col]
    df[f'{col}_roll6_mean'] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    df[f'{col}_roll6_std']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
    df[f'{col}_delta']      = grp.transform(lambda x: x.diff().fillna(0))
    df[f'{col}_delta2']     = grp.transform(lambda x: x.diff().diff().fillna(0))
    new_features += [f'{col}_roll6_mean', f'{col}_roll6_std',
                     f'{col}_delta', f'{col}_delta2']

# Lab deltas
for col in LAB_COLS:
    df[f'{col}_delta'] = df.groupby('patientunitstayid')[col].transform(
        lambda x: x.diff().fillna(0))
    new_features.append(f'{col}_delta')

# Composite clinical scores
if 'heartrate' in df.columns and 'systemicsystolic' in df.columns:
    df['shock_index'] = (df['heartrate'] /
                         df['systemicsystolic'].replace(0, np.nan)).clip(0, 5).fillna(1.0)
    df['shock_index_trend'] = df.groupby('patientunitstayid')['shock_index'].transform(
        lambda x: x.diff().fillna(0))
    new_features += ['shock_index', 'shock_index_trend']

if 'systemicsystolic' in df.columns and 'systemicdiastolic' in df.columns:
    src_col = 'pulse_pressure' if 'pulse_pressure' in df.columns else 'systemicsystolic'
    df['pulse_pressure_trend'] = df.groupby('patientunitstayid')[src_col].transform(
        lambda x: x.diff().fillna(0))
    new_features.append('pulse_pressure_trend')

# Mean Arterial Pressure trend
if 'systemicmean' in df.columns:
    df['map_trend'] = df.groupby('patientunitstayid')['systemicmean'].transform(
        lambda x: x.diff().fillna(0))
    new_features.append('map_trend')

# Respiration / Heart Rate ratio (tachypnea indicator)
if 'respiration' in df.columns and 'heartrate' in df.columns:
    df['resp_hr_ratio'] = (df['respiration'] / df['heartrate'].replace(0, np.nan)).fillna(0).clip(0, 5)
    new_features.append('resp_hr_ratio')

print(f'  Added {len(new_features)} new engineered features')

# Feature columns
exclude = {
    'patientunitstayid', 'hour', 'unitdischargeoffset', 'unitdischargestatus',
    'hospitaldischargestatus', 'is_deteriorated', 'hours_until_discharge',
    'target_6h', 'target_12h', 'target_24h', 'patienthealthsystemstayid',
    'ethnicity', 'unitadmitsource', 'unitvisitnumber', 'hospitaladmitoffset',
}
numeric_types = ['float64', 'float32', 'int64', 'int32', 'uint8', 'bool']
feature_cols = [c for c in df.columns
                if c not in exclude
                and str(df[c].dtype) in numeric_types
                and df[c].nunique() > 1]

df[feature_cols] = (df[feature_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(df[feature_cols].median()))

target_col = 'target_24h' if 'target_24h' in df.columns else None
assert target_col, "target_24h column missing from CSV!"
print(f'  Total features: {len(feature_cols)}')

# Patient split
patient_ids = df['patientunitstayid'].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(patient_ids)
n          = len(patient_ids)
train_ids  = patient_ids[:int(n * 0.70)]
val_ids    = patient_ids[int(n * 0.70):int(n * 0.85)]
test_ids   = patient_ids[int(n * 0.85):]
print(f'  Patients → Train:{len(train_ids)} | Val:{len(val_ids)} | Test:{len(test_ids)}')


def create_sequences(df, pid_list, feat_cols, tgt_col, seq_len=24):
    Xs, ys = [], []
    for pid in pid_list:
        p     = df[df['patientunitstayid'] == pid].sort_values('hour')
        feats = p[feat_cols].values
        tgts  = p[tgt_col].values
        if len(feats) >= seq_len:
            for i in range(len(feats) - seq_len + 1):
                Xs.append(feats[i:i + seq_len])
                ys.append(tgts[i + seq_len - 1])
        elif len(feats) > 0:
            pad = np.zeros((seq_len, feats.shape[1]))
            pad[-len(feats):] = feats
            Xs.append(pad)
            ys.append(tgts[-1])
    if not Xs:
        return (np.empty((0, seq_len, len(feat_cols)), dtype=np.float32),
                np.empty(0, dtype=np.int32))
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


print('  Building sequences (may take 60–90 seconds)...')
t0 = time.time()
Xe_train, ye_train = create_sequences(df, train_ids, feature_cols, target_col)
Xe_val,   ye_val   = create_sequences(df, val_ids,   feature_cols, target_col)
Xe_test,  ye_test  = create_sequences(df, test_ids,  feature_cols, target_col)
print(f'  Done in {time.time() - t0:.1f}s')
print(f'  Xe_train {Xe_train.shape} | pos {ye_train.sum()}/{len(ye_train)} ({ye_train.mean()*100:.1f}%)')
print(f'  Xe_val   {Xe_val.shape}   | pos {ye_val.sum()}/{len(ye_val)} ({ye_val.mean()*100:.1f}%)')
print(f'  Xe_test  {Xe_test.shape}  | pos {ye_test.sum()}/{len(ye_test)} ({ye_test.mean()*100:.1f}%)')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — DataLoaders with WeightedRandomSampler
# ════════════════════════════════════════════════════════════════════════════════
print('\n[2/6] Creating DataLoaders with WeightedRandomSampler...')

X_tr_t  = torch.tensor(Xe_train, dtype=torch.float32)
y_tr_t  = torch.tensor(ye_train, dtype=torch.float32)
X_val_t = torch.tensor(Xe_val,   dtype=torch.float32)
y_val_t = torch.tensor(ye_val,   dtype=torch.float32)
X_te_t  = torch.tensor(Xe_test,  dtype=torch.float32)
y_te_t  = torch.tensor(ye_test,  dtype=torch.float32)

# ─── WeightedRandomSampler: force each batch to have ~50% positives ───────────
n_pos      = int(ye_train.sum())
n_neg      = len(ye_train) - n_pos
# Assign higher weight to minority class samples
class_weights   = [1.0 / n_neg, 1.0 / n_pos]
sample_weights  = [class_weights[int(label)] for label in ye_train]
sampler         = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights, dtype=torch.float64),
    num_samples = len(sample_weights),
    replacement = True
)
print(f'  WeightedRandomSampler: {n_neg} neg / {n_pos} pos → balanced batches')

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Focal Loss + Model Architecture
# ════════════════════════════════════════════════════════════════════════════════
print('\n[3/6] Building Conv1D + BiLSTM + Attention model...')


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    For imbalanced datasets:
    - gamma > 0 reduces relative loss for well-classified examples,
      focusing training on hard, misclassified examples (typically the minority class).
    - alpha acts as a class-specific prior weight.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs      = torch.sigmoid(logits)
        # p_t: probability of the true class
        p_t        = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t: class-specific weight
        alpha_t    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention over temporal dimension of LSTM output."""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        B, T, H = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scale   = math.sqrt(self.head_dim)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = torch.softmax(scores, dim=-1)
        ctx     = torch.matmul(weights, V)               # (B, heads, T, head_dim)
        ctx     = ctx.transpose(1, 2).contiguous().view(B, T, H)
        ctx     = self.out_proj(ctx)
        # Global average pooling over time
        return ctx.mean(dim=1)                            # (B, H)


class ConvBiLSTMAttention(nn.Module):
    """
    Conv1D  →  BiLSTM  →  Multi-Head Attention  →  Classifier

    Conv1D  : local pattern detector (short-term vital-sign spikes).
    BiLSTM  : long-range temporal dependency modeller.
    Attention: learns which time steps matter most for the prediction.
    """
    def __init__(self, input_size, hidden_size=192, num_layers=2,
                 dropout=0.4, num_heads=4, conv_channels=64, conv_kernel=3):
        super().__init__()

        # ── Conv1D front-end ────────────────────────────────────────────────────
        # Conv1d expects (batch, channels, seq_len); we permute input before it.
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )

        # ── BiLSTM ─────────────────────────────────────────────────────────────
        self.bilstm = nn.LSTM(conv_channels, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        lstm_out_size = hidden_size * 2   # bidirectional doubles width

        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # ── Multi-Head Attention ────────────────────────────────────────────────
        self.attention = MultiHeadSelfAttention(lstm_out_size, num_heads=num_heads)

        # ── Classifier head ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, F)  →  conv needs (B, F, T)
        xc  = self.conv(x.permute(0, 2, 1))    # (B, conv_channels, T)
        xc  = xc.permute(0, 2, 1)              # back to (B, T, conv_channels)

        out, _ = self.bilstm(xc)               # (B, T, H*2)
        out    = self.layer_norm(out)

        ctx    = self.attention(out)            # (B, H*2)
        return self.classifier(ctx).squeeze(-1)


INPUT_SIZE = Xe_train.shape[2]
model      = ConvBiLSTMAttention(
    input_size    = INPUT_SIZE,
    hidden_size   = HIDDEN_SIZE,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    num_heads     = 4,
    conv_channels = 64,
    conv_kernel   = 3,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Input features   : {INPUT_SIZE}')
print(f'  Trainable params : {total_params:,}')
print(model)

# ─── Loss, Optimizer, Scheduler ────────────────────────────────────────────────
criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# CosineAnnealingWarmRestarts: periodically resets LR → escapes local minima
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-6
)
print(f'\n  Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})')
print(f'  Optimizer: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})')
print(f'  Scheduler: CosineAnnealingWarmRestarts (T0={COSINE_T0}, Tmult={COSINE_T_MULT})')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training Loop
# ════════════════════════════════════════════════════════════════════════════════
print('\n[4/6] Training ConvBiLSTM + Attention...')
print('=' * 65)

best_val_auc      = 0.0
epochs_no_improve = 0
history           = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────────
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

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    val_loss, all_probs, all_tgts = 0.0, [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds   = model(Xb)
            val_loss += criterion(preds, yb).item()
            all_probs.extend(torch.sigmoid(preds).cpu().numpy())
            all_tgts.extend(yb.cpu().numpy())

    avg_val  = val_loss / len(val_loader)
    val_auc  = roc_auc_score(all_tgts, all_probs)
    curr_lr  = optimizer.param_groups[0]['lr']

    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    history['val_auc'].append(val_auc)
    history['lr'].append(curr_lr)

    # Cosine scheduler steps every epoch
    scheduler.step()

    print(f'Epoch {epoch+1:3d}/{EPOCHS} | '
          f'Train: {avg_train:.4f} | Val: {avg_val:.4f} | '
          f'AUC: {val_auc:.4f} | LR: {curr_lr:.6f}')

    if val_auc > best_val_auc:
        best_val_auc      = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  ✓ Best model saved (AUC={best_val_auc:.4f})')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1} '
                  f'(no improvement for {PATIENCE} epochs)')
            break

print(f'\nBest Validation AUC: {best_val_auc:.4f}')

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

# Optimal threshold from PR curve (maximise F1)
precision, recall, thresholds = precision_recall_curve(test_tgts, test_probs)
f1s        = 2 * precision * recall / (precision + recall + 1e-8)
opt_thresh = thresholds[np.argmax(f1s)]
binary_preds = (test_probs >= opt_thresh).astype(int)

print(f'\n{"=" * 60}')
print(f'  FINAL TEST SET RESULTS — ConvBiLSTM + Attention')
print(f'{"=" * 60}')
print(f'  Baseline LSTM (NB5)   : 0.6324  ROC-AUC')
print(f'  Tuned RF (NB6)        : 0.6860  ROC-AUC')
print(f'  This model (Test AUC) : {test_auc:.4f}  ROC-AUC')
print(f'  Improvement vs base   : +{test_auc - 0.6324:.4f}')
print(f'  Optimal threshold     : {opt_thresh:.3f}')
print(f'\n{classification_report(test_tgts, binary_preds, target_names=["Stable", "Deteriorating"])}')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save Plots
# ════════════════════════════════════════════════════════════════════════════════
print('[6/6] Saving plots...')
plt.style.use('dark_background')

# ── Training curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor('#0d0d1a')
for ax in axes:
    ax.set_facecolor('#12122a')

ep = range(1, len(history['val_auc']) + 1)

axes[0].plot(ep, history['train_loss'], '#FF6B6B', lw=2, label='Train Loss')
axes[0].plot(ep, history['val_loss'],   '#4ECDC4', lw=2, label='Val Loss')
axes[0].set_title('Focal Loss: Train vs Val', color='white', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', color='white')
axes[0].set_ylabel('Loss',  color='white')
axes[0].legend(facecolor='#1a1a3e', edgecolor='#555')
axes[0].tick_params(colors='white')
axes[0].grid(True, alpha=0.15)

axes[1].plot(ep, history['val_auc'], '#FFEAA7', lw=2.5, label='Val AUC')
axes[1].axhline(0.6324, color='#FF6B6B', ls='--', lw=1.2, label='Baseline LSTM (0.6324)')
axes[1].axhline(0.6860, color='#A29BFE', ls='--', lw=1.2, label='Tuned RF (0.6860)')
axes[1].axhline(0.8500, color='#00FF7F', ls='--', lw=1.5, alpha=0.9, label='Target (0.85)')
axes[1].set_title('ConvBiLSTM Validation AUC', color='white', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', color='white')
axes[1].set_ylabel('ROC-AUC', color='white')
axes[1].legend(facecolor='#1a1a3e', edgecolor='#555', fontsize=8)
axes[1].tick_params(colors='white')
axes[1].grid(True, alpha=0.15)
axes[1].set_ylim(0.5, 1.0)

axes[2].plot(ep, history['lr'], '#FD79A8', lw=2, label='Learning Rate')
axes[2].set_title('Cosine Warm Restart LR Schedule', color='white', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Epoch', color='white')
axes[2].set_ylabel('LR',    color='white')
axes[2].tick_params(colors='white')
axes[2].grid(True, alpha=0.15)
axes[2].legend(facecolor='#1a1a3e', edgecolor='#555')

plt.tight_layout()
curve_path = os.path.join(REPORTS_DIR, 'convbilstm_training_curves.png')
plt.savefig(curve_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f'  Saved: {curve_path}')

# ── ROC Curve ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(test_tgts, test_probs)
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#12122a')
ax.plot([0, 1], [0, 1], '--', color='#555', lw=1,    label='Random (AUC=0.50)')
ax.plot(fpr, tpr, '#00CEC9', lw=2.5,
        label=f'ConvBiLSTM+Attention (AUC={test_auc:.4f})')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate',  color='white')
ax.set_title(f'ROC Curve — ConvBiLSTM + Attention\n(Test Set | AUC = {test_auc:.4f})',
             color='white', fontsize=12, fontweight='bold')
ax.legend(facecolor='#1a1a3e', edgecolor='#555')
ax.tick_params(colors='white')
ax.grid(True, alpha=0.15)
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, 'convbilstm_roc_curve.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f'  Saved: {roc_path}')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"#" * 60}')
print(f'  TRAINING COMPLETE')
print(f'{"#" * 60}')
print(f'  Architecture       : Conv1D + BiLSTM + Multi-Head Attention')
print(f'  Loss               : Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})')
print(f'  Sampling           : WeightedRandomSampler (balanced batches)')
print(f'  Scheduler          : CosineAnnealingWarmRestarts')
print(f'  Baseline LSTM AUC  : 0.6324')
print(f'  BiLSTM (NB6) AUC   : ~0.68')
print(f'  This Model AUC     : {test_auc:.4f}')
print(f'  Target             : 0.8500')
print(f'  {"ACHIEVED!" if test_auc >= 0.85 else "Continue tuning if not yet reached"}')
print(f'  Model saved to     : {MODEL_PATH}')
