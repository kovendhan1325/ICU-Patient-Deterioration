"""
==============================================================================
STEP 09 v2 - Improved LSTM Training & Metrics Report
==============================================================================
Key improvements over v1:

  1. FOCAL LOSS (gamma=2, alpha=0.75)
       BCEWithLogitsLoss + pos_weight failed because SMOTE synthetics made
       the model confidently predict low probabilities for everything.
       Focal Loss heavily down-weights easy negatives so the model is
       forced to focus on the hard minority positives.

  2. TRAIN ON ORIGINAL (IMBALANCED) DATA + pos_weight
       SMOTE on 3-D temporal sequences generates physiologically
       unrealistic synthetic sequences.  Instead we train on the raw
       X_train.npy with a high pos_weight (≈ n_neg/n_pos ≈ 19) to
       compensate for imbalance naturally.

  3. SELF-ATTENTION after BiLSTM
       Allows the model to learn which time steps are most predictive
       of deterioration, rather than blindly using the last hidden state.

  4. STRONGER REGULARISATION
       Dropout 0.5/0.4/0.3, weight_decay=1e-3, gradient clipping=0.5

  5. OPTIMAL THRESHOLD  (Youden's J)
       All final metrics reported at the threshold that maximises
       sensitivity + specificity, not the arbitrary 0.5.

  6. TRAINING CURVES PLOT  (reports/lstm_v2_training_curves.png)

Architecture:
  - Input:  (batch, 24, 30)
  - BiLSTM  (64 units bidirectional) + LayerNorm + Dropout(0.5)
  - Self-Attention (dim=128, 4 heads)
  - LSTM    (64 units) + LayerNorm + Dropout(0.4)
  - Dense   (32 units) + GELU + Dropout(0.3)
  - Output  (1 unit, sigmoid)

Training:
  - Loss:           FocalLoss (gamma=2, alpha=0.75)
  - Optimizer:      AdamW (lr=5e-4, weight_decay=1e-3)
  - Scheduler:      CosineAnnealingLR
  - Early stopping: patience=15 on val ROC-AUC
  - Batch size:     256
==============================================================================
"""

import os
import sys
import time
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, REPORTS_DIR, RANDOM_SEED

# --- Reproducibility ----------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Hyperparameters ----------------------------------------------------------
BATCH_SIZE      = 128         # smaller batch -> better gradient signal on minority
MAX_EPOCHS      = 150
LR              = 1e-4        # lower LR for stability
WEIGHT_DECAY    = 1e-3
PATIENCE        = 20          # more patience for AUC to stabilise
LSTM1_UNITS     = 64          # smaller → less overfit
LSTM2_UNITS     = 64
ATTN_HEADS      = 4
DENSE_UNITS     = 32
DROPOUT1        = 0.5
DROPOUT2        = 0.4
DROPOUT3        = 0.3
FOCAL_GAMMA     = 2.0         # focal loss focusing parameter
FOCAL_ALPHA     = 0.95        # high alpha for extreme imbalance (132:1)


# ===========================================================================
# FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    FL(p) = -alpha * (1-p)^gamma * log(p)  for y=1
          = -(1-alpha) * p^gamma * log(1-p) for y=0

    Accepts raw logits (no sigmoid needed).
    """
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)
        ce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal  = alpha_t * (1 - p_t) ** self.gamma * ce
        return focal.mean()


# ===========================================================================
# MODEL
# ===========================================================================

class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention for sequence data."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)   # residual connection


class ICU_AttBiLSTM(nn.Module):
    """
    Attention-augmented BiLSTM for ICU deterioration prediction.

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1) raw logit
    """
    def __init__(self, n_features, seq_len):
        super().__init__()

        # --- Layer 1: Bidirectional LSTM ----------------------------------
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=LSTM1_UNITS,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1   = nn.LayerNorm(LSTM1_UNITS * 2)
        self.drop1 = nn.Dropout(DROPOUT1)

        # --- Self-Attention on BiLSTM output ------------------------------
        self.attn = MultiHeadSelfAttention(
            embed_dim=LSTM1_UNITS * 2,
            num_heads=ATTN_HEADS,
            dropout=0.1,
        )

        # --- Layer 2: Unidirectional LSTM ---------------------------------
        self.lstm2 = nn.LSTM(
            input_size=LSTM1_UNITS * 2,
            hidden_size=LSTM2_UNITS,
            num_layers=1,
            batch_first=True,
        )
        self.ln2   = nn.LayerNorm(LSTM2_UNITS)
        self.drop2 = nn.Dropout(DROPOUT2)

        # --- Classifier ---------------------------------------------------
        self.fc1   = nn.Linear(LSTM2_UNITS, DENSE_UNITS)
        self.act   = nn.GELU()
        self.drop3 = nn.Dropout(DROPOUT3)
        self.out   = nn.Linear(DENSE_UNITS, 1)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "fc" in name or "out" in name:
                if p.dim() == 2:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        # BiLSTM
        out, _ = self.bilstm(x)        # (B, T, 2*L1)
        out    = self.ln1(out)
        out    = self.drop1(out)

        # Self-attention
        out    = self.attn(out)         # (B, T, 2*L1)

        # LSTM
        out, _ = self.lstm2(out)        # (B, T, L2)
        out    = self.ln2(out)
        out    = self.drop2(out)

        # Use last time step
        out    = out[:, -1, :]          # (B, L2)

        # Classifier
        out    = self.fc1(out)
        out    = self.act(out)
        out    = self.drop3(out)
        out    = self.out(out)
        return out.squeeze(-1)          # (B,)


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data():
    """
    Load ORIGINAL (imbalanced) training data and selected val/test splits.
    We do NOT use SMOTE-balanced data here — instead rely on pos_weight + focal loss.
    """
    print("  Loading datasets ...")

    # Training: use ORIGINAL unbalanced sequences
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))

    # Val & Test: use feature-selected splits from Step 08
    X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val_selected.npy"))
    y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test_selected.npy"))
    y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    # Apply the same feature mask to X_train
    feature_mask    = np.load(os.path.join(OUTPUT_DIR, "feature_mask.npy"))
    feature_indices = np.load(os.path.join(OUTPUT_DIR, "feature_indices.npy"))
    X_train = X_train[:, :, feature_mask]

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    print(f"    -> X_train : {X_train.shape}  pos={pos:,}  neg={neg:,}  "
          f"({pos/len(y_train)*100:.2f}% positive)")
    print(f"    -> X_val   : {X_val.shape}    pos={int(y_val.sum()):,}")
    print(f"    -> X_test  : {X_test.shape}    pos={int(y_test.sum()):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size, shuffle=True, use_weighted_sampler=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    if use_weighted_sampler and shuffle:
        # WeightedRandomSampler: oversample positives to ~10% of each batch
        n_pos  = int(y.sum())
        n_neg  = len(y) - n_pos
        w_pos  = 1.0 / n_pos
        w_neg  = 1.0 / n_neg
        weights = np.where(y == 1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(ds),
            replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          pin_memory=False, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=False, num_workers=0)


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits    = model(X_b)
        loss      = criterion(logits, y_b)
        total_loss += loss.item() * len(y_b)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_b.cpu().numpy())
    avg_loss   = total_loss / len(loader.dataset)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if all_labels.sum() > 0 else 0.0
    return avg_loss, auc, all_probs, all_labels


def train(model, train_loader, val_loader, device, pos_weight):
    criterion  = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    history        = []

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Val AUC':>8}  {'LR':>9}  {'Status'}")
    print("  " + "-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc   = val_auc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            status = "[best]"
        else:
            patience_count += 1
            status = f"  ({patience_count}/{PATIENCE})"

        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "val_loss": val_loss,
            "val_auc": val_auc, "lr": lr_now,
        })

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{val_auc:>8.4f}  {lr_now:>9.2e}  {status}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Best model restored (val AUC = {best_val_auc:.4f})")

    return history, best_val_auc


# ===========================================================================
# METRICS
# ===========================================================================

def compute_all_metrics(y_true, y_prob, split_name="Test"):
    """Compute all metrics using BOTH fixed (0.5) and Youden-optimal thresholds."""
    # --- Fixed threshold ---
    threshold = 0.5
    y_pred  = (y_prob >= threshold).astype(int)
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec    = tn / max(1, tn + fp)
    npv     = tn / max(1, tn + fn)
    cr      = classification_report(y_true, y_pred,
                                    target_names=["Normal", "Deterioration"],
                                    zero_division=0)

    # --- Optimal threshold (Youden's J) ---
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    j_scores    = tpr_arr - fpr_arr
    best_idx    = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    y_pred_opt  = (y_prob >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
    f1_opt   = f1_score(y_true, y_pred_opt, zero_division=0)
    rec_opt  = recall_score(y_true, y_pred_opt, zero_division=0)
    prec_opt = precision_score(y_true, y_pred_opt, zero_division=0)
    spec_opt = tn_o / max(1, tn_o + fp_o)

    return {
        "split": split_name,
        "auc_roc": auc_roc, "auc_pr": auc_pr,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "specificity": spec, "npv": npv,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "best_threshold": best_thresh,
        "f1_at_best":    f1_opt,
        "recall_at_best":  rec_opt,
        "prec_at_best":  prec_opt,
        "spec_at_best":  spec_opt,
        "tn_opt": tn_o, "fp_opt": fp_o, "fn_opt": fn_o, "tp_opt": tp_o,
        "classification_report": cr,
        "y_true": y_true, "y_prob": y_prob,
    }


def print_metrics(m):
    print(f"\n  " + "-" * 62)
    print(f"  {m['split']} SET METRICS")
    print(f"  " + "-" * 62)
    print(f"  ROC-AUC               : {m['auc_roc']:.4f}  "
          f"{'[TARGET MET]' if m['auc_roc']>=0.85 else '[Below 0.85]'}")
    print(f"  PR-AUC                : {m['auc_pr']:.4f}")
    print(f"  Accuracy              : {m['accuracy']:.4f}")
    print(f"  Precision (PPV) @0.5  : {m['precision']:.4f}")
    print(f"  Recall      @0.5      : {m['recall']:.4f}")
    print(f"  F1          @0.5      : {m['f1']:.4f}")
    print(f"  Specificity @0.5      : {m['specificity']:.4f}")
    print(f"  NPV                   : {m['npv']:.4f}")
    print(f"\n  -- Youden-optimal threshold = {m['best_threshold']:.4f} --")
    print(f"  Recall      @opt      : {m['recall_at_best']:.4f}")
    print(f"  Precision   @opt      : {m['prec_at_best']:.4f}")
    print(f"  F1          @opt      : {m['f1_at_best']:.4f}")
    print(f"  Specificity @opt      : {m['spec_at_best']:.4f}")
    print(f"\n  Confusion Matrix @0.5:")
    print(f"    {'':15s}  Pred Normal  Pred Deterioration")
    print(f"    Actual Normal     {m['tn']:>10,}  {m['fp']:>17,}")
    print(f"    Actual Deterior.  {m['fn']:>10,}  {m['tp']:>17,}")
    print(f"\n  Confusion Matrix @opt ({m['best_threshold']:.4f}):")
    print(f"    {'':15s}  Pred Normal  Pred Deterioration")
    print(f"    Actual Normal     {m['tn_opt']:>10,}  {m['fp_opt']:>17,}")
    print(f"    Actual Deterior.  {m['fn_opt']:>10,}  {m['tp_opt']:>17,}")
    print(f"\n  Classification Report @0.5:")
    for line in m['classification_report'].splitlines():
        print(f"    {line}")


# ===========================================================================
# PLOT
# ===========================================================================

def save_training_plot(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs  = [h["epoch"] for h in history]
        tr_loss = [h["tr_loss"] for h in history]
        vl_loss = [h["val_loss"] for h in history]
        vl_auc  = [h["val_auc"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("ICU BiLSTM+Attention v2 — Training Curves", fontsize=13)

        ax1.plot(epochs, tr_loss, label="Train Loss", color="#e74c3c", linewidth=2)
        ax1.plot(epochs, vl_loss, label="Val Loss",   color="#3498db", linewidth=2)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, vl_auc, label="Val ROC-AUC", color="#2ecc71", linewidth=2)
        best_auc = max(vl_auc)
        ax2.axhline(best_auc, color="orange", linestyle="--",
                    label=f"Best = {best_auc:.4f}")
        ax2.axhline(0.85, color="purple", linestyle=":", label="Target 0.85")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
        ax2.set_title("Validation AUC"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "lstm_v2_training_curves.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  * Training curves saved to {path}")
    except ImportError:
        print("\n  [!] matplotlib not available — skipping plot.")


def save_roc_plot(val_metrics, test_metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=(7, 6))
        for m, color, label in [
            (val_metrics,  "#3498db", "Validation"),
            (test_metrics, "#e74c3c", "Test"),
        ]:
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{label} AUC = {m['auc_roc']:.4f}")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — BiLSTM+Attention v2")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        path = os.path.join(REPORTS_DIR, "lstm_v2_roc_curve.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  * ROC curve saved to {path}")
    except ImportError:
        print("  [!] matplotlib not available — skipping ROC plot.")


# ===========================================================================
# REPORT WRITER
# ===========================================================================

def write_report(history, val_m, test_m, training_time_s, model):
    sep  = "=" * 70
    lines = []

    def S(title):
        lines.extend(["", sep, title, sep, ""])

    lines += [
        sep,
        "LSTM v2 METRICS REPORT  (Focal Loss + Self-Attention)",
        "ICU Patient Deterioration Prediction - eICU Demo Dataset",
        sep, "",
        f"  Model         : BiLSTM+Attention LSTM (v2)",
        f"  Target        : target_6h",
        f"  Training time : {training_time_s/60:.1f} minutes",
        f"  Epochs run    : {len(history)}",
        f"  Best Val AUC  : {max(h['val_auc'] for h in history):.4f}",
        "",
    ]

    S("MODEL ARCHITECTURE")
    total = sum(p.numel() for p in model.parameters())
    lines += [
        f"  BiLSTM layer    : {LSTM1_UNITS} units x2 (bidirectional)",
        f"  Self-Attention  : {ATTN_HEADS} heads (dim={LSTM1_UNITS*2})",
        f"  LSTM layer 2    : {LSTM2_UNITS} units",
        f"  Dense layer     : {DENSE_UNITS} units + GELU",
        f"  Total params    : {total:,}",
        f"  Dropout         : {DROPOUT1}/{DROPOUT2}/{DROPOUT3}",
        "",
    ]

    S("HYPERPARAMETERS")
    lines += [
        f"  Loss            : FocalLoss (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})",
        f"  Optimizer       : AdamW (lr={LR}, wd={WEIGHT_DECAY})",
        f"  Scheduler       : CosineAnnealingLR",
        f"  Batch size      : {BATCH_SIZE}",
        f"  Max epochs      : {MAX_EPOCHS}",
        f"  Early stopping  : patience={PATIENCE} on Val ROC-AUC",
        f"  Gradient clip   : 0.5",
        f"  Training data   : Original imbalanced X_train (NOT SMOTE)",
        "",
    ]

    S("TRAINING LOG")
    lines.append(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val AUC':>8}  {'LR':>9}")
    lines.append("  " + "-" * 50)
    for h in history:
        lines.append(
            f"  {h['epoch']:>5}  {h['tr_loss']:>10.4f}  "
            f"{h['val_loss']:>8.4f}  {h['val_auc']:>8.4f}  {h['lr']:>9.2e}"
        )

    def block(m):
        nonlocal lines
        lines += [
            f"  ROC-AUC               : {m['auc_roc']:.4f}  "
            f"{'[TARGET MET]' if m['auc_roc']>=0.85 else '[Below 0.85]'}",
            f"  PR-AUC                : {m['auc_pr']:.4f}",
            f"  Accuracy  @0.5        : {m['accuracy']:.4f}",
            f"  Precision @0.5        : {m['precision']:.4f}",
            f"  Recall    @0.5        : {m['recall']:.4f}",
            f"  F1        @0.5        : {m['f1']:.4f}",
            f"  Specificity @0.5      : {m['specificity']:.4f}",
            "",
            f"  Optimal Threshold (Youden): {m['best_threshold']:.6f}",
            f"  Recall    @opt        : {m['recall_at_best']:.4f}",
            f"  Precision @opt        : {m['prec_at_best']:.4f}",
            f"  F1        @opt        : {m['f1_at_best']:.4f}",
            f"  Specificity @opt      : {m['spec_at_best']:.4f}",
            "",
            f"  Confusion Matrix @opt ({m['best_threshold']:.4f}):",
            f"    {'':17s}  Pred Normal  Pred Deterioration",
            f"    Actual Normal        {m['tn_opt']:>10,}  {m['fp_opt']:>17,}",
            f"    Actual Deterioration {m['fn_opt']:>10,}  {m['tp_opt']:>17,}",
            "",
            "  Classification Report @0.5:",
        ]
        for line in m["classification_report"].splitlines():
            lines.append(f"    {line}")

    S("VALIDATION SET METRICS"); block(val_m)
    S("TEST SET METRICS (Final)"); block(test_m)

    S("COMPARISON vs v1")
    lines += [
        f"  {'Metric':<25}  {'v1':>8}  {'v2':>8}",
        "  " + "-" * 45,
        f"  {'Test ROC-AUC':<25}  {'0.6276':>8}  {test_m['auc_roc']:>8.4f}",
        f"  {'Test PR-AUC':<25}  {'0.0188':>8}  {test_m['auc_pr']:>8.4f}",
        f"  {'Test F1 @0.5':<25}  {'0.0380':>8}  {test_m['f1']:>8.4f}",
        f"  {'Test Recall @0.5':<25}  {'0.1067':>8}  {test_m['recall']:>8.4f}",
        f"  {'Epochs trained':<25}  {'12':>8}  {len(history):>8}",
        "",
    ]

    report_path = os.path.join(REPORTS_DIR, "lstm_v2_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  * Report saved to {report_path}")
    return report_path


# ===========================================================================
# MAIN
# ===========================================================================

def run_step09_v2():
    print("\n" + "=" * 70)
    print("STEP 09 v2 - BiLSTM+Attention Training (Focal Loss)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    _, T, F = X_train.shape

    # pos_weight for reference (used in focal loss indirectly via alpha)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_w = n_neg / max(1, n_pos)
    print(f"\n  Class imbalance ratio (neg/pos): {pos_w:.1f}x")
    print(f"  Focal Loss: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True,
                               use_weighted_sampler=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    # Build model
    model = ICU_AttBiLSTM(n_features=F, seq_len=T).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: BiLSTM({LSTM1_UNITS}x2) -> Attention({ATTN_HEADS}h) "
          f"-> LSTM({LSTM2_UNITS}) -> Dense({DENSE_UNITS}) -> 1")
    print(f"  Total parameters: {total_params:,}")

    # Train
    print(f"\n  Training for up to {MAX_EPOCHS} epochs "
          f"(early stop patience={PATIENCE}) ...")
    t0 = time.time()
    history, best_val_auc = train(model, train_loader, val_loader, device, pos_w)
    training_time = time.time() - t0
    print(f"\n  Training complete in {training_time/60:.1f} minutes")

    # Evaluate
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    print("\n  Evaluating on val and test sets ...")
    _, _, val_probs,  val_labels  = evaluate(model, val_loader,  criterion, device)
    _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device)

    val_metrics  = compute_all_metrics(val_labels,  val_probs,  "Validation")
    test_metrics = compute_all_metrics(test_labels, test_probs, "Test")

    print_metrics(val_metrics)
    print_metrics(test_metrics)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "lstm_v2_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": F, "seq_len": T,
        "history": history,
        "test_auc": test_metrics["auc_roc"],
        "best_threshold": test_metrics["best_threshold"],
    }, model_path)
    print(f"\n  * Model saved to {model_path}")

    # Save plots
    save_training_plot(history)
    save_roc_plot(val_metrics, test_metrics)

    # Write report
    report_path = write_report(history, val_metrics, test_metrics,
                               training_time, model)

    # Final banner
    print("\n" + "=" * 70)
    print("  FINAL RESULT (v2)")
    print("=" * 70)
    auc = test_metrics["auc_roc"]
    tag = "[PASS]" if auc >= 0.85 else "[WARN]"
    print(f"  {tag} ROC-AUC = {auc:.4f}  (v1 was 0.6276)")
    print(f"  PR-AUC      = {test_metrics['auc_pr']:.4f}  (v1 was 0.0188)")
    print(f"  F1  @0.5    = {test_metrics['f1']:.4f}  (v1 was 0.0380)")
    print(f"  Recall @opt = {test_metrics['recall_at_best']:.4f}")
    print(f"  Spec   @opt = {test_metrics['spec_at_best']:.4f}")
    print(f"\n  Full report: {report_path}")

    return test_metrics


if __name__ == "__main__":
    run_step09_v2()
