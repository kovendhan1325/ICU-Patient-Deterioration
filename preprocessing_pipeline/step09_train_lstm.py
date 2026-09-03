"""
==============================================================================
STEP 09 - LSTM Training & Metrics Report
==============================================================================
Trains a Bidirectional LSTM model on the balanced dataset from Step 08 and
generates a comprehensive metrics report.

Architecture:
  - Input:  (batch, 24, 30)   [24 time steps, 30 selected features]
  - BiLSTM  (128 units) + Dropout(0.4)
  - LSTM    (64 units)  + Dropout(0.3)
  - Dense   (32)        + ReLU + Dropout(0.2)
  - Output  (1)         + Sigmoid

Training:
  - Loss:           BCEWithLogitsLoss (with pos_weight for residual imbalance)
  - Optimizer:      Adam (lr=1e-3, weight_decay=1e-4)
  - Scheduler:      ReduceLROnPlateau
  - Early Stopping: patience=10 on val ROC-AUC
  - Batch size:     512

Metrics Report (reports/lstm_metrics_report.txt):
  - ROC-AUC, PR-AUC
  - Accuracy, Precision, Recall, F1, Specificity
  - Confusion Matrix
  - Classification Report
  - Per-epoch training log
==============================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
BATCH_SIZE      = 512
MAX_EPOCHS      = 80
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 10        # early stopping patience (on val ROC-AUC)
LSTM1_UNITS     = 128
LSTM2_UNITS     = 64
DENSE_UNITS     = 32
DROPOUT1        = 0.4
DROPOUT2        = 0.3
DROPOUT3        = 0.2
THRESHOLD       = 0.5       # decision threshold for binary metrics


# ===========================================================================
# MODEL
# ===========================================================================

class ICU_BiLSTM(nn.Module):
    """
    Bidirectional LSTM for ICU patient deterioration prediction.

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1)  -- raw logit (apply sigmoid for probability)
    """
    def __init__(self, n_features, seq_len,
                 lstm1=LSTM1_UNITS, lstm2=LSTM2_UNITS, dense=DENSE_UNITS,
                 drop1=DROPOUT1, drop2=DROPOUT2, drop3=DROPOUT3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(drop1)

        self.lstm2 = nn.LSTM(
            input_size=lstm1 * 2,   # *2 because bidirectional
            hidden_size=lstm2,
            num_layers=1,
            batch_first=True,
        )
        self.drop2 = nn.Dropout(drop2)

        self.fc1  = nn.Linear(lstm2, dense)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(drop3)
        self.out  = nn.Linear(dense, 1)

        # Layer norm for stability
        self.ln1 = nn.LayerNorm(lstm1 * 2)
        self.ln2 = nn.LayerNorm(lstm2)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.bilstm(x)          # (B, T, 2*L1)
        out = self.ln1(out)
        out = self.drop1(out)

        out, _ = self.lstm2(out)          # (B, T, L2)
        out = self.ln2(out)
        out = self.drop2(out)

        out = out[:, -1, :]               # take last timestep: (B, L2)

        out = self.fc1(out)               # (B, dense)
        out = self.relu(out)
        out = self.drop3(out)
        out = self.out(out)               # (B, 1)
        return out.squeeze(-1)            # (B,)


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data():
    """Load balanced training data and selected val/test splits."""
    print("  Loading balanced datasets ...")
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_balanced.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train_balanced.npy"))
    X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val_selected.npy"))
    y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test_selected.npy"))
    y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    print(f"    -> X_train : {X_train.shape}  positives: {y_train.sum():,}/{len(y_train):,} ({y_train.mean()*100:.1f}%)")
    print(f"    -> X_val   : {X_val.shape}")
    print(f"    -> X_test  : {X_test.shape}")
    print(f"  Verified: Step 08 ran successfully [OK]")
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size, shuffle=True):
    """Wrap numpy arrays in a DataLoader."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


# ===========================================================================
# TRAINING
# ===========================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if all_labels.sum() > 0 else 0.0
    return avg_loss, auc, all_probs, all_labels


def train(model, train_loader, val_loader, device):
    """Full training loop with early stopping on val ROC-AUC."""
    # pos_weight: ratio of negatives to positives in BALANCED train
    n_pos = int(train_loader.dataset.tensors[1].sum().item())
    n_neg = len(train_loader.dataset) - n_pos
    pos_w = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    history        = []

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val AUC':>8}  {'LR':>9}  {'Status'}")
    print("  " + "-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss          = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
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

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {val_loss:>8.4f}  {val_auc:>8.4f}  {lr_now:>9.2e}  {status}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Best model restored (val AUC = {best_val_auc:.4f})")

    return history, best_val_auc


# ===========================================================================
# METRICS
# ===========================================================================

def compute_all_metrics(y_true, y_prob, threshold=THRESHOLD, split_name="Test"):
    """Compute and return a dict of all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec    = tn / max(1, tn + fp)            # Specificity
    npv     = tn / max(1, tn + fn)            # Negative Predictive Value
    ppv     = tp / max(1, tp + fp)            # Positive Predictive Value (= precision)

    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    y_pred_opt  = (y_prob >= best_thresh).astype(int)
    f1_opt      = f1_score(y_true, y_pred_opt, zero_division=0)
    rec_opt     = recall_score(y_true, y_pred_opt, zero_division=0)
    prec_opt    = precision_score(y_true, y_pred_opt, zero_division=0)

    cr = classification_report(y_true, y_pred, target_names=["Normal", "Deterioration"], zero_division=0)

    return {
        "split":       split_name,
        "auc_roc":     auc_roc,
        "auc_pr":      auc_pr,
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "specificity": spec,
        "npv":         npv,
        "ppv":         ppv,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "best_threshold": best_thresh,
        "f1_at_best_thresh":   f1_opt,
        "recall_at_best_thresh": rec_opt,
        "prec_at_best_thresh": prec_opt,
        "classification_report": cr,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def print_metrics(m):
    print(f"\n  " + "-" * 60)
    print(f"  {m['split']} SET METRICS")
    print(f"  " + "-" * 60)
    print(f"  ROC-AUC               : {m['auc_roc']:.4f}")
    print(f"  PR-AUC                : {m['auc_pr']:.4f}")
    print(f"  Accuracy              : {m['accuracy']:.4f}")
    print(f"  Precision (PPV)       : {m['precision']:.4f}")
    print(f"  Recall (Sensitivity)  : {m['recall']:.4f}")
    print(f"  F1-Score              : {m['f1']:.4f}")
    print(f"  Specificity           : {m['specificity']:.4f}")
    print(f"  NPV                   : {m['npv']:.4f}")
    print(f"\n  Optimal Threshold (Youden's J) : {m['best_threshold']:.4f}")
    print(f"    Recall    @ optimal thresh   : {m['recall_at_best_thresh']:.4f}")
    print(f"    Precision @ optimal thresh   : {m['prec_at_best_thresh']:.4f}")
    print(f"    F1        @ optimal thresh   : {m['f1_at_best_thresh']:.4f}")
    print(f"\n  Confusion Matrix (threshold={THRESHOLD}):")
    print(f"    {'':15s}  Pred Normal  Pred Deterioration")
    print(f"    Actual Normal     {m['tn']:>10,}  {m['fp']:>17,}")
    print(f"    Actual Deterior.  {m['fn']:>10,}  {m['tp']:>17,}")
    print(f"\n  Classification Report:")
    for line in m['classification_report'].splitlines():
        print(f"    {line}")


# ===========================================================================
# REPORT WRITER
# ===========================================================================

def write_report(history, val_metrics, test_metrics, training_time_s, model):
    """Write a comprehensive metrics report to reports/lstm_metrics_report.txt."""
    lines = []
    sep   = "=" * 70

    def section(title):
        lines.extend(["", sep, title, sep, ""])

    lines.append(sep)
    lines.append("LSTM METRICS REPORT")
    lines.append("ICU Patient Deterioration Prediction -- eICU Demo Dataset")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Model         : Bidirectional LSTM + LSTM")
    lines.append(f"  Target        : target_6h (deterioration within 6 hours)")
    lines.append(f"  Training time : {training_time_s/60:.1f} minutes")
    lines.append(f"  Epochs run    : {len(history)}")
    lines.append(f"  Best Val AUC  : {max(h['val_auc'] for h in history):.4f}")
    lines.append("")

    # -- Architecture ----------------------------------------------
    section("MODEL ARCHITECTURE")
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"  Input shape     : (batch, 24, 30)  [24 timesteps, 30 features]")
    lines.append(f"  BiLSTM layer 1  : {LSTM1_UNITS} units x2 (bidirectional)")
    lines.append(f"  Dropout 1       : {DROPOUT1}")
    lines.append(f"  LSTM layer 2    : {LSTM2_UNITS} units")
    lines.append(f"  Dropout 2       : {DROPOUT2}")
    lines.append(f"  Dense layer     : {DENSE_UNITS} units + ReLU")
    lines.append(f"  Dropout 3       : {DROPOUT3}")
    lines.append(f"  Output          : 1 unit (Sigmoid)")
    lines.append(f"  Total params    : {total_params:,}")
    lines.append(f"  Trainable       : {trainable:,}")
    lines.append("")

    # -- Hyperparameters -------------------------------------------
    section("HYPERPARAMETERS")
    lines.append(f"  Optimizer       : Adam")
    lines.append(f"  Learning rate   : {LR}")
    lines.append(f"  Weight decay    : {WEIGHT_DECAY}")
    lines.append(f"  Batch size      : {BATCH_SIZE}")
    lines.append(f"  Max epochs      : {MAX_EPOCHS}")
    lines.append(f"  Early stopping  : patience={PATIENCE} on Val ROC-AUC")
    lines.append(f"  LR scheduler    : ReduceLROnPlateau (factor=0.5, patience=4)")
    lines.append(f"  Loss            : BCEWithLogitsLoss (pos_weight=n_neg/n_pos)")
    lines.append(f"  Decision thresh : {THRESHOLD}")
    lines.append("")

    # -- Per-epoch log ---------------------------------------------
    section("TRAINING LOG (Per Epoch)")
    lines.append(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val AUC':>8}  {'LR':>9}")
    lines.append("  " + "-" * 50)
    for h in history:
        lines.append(
            f"  {h['epoch']:>5}  {h['tr_loss']:>10.4f}  "
            f"{h['val_loss']:>8.4f}  {h['val_auc']:>8.4f}  {h['lr']:>9.2e}"
        )
    lines.append("")

    # -- Validation Metrics ----------------------------------------
    def write_metric_block(m):
        lines.append(f"  ROC-AUC               : {m['auc_roc']:.4f}  {'[TARGET MET]' if m['auc_roc']>=0.85 else '[Below 0.85]'}")
        lines.append(f"  PR-AUC                : {m['auc_pr']:.4f}")
        lines.append(f"  Accuracy              : {m['accuracy']:.4f}")
        lines.append(f"  Precision (PPV)       : {m['precision']:.4f}")
        lines.append(f"  Recall (Sensitivity)  : {m['recall']:.4f}")
        lines.append(f"  F1-Score              : {m['f1']:.4f}")
        lines.append(f"  Specificity           : {m['specificity']:.4f}")
        lines.append(f"  NPV                   : {m['npv']:.4f}")
        lines.append("")
        lines.append(f"  Optimal Threshold (Youden's J) : {m['best_threshold']:.4f}")
        lines.append(f"    Recall    @ optimal thresh   : {m['recall_at_best_thresh']:.4f}")
        lines.append(f"    Precision @ optimal thresh   : {m['prec_at_best_thresh']:.4f}")
        lines.append(f"    F1        @ optimal thresh   : {m['f1_at_best_thresh']:.4f}")
        lines.append("")
        lines.append(f"  Confusion Matrix (threshold={THRESHOLD}):")
        lines.append(f"    {'':17s}  Pred Normal  Pred Deterioration")
        lines.append(f"    Actual Normal        {m['tn']:>10,}  {m['fp']:>17,}")
        lines.append(f"    Actual Deterioration {m['fn']:>10,}  {m['tp']:>17,}")
        lines.append("")
        lines.append(f"  Classification Report:")
        for line in m['classification_report'].splitlines():
            lines.append(f"    {line}")
        lines.append("")

    section("VALIDATION SET METRICS")
    write_metric_block(val_metrics)

    section("TEST SET METRICS  (Final Evaluation)")
    write_metric_block(test_metrics)

    # -- Summary ---------------------------------------------------
    section("SUMMARY")
    target_met = "[YES]" if test_metrics["auc_roc"] >= 0.85 else "[NO]"
    lines.append(f"  ROC-AUC Target (>= 0.85) : {target_met}")
    lines.append(f"  Test ROC-AUC             : {test_metrics['auc_roc']:.4f}")
    lines.append(f"  Test PR-AUC              : {test_metrics['auc_pr']:.4f}")
    lines.append(f"  Test F1-Score            : {test_metrics['f1']:.4f}")
    lines.append(f"  Test Recall              : {test_metrics['recall']:.4f}")
    lines.append(f"  Test Specificity         : {test_metrics['specificity']:.4f}")
    lines.append("")
    lines.append(f"  Balanced training:  X_train_balanced (95,458 sequences)")
    lines.append(f"  Feature selected:   30 out of 55 original features")
    lines.append(f"  Balancing method:   SMOTETomek (minority -> 33.3%)")
    lines.append("")
    lines.append(sep)

    report_path = os.path.join(REPORTS_DIR, "lstm_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  * Report saved to {report_path}")
    return report_path


# ===========================================================================
# MAIN
# ===========================================================================

def run_step09():
    """Train LSTM and generate metrics report."""
    print("\n" + "=" * 70)
    print("STEP 09 - LSTM Training & Metrics Report")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # -- Load data ------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    _, T, F = X_train.shape

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    # -- Build model ----------------------------------------------
    model = ICU_BiLSTM(n_features=F, seq_len=T).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: BiLSTM({LSTM1_UNITS}x2) -> LSTM({LSTM2_UNITS}) -> Dense({DENSE_UNITS}) -> 1")
    print(f"  Total parameters: {total_params:,}")

    # -- Train ----------------------------------------------------
    print(f"\n  Training for up to {MAX_EPOCHS} epochs (early stop patience={PATIENCE}) ...")
    t0 = time.time()
    history, best_val_auc = train(model, train_loader, val_loader, device)
    training_time = time.time() - t0
    print(f"\n  Training complete in {training_time/60:.1f} minutes")

    # -- Evaluate on Val & Test -----------------------------------
    print("\n  Evaluating on validation set ...")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_w = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    _, _, val_probs, val_labels   = evaluate(model, val_loader,  criterion, device)
    _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device)

    val_metrics  = compute_all_metrics(val_labels,  val_probs,  split_name="Validation")
    test_metrics = compute_all_metrics(test_labels, test_probs, split_name="Test")

    print_metrics(val_metrics)
    print_metrics(test_metrics)

    # -- Save model -----------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, "lstm_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": F,
        "seq_len": T,
        "history": history,
        "test_auc": test_metrics["auc_roc"],
    }, model_path)
    print(f"\n  * Model saved to {model_path}")

    # -- Write report ---------------------------------------------
    report_path = write_report(history, val_metrics, test_metrics, training_time, model)

    # -- Final banner ---------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL RESULT")
    print("=" * 70)
    auc = test_metrics["auc_roc"]
    if auc >= 0.85:
        print(f"  [PASS] ROC-AUC = {auc:.4f}  --- TARGET MET (>= 0.85)")
    else:
        print(f"  [WARN] ROC-AUC = {auc:.4f}  --- Below 0.85 target")
    print(f"  PR-AUC      = {test_metrics['auc_pr']:.4f}")
    print(f"  F1-Score    = {test_metrics['f1']:.4f}")
    print(f"  Recall      = {test_metrics['recall']:.4f}")
    print(f"  Specificity = {test_metrics['specificity']:.4f}")
    print(f"\n  Full report: {report_path}")

    return test_metrics


if __name__ == "__main__":
    run_step09()
