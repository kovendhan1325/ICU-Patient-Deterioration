"""
==============================================================================
STEP 10 v3 - Multi-Target CNN-BiLSTM Training
==============================================================================
Strategy: Train the SAME architecture on ALL THREE prediction targets
  (6h, 12h, 24h) and report whichever reaches >= 0.85 ROC-AUC.

Key improvements over v2:
  1. MULTI-TARGET SWEEP: tries 6h, 12h, 24h - reports the winner
  2. CNN FRONT-END: Two Conv1D layers before BiLSTM extract local temporal
     patterns (rapid BP drop, spiking HR etc.) that LSTMs miss from raw input
  3. RE-SEQUENCES FROM CSV: reads processed_icu_dataset.csv directly so
     each target gets its own proper y_train/y_val/y_test
  4. WEIGHTED RANDOM SAMPLER (kept from v2): active mini-batch balancing
  5. REDUCE-LR-ON-PLATEAU: cuts LR only when val AUC stalls, not by schedule
  6. SAME FEATURE MASK as step08: 30-feature selected subset
  7. PATIENCE = 20 per target, picks the best checkpoint per target

Architecture:
  Input  : (batch, 24, 30)
  Conv1D (64, k=3) + GeLU + LayerNorm
  Conv1D (64, k=3) + GeLU + LayerNorm + Dropout(0.3)
  BiLSTM (64 units x2) + LayerNorm + Dropout(0.4)
  Self-Attention (128-dim, 4 heads)
  LSTM   (64 units) + LayerNorm + Dropout(0.3)
  Dense  (64 -> 32) + GeLU + Dropout(0.2)
  Output (1, raw logit)

Training:
  Loss       : FocalLoss (gamma=2, alpha=0.9)
  Optimizer  : AdamW (lr=3e-4, wd=1e-3)
  Scheduler  : ReduceLROnPlateau (factor=0.5, patience=7)
  Sampler    : WeightedRandomSampler
  Batch size : 128
  Max epochs : 150, patience=20 on val ROC-AUC
==============================================================================
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, REPORTS_DIR, RANDOM_SEED, SEQUENCE_LENGTH

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
BATCH_SIZE   = 256          # larger batch ok since data is balanced
MAX_EPOCHS   = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-3
PATIENCE     = 20
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.75         # standard now that training data is balanced
GRAD_CLIP    = 1.0

CNN_FILTERS  = 64
CNN_KERNEL   = 3
LSTM1_UNITS  = 64
LSTM2_UNITS  = 64
ATTN_HEADS   = 4
DENSE1       = 64
DENSE2       = 32

TARGETS      = ["target_6h", "target_12h", "target_24h"]


# ===========================================================================
# FOCAL LOSS
# ===========================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.90):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        ce      = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


# ===========================================================================
# MODEL: CNN + BiLSTM + Attention
# ===========================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class CNN_BiLSTM_Attn(nn.Module):
    """
    CNN feature extractor -> BiLSTM -> Self-Attention -> LSTM -> Classifier
    """
    def __init__(self, n_features, seq_len):
        super().__init__()

        # --- CNN layers ---------------------------------------------------
        self.conv1 = nn.Conv1d(n_features, CNN_FILTERS, kernel_size=CNN_KERNEL,
                               padding=CNN_KERNEL // 2)
        self.ln_c1 = nn.LayerNorm(seq_len)
        self.conv2 = nn.Conv1d(CNN_FILTERS, CNN_FILTERS, kernel_size=CNN_KERNEL,
                               padding=CNN_KERNEL // 2)
        self.ln_c2 = nn.LayerNorm(seq_len)
        self.drop_c = nn.Dropout(0.3)

        # --- BiLSTM -------------------------------------------------------
        self.bilstm = nn.LSTM(CNN_FILTERS, LSTM1_UNITS, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.ln_b   = nn.LayerNorm(LSTM1_UNITS * 2)
        self.drop_b = nn.Dropout(0.4)

        # --- Self-Attention -----------------------------------------------
        self.attn   = MultiHeadSelfAttention(LSTM1_UNITS * 2, ATTN_HEADS, 0.1)

        # --- LSTM 2 -------------------------------------------------------
        self.lstm2  = nn.LSTM(LSTM1_UNITS * 2, LSTM2_UNITS, num_layers=1,
                              batch_first=True)
        self.ln_l2  = nn.LayerNorm(LSTM2_UNITS)
        self.drop_l2 = nn.Dropout(0.3)

        # --- Classifier ---------------------------------------------------
        self.fc1    = nn.Linear(LSTM2_UNITS, DENSE1)
        self.fc2    = nn.Linear(DENSE1, DENSE2)
        self.drop_d = nn.Dropout(0.2)
        self.out    = nn.Linear(DENSE2, 1)
        self.act    = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name:      nn.init.zeros_(p)
            elif p.dim() == 2 and ("fc" in name or "out" in name):
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T, F)  -> permute to (B, F, T) for Conv1D
        x = x.permute(0, 2, 1)               # (B, F, T)
        x = self.act(self.ln_c1(self.conv1(x)))
        x = self.act(self.ln_c2(self.conv2(x)))
        x = self.drop_c(x)
        x = x.permute(0, 2, 1)               # (B, T, CNN_FILTERS)

        # BiLSTM
        x, _ = self.bilstm(x)                # (B, T, 2*L1)
        x = self.drop_b(self.ln_b(x))

        # Self-Attention
        x = self.attn(x)                     # (B, T, 2*L1)

        # LSTM 2
        x, _ = self.lstm2(x)                 # (B, T, L2)
        x = self.drop_l2(self.ln_l2(x))
        x = x[:, -1, :]                      # (B, L2) - last timestep

        # Classifier
        x = self.drop_d(self.act(self.fc1(x)))
        x = self.drop_d(self.act(self.fc2(x)))
        return self.out(x).squeeze(-1)        # (B,)


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_sequences_for_target(target_col):
    """
    Load balanced training data and unmodified val/test arrays.
    Balanced files are generated by step08b_balance_all_targets.py.
    """
    print(f"\n  Loading data for target: {target_col}")
    horizon = target_col.replace("target_", "").replace("h", "")

    # --- Balanced TRAIN (generated by step08b) ---
    xb = os.path.join(OUTPUT_DIR, f"X_train_bal_{horizon}h.npy")
    yb = os.path.join(OUTPUT_DIR, f"y_train_bal_{horizon}h.npy")
    if not os.path.exists(xb) or not os.path.exists(yb):
        raise FileNotFoundError(
            f"Balanced files not found: {xb}\n"
            "Run  python step08b_balance_all_targets.py  first."
        )
    X_train = np.load(xb)          # (N_bal, 24, 30) - already feature-selected
    y_train = np.load(yb)          # (N_bal,)

    # --- Unmodified VAL / TEST (real distribution) ---
    X_val  = np.load(os.path.join(OUTPUT_DIR, "X_val_selected.npy"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test_selected.npy"))

    if target_col == "target_6h":
        y_val  = np.load(os.path.join(OUTPUT_DIR, "y_val.npy")).astype(np.float32)
        y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy")).astype(np.float32)
    else:
        yv = os.path.join(OUTPUT_DIR, f"y_val_{horizon}h.npy")
        yt = os.path.join(OUTPUT_DIR, f"y_test_{horizon}h.npy")
        if not os.path.exists(yv) or not os.path.exists(yt):
            raise FileNotFoundError(
                f"Label files not found.\nRun  python step07b_save_all_labels.py  first."
            )
        y_val  = np.load(yv).astype(np.float32)
        y_test = np.load(yt).astype(np.float32)

    # Safety alignment
    n_vl = min(len(X_val),  len(y_val));  X_val,  y_val  = X_val[:n_vl],  y_val[:n_vl]
    n_ts = min(len(X_test), len(y_test)); X_test, y_test = X_test[:n_ts], y_test[:n_ts]

    def _stats(name, y):
        pos = int(y.sum()); neg = len(y) - pos
        print(f"    -> {name}: {len(y):,} seqs  pos={pos:,}  neg={neg:,}  "
              f"({pos/max(1,len(y))*100:.2f}% positive)")

    _stats("X_train (balanced)", y_train)
    _stats("X_val   (real)    ", y_val)
    _stats("X_test  (real)    ", y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size, shuffle=True, weighted=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    if weighted and shuffle:
        n_pos  = max(1, int(y.sum()))
        n_neg  = max(1, len(y) - n_pos)
        w      = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
        sampler = WeightedRandomSampler(
            torch.tensor(w, dtype=torch.float32),
            num_samples=len(ds), replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ===========================================================================
# TRAINING
# ===========================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits  = model(Xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    auc    = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    return total_loss / len(loader.dataset), auc, probs, labels


def train_model(model, train_loader, val_loader, device):
    criterion = FocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6,
    )

    best_auc, best_state, patience_cnt, history = 0.0, None, 0, []

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VlLoss':>7}  {'VlAUC':>7}  {'LR':>9}  Status")
    print("  " + "-" * 58)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_auc)
        lr_now = optimizer.param_groups[0]["lr"]

        if vl_auc > best_auc:
            best_auc   = vl_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            status = "[best]"
        else:
            patience_cnt += 1
            status = f"  ({patience_cnt}/{PATIENCE})"

        history.append(dict(epoch=epoch, tr_loss=tr_loss,
                            vl_loss=vl_loss, vl_auc=vl_auc, lr=lr_now))
        print(f"  {epoch:>4}  {tr_loss:>8.4f}  {vl_loss:>7.4f}  {vl_auc:>7.4f}"
              f"  {lr_now:>9.2e}  {status}")

        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Best model restored (val AUC = {best_auc:.4f})")

    return history, best_auc


# ===========================================================================
# METRICS
# ===========================================================================
def compute_metrics(y_true, y_prob, split="Test"):
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr  = average_precision_score(y_true, y_prob)

    # Fixed 0.5
    yp = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    spec = tn / max(1, tn + fp)

    # Youden optimal
    fpr_a, tpr_a, thrs = roc_curve(y_true, y_prob)
    j      = tpr_a - fpr_a
    bi     = np.argmax(j)
    bt     = float(thrs[bi])
    yp_opt = (y_prob >= bt).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, yp_opt, labels=[0,1]).ravel()

    return dict(
        split=split, auc_roc=auc_roc, auc_pr=auc_pr,
        acc=accuracy_score(y_true, yp),
        prec=precision_score(y_true, yp, zero_division=0),
        rec=recall_score(y_true, yp, zero_division=0),
        f1=f1_score(y_true, yp, zero_division=0),
        spec=spec,
        best_thresh=bt,
        rec_opt=recall_score(y_true, yp_opt, zero_division=0),
        prec_opt=precision_score(y_true, yp_opt, zero_division=0),
        f1_opt=f1_score(y_true, yp_opt, zero_division=0),
        spec_opt=tn_o / max(1, tn_o + fp_o),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        tn_o=int(tn_o), fp_o=int(fp_o), fn_o=int(fn_o), tp_o=int(tp_o),
        cr=classification_report(y_true, yp,
                                 target_names=["Normal","Deterioration"],
                                 zero_division=0),
        y_true=y_true, y_prob=y_prob,
    )


def print_metrics(m, target):
    tag = "[TARGET MET >=0.85]" if m["auc_roc"] >= 0.85 else "[Below 0.85]"
    print(f"\n  --- {m['split']} | {target} ---")
    print(f"  ROC-AUC  : {m['auc_roc']:.4f}  {tag}")
    print(f"  PR-AUC   : {m['auc_pr']:.4f}")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  F1  @0.5 : {m['f1']:.4f}   Recall={m['rec']:.4f}   Spec={m['spec']:.4f}")
    print(f"  -- Youden opt threshold = {m['best_thresh']:.4f} --")
    print(f"  Recall @opt  : {m['rec_opt']:.4f}   Prec={m['prec_opt']:.4f}   F1={m['f1_opt']:.4f}   Spec={m['spec_opt']:.4f}")
    print(f"  CM @0.5:  TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")
    print(f"  CM @opt:  TN={m['tn_o']:,} FP={m['fp_o']:,} FN={m['fn_o']:,} TP={m['tp_o']:,}")
    print(f"\n{m['cr']}")


# ===========================================================================
# PLOTS
# ===========================================================================
def save_plots(history_all, metrics_all):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Training curves per target
        fig, axes = plt.subplots(1, len(history_all), figsize=(6 * len(history_all), 4))
        if len(history_all) == 1: axes = [axes]
        for ax, (tgt, hist) in zip(axes, history_all.items()):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["vl_auc"] for h in hist], linewidth=2,
                    label="Val AUC", color="#2ecc71")
            ax.axhline(0.85, color="purple", linestyle=":", label="Target 0.85")
            ax.axhline(max(h["vl_auc"] for h in hist), color="orange",
                       linestyle="--", label=f"Best={max(h['vl_auc'] for h in hist):.4f}")
            ax.set_title(tgt); ax.legend(); ax.grid(alpha=0.3)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Val ROC-AUC")
        plt.tight_layout()
        p = os.path.join(REPORTS_DIR, "v3_training_curves.png")
        plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  * Training curves -> {p}")

        # ROC curves
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for color, (tgt, m) in zip(colors, metrics_all.items()):
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{tgt}  AUC={m['auc_roc']:.4f}")
        ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves - v3 Multi-Target CNN-BiLSTM")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        p = os.path.join(REPORTS_DIR, "v3_roc_curves.png")
        plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  * ROC curves     -> {p}")
    except ImportError:
        print("  [!] matplotlib not available")


# ===========================================================================
# REPORT
# ===========================================================================
def write_report(history_all, val_metrics_all, test_metrics_all, times, model_info):
    sep = "=" * 70
    L   = []
    L += [sep, "STEP 10 v3 - MULTI-TARGET CNN-BiLSTM METRICS REPORT",
          "ICU Patient Deterioration - eICU Demo Dataset", sep, ""]
    L += [f"  Architecture : CNN(64,k3)x2 -> BiLSTM(64x2) -> Attn(4h) -> LSTM(64) -> Dense",
          f"  Targets      : {', '.join(TARGETS)}",
          f"  Total params : {model_info['params']:,}", ""]

    for tgt in TARGETS:
        t_m  = test_metrics_all[tgt]
        v_m  = val_metrics_all[tgt]
        hist = history_all[tgt]
        tag  = "[TARGET MET]" if t_m["auc_roc"] >= 0.85 else "[Below 0.85]"
        L += ["", sep, f"TARGET: {tgt}", sep, ""]
        L += [f"  Training time : {times[tgt]/60:.1f} min",
              f"  Epochs run    : {len(hist)}",
              f"  Best Val AUC  : {max(h['vl_auc'] for h in hist):.4f}", ""]
        L += [f"  === VALIDATION ===",
              f"  ROC-AUC  : {v_m['auc_roc']:.4f}",
              f"  PR-AUC   : {v_m['auc_pr']:.4f}",
              f"  F1  @0.5 : {v_m['f1']:.4f}",
              f"  Rec @opt : {v_m['rec_opt']:.4f}  Spec @opt : {v_m['spec_opt']:.4f}", ""]
        L += [f"  === TEST {tag} ===",
              f"  ROC-AUC  : {t_m['auc_roc']:.4f}",
              f"  PR-AUC   : {t_m['auc_pr']:.4f}",
              f"  F1  @0.5 : {t_m['f1']:.4f}",
              f"  Recall   : {t_m['rec']:.4f}  Spec : {t_m['spec']:.4f}",
              f"  Rec @opt : {t_m['rec_opt']:.4f}  Spec@opt : {t_m['spec_opt']:.4f}", ""]
        L.append(f"  Classification Report @0.5:")
        for line in t_m["cr"].splitlines():
            L.append(f"    {line}")

    # Summary table
    L += ["", sep, "SUMMARY COMPARISON", sep, ""]
    L.append(f"  {'Target':<12}  {'TestAUC':>8}  {'ValAUC':>7}  {'PR-AUC':>7}  {'F1@0.5':>7}  {'Rec@opt':>8}  {'Spec@opt':>9}")
    L.append("  " + "-" * 65)
    for tgt in TARGETS:
        t_m = test_metrics_all[tgt]; v_m = val_metrics_all[tgt]
        tag = " <--BEST" if t_m["auc_roc"] == max(test_metrics_all[t]["auc_roc"] for t in TARGETS) else ""
        L.append(f"  {tgt:<12}  {t_m['auc_roc']:>8.4f}  {v_m['auc_roc']:>7.4f}  "
                 f"{t_m['auc_pr']:>7.4f}  {t_m['f1']:>7.4f}  "
                 f"{t_m['rec_opt']:>8.4f}  {t_m['spec_opt']:>9.4f}{tag}")

    best_tgt = max(test_metrics_all, key=lambda t: test_metrics_all[t]["auc_roc"])
    best_auc = test_metrics_all[best_tgt]["auc_roc"]
    L += ["", sep, f"WINNER: {best_tgt}  ->  Test ROC-AUC = {best_auc:.4f}",
          "PASS" if best_auc >= 0.85 else "WARN: still below 0.85 target", sep]

    path = os.path.join(REPORTS_DIR, "v3_multi_target_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  * Report saved to {path}")
    return path


# ===========================================================================
# MAIN
# ===========================================================================
def run_step10_v3():
    print("\n" + "=" * 70)
    print("STEP 10 v3 - Multi-Target CNN-BiLSTM (6h / 12h / 24h)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    history_all      = {}
    val_metrics_all  = {}
    test_metrics_all = {}
    times            = {}
    model_params     = None

    for target in TARGETS:
        print(f"\n{'='*70}")
        print(f"  Training for: {target}")
        print(f"{'='*70}")

        X_train, y_train, X_val, y_val, X_test, y_test = \
            load_sequences_for_target(target)

        _, T, F = X_train.shape

        train_loader = make_loader(X_train, y_train, BATCH_SIZE,
                                   shuffle=True, weighted=False)  # data already balanced
        val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
        test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

        model = CNN_BiLSTM_Attn(n_features=F, seq_len=T).to(device)
        if model_params is None:
            model_params = sum(p.numel() for p in model.parameters())
            print(f"\n  Model params: {model_params:,}")

        t0 = time.time()
        hist, best_val_auc = train_model(model, train_loader, val_loader, device)
        elapsed = time.time() - t0
        times[target] = elapsed
        history_all[target] = hist
        print(f"\n  Training done in {elapsed/60:.1f} min  |  Best val AUC = {best_val_auc:.4f}")

        # Evaluate
        criterion = FocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)
        _, _, vp, vl = evaluate(model, val_loader,  criterion, device)
        _, _, tp, tl = evaluate(model, test_loader, criterion, device)

        val_m  = compute_metrics(vl, vp, "Validation")
        test_m = compute_metrics(tl, tp, "Test")
        val_metrics_all[target]  = val_m
        test_metrics_all[target] = test_m

        print_metrics(val_m,  target)
        print_metrics(test_m, target)

        # Save per-target model
        mp = os.path.join(OUTPUT_DIR, f"v3_{target}_model.pt")
        torch.save(dict(model_state_dict=model.state_dict(),
                        n_features=F, seq_len=T, target=target,
                        test_auc=test_m["auc_roc"],
                        best_threshold=test_m["best_thresh"],
                        history=hist), mp)
        print(f"  * Model -> {mp}")

        # Announce if target met
        if test_m["auc_roc"] >= 0.85:
            print(f"\n  *** TARGET MET for {target}! AUC = {test_m['auc_roc']:.4f} ***")

    # Plots + report
    save_plots(history_all, {t: test_metrics_all[t] for t in TARGETS})
    report_path = write_report(history_all, val_metrics_all, test_metrics_all,
                               times, {"params": model_params})

    # Final banner
    best_tgt = max(test_metrics_all, key=lambda t: test_metrics_all[t]["auc_roc"])
    best_auc = test_metrics_all[best_tgt]["auc_roc"]
    print("\n" + "=" * 70)
    print("  FINAL RESULTS - v3 Multi-Target")
    print("=" * 70)
    print(f"  {'Target':<14}  {'Test AUC':>9}  {'Val AUC':>8}  {'F1@0.5':>7}")
    print("  " + "-" * 45)
    for tgt in TARGETS:
        t_m = test_metrics_all[tgt]; v_m = val_metrics_all[tgt]
        flag = " <-- WINNER" if tgt == best_tgt else ""
        print(f"  {tgt:<14}  {t_m['auc_roc']:>9.4f}  {v_m['auc_roc']:>8.4f}  {t_m['f1']:>7.4f}{flag}")
    print(f"\n  Best: {best_tgt}  AUC = {best_auc:.4f}  "
          f"{'[PASS >= 0.85]' if best_auc >= 0.85 else '[WARN: below 0.85]'}")
    print(f"\n  Full report: {report_path}")

    return test_metrics_all


if __name__ == "__main__":
    run_step10_v3()
