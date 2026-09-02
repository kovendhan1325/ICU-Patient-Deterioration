import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Define paths to the processed data
data_dir = 'preprocessing_pipeline/output/'

print("Loading datasets...")
X_train_np = np.load(os.path.join(data_dir, 'X_train.npy'))
X_val_np = np.load(os.path.join(data_dir, 'X_val.npy'))
X_test_np = np.load(os.path.join(data_dir, 'X_test.npy'))

y_train_np = np.load(os.path.join(data_dir, 'y_train.npy'))
y_val_np = np.load(os.path.join(data_dir, 'y_val.npy'))
y_test_np = np.load(os.path.join(data_dir, 'y_test.npy'))

print(f"X_train shape: {X_train_np.shape}")
print(f"y_train shape: {y_train_np.shape}")
print(f"X_val shape: {X_val_np.shape}")
print(f"y_val shape: {y_val_np.shape}")
print(f"X_test shape: {X_test_np.shape}")
print(f"y_test shape: {y_test_np.shape}")

# Convert Numpy arrays to PyTorch Tensors
print("Converting arrays to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

# --- CLASS IMBALANCE HANDLING (WeightedRandomSampler) ---
# Create a sampler that oversamples the minority class (deterioration)
n_pos = int(y_train_np.sum())
n_neg = len(y_train_np) - n_pos

class_weights = [1.0 / n_neg, 1.0 / n_pos]
sample_weights = [class_weights[int(label)] for label in y_train_np]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float64),
    num_samples=len(sample_weights),
    replacement=True
)
print("Configured WeightedRandomSampler for class imbalance handling.")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128  # Tuned batch size
INPUT_SIZE = X_train_np.shape[2] 
HIDDEN_SIZE = 128 # Increased capacity
NUM_LAYERS = 2
DROPOUT = 0.4     # Tuned dropout
LEARNING_RATE = 5e-4 # Tuned learning rate
EPOCHS = 100      # Increased max epochs
PATIENCE = 10     # Early stopping patience

print(f"Model Architecture: Hidden={HIDDEN_SIZE}, Layers={NUM_LAYERS}, Dropout={DROPOUT}")
print(f"Training Config: LR={LEARNING_RATE}, Batch Size={BATCH_SIZE}, Max Epochs={EPOCHS}")

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ICUDeteriorationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(ICUDeteriorationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :]
        out = self.dropout(last_timestep_out)
        out = self.fc(out)
        return out.squeeze()

# Initialize Model
model = ICUDeteriorationLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

# Loss Function: BCEWithLogitsLoss
# We rely primarily on the WeightedRandomSampler for class balance, 
# so we use a standard loss.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_pr_auc = 0
epochs_no_improve = 0

print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
            
            probs = torch.sigmoid(preds)
            all_val_preds.extend(probs.cpu().numpy())
            all_val_targets.extend(y_batch.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate Validation PR-AUC instead of just ROC-AUC
    precision_val, recall_val, _ = precision_recall_curve(all_val_targets, all_val_preds)
    val_pr_auc = auc(recall_val, precision_val)
    val_roc_auc = roc_auc_score(all_val_targets, all_val_preds)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
    
    # Early Stopping based on PR-AUC
    if val_pr_auc > best_val_pr_auc:
        best_val_pr_auc = val_pr_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

print("Training Complete!")

# --- FINAL EVALUATION & THRESHOLD TUNING ---
print("\n--- Final Test Metrics & Dynamic Threshold Tuning ---")
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# Re-evaluate on validation set to find the optimal threshold for F1-score
all_val_preds = []
all_val_targets = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        probs = torch.sigmoid(model(X_batch))
        all_val_preds.extend(probs.cpu().numpy())
        all_val_targets.extend(y_batch.numpy())

# Calculate PR curve to find optimal threshold (maximizes F1-score)
precision, recall, thresholds = precision_recall_curve(all_val_targets, all_val_preds)
# Calculate F1 scores (avoid division by zero)
f1_scores = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Evaluated Thresholds. Selected Optimal Threshold (max F1): {optimal_threshold:.4f}")
print("This avoids using a highly conservative threshold (like 0.850 or 0.5) and ensures a better Recall-Precision balance.\n")

# Evaluate on Test Set
all_test_preds = []
all_test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        probs = torch.sigmoid(preds)
        
        all_test_preds.extend(probs.cpu().numpy())
        all_test_targets.extend(y_batch.cpu().numpy())

test_roc_auc = roc_auc_score(all_test_targets, all_test_preds)
test_precision, test_recall, _ = precision_recall_curve(all_test_targets, all_test_preds)
test_pr_auc = auc(test_recall, test_precision)

print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test PR-AUC:  {test_pr_auc:.4f}")

# Apply optimal threshold
binary_preds = [1 if p >= optimal_threshold else 0 for p in all_test_preds]
print(f"\nClassification Report (Threshold = {optimal_threshold:.4f}):\n")
print(classification_report(all_test_targets, binary_preds))