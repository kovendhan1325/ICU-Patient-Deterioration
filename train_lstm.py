import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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

# X is float32, y is float32 (for BCEWithLogitsLoss)
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

print("Conversion complete.")
# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
BATCH_SIZE = 128  # You can tune this (e.g., 64, 128, 256)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Created DataLoaders with Batch Size: {BATCH_SIZE}")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in val_loader: {len(val_loader)}")
import torch.nn as nn

class ICUDeteriorationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(ICUDeteriorationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer
        # batch_first=True means input tensor is of shape (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected (Linear) layer for final output
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length=24, features)
        
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # We only care about the output from the final timestep (sequence length end)
        # lstm_out[:, -1, :] grabs the last 24th hour state for all batches
        last_timestep_out = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_timestep_out)
        
        # Pass through the linear layer
        out = self.fc(out)
        return out.squeeze() # Remove extra dimensions so shape is (batch_size,)
import torch.optim as optim

# Hyperparameters
INPUT_SIZE = X_train_np.shape[2]  # Should be ~55 depending on your preprocessed features
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001

# Initialize Model
model = ICUDeteriorationLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

# Loss Function: BCEWithLogitsLoss is perfect for binary classification 
# We also calculate pos_weight to handle class imbalance (deterioration is rare)
pos_weight = (len(y_train_np) - y_train_np.sum()) / y_train_np.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)
from sklearn.metrics import roc_auc_score

EPOCHS = 50
patience = 5
best_val_auc = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 1. Forward pass
        predictions = model(X_batch)
        
        # 2. Calculate loss
        loss = criterion(predictions, y_batch)
        train_loss += loss.item()
        
        # 3. Backpropagation
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
            
            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(preds)
            all_val_preds.extend(probs.cpu().numpy())
            all_val_targets.extend(y_batch.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(all_val_targets, all_val_preds)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Early Stopping check
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

print("Training Complete!")
from sklearn.metrics import average_precision_score, classification_report

# Load best model weights
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

all_test_preds = []
all_test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        probs = torch.sigmoid(preds)
        
        all_test_preds.extend(probs.cpu().numpy())
        all_test_targets.extend(y_batch.cpu().numpy())

test_auc = roc_auc_score(all_test_targets, all_test_preds)
test_pr_auc = average_precision_score(all_test_targets, all_test_preds)

print(f"\n--- FINAL TEST METRICS ---")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC: {test_pr_auc:.4f}")

# Threshold at 0.5 (You can adjust this based on PR curve)
binary_preds = [1 if p >= 0.5 else 0 for p in all_test_preds]
print("\nClassification Report:\n", classification_report(all_test_targets, binary_preds))