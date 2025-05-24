import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Directory containing all CSV files
DATA_DIR = "/Users/avneh/Code/HackSFProject/stockbt/testing_bs/data_folder"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
file = random.choice(csv_files)
file_path = os.path.join(DATA_DIR, file)

# Create lag features to avoid correlation leak
df = pd.read_csv(file_path)
df['Price_Lag1'] = df['Price'].shift(1)
df['Price_Lag2'] = df['Price'].shift(2) 
df['Price_Return'] = df['Price'].pct_change()

# FIXED: Use ONLY historical features - no current price leak
FEATURES1 = ["Price_Lag1", "Price_Lag2", "Price_Return"]

print(f"Using columns for Model 1 input: {FEATURES1}")

# Drop NaN rows from lag creation
df = df.dropna().reset_index(drop=True)

# Set dynamic end based on CSV length
num_rows = len(df)

# Define the moving window parameters
LOOKBACK_WINDOW = 50  # Train until n-50, then predict n-49 to n-1
if num_rows < LOOKBACK_WINDOW + 10:  # Need at least some points for training and prediction window
    raise ValueError(f"Not enough data. Need at least {LOOKBACK_WINDOW + 10} rows for moving window prediction.")

train_until_idx = num_rows - LOOKBACK_WINDOW  # Train on data up to this index (exclusive)
predict_start_idx = train_until_idx  # Start predicting from this index
predict_end_idx = num_rows - 1  # Predict up to (but not including) the last point

print(f"Total rows in dataset: {num_rows}")
print(f"Training on data from index 0 to {train_until_idx-1} (inclusive)")
print(f"Moving window prediction from index {predict_start_idx} to {predict_end_idx} ({predict_end_idx - predict_start_idx + 1} predictions)")

# FIXED: Create data with features and separate price target
data1_features = df[FEATURES1].values.astype(np.float32)  # Historical features only
data1_prices = df['Price'].values.astype(np.float32)  # Price targets
feature_indices1 = {f: i for i, f in enumerate(FEATURES1)}

# Model 1: [Price] at t -> Price at t+1
class PriceNet(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.final = nn.Linear(input_dim, 1)
    def forward(self, x):
        out = x
        for layer in self.layers:
            h = layer(out)
            out = out - h
        return self.final(out).squeeze(-1)

# --- Dynamic hyperparameters ---
INIT_NUM_LAYERS = 1
INIT_LR = 1e-5
INIT_TARGET_ERROR = 0.05
SWITCH_EPOCH = 500000
NEW_NUM_LAYERS = 5
NEW_LR = 5*1e-6
NEW_TARGET_ERROR = 2

# --- Train Model 1: Predict next price ---
# Train on data up to train_until_idx
train_data1 = data1_features[:train_until_idx]
if len(train_data1) < 2:
    raise ValueError("Not enough data for Model 1 training.")

# FIXED: Create targets using separate price array
X_train1 = torch.tensor(train_data1[:-1], dtype=torch.float32)  # Features from 0 to train_until_idx-2
y_train1 = torch.tensor(data1_prices[1:train_until_idx], dtype=torch.float32)  # Price targets from 1 to train_until_idx-1

if X_train1.shape[0] == 0:
    raise ValueError("Model 1 training input X_train1 is empty.")

num_layers = INIT_NUM_LAYERS
lr = INIT_LR
target_error = INIT_TARGET_ERROR
model1 = PriceNet(input_dim=len(FEATURES1), num_layers=num_layers)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=500, min_lr=1e-7)
loss_fn = nn.MSELoss()
max_epochs = 1000000
required_good_epochs = 1000
consecutive_good_epochs = 0

# Validation for Model 1: Use second-to-last point in training data to predict last point in training data
val_idx_m1 = train_until_idx - 1
if val_idx_m1 <= 0:
    raise ValueError("Not enough data for Model 1 validation.")
val_input_m1 = torch.tensor(data1_features[val_idx_m1-1], dtype=torch.float32).unsqueeze(0)
target_val_m1 = data1_prices[val_idx_m1]  # Price is first column

for epoch in range(max_epochs):
    model1.train()
    if X_train1.shape[0] > 0:
        optimizer1.zero_grad()
        pred = model1(X_train1)
        loss = loss_fn(pred, y_train1)
        loss.backward()
        optimizer1.step()
        scheduler1.step(loss.item())
    else:
        loss = torch.tensor(float('inf'))

    model1.eval()
    with torch.no_grad():
        pred_val = model1(val_input_m1).item()
        pred_error = (pred_val - target_val_m1) ** 2

    if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
        print(f"[Model1] Epoch {epoch}: MSE to point {val_idx_m1} = {pred_error:.6f}, Good epochs: {consecutive_good_epochs}, LR: {optimizer1.param_groups[0]['lr']:.2e}, num_layers: {num_layers}, target_error: {target_error}")
    if pred_error <= target_error:
        consecutive_good_epochs += 1
    else:
        consecutive_good_epochs = 0
    if consecutive_good_epochs >= required_good_epochs:
        print(f"[Model1] Reached {required_good_epochs} consecutive good epochs at epoch {epoch}. Stopping training.")
        break
    # Hyperparameter switching
    if epoch == SWITCH_EPOCH:
        print(f"[Model1] Switching hyperparameters at epoch {epoch}!")
        num_layers = NEW_NUM_LAYERS
        lr = NEW_LR
        target_error = NEW_TARGET_ERROR
        new_model = PriceNet(input_dim=len(FEATURES1), num_layers=num_layers)
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), new_model.named_parameters()):
            if p1.shape == p2.shape:
                p2.data.copy_(p1.data)
        model1 = new_model
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
        scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=500, min_lr=1e-7)

# --- Moving Window Prediction ---
print(f"\n--- Moving Window Prediction ({predict_start_idx} to {predict_end_idx}) ---")

# Storage for results
prediction_indices = []
actual_prices = []
predicted_prices = []  # Model predictions

model1.eval()

print("Generating predictions...")

for current_idx in range(predict_start_idx, predict_end_idx + 1):
    # Use features from the previous point to predict current point's price
    if current_idx == 0:
        continue  # Can't predict index 0 as there's no previous point
    
    prev_features = torch.tensor(data1_features[current_idx-1], dtype=torch.float32).unsqueeze(0)
    
    # Get raw prediction from model
    with torch.no_grad():
        raw_predicted_price = model1(prev_features).item()
    
    # Store results
    prediction_indices.append(current_idx)
    actual_prices.append(data1_prices[current_idx])
    predicted_prices.append(raw_predicted_price)
    
    # Display progress
    if current_idx % 10 == 0 or current_idx == predict_end_idx:
        print(f"Index {current_idx}: Actual={data1_prices[current_idx]:.4f}, Predicted={raw_predicted_price:.4f}")

# --- Calculate Metrics ---
print(f"\n--- MODEL PERFORMANCE METRICS ---")

if len(actual_prices) > 1:
    # Raw performance (no corrections)
    raw_errors = np.abs(np.array(predicted_prices) - np.array(actual_prices))
    raw_price_mse = np.mean((np.array(predicted_prices) - np.array(actual_prices)) ** 2)
    raw_price_mae = np.mean(raw_errors)
    raw_price_accuracy = np.mean(raw_errors / np.array(actual_prices)) * 100
    
    print(f"Model Performance:")
    print(f"   MSE: {raw_price_mse:.6f}")
    print(f"   MAE: {raw_price_mae:.6f}")
    print(f"   Error %: {raw_price_accuracy:.2f}%")
    
    print(f"Evaluation points: {len(actual_prices)}")
else:
    print("⚠️  Not enough points for evaluation")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Price predictions comparison
ax1.plot(prediction_indices, actual_prices, label='Actual Price', color='blue', linewidth=2)
ax1.plot(prediction_indices, predicted_prices, label='Predicted', color='orange', linewidth=1, alpha=0.7)
ax1.axvline(x=train_until_idx, color='green', linestyle=':', alpha=0.7, label=f'Training Cutoff (Index {train_until_idx})')
ax1.set_ylabel('Price')
ax1.set_title('Moving Window Price Predictions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction errors
if len(actual_prices) > 0:
    raw_errors = np.abs(np.array(predicted_prices) - np.array(actual_prices))
    
    ax2.plot(prediction_indices, raw_errors, label='Prediction Error', color='orange', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Prediction Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

ax2.set_xlabel('Data Index')
plt.tight_layout()
plt.show()

print(f"\n✅ Moving window prediction complete!")
print(f"   Trained on indices 0 to {train_until_idx-1}")
print(f"   Predicted indices {predict_start_idx} to {predict_end_idx}")
print(f"   Final model: {model1}") 