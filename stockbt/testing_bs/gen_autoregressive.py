import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from skopt import gp_minimize
from skopt.space import Real
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Directory containing all CSV files
DATA_DIR = "/Users/avneh/Code/HackSFProject/stockbt/testing_bs/data_folder"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
file = random.choice(csv_files)
file_path = os.path.join(DATA_DIR, file)

FEATURES1 = ["Bid_Price", "Ask_Price", "Price"]
FEATURES2 = ["Price"]
TARGET_PRICE = "Price"
TARGET_BIDASK = ["Bid_Price", "Ask_Price"]

print(f"Using columns for Model 1 input: {FEATURES1}")
df = pd.read_csv(file_path)

# Set dynamic end based on CSV length
num_rows = len(df)

if num_rows < 2:
    raise ValueError("Not enough data. Need at least 2 rows to train on all data and predict the next.")

print(f"Total rows in dataset: {num_rows}")
print(f"Training models on all {num_rows} rows to predict the hypothetical next point (row {num_rows}).")

data1 = df[FEATURES1 + [TARGET_PRICE]].values.astype(np.float32) # Used for Model 1
feature_indices1 = {f: i for i, f in enumerate(FEATURES1 + [TARGET_PRICE])} # To map feature names to indices in data1

data2 = df[["Price", "Bid_Price", "Ask_Price"]].values.astype(np.float32) # Used for Model 2

# Model 1: [Bid_Price, Ask_Price, Price] at t -> Price at t+1
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

# Model 2: [Price at t+1] -> [Bid_Price, Ask_Price] at t+1
class BidAskNet(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.final = nn.Linear(input_dim, 2)
    def forward(self, x):
        out = x
        for layer in self.layers:
            h = layer(out)
            out = out - h
        return self.final(out)

# --- Dynamic hyperparameters ---
INIT_NUM_LAYERS = 5
INIT_LR = 1e-5
INIT_TARGET_ERROR = 0.01
SWITCH_EPOCH = 500000
NEW_NUM_LAYERS = 5
NEW_LR = 5*1e-6
NEW_TARGET_ERROR = 2

# --- Train Model 1: Predict next price ---
# Model 1 trains on features from data[0...num_rows-2] to predict prices for data[1...num_rows-1]
train_data1_m1 = data1[:num_rows-1] # Use data up to second to last row for features, last row for target
if len(train_data1_m1) < 1: # Needs at least one pair for training if num_rows = 2
    raise ValueError("Not enough data to form a training pair for Model 1.")

X_train1 = torch.tensor(train_data1_m1[:-1, :-1], dtype=torch.float32)  # Input features: data[0...num_rows-2, features_m1]
y_train1 = torch.tensor(train_data1_m1[1:, -1], dtype=torch.float32)    # Target price: data[1...num_rows-1, price_idx]

if X_train1.shape[0] == 0 and num_rows > 1: # If num_rows is 1, this will be empty but caught by initial check
    # This case implies num_rows = 2, train_data1_m1 has 1 row, so X_train1 is empty. This is expected for num_rows=2.
    # For num_rows=2, training happens effectively via validation step if epochs are run.
    # However, with current early stopping logic, it might stop if target_error is too low.
    # A single point prediction based on that is fine.
    print("[Model1] Warning: X_train1 is empty, this is expected if num_rows=2. Model effectively trains on validation logic.")
elif X_train1.shape[0] == 0 and num_rows <=1:
     raise ValueError("Model 1 training input X_train1 is empty.")

num_layers = INIT_NUM_LAYERS
lr = INIT_LR
target_error = INIT_TARGET_ERROR
model1 = PriceNet(input_dim=len(FEATURES1), num_layers=num_layers)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=500, min_lr=1e-7)
loss_fn = nn.MSELoss()
max_epochs = 1000000
required_good_epochs = 100
consecutive_good_epochs = 0
# Validation for Model 1: Use features from num_rows-2 to predict price at num_rows-1 (last actual point)
val_idx_m1 = num_rows - 1 
val_input_m1 = torch.tensor(data1[val_idx_m1-1, :-1], dtype=torch.float32).unsqueeze(0) # Features from num_rows-2
target_val_m1 = data1[val_idx_m1, -1] # Actual price at num_rows-1

for epoch in range(max_epochs):
    model1.train()
    if X_train1.shape[0] > 0: # Only train if there's training data
        optimizer1.zero_grad()
        pred = model1(X_train1)
        loss = loss_fn(pred, y_train1)
        loss.backward()
        optimizer1.step()
        scheduler1.step(loss.item())
    else: # If no training data (num_rows=2), rely on initial weights or skip optimizer step
        loss = torch.tensor(float('inf')) # Effectively, no training loss to report or step scheduler on

    model1.eval()
    with torch.no_grad():
        pred_val = model1(val_input_m1).item()
        pred_error = (pred_val - target_val_m1) ** 2 # Error for the last point

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

# --- Train Model 2: Predict next Bid/Ask from next Price ---
# Model 2 input: Price at t, Output: Bid/Ask at t. Trains on all available data [0...num_rows-1]
train_data2_m2 = data2 # All data from index 0 to num_rows-1
X_train2 = torch.tensor(train_data2_m2[:, [0]], dtype=torch.float32)  # Price at t (0 to num_rows-1)
Y_train2 = torch.tensor(train_data2_m2[:, 1:3], dtype=torch.float32)  # Bid/Ask at t (0 to num_rows-1)

if X_train2.shape[0] == 0:
    raise ValueError("Model 2 training input X_train2 is empty.")

num_layers2 = INIT_NUM_LAYERS
lr2 = INIT_LR
target_error2 = INIT_TARGET_ERROR
model2 = BidAskNet(input_dim=1, num_layers=num_layers2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2)
scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=500, min_lr=1e-7)
consecutive_good_epochs = 0
# Validation for Model 2: Use Price at num_rows-1 to predict Bid/Ask at num_rows-1 (last actual point)
val_idx_m2 = num_rows - 1 
val_input_m2 = torch.tensor(data2[val_idx_m2, [0]], dtype=torch.float32).unsqueeze(0) # Price at num_rows-1
target_val_m2 = data2[val_idx_m2, 1:3] # Actual Bid/Ask at num_rows-1

for epoch in range(max_epochs):
    model2.train()
    optimizer2.zero_grad()
    pred = model2(X_train2)
    loss = loss_fn(pred, Y_train2)
    loss.backward()
    optimizer2.step()

    model2.eval()
    with torch.no_grad():
        pred_val2 = model2(val_input_m2).squeeze(0).numpy()
        pred_error = np.mean((pred_val2 - target_val_m2) ** 2) # Error for point predict_idx-1

    scheduler2.step(loss.item())
    if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
        print(f"[Model2] Epoch {epoch}: MSE to point {val_idx_m2} = {pred_error:.6f}, Good epochs: {consecutive_good_epochs}, LR: {optimizer2.param_groups[0]['lr']:.2e}, num_layers: {num_layers2}, target_error: {target_error2}")
    if pred_error <= target_error2:
        consecutive_good_epochs += 1
    else:
        consecutive_good_epochs = 0
    if consecutive_good_epochs >= required_good_epochs:
        print(f"[Model2] Reached {required_good_epochs} consecutive good epochs at epoch {epoch}. Stopping training.")
        break
    # Hyperparameter switching
    if epoch == SWITCH_EPOCH:
        print(f"[Model2] Switching hyperparameters at epoch {epoch}!")
        num_layers2 = NEW_NUM_LAYERS
        lr2 = NEW_LR
        target_error2 = NEW_TARGET_ERROR
        new_model2 = BidAskNet(input_dim=1, num_layers=num_layers2)
        for (n1, p1), (n2, p2) in zip(model2.named_parameters(), new_model2.named_parameters()):
            if p1.shape == p2.shape:
                p2.data.copy_(p1.data)
        model2 = new_model2
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2)
        scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=500, min_lr=1e-7)

# --- Predict the NEXT HYPOTHETICAL point (index num_rows) ---
print(f"\n--- Predicting the HYPOTHETICAL next point (after index {num_rows-1}) ---")

# Get the features from the VERY LAST actual data point to predict the next price
# These are from data1 at row num_rows-1
input_features_m1_future = torch.tensor(data1[num_rows-1, :-1], dtype=torch.float32).unsqueeze(0)

# Predict the NEXT price using Model 1
model1.eval()
with torch.no_grad():
    predicted_next_price = model1(input_features_m1_future).item()

print(f"Predicted Next Hypothetical Price: {predicted_next_price:.4f}")

# Prepare input for Model 2: the predicted_next_price
input_price_m2_future = torch.tensor([[predicted_next_price]], dtype=torch.float32)

# Predict the NEXT Bid/Ask using Model 2 with the predicted price
model2.eval()
with torch.no_grad():
    predicted_next_bidask = model2(input_price_m2_future).squeeze(0).numpy()

predicted_next_bid = predicted_next_bidask[0]
predicted_next_ask = predicted_next_bidask[1]

print(f"Predicted Next Hypothetical Bid: {predicted_next_bid:.4f}")
print(f"Predicted Next Hypothetical Ask: {predicted_next_ask:.4f}")

# --- Performance Metrics for the Last Point Prediction ---
# No actual data for the hypothetical next point, so no direct MSE here.
# Training validation errors give an indication of model fit to historical data.
print(f"\n--- METRICS FOR LAST ACTUAL POINT (VALIDATION DURING TRAINING) ---")
# Re-calculate validation error for the last actual point for clarity, if desired, or use stored values.
# For Model 1 (Price prediction for point num_rows-1)
val_input_m1_check = torch.tensor(data1[num_rows-2, :-1], dtype=torch.float32).unsqueeze(0)
actual_last_price_val = data1[num_rows-1, -1]
with torch.no_grad():
    pred_last_price_val = model1(val_input_m1_check).item()
price_val_mse = (pred_last_price_val - actual_last_price_val) ** 2
print(f"Model 1 Validation MSE (predicting price for point {num_rows-1}): {price_val_mse:.6f}")

# For Model 2 (Bid/Ask prediction for point num_rows-1, using its actual price)
val_input_m2_check = torch.tensor(data2[num_rows-1, [0]], dtype=torch.float32).unsqueeze(0)
actual_last_bidask_val = data2[num_rows-1, 1:3]
with torch.no_grad():
    pred_last_bidask_val = model2(val_input_m2_check).squeeze(0).numpy()
bidask_val_mse = np.mean((pred_last_bidask_val - actual_last_bidask_val) ** 2)
print(f"Model 2 Validation MSE (predicting bid/ask for point {num_rows-1}): {bidask_val_mse:.6f}")


# --- Plotting the Predicted Next Point ---
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

bar_labels = [f'Predicted P(N+1)']
price_value = [predicted_next_price]
axs[0].bar(bar_labels, price_value, color=['cyan'])
axs[0].set_ylabel('Price')
axs[0].set_title(f'Hypothetical Next Point (after Index {num_rows-1}) - Price Prediction')
for i, v in enumerate(price_value):
    axs[0].text(i, v + 0.001 * abs(v) if v != 0 else 0.01, f"{v:.4f}", color='black', ha='center')

bid_value = [predicted_next_bid]
axs[1].bar([f'Predicted B(N+1)'], bid_value, color=['lime'])
axs[1].set_ylabel('Price')
axs[1].set_title(f'Hypothetical Next Point - Bid Prediction')
for i, v in enumerate(bid_value):
    axs[1].text(i, v + 0.001 * abs(v) if v != 0 else 0.01, f"{v:.4f}", color='black', ha='center')

ask_value = [predicted_next_ask]
axs[2].bar([f'Predicted A(N+1)'], ask_value, color=['magenta'])
axs[2].set_ylabel('Price')
axs[2].set_title(f'Hypothetical Next Point - Ask Prediction')
for i, v in enumerate(ask_value):
    axs[2].text(i, v + 0.001 * abs(v) if v != 0 else 0.01, f"{v:.4f}", color='black', ha='center')

plt.xlabel(f'Prediction for Hypothetical Point after Index {num_rows-1}')
plt.tight_layout()
plt.show()

print(f"\n✅ Hypothetical next point prediction complete.") 