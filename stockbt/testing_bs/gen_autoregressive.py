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

print(f"Using columns: {FEATURES1 + [TARGET_PRICE]}")
df = pd.read_csv(file_path)

data1 = df[FEATURES1 + [TARGET_PRICE]].values.astype(np.float32)
feature_indices1 = {f: i for i, f in enumerate(FEATURES1 + [TARGET_PRICE])}

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
INIT_NUM_LAYERS = 1
INIT_LR = 1e-5
INIT_TARGET_ERROR = 11
SWITCH_EPOCH = 500000
NEW_NUM_LAYERS = 5
NEW_LR = 5*1e-6
NEW_TARGET_ERROR = 2

# --- Data split for new regime ---
train_end = 250
pid_start = 251
pid_end = 305  # inclusive
pred_start = 306
pred_end = 499  # inclusive (or min(pred_end, data1.shape[0]-1) if you want to be robust)

# --- Train Model 1: Predict next price ---
train_data1 = data1[:train_end]
X_train1 = torch.tensor(train_data1[:-1, :-1], dtype=torch.float32)  # All features except last col (Price at t)
y_train1 = torch.tensor(train_data1[1:, -1], dtype=torch.float32)    # Price at t+1
num_layers = INIT_NUM_LAYERS
lr = INIT_LR
target_error = INIT_TARGET_ERROR
model1 = PriceNet(input_dim=len(FEATURES1), num_layers=num_layers)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=500, min_lr=1e-7)
loss_fn = nn.MSELoss()
max_epochs = 1000000
required_good_epochs = 10
consecutive_good_epochs = 0
# Use train_end for validation/checkpoint
val_idx = train_end
val_input = torch.tensor(data1[val_idx-1, :-1], dtype=torch.float32).unsqueeze(0)
target_val = data1[val_idx, -1]
for epoch in range(max_epochs):
    model1.train()
    optimizer1.zero_grad()
    pred = model1(X_train1)
    loss = loss_fn(pred, y_train1)
    loss.backward()
    optimizer1.step()
    model1.eval()
    with torch.no_grad():
        pred_val = model1(val_input).item()
        pred_error = (pred_val - target_val) ** 2
    scheduler1.step(loss.item())
    if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
        print(f"[Model1] Epoch {epoch}: MSE to point {val_idx} = {pred_error:.6f}, Good epochs: {consecutive_good_epochs}, LR: {optimizer1.param_groups[0]['lr']:.2e}, num_layers: {num_layers}, target_error: {target_error}")
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
data2 = df[["Price", "Bid_Price", "Ask_Price"]].values.astype(np.float32)
train_data2 = data2[:train_end+1]  # 0-train_end for t+1
X_train2 = torch.tensor(train_data2[1:, [0]], dtype=torch.float32)  # Price at t+1
Y_train2 = torch.tensor(train_data2[1:, 1:3], dtype=torch.float32)  # Bid/Ask at t+1
num_layers2 = INIT_NUM_LAYERS
lr2 = INIT_LR
target_error2 = INIT_TARGET_ERROR
model2 = BidAskNet(input_dim=1, num_layers=num_layers2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2)
scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=500, min_lr=1e-7)
consecutive_good_epochs = 0
# Use train_end for validation/checkpoint
val_idx2 = train_end
val_input2 = torch.tensor(data2[val_idx2, [0]], dtype=torch.float32).unsqueeze(0)
target_val2 = data2[val_idx2, 1:3]
for epoch in range(max_epochs):
    model2.train()
    optimizer2.zero_grad()
    pred = model2(X_train2)
    loss = loss_fn(pred, Y_train2)
    loss.backward()
    optimizer2.step()
    model2.eval()
    with torch.no_grad():
        pred_val2 = model2(val_input2).squeeze(0).numpy()
        pred_error = np.mean((pred_val2 - target_val2) ** 2)
    scheduler2.step(loss.item())
    if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
        print(f"[Model2] Epoch {epoch}: MSE to point {val_idx2} = {pred_error:.6f}, Good epochs: {consecutive_good_epochs}, LR: {optimizer2.param_groups[0]['lr']:.2e}, num_layers: {num_layers2}, target_error: {target_error2}")
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

# --- Autoregressive + PID tuning and prediction ---
print(f"\n--- Bidirectional Autoregressive + PID tuning (train till {train_end}, PID {pid_start}-{pid_end}, predict {pred_start}-{pred_end}) ---")

start_idx = pid_start
max_idx = data1.shape[0] - 1
pred_end = min(pred_end, max_idx)
end_idx = pred_end + 1

# Prepare actuals
actual_price = [data1[i, -1] for i in range(start_idx, end_idx)]
actual_bid = [data1[i, feature_indices1["Bid_Price"]] for i in range(start_idx, end_idx)]
actual_ask = [data1[i, feature_indices1["Ask_Price"]] for i in range(start_idx, end_idx)]

# Initial input for walk-forward
input_seq = [row.copy() for row in data1[:start_idx]]
ar_price = []
ar_bid = []
ar_ask = []

# Generate pure autoregressive forecast for points pid_start to pred_end
for i in range(start_idx, end_idx):
    prev_vec = input_seq[-1].copy()
    X_input1 = torch.tensor(prev_vec[:-1], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        price_pred = model1(X_input1).item()
    X_input2 = torch.tensor([[price_pred]], dtype=torch.float32)
    with torch.no_grad():
        bidask_pred = model2(X_input2).squeeze(0).numpy()
    new_vec = prev_vec.copy()
    new_vec[feature_indices1["Price"]] = price_pred
    new_vec[feature_indices1["Bid_Price"]] = bidask_pred[0]
    new_vec[feature_indices1["Ask_Price"]] = bidask_pred[1]
    ar_price.append(price_pred)
    ar_bid.append(bidask_pred[0])
    ar_ask.append(bidask_pred[1])
    input_seq.append(new_vec)

# --- PID tuning on points 251-305 (first 55 of forecast) with decaying integral ---
def pid_tune_on_first_n_decay(pred_series, actual_series, n, label, decay=0.98):
    def pid_objective(params):
        Kp, Ki, Kd = params
        integral = 0
        prev_error = 0
        pid_preds = []
        for idx in range(n):
            pred_val = pred_series[idx]
            actual_val = actual_series[idx]
            error = actual_val - pred_val
            integral = decay * integral + error
            derivative = error - prev_error
            pid_correction = Kp * error + Ki * integral + Kd * derivative
            prev_error = error
            pid_pred = pred_val + pid_correction
            pid_preds.append(pid_pred)
        mse = np.mean((np.array(pid_preds) - np.array(actual_series[:n])) ** 2)
        return mse
    space = [
        Real(0.0, 2.0, name='Kp'),
        Real(0.0, 0.2, name='Ki'),
        Real(0.0, 1.0, name='Kd'),
    ]
    res = gp_minimize(pid_objective, space, n_calls=30, random_state=42, verbose=True)
    best_pid = {'Kp': res.x[0], 'Ki': res.x[1], 'Kd': res.x[2]}
    print(f"Best PID parameters for {label} (first {n}, decay={decay}): Kp={best_pid['Kp']:.4f}, Ki={best_pid['Ki']:.4f}, Kd={best_pid['Kd']:.4f}")
    # Apply PID correction to the full sequence
    pid_preds = []
    integral = 0
    prev_error = 0
    for idx in range(len(pred_series)):
        pred_val = pred_series[idx]
        actual_val = actual_series[idx]
        error = actual_val - pred_val
        integral = decay * integral + error
        derivative = error - prev_error
        pid_correction = best_pid['Kp'] * error + best_pid['Ki'] * integral + best_pid['Kd'] * derivative
        prev_error = error
        pid_pred = pred_val + pid_correction
        pid_preds.append(pid_pred)
    return pid_preds

ar_pid_price = pid_tune_on_first_n_decay(ar_price, actual_price, pid_end-pid_start+1, 'ar_price', decay=0.98)

# --- Correction: 301-305 average abs diff adjustment (now within PID/correction window, so no leakage) ---
corr_start = 301
corr_end = 305
corr_idx0 = corr_start - pid_start  # index in ar_price/actual_price
corr_idx1 = corr_end - pid_start + 1
avg_abs_diff = np.mean(np.abs(np.array(actual_price[corr_idx0:corr_idx1]) - np.array(ar_pid_price[corr_idx0:corr_idx1])))
print(f"Average abs diff (|actual - predicted|) for {corr_start}-{corr_end}: {avg_abs_diff:.6f}")
adjusted_ar_pid_price = [p - avg_abs_diff for p in ar_pid_price]

# Compute and print MSE for points 306-499 only (with correction)
pred_eval_start = pred_start - pid_start  # index in ar_price/actual_price
mse_ar_pid = np.mean((np.array(adjusted_ar_pid_price[pred_eval_start:]) - np.array(actual_price[pred_eval_start:])) ** 2)
print(f"MSE of corrected pure autoregressive forecast (points {pred_start}-{pred_end}): {mse_ar_pid:.6f}")

# Plot only the pure autoregressive forecast vs actual, with correction (exclude PID-tuning region)
plt.figure(figsize=(10, 5))
plt.plot(range(pred_start, pred_end+1), actual_price[pred_eval_start:], label='Actual Price', linestyle='-')
plt.plot(range(pred_start, pred_end+1), adjusted_ar_pid_price[pred_eval_start:], label='Autoregressive Forecast Price (PID+avg abs diff corrected)', linestyle=':')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.title(f'Pure Autoregressive Forecast (PID-tuned on {pid_start}-{pid_end}, avg abs diff {corr_start}-{corr_end}) vs Actual (points {pred_start} to {pred_end}) for {file}')
plt.legend()
plt.tight_layout()
plt.show() 