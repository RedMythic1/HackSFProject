import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Cursor

# Try to import skopt for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Directory containing all CSV files
data_dir = "/Users/avneh/Code/HackSFProject/stockbt/testing_bs/data_folder"

# Get list of all CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise RuntimeError('No CSV files found in data_folder.')

# Select a random file
file = random.choice(csv_files)

# Check if the file exists
file_path = os.path.join(data_dir, file)
if not os.path.exists(file_path):
    raise RuntimeError(f'{file} not found in data_folder.')

class ResidualSubtractionNet(nn.Module):
    def __init__(self, input_dim, num_layers=10):
        super().__init__()
        self.num_layers = num_layers
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

print(f"\n=== Processing {file} ===")
df = pd.read_csv(os.path.join(data_dir, file))
df["PriceChange"] = df["Price"].diff().fillna(0)
df["TotalVolume"] = df["Buy_Vol"] + df["Sell_Vol"]
cols = ["Price", "PriceChange", "Bid_Price", "Ask_Price", "TotalVolume"]
data = df[cols].values.astype(np.float32)

# PHASE 1: Train intensively ONLY on points 0-199 until prediction for point 200 is very good
train_data = data[:200]  # Points 0-199
X_train = torch.tensor(train_data[:-1], dtype=torch.float32)  # Points 0-198
y_train = torch.tensor([row[0] for row in train_data[1:]], dtype=torch.float32)  # Points 1-199 (total volumes)

# Create and train the model
model = ResidualSubtractionNet(input_dim=5, num_layers=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.MSELoss()

# Training until prediction for point 200 is very good
target_point_200 = data[200][0]  # The actual total volume at point 200
max_epochs = 1000000
required_good_epochs = 10
target_error = 0.5  # Target squared error threshold
pred_error = float('inf')

# New: Track consecutive epochs below error threshold
consecutive_good_epochs = 0

print(f"Beginning intensive training on points 0-199 to predict point 200...")
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        X_200_input = torch.tensor(data[:200], dtype=torch.float32)  # Input for predicting point 200 is points 0-199
        pred_200 = model(X_200_input)[-1].item()
        pred_error = (pred_200 - target_point_200) ** 2
    
    if pred_error <= target_error:
        consecutive_good_epochs += 1
    else:
        consecutive_good_epochs = 0
    
    if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
        print(f"  Epoch {epoch}: Predicted={pred_200:.4f}, Actual={target_point_200:.4f}, Error={pred_error:.6f}, Good epochs: {consecutive_good_epochs}")
    
    if consecutive_good_epochs >= required_good_epochs:
        print(f"  Reached {required_good_epochs} consecutive good epochs at epoch {epoch}. Stopping training.")
        break

# PHASE 2: Use trained model to predict points 200-250 WITHOUT further training, using expanding window of actuals
squared_errors = []
absolute_errors = []
percentage_errors = []

print("\nUsing trained model to predict points 200-250 (walk-forward with actuals):")
model.eval()
with torch.no_grad():
    for i in range(200, 251): # Predict points 200 to 250
        # Input for predicting point 'i' is all actual data up to (but not including) 'i'
        X_pred_input = torch.tensor(data[:i], dtype=torch.float32) 
        pred_next = model(X_pred_input)[-1].item() # Predict total volume at point 'i'
        real_next = data[i][0] # Actual total volume at point 'i'
        sq_err = (pred_next - real_next) ** 2
        abs_err = abs(pred_next - real_next)
        pct_err = (abs_err / real_next) * 100
        
        squared_errors.append(sq_err)
        absolute_errors.append(abs_err)
        percentage_errors.append(pct_err)
        
        print(f"Point {i}: Predicted={pred_next:.4f}, Actual={real_next:.4f}, Squared Error={sq_err:.4f}")

# Compute mean squared error over all predictions
squared_errors = np.array(squared_errors)
absolute_errors = np.array(absolute_errors)
percentage_errors = np.array(percentage_errors)

mse = np.mean(squared_errors)
mae = np.mean(absolute_errors)
mape = np.mean(percentage_errors)
total_error = np.sum(squared_errors)

print(f"\nResults over points 200-250:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Sum of Squared Errors: {total_error:.6f}")
print(f"Average Percentage Error: {mape:.2f}%")
print(f"Number of predictions: {len(squared_errors)}")

# --- Walk-forward simulation for points 201-250 (no PID) ---
walk_preds_201_250 = []
walk_actuals_201_250 = []
walk_percentage_errors_201_250 = []

print("\nWalk-forward simulation for points 201-250 (no PID):")
with torch.no_grad():
    for i in range(201, 251):
        X_input = torch.tensor(data[:i], dtype=torch.float32)
        pred_val = model(X_input)[-1].item()
        actual_val = data[i][0]
        sq_err = (pred_val - actual_val) ** 2
        pct_err = (abs(pred_val - actual_val) / actual_val) * 100
        
        walk_preds_201_250.append(pred_val)
        walk_actuals_201_250.append(actual_val)
        walk_percentage_errors_201_250.append(pct_err)
        
        print(f"  Point {i}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}, Squared Error={sq_err:.4f}, Percentage Error={pct_err:.2f}%")

    # Show average percentage error for this range
    avg_pct_err_201_250 = np.mean(walk_percentage_errors_201_250)
    print(f"  Average Percentage Error (points 201-250): {avg_pct_err_201_250:.2f}%")

# --- PID parameter tuning using Bayesian optimization on points 201-250 ---
best_pid = {'Kp': 0.5, 'Ki': 0.01, 'Kd': 0.1}  # Default values
if SKOPT_AVAILABLE:
    print("\nTuning PID parameters on points 201-250 using Bayesian optimization...")
    def pid_objective(params):
        Kp, Ki, Kd = params
        integral = 0
        prev_error = 0
        pid_preds = []
        for idx in range(50):
            pred_val = walk_preds_201_250[idx]
            actual_val = walk_actuals_201_250[idx]
            error = actual_val - pred_val
            integral += error
            derivative = error - prev_error
            pid_correction = Kp * error + Ki * integral + Kd * derivative
            prev_error = error
            pid_pred = pred_val + pid_correction
            pid_preds.append(pid_pred)
        mse = np.mean((np.array(pid_preds) - np.array(walk_actuals_201_250)) ** 2)
        return mse
    # Search space for Kp, Ki, Kd
    space = [
        Real(0.0, 2.0, name='Kp'),
        Real(0.0, 0.2, name='Ki'),
        Real(0.0, 1.0, name='Kd'),
    ]
    res = gp_minimize(pid_objective, space, n_calls=30, random_state=42, verbose=True)
    best_pid = {'Kp': res.x[0], 'Ki': res.x[1], 'Kd': res.x[2]}
    print(f"Best PID parameters: Kp={best_pid['Kp']:.4f}, Ki={best_pid['Ki']:.4f}, Kd={best_pid['Kd']:.4f}")
else:
    print("\n[WARNING] scikit-optimize (skopt) is not installed. Using default PID parameters. To enable Bayesian tuning, run: pip install scikit-optimize\n")

# --- PID-corrected walk-forward prediction for points 251+ ---
Kp = best_pid['Kp']
Ki = best_pid['Ki']
Kd = best_pid['Kd']

integral = 0
prev_error = 0
pid_walk_preds = []
pid_walk_actuals = []
pid_errors = []
pid_percentage_errors = []

print("\nWalk-forward prediction for points 251+ (with tuned PID correction):")
with torch.no_grad():
    for i in range(251, len(data)):
        X_input = torch.tensor(data[:i], dtype=torch.float32)
        pred_val = model(X_input)[-1].item()
        actual_val = data[i][0]
        error = actual_val - pred_val
        integral += error
        derivative = error - prev_error
        pid_correction = Kp * error + Ki * integral + Kd * derivative
        prev_error = error
        pid_pred = pred_val + pid_correction
        
        pid_walk_preds.append(pid_pred)
        pid_walk_actuals.append(actual_val)
        
        sq_err = (pid_pred - actual_val) ** 2
        pct_err = (abs(pid_pred - actual_val) / actual_val) * 100
        
        pid_errors.append(sq_err)
        pid_percentage_errors.append(pct_err)
        
        print(f"  Point {i}: Model={pred_val:.4f}, PID Adjusted={pid_pred:.4f}, Actual={actual_val:.4f}, Squared Error={sq_err:.4f}, Percentage Error={pct_err:.2f}%")

n_points = len(pid_walk_preds)
pid_mse = np.sum(pid_errors) / n_points if n_points > 0 else float('nan')
pid_mape = np.mean(pid_percentage_errors) if n_points > 0 else float('nan')

print(f"\nPID Walk-forward MSE (points 251+): {pid_mse:.6f}")
print(f"PID Walk-forward MAPE (Mean Absolute Percentage Error): {pid_mape:.2f}%")

# Calculate improvement compared to non-PID predictions
model_only_errors = []
model_only_pct_errors = []

with torch.no_grad():
    for i in range(251, len(data)):
        X_input = torch.tensor(data[:i], dtype=torch.float32)
        pred_val = model(X_input)[-1].item()
        actual_val = data[i][0]
        sq_err = (pred_val - actual_val) ** 2
        pct_err = (abs(pred_val - actual_val) / actual_val) * 100
        model_only_errors.append(sq_err)
        model_only_pct_errors.append(pct_err)

model_only_mse = np.mean(model_only_errors) if len(model_only_errors) > 0 else float('nan')
model_only_mape = np.mean(model_only_pct_errors) if len(model_only_pct_errors) > 0 else float('nan')

improvement_mse = (1 - (pid_mse / model_only_mse)) * 100 if model_only_mse != 0 else float('nan')
improvement_mape = (1 - (pid_mape / model_only_mape)) * 100 if model_only_mape != 0 else float('nan')

print(f"\nComparison with model without PID:")
print(f"Model-only MSE: {model_only_mse:.6f}")
print(f"Model-only MAPE: {model_only_mape:.2f}%")
print(f"PID improves MSE by: {improvement_mse:.2f}%")
print(f"PID improves percentage error by: {improvement_mape:.2f}%")

# --- Combined 2-graph plot: normal, PID, and difference ---
fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# 1. Normal predictions vs actual
axs[0].plot(range(251, 251+len(pid_walk_actuals)), pid_walk_actuals, label='Actual', linestyle='-')
axs[0].plot(range(251, 251+len(pid_walk_preds)), pid_walk_preds, label='PID Corrected Predicted', linestyle=':')
axs[0].set_ylabel('Value')
axs[0].set_title('Predictions vs Actual')
axs[0].legend()

# 2. Difference between PID and normal predictions
diff = np.array(pid_walk_preds) - np.array(pid_walk_actuals)
axs[1].plot(range(251, 251+len(diff)), diff, color='purple', label='PID - Actual')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Difference')
axs[1].set_title('Difference: PID - Actual')
axs[1].legend()

# Add interactive crosshair (XY follower) to all subplots
for ax in axs:
    Cursor(ax, horizOn=True, vertOn=True, useblit=True, color='red', linewidth=1)

plt.tight_layout()
plt.show() 
