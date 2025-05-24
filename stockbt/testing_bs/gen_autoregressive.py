import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import minimize

# 🔧 MISSING CLASS DEFINITION: PIDController
# The original code referenced PIDController but never defined it.
# Adding basic implementation to prevent runtime errors.

class PIDController:
    """Basic PID Controller implementation"""
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        self.previous_error = 0
        self.integral = 0
    
    def update(self, error):
        """Update PID controller with new error and return correction"""
        self.integral += error
        derivative = error - self.previous_error
        
        correction = (self.kp * error + 
                     self.ki * self.integral + 
                     self.kd * derivative)
        
        self.previous_error = error
        return correction
    
    def reset(self):
        """Reset controller state"""
        self.previous_error = 0
        self.integral = 0

# Directory containing all CSV files
DATA_DIR = "/Users/avneh/Code/HackSFProject/stockbt/datasets"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
file = random.choice(csv_files)
file_path = os.path.join(DATA_DIR, file)

df = pd.read_csv(file_path)

print(f"=== ML PROGRESSIVE TRAINING ===")
print(f"Using file: {file}")

# 🚨 SYNTHETIC DATA WARNING: Artificial Bid/Ask Spread
# The following creates artificial bid/ask prices from close prices with fixed spread.
# This creates unrealistic perfect relationships that won't exist in real market data.
# Real bid/ask spreads vary dynamically based on volatility, volume, market conditions.
# 💡 Consider: Use real bid/ask data or add noise/variability to spreads.

# Create synthetic bid/ask prices from OHLC data
# Bid typically slightly below close, Ask slightly above close
spread_pct = 0.001  # 0.1% spread
df['Bid_Price'] = df['Close'] * (1 - spread_pct/2)
df['Ask_Price'] = df['Close'] * (1 + spread_pct/2)
df['Price'] = df['Close']  # Use Close as the main price

num_rows = len(df)

# Define the progressive training parameters
LOOKBACK_WINDOW = 50
STARTING_POINT = 3
final_train_until_idx = num_rows - LOOKBACK_WINDOW

if num_rows < LOOKBACK_WINDOW + 10:
    raise ValueError(f"Not enough data. Need at least {LOOKBACK_WINDOW + 10} rows.")

print(f"Total rows in dataset: {num_rows}")
print(f"Will progressively train from point {STARTING_POINT} to point {final_train_until_idx-1}")

# 🚨 DATA LEAK WARNING: Feature-Target Contamination Risk
# Using 'Price' as both feature and target creates potential artificial correlation.
# At time t-1, we use [Bid_Price, Ask_Price, Price] to predict Price at time t.
# Since Price is derived from Close, this might create unrealistic patterns.
# 💡 Consider: Remove 'Price' from features or use different target variable.

FEATURES1 = ["Bid_Price", "Ask_Price", "Price"]
TARGET_PRICE = "Price"
TARGET_BIDASK = ["Bid_Price", "Ask_Price"]

data1 = df[FEATURES1 + [TARGET_PRICE]].values.astype(np.float32)
data2 = df[["Price", "Bid_Price", "Ask_Price"]].values.astype(np.float32)

# Model architectures
class PriceNet(nn.Module):
    def __init__(self, input_dim, num_layers=5):
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

class BidAskNet(nn.Module):
    def __init__(self, input_dim, num_layers=5):
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

def train_to_convergence_for_point(model1, model2, data1, data2, target_point, 
                                   max_epochs=2000, target_error=0.001, required_good_epochs=20):
    """Train models to predict specific point"""
    print(f"\nTraining to predict point {target_point}...")
    
    train_data1 = data1[:target_point]
    train_data2 = data2[:target_point]
    
    if len(train_data1) < 2:
        raise ValueError(f"Not enough data to train for point {target_point}")
    
    X_train1 = torch.tensor(train_data1[:-1, :-1], dtype=torch.float32)
    y_train1 = torch.tensor(train_data1[1:, -1], dtype=torch.float32)
    X_train2 = torch.tensor(train_data2[1:, [0]], dtype=torch.float32)
    Y_train2 = torch.tensor(train_data2[1:, 1:3], dtype=torch.float32)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=100, min_lr=1e-7)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=100, min_lr=1e-7)
    loss_fn = nn.MSELoss()
    
    consecutive_good_epochs = 0
    
    val_input_m1 = torch.tensor(data1[target_point-1, :-1], dtype=torch.float32).unsqueeze(0)
    target_val_m1 = data1[target_point, -1]
    val_input_m2 = torch.tensor(data2[target_point, [0]], dtype=torch.float32).unsqueeze(0)
    target_val_m2 = data2[target_point, 1:3]
    
    for epoch in range(max_epochs):
        model1.train()
        model2.train()
        
        # Train Model 1 (Price prediction)
        if X_train1.shape[0] > 0:
            optimizer1.zero_grad()
            pred1 = model1(X_train1)
            loss1 = loss_fn(pred1, y_train1)
            loss1.backward()
            optimizer1.step()
            scheduler1.step(loss1.item())
        
        # Train Model 2 (Bid/Ask prediction)
        if X_train2.shape[0] > 0:
            optimizer2.zero_grad()
            pred2 = model2(X_train2)
            loss2 = loss_fn(pred2, Y_train2)
            loss2.backward()
            optimizer2.step()
            scheduler2.step(loss2.item())
        
        # Validation check
        model1.eval()
        model2.eval()
        with torch.no_grad():
            pred_val1 = model1(val_input_m1).item()
            pred_error1 = (pred_val1 - target_val_m1) ** 2
            pred_val2 = model2(val_input_m2).squeeze(0).numpy()
            pred_error2 = np.mean((pred_val2 - target_val_m2) ** 2)
            combined_error = pred_error1 + pred_error2
        
        if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
            print(f"  Epoch {epoch:4d}: Val_error={combined_error:.6f}, Good={consecutive_good_epochs}")
        
        if combined_error <= target_error:
            consecutive_good_epochs += 1
        else:
            consecutive_good_epochs = 0
        
        if consecutive_good_epochs >= required_good_epochs:
            print(f"  ✅ CONVERGED at epoch {epoch}!")
            break
    
    return model1, model2, combined_error

def tune_pid_parameters(errors):
    """Optimize PID parameters based on error signal"""
    def pid_cost_function(params):
        kp, ki, kd = params
        pid = PIDController(kp, ki, kd)
        
        total_cost = 0
        for error in errors:
            correction = pid.update(error)
            corrected_error = error - correction
            total_cost += corrected_error ** 2
        
        return total_cost
    
    initial_params = [1.0, 0.1, 0.05]
    
    result = minimize(pid_cost_function, initial_params, 
                     bounds=[(0.1, 10.0), (0.01, 1.0), (0.001, 0.5)],
                     method='L-BFGS-B')
    
    return result.x

def generate_pid_profile(model1, model2, data1, data2, calibration_points=5):
    """Generate PID correction profile from training data"""
    print(f"\nGenerating PID profile from training data...")
    
    model1.eval()
    model2.eval()
    
    price_errors = []
    bid_errors = []
    ask_errors = []
    
    # Use end of training data for calibration
    start_idx = max(0, final_train_until_idx - calibration_points - 5)
    end_idx = min(start_idx + calibration_points, final_train_until_idx)
    
    print(f"Using indices {start_idx} to {end_idx-1} for calibration")
    
    for i in range(calibration_points):
        current_idx = start_idx + i
        if current_idx >= end_idx or current_idx >= len(data1):
            break
            
        prev_features = torch.tensor(data1[current_idx-1, :-1], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            predicted_price = model1(prev_features).item()
            predicted_bidask = model2(torch.tensor([[predicted_price]], dtype=torch.float32)).squeeze(0).numpy()
        
        actual_price = data1[current_idx, -1]
        actual_bid = data2[current_idx, 1]
        actual_ask = data2[current_idx, 2]
        
        price_error = actual_price - predicted_price
        bid_error = actual_bid - predicted_bidask[0]
        ask_error = actual_ask - predicted_bidask[1]
        
        price_errors.append(price_error)
        bid_errors.append(bid_error)
        ask_errors.append(ask_error)
        
        print(f"  Calibration {i+1}: Price_error={price_error:.6f}, Bid_error={bid_error:.6f}, Ask_error={ask_error:.6f}")
    
    if len(price_errors) == 0:
        print("  Warning: No calibration data available")
        return None, None, None
    
    print(f"Tuning PID parameters on {len(price_errors)} samples...")
    
    price_pid_params = tune_pid_parameters(price_errors)
    bid_pid_params = tune_pid_parameters(bid_errors)
    ask_pid_params = tune_pid_parameters(ask_errors)
    
    print(f"Price PID: Kp={price_pid_params[0]:.4f}, Ki={price_pid_params[1]:.4f}, Kd={price_pid_params[2]:.4f}")
    print(f"Bid PID:   Kp={bid_pid_params[0]:.4f}, Ki={bid_pid_params[1]:.4f}, Kd={bid_pid_params[2]:.4f}")
    print(f"Ask PID:   Kp={ask_pid_params[0]:.4f}, Ki={ask_pid_params[1]:.4f}, Kd={ask_pid_params[2]:.4f}")
    
    price_pid = PIDController(*price_pid_params)
    bid_pid = PIDController(*bid_pid_params)
    ask_pid = PIDController(*ask_pid_params)
    
    return price_pid, bid_pid, ask_pid

# === MAIN TRAINING ===

print(f"\n=== Phase 1: Progressive Training ===")

# Initialize models
model1 = PriceNet(input_dim=len(FEATURES1), num_layers=5)
model2 = BidAskNet(input_dim=1, num_layers=5)

# Progressive training for first 10 points
quick_train_end = min(STARTING_POINT + 10, final_train_until_idx)

for target_point in range(STARTING_POINT, quick_train_end):
    model1, model2, final_error = train_to_convergence_for_point(
        model1, model2, data1, data2, target_point,
        max_epochs=1000, target_error=0.01, required_good_epochs=10
    )

print(f"\n=== Phase 2: Testing on Future Data ===")

# Test on unseen future data
test_start_idx = final_train_until_idx
test_end_idx = min(final_train_until_idx + 100, num_rows - 1)

prediction_indices = []
actual_prices = []
predicted_prices = []
actual_bids = []
predicted_bids = []
actual_asks = []
predicted_asks = []

model1.eval()
model2.eval()

print(f"Testing on indices {test_start_idx} to {test_end_idx}")

for current_idx in range(test_start_idx, test_end_idx + 1):
    if current_idx == 0:
        continue
    
    prev_features = torch.tensor(data1[current_idx-1, :-1], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        predicted_price = model1(prev_features).item()
        predicted_bidask = model2(torch.tensor([[predicted_price]], dtype=torch.float32)).squeeze(0).numpy()
    
    actual_price = data1[current_idx, -1]
    actual_bid = data2[current_idx, 1]
    actual_ask = data2[current_idx, 2]
    
    prediction_indices.append(current_idx)
    actual_prices.append(actual_price)
    predicted_prices.append(predicted_price)
    actual_bids.append(actual_bid)
    predicted_bids.append(predicted_bidask[0])
    actual_asks.append(actual_ask)
    predicted_asks.append(predicted_bidask[1])
    
    if current_idx % 20 == 0 or current_idx == test_end_idx:
        print(f"Index {current_idx}: Actual={actual_price:.2f}, Predicted={predicted_price:.2f}, Error={abs(actual_price-predicted_price):.3f}")

# Bias correction using first 5 predictions
print(f"\n=== Bias Correction ===")
if len(prediction_indices) >= 5:
    first_5_actual = actual_prices[:5]
    first_5_predicted = predicted_prices[:5]
    bias_offset = np.mean(np.array(first_5_actual) - np.array(first_5_predicted))
    
    print(f"First 5 predictions bias analysis:")
    for i in range(5):
        diff = first_5_actual[i] - first_5_predicted[i]
        print(f"  Point {i+1}: Actual={first_5_actual[i]:.2f}, Predicted={first_5_predicted[i]:.2f}, Diff={diff:.2f}")
    
    print(f"Average bias offset: {bias_offset:.3f}")
    
    # Apply bias correction
    corrected_predicted_prices = [p + bias_offset for p in predicted_prices]
    corrected_predicted_bids = [p + bias_offset for p in predicted_bids]
    corrected_predicted_asks = [p + bias_offset for p in predicted_asks]
    
    # Analysis on remaining points (excluding first 5)
    analysis_indices = prediction_indices[5:]
    analysis_actual_prices = actual_prices[5:]
    analysis_predicted_prices = predicted_prices[5:]
    analysis_corrected_prices = corrected_predicted_prices[5:]
    analysis_actual_bids = actual_bids[5:]
    analysis_predicted_bids = predicted_bids[5:]
    analysis_corrected_bids = corrected_predicted_bids[5:]
    analysis_actual_asks = actual_asks[5:]
    analysis_predicted_asks = predicted_asks[5:]
    analysis_corrected_asks = corrected_predicted_asks[5:]
    
    print(f"Analysis on {len(analysis_indices)} points (excluding first 5)")
    
else:
    print(f"Not enough predictions for bias correction")
    bias_offset = 0
    analysis_indices = prediction_indices
    analysis_actual_prices = actual_prices
    analysis_predicted_prices = predicted_prices
    analysis_corrected_prices = predicted_prices
    analysis_actual_bids = actual_bids
    analysis_predicted_bids = predicted_bids
    analysis_corrected_bids = predicted_bids
    analysis_actual_asks = actual_asks
    analysis_predicted_asks = predicted_asks
    analysis_corrected_asks = predicted_asks

# Calculate performance metrics
raw_price_mse = np.mean((np.array(analysis_predicted_prices) - np.array(analysis_actual_prices)) ** 2)
raw_price_mae = np.mean(np.abs(np.array(analysis_predicted_prices) - np.array(analysis_actual_prices)))
corrected_price_mse = np.mean((np.array(analysis_corrected_prices) - np.array(analysis_actual_prices)) ** 2)
corrected_price_mae = np.mean(np.abs(np.array(analysis_corrected_prices) - np.array(analysis_actual_prices)))
bid_mse = np.mean((np.array(analysis_corrected_bids) - np.array(analysis_actual_bids)) ** 2)

print(f"\n=== RESULTS ===")
print(f"Predictions analyzed: {len(analysis_indices)}")
print(f"Raw Price MSE: {raw_price_mse:.6f}")
print(f"Raw Price MAE: {raw_price_mae:.6f}")
print(f"Bias-Corrected Price MSE: {corrected_price_mse:.6f}")
print(f"Bias-Corrected Price MAE: {corrected_price_mae:.6f}")
print(f"MSE Improvement: {((raw_price_mse - corrected_price_mse) / raw_price_mse * 100):.1f}%")
print(f"Bid MSE: {bid_mse:.6f}")

# Generate plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Price predictions
ax1.plot(analysis_indices, analysis_actual_prices, label='Actual Price', color='blue', linewidth=2)
ax1.plot(analysis_indices, analysis_predicted_prices, label='Raw Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
ax1.plot(analysis_indices, analysis_corrected_prices, label='Bias-Corrected', color='green', linewidth=2)
ax1.axvline(x=final_train_until_idx, color='purple', linestyle=':', alpha=0.7, label='Training Cutoff')
ax1.set_ylabel('Price')
ax1.set_title('Price Predictions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error comparison
raw_price_errors = np.abs(np.array(analysis_predicted_prices) - np.array(analysis_actual_prices))
corrected_price_errors = np.abs(np.array(analysis_corrected_prices) - np.array(analysis_actual_prices))
ax2.plot(analysis_indices, raw_price_errors, label='Raw Error', color='red', linewidth=2, alpha=0.7)
ax2.plot(analysis_indices, corrected_price_errors, label='Corrected Error', color='green', linewidth=2)
ax2.set_ylabel('Absolute Error')
ax2.set_title('Error Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bid predictions
ax3.plot(analysis_indices, analysis_actual_bids, label='Actual Bid', color='blue', linewidth=2)
ax3.plot(analysis_indices, analysis_predicted_bids, label='Raw Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
ax3.plot(analysis_indices, analysis_corrected_bids, label='Bias-Corrected', color='green', linewidth=2)
ax3.set_ylabel('Bid Price')
ax3.set_title('Bid Predictions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Ask predictions
ax4.plot(analysis_indices, analysis_actual_asks, label='Actual Ask', color='blue', linewidth=2)
ax4.plot(analysis_indices, analysis_predicted_asks, label='Raw Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
ax4.plot(analysis_indices, analysis_corrected_asks, label='Bias-Corrected', color='green', linewidth=2)
ax4.set_xlabel('Data Index')
ax4.set_ylabel('Ask Price')
ax4.set_title('Ask Predictions')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bias_corrected_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== SUMMARY ===")
print(f"✅ Progressive training complete")
print(f"✅ Bias correction applied (offset: {bias_offset:.3f})")
print(f"✅ Analysis on {len(analysis_indices)} test points")
print(f"✅ Raw MSE: {raw_price_mse:.6f}, Corrected MSE: {corrected_price_mse:.6f}")
print(f"✅ Results saved to 'bias_corrected_predictions.png'")
print(f"✅ Neural network prediction system ready!") 