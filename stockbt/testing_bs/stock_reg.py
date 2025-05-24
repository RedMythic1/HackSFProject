import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from matplotlib.widgets import Cursor

# Optional Bayesian optimization for PID tuning
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not available. PID tuning will use default parameters.")

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        
        # Adjust num_heads to be compatible with d_model
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        print(f"Attention: d_model={d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Generate queries, keys, values
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attended)
        
        return output

class EnhancedResidualBlock(nn.Module):
    def __init__(self, input_dim, use_attention=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.use_attention = use_attention
        
        # Enhanced residual connection with learnable parameters
        self.alpha = nn.Parameter(torch.ones(input_dim))  # Multiplier for main path
        self.beta = nn.Parameter(torch.ones(input_dim))   # Multiplier for residual path
        self.bias = nn.Parameter(torch.zeros(input_dim))  # Learnable bias
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(input_dim, num_heads=4)
            self.layer_norm1 = nn.LayerNorm(input_dim)
            self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Activation function
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) or (seq_len, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            squeeze_output = True
        else:
            squeeze_output = False
            
        residual = x
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.layer_norm1(x)
            x = x + self.attention(x)  # Skip connection around attention
            x = self.layer_norm2(x)
        
        # Main transformation
        h = self.linear(x)
        h = self.activation(h)
        
        # Enhanced residual connection: out = alpha * residual - beta * h + bias
        out = self.alpha * residual - self.beta * h + self.bias
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out

class EnhancedResidualSubtractionNet(nn.Module):
    def __init__(self, input_dim, num_layers=5, use_attention=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Enhanced residual blocks
        self.blocks = nn.ModuleList([
            EnhancedResidualBlock(input_dim, use_attention=use_attention) 
            for _ in range(num_layers)
        ])
        
        # Final projection layers
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        self.final = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x shape: (seq_len, input_dim) for single sequence
        out = x
        
        # Apply enhanced residual blocks
        for block in self.blocks:
            out = block(out)
        
        # Final processing
        out = self.layer_norm(out)
        out = self.dropout(out)
        output = self.final(out)
        
        # Return last time step prediction
        if len(output.shape) > 1:
            return output[-1].squeeze(-1)
        else:
            return output.squeeze(-1)

print(f"\n=== Processing {file} ===")
df = pd.read_csv(os.path.join(data_dir, file))
df["PriceChange"] = df["Price"].diff().fillna(0)
df["Spread"] = df["Ask_Price"] - df["Bid_Price"]  # Use spread instead of volume
df["MidPrice"] = (df["Ask_Price"] + df["Bid_Price"]) / 2  # Add mid price feature
cols = ["Price", "PriceChange", "Bid_Price", "Ask_Price", "Spread"]
data = df[cols].values.astype(np.float32)

# Determine training cutoff - find the maximum prediction point we can train for
training_cutoff = len(data) - 50  # Leave some data for final evaluation
prediction_points = []
point = 100
while point <= training_cutoff:
    prediction_points.append(point)
    point += 100

print(f"Data length: {len(data)}")
print(f"Training cutoff: {training_cutoff}")
print(f"Will train models to predict points: {prediction_points}")

# Storage for all trained models and their results
trained_models = {}
all_predictions = {}
all_errors = {}

# Train separate models for each prediction point
print("\n" + "="*80)
print("COMPARISON: Training both Original and Enhanced Models")
print("="*80)

# Storage for comparison
original_models = {}
enhanced_models = {}
original_predictions = {}
enhanced_predictions = {}
original_errors = {}
enhanced_errors = {}

for target_point in prediction_points:
    print(f"\n=== Training Models for point {target_point} ===")
    
    # Use all data up to target_point-1 for training
    train_data = data[:target_point]
    X_train = torch.tensor(train_data[:-1], dtype=torch.float32)  # Points 0 to target_point-2
    y_train = torch.tensor([row[0] for row in train_data[1:]], dtype=torch.float32)  # Points 1 to target_point-1
    
    target_value = data[target_point][0]  # The actual value at target_point
    
    # TRAIN ORIGINAL MODEL
    print(f"\n--- Training Original Model for point {target_point} ---")
    class OriginalResidualSubtractionNet(nn.Module):
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

    original_model = OriginalResidualSubtractionNet(input_dim=5, num_layers=5)
    original_optimizer = torch.optim.Adam(original_model.parameters(), lr=0.00001)
loss_fn = nn.MSELoss()

    max_epochs = 50000  # Reduced for comparison
required_good_epochs = 10
    target_error = 0.5
consecutive_good_epochs = 0

for epoch in range(max_epochs):
        original_model.train()
        original_optimizer.zero_grad()
        pred = original_model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
        original_optimizer.step()
    
        original_model.eval()
    with torch.no_grad():
            X_target_input = torch.tensor(data[:target_point], dtype=torch.float32)
            pred_target = original_model(X_target_input)[-1].item()
            pred_error = (pred_target - target_value) ** 2
    
    if pred_error <= target_error:
        consecutive_good_epochs += 1
    else:
        consecutive_good_epochs = 0
    
        if epoch % 5000 == 0 or consecutive_good_epochs >= required_good_epochs:
            print(f"  Original Epoch {epoch}: Predicted={pred_target:.4f}, Actual={target_value:.4f}, Error={pred_error:.6f}")
    
    if consecutive_good_epochs >= required_good_epochs:
            print(f"  Original model converged after {epoch} epochs")
        break

    original_models[target_point] = original_model
    original_predictions[target_point] = pred_target
    original_errors[target_point] = pred_error
    
    # TRAIN ENHANCED MODEL
    print(f"\n--- Training Enhanced Model for point {target_point} ---")
    enhanced_model = EnhancedResidualSubtractionNet(input_dim=5, num_layers=6, use_attention=True)
    enhanced_optimizer = torch.optim.AdamW(enhanced_model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(enhanced_optimizer, mode='min', factor=0.5, patience=1000)
    
    max_epochs = 75000  # Moderate epochs for comparison
    required_good_epochs = 15
    target_error = 0.3
    consecutive_good_epochs = 0
    best_error = float('inf')
    
    for epoch in range(max_epochs):
        enhanced_model.train()
        enhanced_optimizer.zero_grad()
        pred = enhanced_model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), max_norm=1.0)
        enhanced_optimizer.step()
        scheduler.step(loss.item())
        
        enhanced_model.eval()
with torch.no_grad():
            X_target_input = torch.tensor(data[:target_point], dtype=torch.float32)
            pred_target = enhanced_model(X_target_input).item()
            pred_error = (pred_target - target_value) ** 2
        
        if pred_error < best_error:
            best_error = pred_error
        
        if pred_error <= target_error:
            consecutive_good_epochs += 1
        else:
            consecutive_good_epochs = 0
        
        if epoch % 5000 == 0 or consecutive_good_epochs >= required_good_epochs:
            current_lr = enhanced_optimizer.param_groups[0]['lr']
            print(f"  Enhanced Epoch {epoch}: Predicted={pred_target:.4f}, Actual={target_value:.4f}, Error={pred_error:.6f}, Best={best_error:.6f}, LR: {current_lr:.2e}")
        
        if consecutive_good_epochs >= required_good_epochs:
            print(f"  Enhanced model converged after {epoch} epochs")
            break
    
    enhanced_models[target_point] = enhanced_model
    enhanced_predictions[target_point] = pred_target
    enhanced_errors[target_point] = pred_error
    
    # Store the better performing model as the main one
    if pred_error < original_errors[target_point]:
        trained_models[target_point] = enhanced_model
        all_predictions[target_point] = pred_target
        all_errors[target_point] = pred_error
        print(f"  ✅ Enhanced model selected for point {target_point} (Error: {pred_error:.6f} vs {original_errors[target_point]:.6f})")
    else:
        trained_models[target_point] = original_model
        all_predictions[target_point] = original_predictions[target_point]
        all_errors[target_point] = original_errors[target_point]
        print(f"  ⚠️ Original model selected for point {target_point} (Error: {original_errors[target_point]:.6f} vs {pred_error:.6f})")

# COMPARISON RESULTS
print(f"\n" + "="*80)
print("DETAILED COMPARISON RESULTS")
print("="*80)
print("Point\tOriginal Pred\tEnhanced Pred\tActual\t\tOrig Error\tEnh Error\tImprovement")
print("-" * 95)

total_orig_error = 0
total_enh_error = 0
improvements = []

for target_point in prediction_points:
    actual_value = data[target_point][0]
    orig_pred = original_predictions[target_point]
    enh_pred = enhanced_predictions[target_point]
    orig_err = original_errors[target_point]
    enh_err = enhanced_errors[target_point]
    
    improvement = ((orig_err - enh_err) / orig_err) * 100 if orig_err > 0 else 0
    improvements.append(improvement)
    
    total_orig_error += orig_err
    total_enh_error += enh_err
    
    print(f"{target_point}\t{orig_pred:.4f}\t\t{enh_pred:.4f}\t\t{actual_value:.4f}\t\t{orig_err:.6f}\t{enh_err:.6f}\t{improvement:+.1f}%")

avg_improvement = np.mean(improvements)
total_improvement = ((total_orig_error - total_enh_error) / total_orig_error) * 100

print(f"\nSUMMARY:")
print(f"Average improvement per point: {avg_improvement:+.1f}%")
print(f"Total error improvement: {total_improvement:+.1f}%")
print(f"Original total error: {total_orig_error:.6f}")
print(f"Enhanced total error: {total_enh_error:.6f}")

# EVALUATION PHASE: Test each model on its target point
print(f"\n=== EVALUATION RESULTS ===")
print("Point\tPredicted\tActual\t\tSquared Error\tAbs Error\tPct Error")
print("-" * 70)

total_squared_error = 0
total_absolute_error = 0
total_percentage_error = 0

for target_point in prediction_points:
    model = trained_models[target_point]
    actual_value = data[target_point][0]
    predicted_value = all_predictions[target_point]
    
    squared_error = (predicted_value - actual_value) ** 2
    absolute_error = abs(predicted_value - actual_value)
    percentage_error = (absolute_error / actual_value) * 100
    
    total_squared_error += squared_error
    total_absolute_error += absolute_error
    total_percentage_error += percentage_error
    
    print(f"{target_point}\t{predicted_value:.4f}\t\t{actual_value:.4f}\t\t{squared_error:.6f}\t{absolute_error:.4f}\t{percentage_error:.2f}%")

num_predictions = len(prediction_points)
mse = total_squared_error / num_predictions
mae = total_absolute_error / num_predictions
mape = total_percentage_error / num_predictions

print(f"\n=== AGGREGATE METRICS ===")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Total Squared Error: {total_squared_error:.6f}")
print(f"Number of predictions: {num_predictions}")

# EXTENDED EVALUATION: Use the last trained model to predict beyond training cutoff
if prediction_points:
    last_point = max(prediction_points)
    last_model = trained_models[last_point]
    
    print(f"\n=== EXTENDED PREDICTIONS (using model trained for point {last_point}) ===")
    extended_predictions = []
    extended_actuals = []
    extended_errors = []
    
    # Predict points from training_cutoff to end of data
    last_model.eval()
with torch.no_grad():
        for i in range(training_cutoff, len(data)):
            if i >= len(data):
                break
                
        X_input = torch.tensor(data[:i], dtype=torch.float32)
            pred_val = last_model(X_input)[-1].item()
        actual_val = data[i][0]
        sq_err = (pred_val - actual_val) ** 2
        pct_err = (abs(pred_val - actual_val) / actual_val) * 100
        
            extended_predictions.append(pred_val)
            extended_actuals.append(actual_val)
            extended_errors.append(sq_err)
        
            print(f"Point {i}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}, Squared Error={sq_err:.4f}, Percentage Error={pct_err:.2f}%")

    if extended_predictions:
        extended_mse = np.mean(extended_errors)
        print(f"\nExtended prediction MSE: {extended_mse:.6f}")

# Visualization
plt.figure(figsize=(12, 8))

# Plot 1: Target point predictions
plt.subplot(2, 1, 1)
actual_values = [data[point][0] for point in prediction_points]
predicted_values = [all_predictions[point] for point in prediction_points]

plt.plot(prediction_points, actual_values, 'bo-', label='Actual Values', linewidth=2, markersize=8)
plt.plot(prediction_points, predicted_values, 'ro-', label='Predicted Values', linewidth=2, markersize=8)
plt.xlabel('Prediction Point')
plt.ylabel('Value')
plt.title(f'Fixed Point Predictions - {file}')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Prediction errors
plt.subplot(2, 1, 2)
errors = [all_errors[point] for point in prediction_points]
plt.plot(prediction_points, errors, 'go-', label='Squared Errors', linewidth=2, markersize=6)
plt.xlabel('Prediction Point')
plt.ylabel('Squared Error')
plt.title('Prediction Errors by Point')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() 

print(f"\n✅ Fixed point prediction training complete!")
print(f"   Trained {len(prediction_points)} separate models")
print(f"   Average MSE across all points: {mse:.6f}")
print(f"   Average MAPE: {mape:.2f}%") 
