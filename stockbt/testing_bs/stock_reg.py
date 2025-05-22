import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Attention block similar to your original, with bias and num_layers for dense layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model, num_dense_layers=3, num_qkv_projections=1): # Renamed num_layers to num_dense_layers
        super().__init__()
        self.num_qkv_projections = num_qkv_projections
        # QKV layers based on num_qkv_projections
        self.queries = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_qkv_projections)])
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_qkv_projections)])
        self.values = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_qkv_projections)])
        
        self.dense_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_dense_layers)])
        self.final_weight = nn.Parameter(torch.randn(d_model, 1))
        self.final_bias = nn.Parameter(torch.randn(1))
        self.act = nn.ReLU()

    def forward(self, x):
        attn_outputs = []
        # Iterate through each QKV projection
        for i in range(self.num_qkv_projections):
            Q = self.queries[i](x)
            K = self.keys[i](x)
            V = self.values[i](x)
            attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
            attn_weights = torch.softmax(attn_scores, dim=1)
            attended = torch.matmul(attn_weights, V)
            attn_outputs.append(attended)
        
        # Average the outputs of all QKV projections
        if self.num_qkv_projections > 0:
            out = torch.stack(attn_outputs).mean(dim=0)
        else: # Should not happen if num_qkv_projections >= 1
            out = x 

        for layer in self.dense_layers:
            out = self.act(layer(out))
        out = out @ self.final_weight + self.final_bias
        return out.squeeze(-1)

for file in files:
    print(f"\n=== Processing {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    df["PriceChange"] = df["Price"].diff().fillna(0)
    cols = ["Price", "PriceChange", "Bid_Price", "Ask_Price", "Buy_Vol", "Sell_Vol"]
    data = df[cols].values.astype(np.float32)

    # PHASE 1: Train intensively ONLY on points 0-399 until prediction for point 400 is very good
    train_data = data[:400]  # Points 0-399
    X_train = torch.tensor(train_data[:-1], dtype=torch.float32)  # Points 0-398
    y_train = torch.tensor([row[0] for row in train_data[1:]], dtype=torch.float32)  # Points 1-399 (prices)
    
    # Create and train the model
    model = DeepSelfAttention(d_model=6, num_dense_layers=1, num_qkv_projections=10) 
    
    # Dynamic learning rate parameters
    base_lr = 1.0
    min_lr = 0.001
    max_lr = 10.0
    lr_momentum = 0.09
    error_history = []
    lr_history = []
    random_step_prob = 0.1  # Probability of taking a random step
    random_step_size = 0.8  # Maximum size of random step
    
    # Plateau detection parameters
    plateau_window = 50  # Number of epochs to check for plateau
    plateau_threshold = 0.1  # Maximum allowed error change to consider as plateau
    plateau_spike_factor = 50.0  # How much to multiply LR by when plateau detected
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    loss_fn = nn.MSELoss()
    
    # Training until prediction for point 400 is very good
    target_point_400 = data[400][0]  # The actual price at point 400
    max_epochs = 1000000
    target_error = 0.1  # Target squared error threshold
    min_good_epochs = 400 # Minimum epochs for a "good" fit
    pred_error = float('inf')
    epochs_at_target_error = 0
    
    print(f"Beginning intensive training on points 0-399 to predict point 400...")
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        
        # Dynamic learning rate adjustment based on error derivative
        current_error = loss.item()
        error_history.append(current_error)
        
        if len(error_history) > 1:
            error_derivative = error_history[-1] - error_history[-2]
            
            # Check for plateau if we have enough history
            is_plateau = False
            if len(error_history) >= plateau_window:
                recent_errors = error_history[-plateau_window:]
                error_range = max(recent_errors) - min(recent_errors)
                error_std = np.std(recent_errors)
                
                # Detect plateau: small range of error values and low standard deviation
                if error_range < plateau_threshold and error_std < plateau_threshold/2:
                    is_plateau = True
                    print(f"  Plateau detected at epoch {epoch}! Spiking learning rate...")
                    new_lr = min(max_lr, optimizer.param_groups[0]['lr'] * plateau_spike_factor)
                else:
                    # Regular learning rate adjustment
                    if error_derivative < 0:  # Error is decreasing
                        new_lr = min(max_lr, optimizer.param_groups[0]['lr'] * (1 + 0.1 * abs(error_derivative)))
                    else:  # Error is increasing
                        new_lr = max(min_lr, optimizer.param_groups[0]['lr'] * (1 - 0.1 * abs(error_derivative)))
            else:
                # Regular learning rate adjustment for early epochs
                if error_derivative < 0:  # Error is decreasing
                    new_lr = min(max_lr, optimizer.param_groups[0]['lr'] * (1 + 0.1 * abs(error_derivative)))
                else:  # Error is increasing
                    new_lr = max(min_lr, optimizer.param_groups[0]['lr'] * (1 - 0.1 * abs(error_derivative)))
            
            # Apply momentum to learning rate changes (less momentum during plateau spikes)
            current_lr = optimizer.param_groups[0]['lr']
            momentum_factor = 0.01 if is_plateau else lr_momentum
            smoothed_lr = momentum_factor * current_lr + (1 - momentum_factor) * new_lr
            
            # Add random perturbation with probability random_step_prob
            # Higher probability during plateaus
            current_random_prob = random_step_prob * 2 if is_plateau else random_step_prob
            if np.random.random() < current_random_prob:
                # More aggressive random steps during plateaus
                current_random_size = random_step_size * 2 if is_plateau else random_step_size
                random_factor = 1 + (np.random.random() * 2 - 1) * current_random_size
                smoothed_lr *= random_factor
            
            # Keep learning rate within bounds
            smoothed_lr = np.clip(smoothed_lr, min_lr, max_lr)
            optimizer.param_groups[0]['lr'] = smoothed_lr
            lr_history.append(smoothed_lr)
        
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_400_input = torch.tensor(data[:400], dtype=torch.float32)  # Input for predicting point 400 is points 0-399
            pred_400 = model(X_400_input)[-1].item()
            current_pred_error = (pred_400 - target_point_400) ** 2
        
        if epoch % 100 == 0 or current_pred_error <= target_error:
            print(f"  Epoch {epoch}: Predicted={pred_400:.4f}, Actual={target_point_400:.4f}, Error={current_pred_error:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
            
        if current_pred_error <= target_error:
            epochs_at_target_error += 1
            if epochs_at_target_error >= min_good_epochs:
                print(f"  Reached target error for {min_good_epochs} consecutive epochs. Stopping training at epoch {epoch}.")
                break
        else:
            # Reset counter if error goes above target
            epochs_at_target_error = 0 
            
    # Check if training finished due to max_epochs without satisfying min_good_epochs
    if epochs_at_target_error < min_good_epochs and epoch == max_epochs -1:
        print(f"  Warning: Max epochs reached, but target error not held for {min_good_epochs} epochs.")
    
    # Plot learning rate and error history
    if len(lr_history) > 0:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(lr_history)
        plt.title('Learning Rate History')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(error_history)
        plt.title('Training Error History')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # PHASE 2: Use trained model to predict points 400-499 WITHOUT further training, using expanding window of actuals
    squared_errors = []
    
    print("\nUsing trained model to predict points 400-499 (walk-forward with actuals):")
    model.eval()
    with torch.no_grad():
        for i in range(400, 500): # Predict points 400 to 499
            # Input for predicting point 'i' is all actual data up to (but not including) 'i'
            X_pred_input = torch.tensor(data[:i], dtype=torch.float32) 
            pred_next = model(X_pred_input)[-1].item() # Predict price at point 'i'
            real_next = data[i][0] # Actual price at point 'i'
            sq_err = (pred_next - real_next) ** 2
            squared_errors.append(sq_err)
            print(f"Point {i}: Predicted={pred_next:.4f}, Actual={real_next:.4f}, Squared Error={sq_err:.4f}")
    
    # Compute mean squared error over all predictions
    squared_errors = np.array(squared_errors)
    mse = np.mean(squared_errors)
    total_error = np.sum(squared_errors)
    print(f"\nResults over points 400-499:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Sum of Squared Errors: {total_error:.6f}")
    print(f"Number of predictions: {len(squared_errors)}")

    # Predict the final price using all previous data (for bar plot)
    model.eval()
    with torch.no_grad():
        X_pred_all = torch.stack([torch.tensor(row) for row in data[:-1]])
        pred_last_val = model(X_pred_all)[-1].item()
        actual_last_val = data[-1][0]
        print(f"\nPredicted final price (using all data): {pred_last_val:.4f}")
        print(f"Actual final price:    {actual_last_val:.4f}")
        print(f"Absolute error: {np.abs(pred_last_val - actual_last_val):.4f}")

    # --- PLOT: Bar plot for final price ---
    plt.figure(figsize=(7, 4))
    x_bar = np.arange(1) # Use a different variable name for bar plot x-axis
    width = 0.35
    plt.bar(x_bar - width/2, pred_last_val, width, label='Predicted')
    plt.bar(x_bar + width/2, actual_last_val, width, label='Actual')
    plt.xticks(x_bar, ["Price"], rotation=20)
    plt.ylabel('Value')
    plt.title(f'Predicted vs Actual Final Price\n{file}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Autoregressive prediction for last 100 points (separate analysis) ---
    start_idx_autoreg = max(0, len(data) - 100)
    if len(data) > start_idx_autoreg + 1:
        auto_preds = []
        auto_actuals = []
        # Initialize history with actual data up to start_idx_autoreg
        auto_history = [torch.tensor(row) for row in data[:start_idx_autoreg]] 
        for i in range(start_idx_autoreg, len(data)):
            X_auto = torch.stack(auto_history) 
            with torch.no_grad():
                pred_val = model(X_auto)[-1].item()
            auto_preds.append(pred_val)
            auto_actuals.append(data[i][0])
            # For next step, use the predicted price in a full 6D vector based on current real data
            next_vec_autoreg = data[i].copy() 
            next_vec_autoreg[0] = pred_val 
            auto_history.append(torch.tensor(next_vec_autoreg))
        plt.figure(figsize=(10, 5))
        plt.plot(range(start_idx_autoreg, len(data)), auto_actuals, label='Actual', linestyle='-')
        plt.plot(range(start_idx_autoreg, len(data)), auto_preds, label='Predicted', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Autoregressive Prediction vs Actual (1 dim) for {file}')
        plt.legend()
        plt.tight_layout()
        plt.show() 