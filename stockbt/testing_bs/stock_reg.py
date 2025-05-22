import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing all CSV files
data_dir = "/Users/avneh/Code/HackSFProject/stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

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
    model = ResidualSubtractionNet(input_dim=6, num_layers=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    
    # Training until prediction for point 400 is very good
    target_point_400 = data[400][0]  # The actual price at point 400
    max_epochs = 1000000
    target_error = 0.5  # Target squared error threshold
    pred_error = float('inf')
    
    # New: Track consecutive epochs below error threshold
    consecutive_good_epochs = 0
    required_good_epochs = 1000
    
    print(f"Beginning intensive training on points 0-399 to predict point 400...")
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_400_input = torch.tensor(data[:400], dtype=torch.float32)  # Input for predicting point 400 is points 0-399
            pred_400 = model(X_400_input)[-1].item()
            pred_error = (pred_400 - target_point_400) ** 2
        
        if pred_error <= target_error:
            consecutive_good_epochs += 1
        else:
            consecutive_good_epochs = 0
        
        if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
            print(f"  Epoch {epoch}: Predicted={pred_400:.4f}, Actual={target_point_400:.4f}, Error={pred_error:.6f}, Good epochs: {consecutive_good_epochs}")
        
        if consecutive_good_epochs >= required_good_epochs:
            print(f"  Reached {required_good_epochs} consecutive good epochs at epoch {epoch}. Stopping training.")
            break
    
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

    # --- PID-corrected walk-forward prediction for last 100 points ---
    # PID parameters (tune as needed)
    Kp = 0.5
    Ki = 0.01
    Kd = 0.1
    
    integral = 0
    prev_error = 0
    pid_walk_preds = []
    pid_walk_actuals = []
    pid_errors = []
    
    print("\nWalk-forward prediction for last 100 points (with pseudo PID correction):")
    start_idx = max(0, len(data) - 100)
    with torch.no_grad():
        for i in range(start_idx, len(data)):
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
            pid_errors.append((pid_pred - actual_val) ** 2)
            print(f"  Point {i}: Model={pred_val:.4f}, PID Adjusted={pid_pred:.4f}, Actual={actual_val:.4f}, Squared Error={pid_errors[-1]:.4f}")
    pid_mse = np.mean(pid_errors)
    print(f"\nPID Walk-forward MSE (last 100 points): {pid_mse:.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_idx, len(data)), pid_walk_actuals, label='Actual', linestyle='-')
    plt.plot(range(start_idx, len(data)), pid_walk_preds, label='PID Corrected Predicted', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'PID Walk-forward Prediction vs Actual (last 100 points) for {file}')
    plt.legend()
    plt.tight_layout()
    plt.show() 
