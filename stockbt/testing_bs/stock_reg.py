import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Attention block with configurable QKV and dense layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model, qkv_layers=1, dense_layers=3):
        super().__init__()
        self.qkv_layers = qkv_layers
        self.queries = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(qkv_layers)])
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(qkv_layers)])
        self.values = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(qkv_layers)])
        self.dense_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(dense_layers)])
        self.final_proj = nn.Linear(d_model, 1)  # Output scalar
        self.act = nn.ReLU()

    def forward(self, x):
        attn_outs = []
        for q, k, v in zip(self.queries, self.keys, self.values):
            Q = q(x)
            K = k(x)
            V = v(x)
            attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
            attn_weights = torch.softmax(attn_scores, dim=1)
            attended = torch.matmul(attn_weights, V)
            attn_outs.append(attended)
        # Average all QKV projections
        out = torch.stack(attn_outs, dim=0).mean(dim=0)
        # Pass through all dense layers
        for layer in self.dense_layers:
            out = self.act(layer(out))
        out = self.final_proj(out)  # shape: (seq_len, 1)
        return out.squeeze(-1)  # shape: (seq_len,)

# You can change qkv_layers and dense_layers here
model = DeepSelfAttention(d_model=6, qkv_layers=3, dense_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

for file in files:
    print(f"\n=== Processing {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    df["PriceChange"] = df["Price"].diff().fillna(0)
    cols = ["Price", "PriceChange", "Bid_Price", "Ask_Price", "Buy_Vol", "Sell_Vol"]
    data = df[cols].values.astype(np.float32)

    # True walk-forward: train on 0:399, predict 400, then train on 0:400, predict 401, etc.
    start_idx = 400
    end_idx = 500
    squared_errors = []

    for i in range(start_idx, end_idx):
        # Re-initialize model and optimizer for each step (optional, for strictest walk-forward)
        model = DeepSelfAttention(d_model=6, qkv_layers=3, dense_layers=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        # Train on all data up to i (not including i)
        X_train = torch.tensor(data[:i], dtype=torch.float32)
        y_train = torch.tensor([row[0] for row in data[1:i+1]], dtype=torch.float32)  # Only price
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
        # Predict the price at i (using all data up to i)
        model.eval()
        with torch.no_grad():
            X_pred = torch.tensor(data[:i], dtype=torch.float32)
            pred_next = model(X_pred)[-1].item()
        real_next = data[i][0]
        sq_err = (pred_next - real_next) ** 2
        squared_errors.append(sq_err)
        print(f"Step {i}: Predicted={pred_next:.4f}, Actual={real_next:.4f}, Squared Error={sq_err:.4f}")

    # Compute mean squared error over all predictions (price only)
    squared_errors = np.array(squared_errors)  # shape (100,)
    mse = np.mean(squared_errors)
    print(f"\nTrue Walk-forward 100-step MSE (price only): {mse:.6f}\n")

    # Predict the last vector using all previous data
    model.eval()
    with torch.no_grad():
        X_pred = torch.stack([torch.tensor(row) for row in data[:-1]])
        pred_last = model(X_pred)[-1].numpy()
        actual_last = data[-1][0]
        print(f"\nPredicted last vector: {pred_last}")
        print(f"Actual last vector:    {actual_last}")
        print(f"Per-dimension abs error: {np.abs(pred_last - actual_last)}")

    # --- PLOT: Bar plot for last vector ---
    plt.figure(figsize=(7, 4))
    x = np.arange(1)
    width = 0.35
    plt.bar(x - width/2, pred_last, width, label='Predicted')
    plt.bar(x + width/2, actual_last, width, label='Actual')
    plt.xticks(x, ["Price"], rotation=20)
    plt.ylabel('Value')
    plt.title(f'Predicted vs Actual Last Vector (1 dim)\n{file}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Autoregressive prediction for last 100 points ---
    start_idx = max(0, len(data) - 100)
    if len(data) > start_idx + 1:
        auto_preds = []
        auto_actuals = []
        auto_history = [torch.tensor(row) for row in data[:start_idx]]
        for i in range(start_idx, len(data)):
            X_auto = torch.stack(auto_history)
            with torch.no_grad():
                pred_val = model(X_auto)[-1].numpy()
            auto_preds.append(pred_val)
            auto_actuals.append(data[i][0])
            next_val = data[i][0]
            next_val = pred_val
            auto_history.append(torch.tensor(next_val))
        plt.figure(figsize=(10, 5))
        plt.plot(range(start_idx, len(data)), auto_actuals, label='Actual', linestyle='-')
        plt.plot(range(start_idx, len(data)), auto_preds, label='Predicted', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Autoregressive Prediction vs Actual (1 dim) for {file}')
        plt.legend()
        plt.tight_layout()
        plt.show() 