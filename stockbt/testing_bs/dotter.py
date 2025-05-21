import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Define attention block with extra layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.layer3 = nn.Linear(d_model, d_model)
        self.final_weight = nn.Parameter(torch.randn(d_model, 1))  # 6d weight vector
        self.act = nn.ReLU()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended = torch.matmul(attn_weights, V)
        # Pass through 3 dense layers with activation
        out = self.act(self.layer1(attended))
        out = self.act(self.layer2(out))
        out = self.act(self.layer3(out))
        # Collapse to scalar for each vector
        out = out @ self.final_weight
        return out.squeeze(-1)

# Initialize model and optimizer ONCE
model = DeepSelfAttention(d_model=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for file in files:
    print(f"\n=== Processing {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    cols = ["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]
    data = df[cols].values.astype(np.float32)

    history_X = [torch.tensor(data[0])]
    history_y = [data[1, 0]]  # The price of the 2nd point

    predictions = []
    actuals = []

    for i in range(1, len(data)-1):
        X_train = torch.stack(history_X)
        y_train = torch.tensor(history_y)
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred[-1:], y_train[-1:])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            X_pred = torch.stack(history_X + [torch.tensor(data[i])])
            pred_next = model(X_pred)[-1].item()
        predictions.append(pred_next)
        actuals.append(data[i+1, 0])
        history_X.append(torch.tensor(data[i]))
        history_y = list(y_train.numpy()) + [data[i+1, 0]]

    print("Walk-forward predictions vs actuals:")
    for i, (p, a) in enumerate(zip(predictions, actuals), start=2):
        print(f"Step {i}: Predicted={p:.2f}, Actual={a:.2f}")

    mse = np.mean((np.array(predictions) - np.array(actuals))**2)
    print(f"\nWalk-forward MSE: {mse:.4f}")

    # Predict the last price using all previous data
    model.eval()
    with torch.no_grad():
        X_pred = torch.stack([torch.tensor(row) for row in data[:-1]])
        pred_last = model(X_pred)[-1].item()
        actual_last = data[-1, 0]
        pct_diff = 100 * (pred_last - actual_last) / actual_last
        print(f"\nPredicted last price: {pred_last:.4f}")
        print(f"Actual last price:    {actual_last:.4f}")
        print(f"Percentage difference: {pct_diff:.2f}%")

    # --- Autoregressive prediction for last 100 points ---
    start_idx = 400
    if len(data) > start_idx + 1:
        auto_preds = []
        auto_actuals = []
        # Use real data up to start_idx as history
        auto_history = [torch.tensor(row) for row in data[:start_idx]]
        for i in range(start_idx, len(data)):
            X_auto = torch.stack(auto_history)
            with torch.no_grad():
                pred_val = model(X_auto)[-1].item()
            auto_preds.append(pred_val)
            auto_actuals.append(data[i, 0])
            # For next step, use the predicted value in place of the real one
            # Copy the last row, but replace the price with the predicted value
            next_vec = data[i].copy()
            next_vec[0] = pred_val
            auto_history.append(torch.tensor(next_vec))
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_idx, len(data)), auto_actuals, label='Actual Price', linewidth=1)
        plt.plot(range(start_idx, len(data)), auto_preds, label='Predicted Price (autoregressive)', linewidth=1)
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title(f'Autoregressive Prediction vs Actual for {file}')
        plt.legend()
        plt.tight_layout()

        # Zoom in on the y-axis to focus on micro fluctuations
        all_prices = np.concatenate([auto_actuals, auto_preds])
        ymin, ymax = np.min(all_prices), np.max(all_prices)
        yrange = ymax - ymin
        plt.ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)  # 5% padding

        # Optionally, plot the error
        plt.twinx()
        plt.plot(range(start_idx, len(data)), np.array(auto_preds) - np.array(auto_actuals), 
                 color='gray', alpha=0.3, label='Prediction Error')
        plt.ylabel('Prediction Error')
        plt.legend(loc='upper right')

        print("auto_preds[:10]:", auto_preds[:10])
        print("auto_actuals[:10]:", auto_actuals[:10])
        print("auto_preds unique values:", np.unique(auto_preds))

        plt.show()
