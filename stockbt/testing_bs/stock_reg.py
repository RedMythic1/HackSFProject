import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing all CSV files
# data_dir = "stockbt/testing_bs/data_folder"
# files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
#
# # Define attention block with extra layers
# class DeepSelfAttention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
#         self.layer1 = nn.Linear(d_model, d_model)
#         self.layer2 = nn.Linear(d_model, d_model)
#         self.layer3 = nn.Linear(d_model, d_model)
#         self.final_weight = nn.Parameter(torch.randn(d_model, 1))  # 6d weight vector
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
#         attn_weights = torch.softmax(attn_scores, dim=1)
#         attended = torch.matmul(attn_weights, V)
#         # Pass through 3 dense layers with activation
#         out = self.act(self.layer1(attended))
#         out = self.act(self.layer2(out))
#         out = self.act(self.layer3(out))
#         # Collapse to scalar for each vector
#         out = out @ self.final_weight
#         return out.squeeze(-1)
#
# # Initialize model and optimizer ONCE
# model = DeepSelfAttention(d_model=6)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = nn.MSELoss()
#
# for file in files:
#     print(f"\n=== Processing {file} ===")
#     df = pd.read_csv(os.path.join(data_dir, file))
#     cols = ["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]
#     data = df[cols].values.astype(np.float32)
#
#     history_X = [torch.tensor(data[0])]
#     history_y = [data[1, 0]]  # The price of the 2nd point
#
#     predictions = []
#     actuals = []
#
#     for i in range(1, len(data)-1):
#         X_train = torch.stack(history_X)
#         y_train = torch.tensor(history_y)
#         for _ in range(10):
#             model.train()
#             optimizer.zero_grad()
#             pred = model(X_train)
#             loss = loss_fn(pred[-1:], y_train[-1:])
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         with torch.no_grad():
#             X_pred = torch.stack(history_X + [torch.tensor(data[i])])
#             pred_next = model(X_pred)[-1].item()
#         predictions.append(pred_next)
#         actuals.append(data[i+1, 0])
#         history_X.append(torch.tensor(data[i]))
#         history_y = list(y_train.numpy()) + [data[i+1, 0]]
#
#     print("Walk-forward predictions vs actuals:")
#     for i, (p, a) in enumerate(zip(predictions, actuals), start=2):
#         print(f"Step {i}: Predicted={p:.2f}, Actual={a:.2f}")
#
#     mse = np.mean((np.array(predictions) - np.array(actuals))**2)
#     print(f"\nWalk-forward MSE: {mse:.4f}")
#
#     # Predict the last price using all previous data
#     model.eval()
#     with torch.no_grad():
#         X_pred = torch.stack([torch.tensor(row) for row in data[:-1]])
#         pred_last = model(X_pred)[-1].item()
#         actual_last = data[-1, 0]
#         pct_diff = 100 * (pred_last - actual_last) / actual_last
#         print(f"\nPredicted last price: {pred_last:.4f}")
#         print(f"Actual last price:    {actual_last:.4f}")
#         print(f"Percentage difference: {pct_diff:.2f}%")
#
#     # --- Autoregressive prediction for last 100 points ---
#     start_idx = 400
#     if len(data) > start_idx + 1:
#         auto_preds = []
#         auto_actuals = []
#         # Use real data up to start_idx as history
#         auto_history = [torch.tensor(row) for row in data[:start_idx]]
#         for i in range(start_idx, len(data)):
#             X_auto = torch.stack(auto_history)
#             with torch.no_grad():
#                 pred_val = model(X_auto)[-1].item()
#             auto_preds.append(pred_val)
#             auto_actuals.append(data[i, 0])
#             # For next step, use the predicted value in place of the real one
#             # Copy the last row, but replace the price with the predicted value
#             next_vec = data[i].copy()
#             next_vec[0] = pred_val
#             auto_history.append(torch.tensor(next_vec))
#         # Plot
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(start_idx, len(data)), auto_actuals, label='Actual Price')
#         plt.plot(range(start_idx, len(data)), auto_preds, label='Predicted Price (autoregressive)')
#         plt.xlabel('Time Step')
#         plt.ylabel('Price')
#         plt.title(f'Autoregressive Prediction vs Actual for {file}')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

# === New autoregressive 6D vector generation and plotting ===
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory and file setup (adjust as needed)
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

class DeepSelfAttention6D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.layer3 = nn.Linear(d_model, d_model)
        self.final_matrix = nn.Parameter(torch.randn(d_model, 6))  # 6x6 matrix for 6D output
        self.act = nn.ReLU()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended = torch.matmul(attn_weights, V)
        out = self.act(self.layer1(attended))
        out = self.act(self.layer2(out))
        out = self.act(self.layer3(out))
        out = out @ self.final_matrix  # shape: (seq_len, 6)
        return out  # shape: (seq_len, 6)

model = DeepSelfAttention6D(d_model=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for file in files:
    print(f"\n=== Processing {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    cols = ["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]
    data = df[cols].values.astype(np.float32)

    # Train model as before (walk-forward, but for 6D output)
    history_X = [torch.tensor(data[0])]
    # Prepare first target: next 5 features + price diff
    price_diff = data[1][0] - data[0][0]
    history_y = [np.concatenate([data[1][:5], [price_diff]])]
    for i in range(1, 400):
        X_train = torch.stack(history_X)
        y_train = torch.tensor(history_y)
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred[-1:], y_train[-1:])
            loss.backward()
            optimizer.step()
        print(f"Step {i}: Training loss = {loss.item():.4f}")
        with torch.no_grad():
            per_dim_loss = ((pred[-1] - y_train[-1]) ** 2).cpu().numpy()
            print(f"Step {i}: Per-dimension MSE = {per_dim_loss}")
        model.eval()
        with torch.no_grad():
            X_pred = torch.stack(history_X + [torch.tensor(data[i])])
            pred_next = model(X_pred)[-1].detach().numpy()
        history_X.append(torch.tensor(data[i]))
        # Prepare next target: next 5 features + price diff
        price_diff = data[i+1][0] - data[i][0]
        history_y.append(np.concatenate([data[i+1][:5], [price_diff]]))

    # Autoregressive generation from 400 to 500
    gen_vectors = [torch.tensor(data[399])]  # Start from the 400th real vector
    for i in range(400, 500):
        X_gen = torch.stack(gen_vectors)
        with torch.no_grad():
            pred_vec6 = model(X_gen)[-1].detach().numpy()  # 6D prediction
        prev_price = gen_vectors[-1][0].item()
        # For the 6th value, set it to pred_price - prev_price (should match model output)
        pred_vec6[5] = pred_vec6[0] - prev_price
        print(f"Step {i}: Predicted price = {pred_vec6[0]:.2f}, Previous price = {prev_price:.2f}, Change = {pred_vec6[5]:.2f}")
        print(f"Step {i}: Predicted vector = {pred_vec6}")
        gen_vectors.append(torch.tensor(pred_vec6))
    # Remove the initial seed
    gen_vectors = gen_vectors[1:]
    # Gather generated and real prices
    gen_prices = [vec[0].item() for vec in gen_vectors]
    real_prices = data[400:500, 0]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(400, 500), real_prices, label='Actual Price')
    plt.plot(range(400, 500), gen_prices, label='Generated Price (autoregressive 6D)')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title(f'6D Autoregressive Generation vs Actual for {file}')
    plt.legend()
    plt.tight_layout()
    input("\nPress Enter to show the graph...")
    plt.show()