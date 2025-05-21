import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Define attention block with extra layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model, num_layers=20):
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=True)
        self.key = nn.Linear(d_model, d_model, bias=True)
        self.value = nn.Linear(d_model, d_model, bias=True)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        self.final_weight = nn.Parameter(torch.randn(d_model, d_model))  # output d_model-dim vector
        self.final_bias = nn.Parameter(torch.randn(d_model))
        self.act = nn.ReLU()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended = torch.matmul(attn_weights, V)
        out = attended
        for layer in self.layers:
            out = self.act(layer(out))
        out = out @ self.final_weight + self.final_bias  # shape: (seq_len, d_model)
        return out.squeeze(-1)

print(f"Found {len(files)} CSV files: {files}")
print(f"Model: DeepSelfAttention(d_model=10, num_layers=10, output_dim=10)")

# Initialize model and optimizer ONCE
model = DeepSelfAttention(d_model=10, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

SCHEDULED_SAMPLING_PROB = 0.2

# --- SEQUENTIAL FILE TRAINING PHASE ---
for file_idx, file in enumerate(files[:-1]):
    print(f"\n=== Training on file {file_idx+1}/{len(files)}: {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    # Calculate changes on the spot
    df["PriceChange"] = df["Price"].diff().fillna(0)
    df["BuyVolChange"] = df["Buy_Vol"].diff().fillna(0)
    df["SellVolChange"] = df["Sell_Vol"].diff().fillna(0)
    df["BidPriceChange"] = df["Bid_Price"].diff().fillna(0)
    df["AskPriceChange"] = df["Ask_Price"].diff().fillna(0)
    cols = [
        "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price",
        "PriceChange", "BuyVolChange", "SellVolChange", "BidPriceChange", "AskPriceChange", "Price"
    ]
    data = df[cols].values.astype(np.float32)
    history_X = [torch.tensor(data[0])]
    history_y = [data[1]]
    for i in range(1, len(data)-1):
        X_train = torch.stack(history_X)
        y_train = torch.stack([torch.tensor(vec) for vec in history_y])
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            # Target: [Buy_Vol, Bid_Price, Sell_Vol, Ask_Price, PriceChange, BuyVolChange, SellVolChange, BidPriceChange, AskPriceChange, Price]
            target = torch.cat([
                y_train[-1, 0:9],  # first 9 features
                torch.tensor([
                    y_train[-1, 9]  # Price
                ])
            ]).float()
            loss = loss_fn(pred[-1], target)
            loss.backward()
            optimizer.step()
            if epoch == 49 and i % max(1, (len(data)-2)//5) == 0:
                print(f"  Step {i}/{len(data)-2}, Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
        # Scheduled sampling
        use_pred = random.random() < SCHEDULED_SAMPLING_PROB
        if use_pred:
            model.eval()
            with torch.no_grad():
                pred_vec = model(X_train)[-1].cpu().numpy()
            prev = history_X[-1].numpy()
            pred_price = prev[0] + pred_vec[9]
            pred_buy_vol = prev[1] + pred_vec[5]
            pred_sell_vol = prev[2] + pred_vec[6]
            pred_buy_price = prev[3] + pred_vec[7]
            pred_sell_price = prev[4] + pred_vec[8]
            # Reconstruct next vector in correct 10D order
            # [Buy_Vol, Bid_Price, Sell_Vol, Ask_Price, PriceChange, BuyVolChange, SellVolChange, BidPriceChange, AskPriceChange, Price]
            next_vec = np.array([
                pred_vec[0],  # Buy_Vol
                pred_vec[1],  # Bid_Price
                pred_vec[2],  # Sell_Vol
                pred_vec[3],  # Ask_Price
                pred_vec[4],  # PriceChange
                pred_vec[5],  # BuyVolChange
                pred_vec[6],  # SellVolChange
                pred_vec[7],  # BidPriceChange
                pred_vec[8],  # AskPriceChange
                pred_vec[9],  # Price
            ])
            history_X.append(torch.tensor(next_vec))
            history_y.append(next_vec)
            print(f"    [Scheduled Sampling] Used model prediction as next input at step {i}")
        else:
            history_X.append(torch.tensor(data[i]))
            history_y.append(data[i+1])
    print(f"  Finished training on {file}. Passing weights to next file.")

# --- LAST FILE: TRAIN UP TO LAST 50, THEN AUTOREGRESSIVE ---
last_file = files[-1]
print(f"\n=== Training (walk-forward) on last file up to last 50 rows: {last_file} ===")
df = pd.read_csv(os.path.join(data_dir, last_file))
# Calculate changes on the spot
df["PriceChange"] = df["Price"].diff().fillna(0)
df["BuyVolChange"] = df["Buy_Vol"].diff().fillna(0)
df["SellVolChange"] = df["Sell_Vol"].diff().fillna(0)
df["BidPriceChange"] = df["Bid_Price"].diff().fillna(0)
df["AskPriceChange"] = df["Ask_Price"].diff().fillna(0)
cols = [
    "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price",
    "PriceChange", "BuyVolChange", "SellVolChange", "BidPriceChange", "AskPriceChange", "Price"
]
data = df[cols].values.astype(np.float32)
N = data.shape[0]
train_cutoff = N - 50
history_X = [torch.tensor(data[0])]
history_y = [data[1]]
for i in range(1, train_cutoff-1):
    X_train = torch.stack(history_X)
    y_train = torch.stack([torch.tensor(vec) for vec in history_y])
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        target = torch.cat([
            y_train[-1, 0:9],
            torch.tensor([
                y_train[-1, 9]
            ])
        ]).float()
        loss = loss_fn(pred[-1], target)
        loss.backward()
        optimizer.step()
        if epoch == 49 and i % max(1, (train_cutoff-2)//5) == 0:
            print(f"  Step {i}/{train_cutoff-2}, Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
    use_pred = random.random() < SCHEDULED_SAMPLING_PROB
    if use_pred:
        model.eval()
        with torch.no_grad():
            pred_vec = model(X_train)[-1].cpu().numpy()
        prev = history_X[-1].numpy()
        pred_price = prev[0] + pred_vec[9]
        pred_buy_vol = prev[1] + pred_vec[5]
        pred_sell_vol = prev[2] + pred_vec[6]
        pred_buy_price = prev[3] + pred_vec[7]
        pred_sell_price = prev[4] + pred_vec[8]
        # Reconstruct next vector in correct 10D order
        next_vec = np.array([
            pred_vec[0],  # Buy_Vol
            pred_vec[1],  # Bid_Price
            pred_vec[2],  # Sell_Vol
            pred_vec[3],  # Ask_Price
            pred_vec[4],  # PriceChange
            pred_vec[5],  # BuyVolChange
            pred_vec[6],  # SellVolChange
            pred_vec[7],  # BidPriceChange
            pred_vec[8],  # AskPriceChange
            pred_vec[9],  # Price
        ])
        history_X.append(torch.tensor(next_vec))
        history_y.append(next_vec)
        print(f"    [Scheduled Sampling] Used model prediction as next input at step {i}")
    else:
        history_X.append(torch.tensor(data[i]))
        history_y.append(data[i+1])
print(f"  Finished walk-forward training on {last_file} up to last 50 rows.")

# --- AUTOREGRESSIVE PREDICTION PHASE ---
print("\n=== AUTOREGRESSIVE PREDICTION ON LAST 50 ROWS OF LAST FILE ===")
auto_history = [torch.tensor(row) for row in data[:train_cutoff]]
predicted_vectors = []
real_vectors = []
predicted_prices_log = [] # To store predicted prices at each autoregressive step

for i in range(train_cutoff, N):
    X_auto = torch.stack(auto_history)
    model.eval()
    with torch.no_grad():
        pred_vec = model(X_auto)[-1].cpu().numpy()
    prev = auto_history[-1].numpy()
    # Reconstruct next vector in correct 10D order
    next_vec = np.array([
        pred_vec[0],  # Buy_Vol
        pred_vec[1],  # Bid_Price
        pred_vec[2],  # Sell_Vol
        pred_vec[3],  # Ask_Price
        pred_vec[4],  # PriceChange
        pred_vec[5],  # BuyVolChange
        pred_vec[6],  # SellVolChange
        pred_vec[7],  # BidPriceChange
        pred_vec[8],  # AskPriceChange
        pred_vec[9],  # Price
    ])
    auto_history.append(torch.tensor(next_vec))
    predicted_vectors.append(next_vec)
    real_vectors.append(data[i])

    # Log and print the current predicted price and the list of predicted prices
    current_predicted_price = next_vec[9]
    predicted_prices_log.append(current_predicted_price)
    print(f"  Current predicted price: {current_predicted_price:.6f}, All predicted prices so far: {predicted_prices_log}")

    # Increased verbosity for debugging
    print(f"\nStep {i-train_cutoff+1}/50:")
    print(f"  Input to model (last input): {X_auto[-1].numpy()}")
    print(f"  Model raw output (pred_vec): {pred_vec}")
    print(f"  Constructed next_vec:        {next_vec}")
    print(f"  Real vector:                 {data[i]}")

predicted_vectors = np.array(predicted_vectors)
real_vectors = np.array(real_vectors)

# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(range(50), predicted_vectors[:, 9], label='Predicted Price', color='red')
plt.plot(range(50), real_vectors[:, 9], label='Actual Price', color='black', linestyle='--')
plt.xlabel('Step (last 50)')
plt.ylabel('Price')
plt.title('Autoregressive Prediction: Last 50 Steps of Last File')
plt.legend()
plt.tight_layout()
plt.show()

# Print final prediction, actual, and error
final_pred = predicted_vectors[-1]
final_real = real_vectors[-1]
final_error = np.abs(final_pred - final_real)
print(f"\n=== FINAL AUTOREGRESSIVE PREDICTION ===")
print(f"Predicted vector: {final_pred}")
print(f"Actual vector:    {final_real}")
print(f"Absolute error:   {final_error}")
print(f"Mean absolute error: {np.mean(final_error):.6f}")
