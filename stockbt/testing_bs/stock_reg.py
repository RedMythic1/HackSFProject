import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import json

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.json")

def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def save_weights_to_json(model, filename):
    weights = {k: v.cpu().detach().numpy().tolist() for k, v in model.state_dict().items()}
    with open(filename, "w") as f:
        json.dump(weights, f)
    print(f"Saved weights to {filename}")

def load_weights_from_json(model, filename):
    with open(filename, "r") as f:
        weights = json.load(f)
    state_dict = {k: torch.tensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {filename}")

# Define attention block with extra layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model, num_layers=20):
        super().__init__()
        self.num_qkv = num_layers * 10
        self.queries = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(self.num_qkv)])
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(self.num_qkv)])
        self.values = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(self.num_qkv)])
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        self.final_weight = nn.Parameter(torch.randn(d_model, d_model))  # output d_model-dim vector
        self.final_bias = nn.Parameter(torch.randn(d_model))
        self.act = nn.ReLU()

    def forward(self, x):
        # Aggregate all QKV attention heads
        attn_outputs = []
        for q, k, v in zip(self.queries, self.keys, self.values):
            Q = q(x)
            K = k(x)
            V = v(x)
            attn_scores = torch.sigmoid(torch.matmul(Q, K.T) / np.sqrt(x.shape[1]))
            attn_weights = torch.softmax(attn_scores, dim=1)
            attended = torch.matmul(attn_weights, V)
            attn_outputs.append(attended)
        # Aggregate outputs (mean over heads)
        out = torch.stack(attn_outputs, dim=0).mean(dim=0)
        for layer in self.layers:
            out = self.act(layer(out))
        out = torch.sigmoid(out @ self.final_weight) + self.final_bias  # shape: (seq_len, d_model)
        return out.squeeze(-1)

print(f"Found {len(files)} CSV files: {files}")
print(f"Model: DeepSelfAttention(d_model=10, num_layers=10, output_dim=10)")

# Initialize model and optimizer ONCE
model = DeepSelfAttention(d_model=10, num_layers=1)
if os.path.exists(WEIGHTS_PATH):
    load_weights_from_json(model, WEIGHTS_PATH)
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
data = df[cols].values.astype(np.float32)  # Convert selected columns to float32 numpy array
N = data.shape[0]  # Total number of rows in the data
train_cutoff = N - 50  # Define the cutoff for training (last 50 rows reserved for prediction)
history_X = [torch.tensor(data[0])]  # Initialize history of input vectors with the first row
history_y = [data[1]]  # Initialize history of target vectors with the second row
for i in range(1, train_cutoff-1):  # Walk-forward through the data up to the cutoff
    X_train = torch.stack(history_X)  # Stack input history into a tensor
    y_train = torch.stack([torch.tensor(vec) for vec in history_y])  # Stack target history into a tensor
    for epoch in range(50):  # Train for 50 epochs at each step
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Reset gradients
        pred = model(X_train)  # Forward pass: predict next step for all history
        target = torch.cat([
            y_train[-1, 0:9],  # Use last target's first 9 features
            torch.tensor([
                y_train[-1, 9]  # Use last target's price
            ])
        ]).float()  # Build the target vector
        loss = loss_fn(pred[-1], target)  # Compute loss on the last prediction
        loss.backward()  # Backpropagate
        optimizer.step()  # Update model weights
        if epoch == 49 and i % max(1, (train_cutoff-2)//5) == 0:
            print(f"  Step {i}/{train_cutoff-2}, Epoch {epoch+1}/50, Loss: {loss.item():.6f}")  # Print progress
    use_pred = random.random() < SCHEDULED_SAMPLING_PROB  # Scheduled sampling: decide whether to use model prediction
    if use_pred:
        model.eval()  # Set model to eval mode
        with torch.no_grad():  # No gradients needed
            pred_vec = model(X_train)[-1].cpu().numpy()  # Get model's prediction for next step
        prev = history_X[-1].numpy()  # Get previous input vector
        pred_price = prev[0] + pred_vec[9]  # (Unused) Example of how to update price
        pred_buy_vol = prev[1] + pred_vec[5]  # (Unused) Example of how to update buy volume
        pred_sell_vol = prev[2] + pred_vec[6]  # (Unused) Example of how to update sell volume
        pred_buy_price = prev[3] + pred_vec[7]  # (Unused) Example of how to update buy price
        pred_sell_price = prev[4] + pred_vec[8]  # (Unused) Example of how to update sell price
        # Reconstruct next vector in correct 10D order
        next_vec = np.array([
            pred_vec[0],  # Buy_Vol (predicted)
            pred_vec[1],  # Bid_Price (predicted)
            pred_vec[2],  # Sell_Vol (predicted)
            pred_vec[3],  # Ask_Price (predicted)
            pred_vec[4],  # PriceChange (predicted)
            pred_vec[5],  # BuyVolChange (predicted)
            pred_vec[6],  # SellVolChange (predicted)
            pred_vec[7],  # BidPriceChange (predicted)
            pred_vec[8],  # AskPriceChange (predicted)
            pred_vec[9],  # Price (predicted)
        ])
        history_X.append(torch.tensor(next_vec))  # Add predicted vector to input history
        history_y.append(next_vec)  # Add predicted vector to target history
        print(f"    [Scheduled Sampling] Used model prediction as next input at step {i}")
    else:
        history_X.append(torch.tensor(data[i]))  # Add actual data to input history
        history_y.append(data[i+1])  # Add next actual data to target history
print(f"  Finished walk-forward training on {last_file} up to last 50 rows.")

# Save weights after all training
save_weights_to_json(model, WEIGHTS_PATH)

# --- AUTOREGRESSIVE PREDICTION PHASE ---
print("\n=== AUTOREGRESSIVE PREDICTION ON LAST 50 ROWS OF LAST FILE ===")
auto_history = [torch.tensor(row) for row in data[:train_cutoff]]  # Initialize history with all training data
predicted_prices_log = [] # To store predicted prices at each autoregressive step

for i in range(train_cutoff, N):  # Loop over the last 50 rows for prediction
    X_auto = torch.stack(auto_history)  # Stack the historical vectors into a tensor for model input
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        pred_vec = model(X_auto)[-1].cpu().numpy()  # Get the model's prediction for the next step (last output)
    next_vec = np.array([
        pred_vec[0],  # Buy_Vol (predicted)
        pred_vec[1],  # Bid_Price (predicted)
        pred_vec[2],  # Sell_Vol (predicted)
        pred_vec[3],  # Ask_Price (predicted)
        pred_vec[4],  # PriceChange (predicted)
        pred_vec[5],  # BuyVolChange (predicted)
        pred_vec[6],  # SellVolChange (predicted)
        pred_vec[7],  # BidPriceChange (predicted)
        pred_vec[8],  # AskPriceChange (predicted)
        pred_vec[9],  # Price (predicted)
    ])
    predicted_prices_log.append(next_vec[9])  # Store only the predicted price for plotting
    auto_history.append(torch.tensor(next_vec))  # Add the predicted next vector to the history for future steps

# --- PLOTTING ---
plt.figure(figsize=(12, 6))  # Create a new figure for plotting
plt.plot(range(50), predicted_prices_log, label='Predicted Price', color='red')  # Plot predicted prices
plt.xlabel('Step (last 50)')  # Label x-axis
plt.ylabel('Price')  # Label y-axis
plt.title('Autoregressive Prediction: Last 50 Steps of Last File')  # Set plot title
plt.legend()  # Show legend
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot

# Print final prediction, actual, and error
final_pred = predicted_prices_log[-1]  # Last predicted price
final_real = data[-50, 9]  # Last actual price
final_error = np.abs(final_pred - final_real)  # Absolute error for each feature
print(f"\n=== FINAL AUTOREGRESSIVE PREDICTION ===")
print(f"Predicted price: {final_pred:.6f}")
print(f"Actual price:    {final_real:.6f}")
print(f"Absolute error:   {final_error:.6f}")
print(f"Mean absolute error: {np.mean(final_error):.6f}")
