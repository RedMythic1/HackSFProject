import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import sys

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.json")

cols = [
    "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price",
    "PriceChange", "BuyVolChange", "SellVolChange", "BidPriceChange", "AskPriceChange", "Price"
]

# Model definition (must match training script)
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model=10, num_layers=2, attn_layers=5, ff_dim=32):
        super().__init__()
        self.num_heads = attn_layers
        self.d_model = d_model
        self.head_dim = d_model // attn_layers
        assert d_model % attn_layers == 0, "d_model must be divisible by attn_layers"
        self.queries = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.keys = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.values = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.final_proj = nn.Linear(d_model, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        heads = []
        for q, k, v in zip(self.queries, self.keys, self.values):
            Q = q(x)
            K = k(x)
            V = v(x)
            attn_scores = torch.matmul(Q, K.T) / np.sqrt(Q.shape[-1])
            attn_weights = torch.softmax(attn_scores, dim=1)
            attended = torch.matmul(attn_weights, V)
            heads.append(attended)
        attn_out = torch.cat(heads, dim=-1)
        attn_out = self.out_proj(attn_out)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        for layer in self.layers:
            x = self.act(layer(x))
        out = self.final_proj(x)
        return out.squeeze(-1)

# Load model and weights
model = DeepSelfAttention(d_model=10, num_layers=4, attn_layers=10)
if not os.path.exists(WEIGHTS_PATH):
    print(f"weights.json not found at {WEIGHTS_PATH}")
    sys.exit(1)
with open(WEIGHTS_PATH, "r") as f:
    weights = json.load(f)
model.load_state_dict({k: torch.tensor(v) for k, v in weights.items()}, strict=False)
model.eval()

# Choose file (user can specify as argument, else random)
if len(sys.argv) > 1:
    target_file = sys.argv[1]
    if target_file not in files:
        print(f"File {target_file} not found in data folder.")
        sys.exit(1)
else:
    target_file = random.choice(files)
print(f"Predicting for file: {target_file}")

# Load data
path = os.path.join(data_dir, target_file)
df = pd.read_csv(path)
df["PriceChange"] = df["Price"].diff().fillna(0)
df["BuyVolChange"] = df["Buy_Vol"].diff().fillna(0)
df["SellVolChange"] = df["Sell_Vol"].diff().fillna(0)
df["BidPriceChange"] = df["Bid_Price"].diff().fillna(0)
df["AskPriceChange"] = df["Ask_Price"].diff().fillna(0)
data = df[cols].values.astype(np.float32)
N = data.shape[0]

# Predict final price using all previous rows
input_history = [torch.tensor(row, dtype=torch.float32) for row in data[:-1]]
X_input = torch.stack(input_history)
with torch.no_grad():
    pred_price = model(X_input)[-1].item()
real_price = data[-1, 9]
loss = abs(pred_price - real_price)

print(f"\nFile: {target_file}")
print(f"Predicted final price: {pred_price:.4f}")
print(f"Actual final price:    {real_price:.4f}")
print(f"Absolute error:        {loss:.4f}")

# Plot
plt.figure(figsize=(5, 4))
plt.bar(['Predicted', 'Actual'], [pred_price, real_price], color=['red', 'black'], alpha=0.7)
plt.ylabel('Price')
plt.title(f'Predicted vs Actual Final Price\n{target_file}\nLoss: {loss:.4f}')
plt.tight_layout()
plt.show() 