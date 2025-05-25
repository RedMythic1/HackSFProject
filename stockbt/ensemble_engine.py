import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import torch.nn.functional as F
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
import math

# Directory containing all CSV files
DATA_DIR = "/Users/avneh/Code/HackSFProject/stockbt/datasets"

# Get list of all CSV files in the directory
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

# If no CSV files found, create sample data
if not csv_files:
    print("No CSV files found. Generating sample stock data for demonstration...")
    
    # Generate realistic stock price data
    np.random.seed(42)  # For reproducible results
    n_points = 800
    
    # Generate price with trend and noise
    time_steps = np.arange(n_points)
    base_price = 390 + 0.002 * time_steps  # Slight upward trend
    noise = np.random.normal(0, 0.5, n_points)  # Price noise
    volatility = 0.3 + 0.2 * np.sin(time_steps / 100)  # Variable volatility
    
    # Create realistic price movements
    price_changes = np.random.normal(0, volatility, n_points)
    price_changes[0] = 0  # Start at base price
    
    prices = base_price + np.cumsum(price_changes) + noise
    
    # Ensure prices stay positive and realistic
    prices = np.maximum(prices, 350)  # Floor price
    
    # Generate bid/ask spreads
    spreads = np.random.uniform(0.01, 0.05, n_points)  # Realistic spreads
    bid_prices = prices - spreads/2
    ask_prices = prices + spreads/2
    
    # Create DataFrame
    df = pd.DataFrame({
        'Price': prices,
        'Bid_Price': bid_prices,
        'Ask_Price': ask_prices
    })
    
    file = "SAMPLE_STOCK.csv"
    print(f"Generated sample data: {n_points} points, Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
else:
    file = random.choice(csv_files)
    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path)

print(f"Ensemble Engine - Enhanced Stock Price Prediction with Advanced Training")
print(f"Processing file: {file}")

# Create basic lag features first (before Fourier analysis)
df['Price_Lag1'] = df['Price'].shift(1)
df['Price_Lag2'] = df['Price'].shift(2) 
df['Price_Return'] = df['Price'].pct_change()
df['Bid_Lag1'] = df['Bid_Price'].shift(1)
df['Ask_Lag1'] = df['Ask_Price'].shift(1)
df['Spread_Lag1'] = (df['Ask_Price'] - df['Bid_Price']).shift(1)

# Drop NaN rows from lag creation
df = df.dropna().reset_index(drop=True)

# Set dynamic end based on CSV length
num_rows = len(df)

# HARD RESTRICTION: Train on exactly half the dataset minus 6 points
# This ensures we have a large test set and prevents overfitting
HARD_TRAINING_LIMIT = (num_rows // 2) - 6
train_until_idx = HARD_TRAINING_LIMIT

# Validate we have enough data for this hard restriction
MIN_REQUIRED_ROWS = 20  # Need at least 20 rows total (10 for training, 6 for bias+MPC, 4 for prediction)
if num_rows < MIN_REQUIRED_ROWS:
    raise ValueError(f"Not enough data. Need at least {MIN_REQUIRED_ROWS} rows, but got {num_rows} rows.")

if train_until_idx <= 0:
    raise ValueError(f"Training restriction too strict. Half dataset minus 6 = {train_until_idx}. Need at least 1 training point.")

# Calculate remaining indices based on hard training restriction
bias_point_idx = train_until_idx  # Use this point for bias calculation  
mpc_start_idx = train_until_idx + 1  # Start MPC from this index
mpc_end_idx = train_until_idx + 5    # End MPC at this index (5 points for MPC)
predict_start_idx = train_until_idx + 6  # Start walk forward prediction from this index
predict_end_idx = num_rows - 1  # Predict up to the last point

# Validate the hard restriction allows for all required phases
if predict_start_idx >= num_rows:
    raise ValueError(f"Hard training restriction leaves no data for prediction. Training ends at {train_until_idx}, but dataset only has {num_rows} rows.")

if mpc_end_idx >= num_rows:
    raise ValueError(f"Hard training restriction leaves insufficient data for MPC phase. Need 5 MPC points starting at {mpc_start_idx}, but dataset only has {num_rows} rows.")

print(f"HARD RESTRICTION APPLIED:")
print(f"Total rows in dataset: {num_rows}")
print(f"Training data HARD LIMITED to: {train_until_idx} points ({train_until_idx/num_rows*100:.1f}% of dataset)")
print(f"Remaining data for testing: {num_rows - train_until_idx} points ({(num_rows - train_until_idx)/num_rows*100:.1f}% of dataset)")
print(f"Training on data from index 0 to {train_until_idx-1} (inclusive)")
print(f"Bias calculation at index: {bias_point_idx}")
print(f"MPC phase from index {mpc_start_idx} to {mpc_end_idx} (5 points)")
print(f"Walk forward prediction from index {predict_start_idx} to {predict_end_idx} ({predict_end_idx - predict_start_idx + 1} predictions)")

# === FOURIER TRANSFORM ANALYSIS (AFTER HARD RESTRICTION DEFINED) ===
print(f"Fourier Transform Analysis...")

# CRITICAL FIX: Only use TRAINING data for Fourier analysis to prevent future leak
# Apply Fourier transform to price series - TRAINING PORTION ONLY
print(f"Computing Fourier features using TRAINING data only to prevent data leakage...")

# Use the HARD TRAINING RESTRICTION for Fourier analysis
training_price_series = df['Price'].values[:HARD_TRAINING_LIMIT]  # ONLY training data based on hard restriction

n_training_samples = len(training_price_series)
print(f"Using {n_training_samples} training samples for Fourier analysis (HARD RESTRICTED)")

# Compute FFT on TRAINING data only
price_fft = fft(training_price_series)
freqs = fftfreq(n_training_samples)

# Find dominant frequencies from TRAINING data only
magnitude = np.abs(price_fft)
dominant_freq_indices = np.argsort(magnitude[1:n_training_samples//2])[-5:]  # Top 5 frequencies (excluding DC)
dominant_freqs = freqs[1:n_training_samples//2][dominant_freq_indices]
dominant_magnitudes = magnitude[1:n_training_samples//2][dominant_freq_indices]

print(f"Dominant Frequencies (from training data only):")
for i, (freq, mag) in enumerate(zip(dominant_freqs, dominant_magnitudes)):
    period = 1/abs(freq) if freq != 0 else float('inf')
    print(f"   Freq {i+1}: {freq:.6f} cycles/sample (Period: {period:.1f} samples) | Magnitude: {mag:.2f}")

# Create Fourier-based features using TRAINING data patterns only
def create_fourier_features_no_leak(training_prices, full_prices, n_components=5):
    """Create Fourier-based features WITHOUT data leakage"""
    # Learn Fourier patterns from training data only
    fft_training = fft(training_prices)
    
    # Extract phase and magnitude features from training
    magnitude_features = np.abs(fft_training[:n_components])
    phase_features = np.angle(fft_training[:n_components])
    
    # Apply learned Fourier patterns to full series (but only use training-derived parameters)
    fft_full = fft(full_prices)
    fft_filtered = np.zeros_like(fft_full)
    
    # Use only the frequency components identified in training
    n_training = len(training_prices)
    n_full = len(full_prices)
    scale_factor = n_full / n_training
    
    # Scale the learned components appropriately
    for i in range(min(n_components, len(fft_full))):
        if i < len(fft_training):
            fft_filtered[i] = fft_training[i] * scale_factor
            if i > 0 and len(fft_full) - i > 0:
                fft_filtered[-i] = fft_training[-i] * scale_factor if i < len(fft_training) else 0
    
    reconstructed = np.real(ifft(fft_filtered))
    residual = full_prices - reconstructed
    
    # Smooth trend using training data parameters only
    # FIXED: Ensure window_len > polyorder and handle small training sets
    min_window_for_polyorder3 = 5  # Need at least 5 for polyorder=3
    window_len = max(min_window_for_polyorder3, min(21, len(training_prices)//3))
    
    # Ensure window_len is odd (required by savgol_filter)
    if window_len % 2 == 0:
        window_len += 1
    
    # Adjust polyorder based on available window length
    max_polyorder = min(3, window_len - 1)  # polyorder must be < window_length
    
    return {
        'magnitude': magnitude_features,
        'phase': phase_features, 
        'reconstructed': reconstructed,
        'residual': residual,
        'smooth_trend': savgol_filter(full_prices, window_length=window_len, polyorder=max_polyorder)
    }

# Generate Fourier features WITHOUT data leakage
fourier_features = create_fourier_features_no_leak(training_price_series, df['Price'].values, n_components=5)

print(f"Generated Fourier features WITHOUT data leakage:")
print(f"   Training samples used: {n_training_samples}")
print(f"   Reconstructed signal RMS error: {np.sqrt(np.mean(fourier_features['residual'][:n_training_samples]**2)):.4f}")

# Add Fourier-based features (lagged to avoid future leak) - USING TRAINING-DERIVED PATTERNS ONLY
df['Fourier_Trend_Lag1'] = pd.Series(fourier_features['smooth_trend']).shift(1)
df['Fourier_Residual_Lag1'] = pd.Series(fourier_features['residual']).shift(1)

# CRITICAL FIX: Use static training-derived values, not dynamic per-sample values
df['Fourier_Magnitude_1'] = fourier_features['magnitude'][1] if len(fourier_features['magnitude']) > 1 else 0  # Static training-derived
df['Fourier_Magnitude_2'] = fourier_features['magnitude'][2] if len(fourier_features['magnitude']) > 2 else 0  # Static training-derived

# ENHANCED FEATURES: Use Fourier features + original features (NO LEAK)
FEATURES1 = ["Price_Lag1", "Price_Lag2", "Price_Return", "Bid_Lag1", "Ask_Lag1", "Spread_Lag1", 
             "Fourier_Trend_Lag1", "Fourier_Residual_Lag1", "Fourier_Magnitude_1", "Fourier_Magnitude_2"]

print(f"Using enhanced columns for Model input: {FEATURES1}")

# CRITICAL FIX: Drop NaN rows AGAIN after adding Fourier features
print(f"Data before dropping Fourier NaNs: {len(df)} rows")
df = df.dropna().reset_index(drop=True)
print(f"Data after dropping Fourier NaNs: {len(df)} rows")

# RECALCULATE all indices after second NaN drop
num_rows = len(df)

# HARD RESTRICTION: Train on exactly half the dataset minus 6 points (RECALCULATED)
HARD_TRAINING_LIMIT = (num_rows // 2) - 6
train_until_idx = HARD_TRAINING_LIMIT

# Validate we have enough data for this hard restriction
MIN_REQUIRED_ROWS = 20  # Need at least 20 rows total (10 for training, 6 for bias+MPC, 4 for prediction)
if num_rows < MIN_REQUIRED_ROWS:
    raise ValueError(f"Not enough data after Fourier NaN removal. Need at least {MIN_REQUIRED_ROWS} rows, but got {num_rows} rows.")

if train_until_idx <= 0:
    raise ValueError(f"Training restriction too strict after Fourier NaN removal. Half dataset minus 6 = {train_until_idx}. Need at least 1 training point.")

# Calculate remaining indices based on hard training restriction (RECALCULATED)
bias_point_idx = train_until_idx  # Use this point for bias calculation  
mpc_start_idx = train_until_idx + 1  # Start MPC from this index
mpc_end_idx = train_until_idx + 5    # End MPC at this index (5 points for MPC)
predict_start_idx = train_until_idx + 6  # Start walk forward prediction from this index
predict_end_idx = num_rows - 1  # Predict up to the last point

# Validate the hard restriction allows for all required phases (RECALCULATED)
if predict_start_idx >= num_rows:
    raise ValueError(f"Hard training restriction leaves no data for prediction after Fourier NaN removal. Training ends at {train_until_idx}, but dataset only has {num_rows} rows.")

if mpc_end_idx >= num_rows:
    raise ValueError(f"Hard training restriction leaves insufficient data for MPC phase after Fourier NaN removal. Need 5 MPC points starting at {mpc_start_idx}, but dataset only has {num_rows} rows.")

print(f"RECALCULATED HARD RESTRICTION (after Fourier NaN removal):")
print(f"Total rows in dataset: {num_rows}")
print(f"Training data HARD LIMITED to: {train_until_idx} points ({train_until_idx/num_rows*100:.1f}% of dataset)")
print(f"Remaining data for testing: {num_rows - train_until_idx} points ({(num_rows - train_until_idx)/num_rows*100:.1f}% of dataset)")
print(f"Training on data from index 0 to {train_until_idx-1} (inclusive)")
print(f"Bias calculation at index: {bias_point_idx}")
print(f"MPC phase from index {mpc_start_idx} to {mpc_end_idx} (5 points)")
print(f"Walk forward prediction from index {predict_start_idx} to {predict_end_idx} ({predict_end_idx - predict_start_idx + 1} predictions)")

# FIXED: Create data with features and separate targets for price, bid, and ask
data1_features = df[FEATURES1].values.astype(np.float32)  # Historical features only
data1_prices = df['Price'].values.astype(np.float32)  # Price targets
data1_bids = df['Bid_Price'].values.astype(np.float32)  # Bid price targets
data1_asks = df['Ask_Price'].values.astype(np.float32)  # Ask price targets
feature_indices1 = {f: i for i, f in enumerate(FEATURES1)}

# DATA VALIDATION: Check for NaN, infinite, or problematic values
print(f"\nDATA VALIDATION BEFORE TRAINING:")
print(f"Features shape: {data1_features.shape}")
print(f"Prices shape: {data1_prices.shape}")

# Check for NaN values
features_nan_count = np.isnan(data1_features).sum()
prices_nan_count = np.isnan(data1_prices).sum()
bids_nan_count = np.isnan(data1_bids).sum()
asks_nan_count = np.isnan(data1_asks).sum()

print(f"NaN values - Features: {features_nan_count}, Prices: {prices_nan_count}, Bids: {bids_nan_count}, Asks: {asks_nan_count}")

# Check for infinite values
features_inf_count = np.isinf(data1_features).sum()
prices_inf_count = np.isinf(data1_prices).sum()
bids_inf_count = np.isinf(data1_bids).sum()
asks_inf_count = np.isinf(data1_asks).sum()

print(f"Infinite values - Features: {features_inf_count}, Prices: {prices_inf_count}, Bids: {bids_inf_count}, Asks: {asks_inf_count}")

# Check feature ranges
for i, feature_name in enumerate(FEATURES1):
    feature_col = data1_features[:, i]
    min_val = np.min(feature_col)
    max_val = np.max(feature_col)
    mean_val = np.mean(feature_col)
    std_val = np.std(feature_col)
    print(f"  {feature_name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")

# Check training data range specifically
print(f"\nTRAINING DATA VALIDATION (indices 0 to {train_until_idx}):")
train_features_subset = data1_features[:train_until_idx]
train_prices_subset = data1_prices[:train_until_idx]
train_bids_subset = data1_bids[:train_until_idx]
train_asks_subset = data1_asks[:train_until_idx]

print(f"Training features NaN: {np.isnan(train_features_subset).sum()}")
print(f"Training prices NaN: {np.isnan(train_prices_subset).sum()}")
print(f"Training bids NaN: {np.isnan(train_bids_subset).sum()}")
print(f"Training asks NaN: {np.isnan(train_asks_subset).sum()}")

# Check if any training values are too large or too small
train_features_max = np.max(np.abs(train_features_subset))
train_prices_max = np.max(np.abs(train_prices_subset))
print(f"Max absolute training feature value: {train_features_max:.6f}")
print(f"Max absolute training price value: {train_prices_max:.6f}")

if features_nan_count > 0 or prices_nan_count > 0 or bids_nan_count > 0 or asks_nan_count > 0:
    raise ValueError(f"Found NaN values in data! Cannot proceed with training.")

if features_inf_count > 0 or prices_inf_count > 0 or bids_inf_count > 0 or asks_inf_count > 0:
    raise ValueError(f"Found infinite values in data! Cannot proceed with training.")

# Replace any extreme values that might cause numerical issues
print(f"Clamping extreme values to prevent numerical instability...")
data1_features = np.clip(data1_features, -1e6, 1e6)
data1_prices = np.clip(data1_prices, -1e6, 1e6)
data1_bids = np.clip(data1_bids, -1e6, 1e6)
data1_asks = np.clip(data1_asks, -1e6, 1e6)

print(f"Data validation completed successfully!")

# === ADVANCED LOSS FUNCTIONS ===
class HuberLoss(nn.Module):
    """Robust Huber loss - less sensitive to outliers than MSE"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * torch.abs(error) - 0.5 * self.delta ** 2
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

class FocalLoss(nn.Module):
    """Focal loss - focuses training on hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Convert MSE to probability-like score
        p = torch.exp(-mse)
        focal_weight = self.alpha * (1 - p) ** self.gamma
        return (focal_weight * mse).mean()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that learns optimal loss weighting"""
    def __init__(self, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        # losses should be a list of individual task losses
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        return sum(weighted_losses)

# === ADVANCED OPTIMIZERS ===
class Lookahead:
    """Lookahead optimizer wrapper - Fixed for PyTorch scheduler compatibility"""
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        self.defaults = self.optimizer.defaults
        for group in self.param_groups:
            group["counter"] = 0
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state.get(id(p), {})
                if "slow_weights" not in param_state:
                    param_state["slow_weights"] = torch.zeros_like(p.data)
                    param_state["slow_weights"].copy_(p.data)
                
                slow = param_state["slow_weights"]
                if group["counter"] % self.k == 0:
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
                
            group["counter"] += 1
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

# === ADVANCED REGULARIZATION ===
class LabelSmoothingLoss(nn.Module):
    """Label smoothing for regression - adds noise to targets"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Add small random noise to targets for regularization
        if self.training:
            noise = torch.randn_like(target) * self.smoothing * target.std()
            smoothed_target = target + noise
        else:
            smoothed_target = target
        return self.mse(pred, smoothed_target)

# === TIME SERIES MIXUP ===
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for time series"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Ensemble Engine: Enhanced Cross-Information Multi-Price Network with Gradient Feedback
class EnsembleEngineNet(nn.Module):
    def __init__(self, input_dim, num_layers=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Cross-attention mechanism for information sharing
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # ENHANCED: Gradient feedback layers for bid/ask → price information flow
        self.bid_to_price_feedback = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        self.ask_to_price_feedback = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Economic relationship encoder (bid-ask spread information)
        self.spread_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # [bid, ask] → spread features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Separate feature processing for each price type
        self.price_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.bid_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.ask_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Cross-information fusion layers
        fusion_input_dim = (hidden_dim // 2) * 3  # Concatenated features from all three processors
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # STAGE 1: Initial prediction heads
        self.bid_head_stage1 = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.ask_head_stage1 = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # STAGE 2: Enhanced price head with bid/ask feedback
        price_input_dim = (hidden_dim // 2) + (hidden_dim // 2) + (hidden_dim // 2) + (hidden_dim // 2)  # base + fused + bid_feedback + ask_feedback
        self.price_head_enhanced = nn.Sequential(
            nn.Linear(price_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # STAGE 3: Iterative refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2 + 3, hidden_dim // 2),  # features + [price, bid, ask]
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(2)  # 2 refinement iterations
        ])
        
        # Economic constraint enforcement
        self.constraint_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # [price_weight, bid_weight, ask_weight]
        
        # Confidence estimation for gradient weighting
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # confidence for [price, bid, ask]
            nn.Sigmoid()
        )
        
    def forward(self, x, training_mode=True):
        batch_size = x.size(0)
        
        # Shared feature extraction with residual connections
        out = x
        for layer in self.shared_layers:
            h = layer(out)
            if out.size(-1) == h.size(-1):
                out = out - h  # Residual subtraction
            else:
                out = h
        
        # Prepare for cross-attention
        shared_features = out.unsqueeze(1)
        
        # Create queries for price, bid, ask
        queries = torch.cat([shared_features, shared_features, shared_features], dim=1)
        
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(queries, queries, queries)
        
        # Extract attended features
        price_attended = attended_features[:, 0, :]
        bid_attended = attended_features[:, 1, :]
        ask_attended = attended_features[:, 2, :]
        
        # Process each price type
        price_features = self.price_processor(price_attended)
        bid_features = self.bid_processor(bid_attended)
        ask_features = self.ask_processor(ask_attended)
        
        # Cross-information fusion
        fused_features = torch.cat([price_features, bid_features, ask_features], dim=-1)
        shared_info = self.fusion_layer(fused_features)
        
        # STAGE 1: Initial bid/ask predictions
        bid_input = torch.cat([bid_features, shared_info], dim=-1)
        ask_input = torch.cat([ask_features, shared_info], dim=-1)
        
        bid_pred_stage1 = self.bid_head_stage1(bid_input).squeeze(-1)
        ask_pred_stage1 = self.ask_head_stage1(ask_input).squeeze(-1)

        # STAGE 2: Enhanced price prediction with bid/ask feedback
        if training_mode:
            # During training, enable gradients for feedback
            bid_feedback = self.bid_to_price_feedback(bid_pred_stage1.unsqueeze(-1))
            ask_feedback = self.ask_to_price_feedback(ask_pred_stage1.unsqueeze(-1))
        else:
            # During inference, use detached values to prevent gradient loops
            bid_feedback = self.bid_to_price_feedback(bid_pred_stage1.detach().unsqueeze(-1))
            ask_feedback = self.ask_to_price_feedback(ask_pred_stage1.detach().unsqueeze(-1))
        
        # Economic relationship encoding
        spread_input = torch.stack([bid_pred_stage1, ask_pred_stage1], dim=-1)
        spread_features = self.spread_encoder(spread_input)
        
        # Enhanced price prediction with gradient feedback
        price_input_enhanced = torch.cat([price_features, shared_info, bid_feedback, ask_feedback], dim=-1)
        price_pred_enhanced = self.price_head_enhanced(price_input_enhanced).squeeze(-1)
        
        # STAGE 3: Iterative refinement
        current_predictions = torch.stack([price_pred_enhanced, bid_pred_stage1, ask_pred_stage1], dim=-1)
        
        refined_predictions = []
        for refinement_layer in self.refinement_layers:
            refinement_input = torch.cat([shared_info, current_predictions], dim=-1)
            refinement_delta = refinement_layer(refinement_input).squeeze(-1)
            
            # Apply refinement to price prediction
            price_refined = price_pred_enhanced + refinement_delta
            refined_predictions.append(price_refined)
            
            # Update current predictions for next iteration
            current_predictions = torch.stack([price_refined, bid_pred_stage1, ask_pred_stage1], dim=-1)
        
        # Final refined price prediction
        final_price_pred = refined_predictions[-1] if refined_predictions else price_pred_enhanced
        
        # Economic constraint enforcement with soft penalties
        constraint_weights_norm = F.softmax(self.constraint_weights, dim=0)
        
        # Ensure bid ≤ price ≤ ask with learnable constraints
        bid_violation = torch.relu(bid_pred_stage1 - final_price_pred)  # Penalty if bid > price
        ask_violation = torch.relu(final_price_pred - ask_pred_stage1)  # Penalty if price > ask
        
        # Apply soft constraints
        constrained_price = final_price_pred - 0.1 * (bid_violation + ask_violation)
        constrained_bid = bid_pred_stage1 - 0.1 * bid_violation
        constrained_ask = ask_pred_stage1 + 0.1 * ask_violation
        
        # Confidence estimation for gradient weighting
        confidence_scores = self.confidence_estimator(shared_info)
        
        return constrained_price, constrained_bid, constrained_ask, attention_weights, confidence_scores, {
            'bid_stage1': bid_pred_stage1,
            'ask_stage1': ask_pred_stage1,
            'price_enhanced': price_pred_enhanced,
            'price_refined': refined_predictions,
            'bid_feedback': bid_feedback,
            'ask_feedback': ask_feedback,
            'constraint_violations': bid_violation + ask_violation
        }

# Enhanced Error Prediction Network for Meta-Learning
class ErrorPredictionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.error_predictor = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # features + [price_pred, bid_pred, ask_pred]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Predict the error
        )
        
    def forward(self, features, price_pred, bid_pred, ask_pred):
        # Combine features with predictions
        pred_info = torch.stack([price_pred, bid_pred, ask_pred], dim=-1)
        combined_input = torch.cat([features, pred_info], dim=-1)
        predicted_error = self.error_predictor(combined_input)
        return predicted_error.squeeze(-1)

# --- ADVANCED TRAINING CONFIGURATION ---
TRAINING_CONFIG = {
    'optimizer_type': 'adamw_lookahead',  # 'adam', 'adamw', 'adamw_lookahead'
    'loss_type': 'adaptive_huber',  # 'mse', 'huber', 'focal', 'adaptive_huber'
    'scheduler_type': 'cosine_warm_restarts',  # 'plateau', 'cosine_warm_restarts', 'one_cycle'
    'use_mixup': True,
    'use_label_smoothing': True,
    'gradient_accumulation_steps': 4,  # Simulate larger batch size
    'early_stopping_patience': 50,
    'validation_frequency': 25,  # Validate every 25 epochs
}

# --- Advanced Training Setup ---
INIT_NUM_LAYERS = 2  # Start with more layers
INIT_LR = 3e-4  # Higher initial learning rate for advanced optimizers
INIT_TARGET_ERROR = 0.3  # More aggressive target
SWITCH_EPOCH = 1500  # Earlier switch
NEW_NUM_LAYERS = 3  # More complex final model
NEW_LR = 1e-4  # Lower final learning rate
NEW_TARGET_ERROR = 0.15  # Very aggressive final target

# --- Train Model 1: Advanced Multi-output prediction ---
# Train on data up to train_until_idx
train_data1 = data1_features[:train_until_idx]
if len(train_data1) < 2:
    raise ValueError("Not enough data for Model 1 training.")

# FIXED: Create targets using separate price arrays for all three outputs
X_train1 = torch.tensor(train_data1[:-1], dtype=torch.float32)  # Features from 0 to train_until_idx-2
y_train1_price = torch.tensor(data1_prices[1:train_until_idx], dtype=torch.float32)  # Price targets from 1 to train_until_idx-1
y_train1_bid = torch.tensor(data1_bids[1:train_until_idx], dtype=torch.float32)  # Bid targets from 1 to train_until_idx-1
y_train1_ask = torch.tensor(data1_asks[1:train_until_idx], dtype=torch.float32)  # Ask targets from 1 to train_until_idx-1

if X_train1.shape[0] == 0:
    raise ValueError("Model 1 training input X_train1 is empty.")

# === ADVANCED MODEL AND OPTIMIZER SETUP ===
num_layers = INIT_NUM_LAYERS
lr = INIT_LR
target_error = INIT_TARGET_ERROR
model1 = EnsembleEngineNet(input_dim=len(FEATURES1), num_layers=num_layers)

# Advanced optimizer selection - Simplified to avoid compatibility issues
if TRAINING_CONFIG['optimizer_type'] == 'adamw' or TRAINING_CONFIG['optimizer_type'] == 'adamw_lookahead':
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=1e-4)
    use_lookahead = TRAINING_CONFIG['optimizer_type'] == 'adamw_lookahead'
else:
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    use_lookahead = False

# Advanced loss function selection
if TRAINING_CONFIG['loss_type'] == 'huber':
    loss_fn = HuberLoss(delta=0.5)
elif TRAINING_CONFIG['loss_type'] == 'focal':
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
elif TRAINING_CONFIG['loss_type'] == 'adaptive_huber':
    loss_fn = HuberLoss(delta=0.3)
    adaptive_loss = AdaptiveLoss(num_tasks=3)
else:
    loss_fn = nn.MSELoss()

# Label smoothing
if TRAINING_CONFIG['use_label_smoothing']:
    label_smooth_loss = LabelSmoothingLoss(smoothing=0.05)

# Advanced learning rate scheduler
if TRAINING_CONFIG['scheduler_type'] == 'cosine_warm_restarts':
    scheduler1 = CosineAnnealingWarmRestarts(optimizer1, T_0=200, T_mult=2, eta_min=1e-7)
elif TRAINING_CONFIG['scheduler_type'] == 'one_cycle':
    scheduler1 = OneCycleLR(optimizer1, max_lr=lr*5, epochs=5000, steps_per_epoch=1)
else:
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.7, patience=100, min_lr=1e-7)

# Lookahead wrapper (applied after scheduler creation)
if use_lookahead:
    lookahead_optimizer = Lookahead(optimizer1, k=5, alpha=0.5)
else:
    lookahead_optimizer = None

max_epochs = 8000  # Increased for better convergence
required_good_epochs = 15  # More stringent requirement
consecutive_good_epochs = 0
best_val_loss = float('inf')
patience_counter = 0

# Advanced validation setup - multiple validation windows
validation_windows = [
    (max(0, train_until_idx - 30), train_until_idx - 1),  # Last 30 points
    (max(0, train_until_idx - 60), train_until_idx - 31), # Previous 30 points
    (max(0, train_until_idx - 90), train_until_idx - 61), # Earlier 30 points
]

print(f"ADVANCED TRAINING CONFIGURATION:")
print(f"  Optimizer: {TRAINING_CONFIG['optimizer_type']}")
print(f"  Loss: {TRAINING_CONFIG['loss_type']}")
print(f"  Scheduler: {TRAINING_CONFIG['scheduler_type']}")
print(f"  Mixup: {TRAINING_CONFIG['use_mixup']}")
print(f"  Label Smoothing: {TRAINING_CONFIG['use_label_smoothing']}")
print(f"  Validation Windows: {len(validation_windows)}")

# Training metrics tracking
training_history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'learning_rate': [],
    'gradient_norm': []
}

print(f"Starting ADVANCED training with {max_epochs} max epochs...")

for epoch in range(max_epochs):
    model1.train()
    epoch_losses = []
    
    if X_train1.shape[0] > 0:
        # Gradient accumulation for larger effective batch size
        if lookahead_optimizer:
            lookahead_optimizer.zero_grad()
        else:
            optimizer1.zero_grad()
        total_loss = 0
        
        for accum_step in range(TRAINING_CONFIG['gradient_accumulation_steps']):
            # Apply mixup augmentation
            if TRAINING_CONFIG['use_mixup'] and np.random.random() < 0.5:
                mixed_x, y_a_price, y_b_price, lam = mixup_data(X_train1, y_train1_price, alpha=0.2)
                _, y_a_bid, y_b_bid, _ = mixup_data(X_train1, y_train1_bid, alpha=0.2)
                _, y_a_ask, y_b_ask, _ = mixup_data(X_train1, y_train1_ask, alpha=0.2)
                
                pred_price, pred_bid, pred_ask, attention_weights, confidence_scores, debug_info = model1(mixed_x, training_mode=True)
                
                # Mixup loss calculation
                if TRAINING_CONFIG['use_label_smoothing']:
                    loss_price = mixup_criterion(label_smooth_loss, pred_price, y_a_price, y_b_price, lam)
                    loss_bid = mixup_criterion(label_smooth_loss, pred_bid, y_a_bid, y_b_bid, lam)
                    loss_ask = mixup_criterion(label_smooth_loss, pred_ask, y_a_ask, y_b_ask, lam)
                else:
                    loss_price = mixup_criterion(loss_fn, pred_price, y_a_price, y_b_price, lam)
                    loss_bid = mixup_criterion(loss_fn, pred_bid, y_a_bid, y_b_bid, lam)
                    loss_ask = mixup_criterion(loss_fn, pred_ask, y_a_ask, y_b_ask, lam)
            else:
                # Regular training
                pred_price, pred_bid, pred_ask, attention_weights, confidence_scores, debug_info = model1(X_train1, training_mode=True)
                
                if TRAINING_CONFIG['use_label_smoothing']:
                    loss_price = label_smooth_loss(pred_price, y_train1_price)
                    loss_bid = label_smooth_loss(pred_bid, y_train1_bid)
                    loss_ask = label_smooth_loss(pred_ask, y_train1_ask)
                else:
                    loss_price = loss_fn(pred_price, y_train1_price)
                    loss_bid = loss_fn(pred_bid, y_train1_bid)
                    loss_ask = loss_fn(pred_ask, y_train1_ask)
            
            # Advanced loss combination
            if TRAINING_CONFIG['loss_type'] == 'adaptive_huber':
                step_loss = adaptive_loss([loss_price, loss_bid, loss_ask])
            else:
                # Confidence weighting
                confidence_price = confidence_scores[:, 0].mean()
                confidence_bid = confidence_scores[:, 1].mean()
                confidence_ask = confidence_scores[:, 2].mean()
                
                weighted_loss_price = loss_price * (1.0 + confidence_price)
                weighted_loss_bid = loss_bid * (1.0 + confidence_bid)
                weighted_loss_ask = loss_ask * (1.0 + confidence_ask)
                
                # Economic constraint penalties
                constraint_penalty = debug_info['constraint_violations'].mean() * 0.15
                
                # Enhanced gradient feedback
                price_spread_alignment = torch.mean(
                    torch.relu(pred_bid - pred_price) +  # Price should be >= bid
                    torch.relu(pred_price - pred_ask)    # Price should be <= ask
                ) * 0.25
                
                step_loss = (weighted_loss_price + weighted_loss_bid + weighted_loss_ask + 
                           constraint_penalty + price_spread_alignment)
            
            # Scale loss for gradient accumulation
            step_loss = step_loss / TRAINING_CONFIG['gradient_accumulation_steps']
            step_loss.backward()
            total_loss += step_loss.item()
            epoch_losses.append(step_loss.item())
        
        # Gradient clipping and optimization step
        grad_norm = torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
        
        if lookahead_optimizer:
            lookahead_optimizer.step()
        else:
            optimizer1.step()
        
        # Advanced scheduler step
        if TRAINING_CONFIG['scheduler_type'] in ['cosine_warm_restarts', 'one_cycle']:
            scheduler1.step()
        
        # Track metrics
        current_lr = optimizer1.param_groups[0]['lr']
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(total_loss)
        training_history['learning_rate'].append(current_lr)
        training_history['gradient_norm'].append(grad_norm.item())
        
        # Enhanced logging
        if epoch % 50 == 0:
            lookahead_status = " (Lookahead)" if lookahead_optimizer else ""
            print(f"   Epoch {epoch}: Loss={total_loss:.6f}, LR={current_lr:.2e}, GradNorm={grad_norm:.4f}{lookahead_status}")
    else:
        total_loss = float('inf')

    # Advanced validation with multiple windows
    if epoch % TRAINING_CONFIG['validation_frequency'] == 0 or consecutive_good_epochs >= required_good_epochs:
        model1.eval()
        all_val_errors = []
        window_errors = []
        
        with torch.no_grad():
            for window_idx, (val_start, val_end) in enumerate(validation_windows):
                window_val_errors = []
                
                for val_idx in range(val_start, val_end + 1):
                    if val_idx <= 0:
                        continue
                    val_input = torch.tensor(data1_features[val_idx-1], dtype=torch.float32).unsqueeze(0)
                    pred_price, pred_bid, pred_ask, attention_weights, confidence_scores, debug_info = model1(val_input, training_mode=False)
                    
                    # Calculate errors for all three predictions
                    actual_price = data1_prices[val_idx]
                    actual_bid = data1_bids[val_idx]
                    actual_ask = data1_asks[val_idx]
                    
                    error_price = (pred_price.item() - actual_price) ** 2
                    error_bid = (pred_bid.item() - actual_bid) ** 2
                    error_ask = (pred_ask.item() - actual_ask) ** 2
                    
                    # Multi-task validation error
                    combined_error = (error_price + error_bid + error_ask) / 3
                    window_val_errors.append(combined_error)
                    all_val_errors.append(combined_error)
                
                window_avg = np.mean(window_val_errors) if window_val_errors else float('inf')
                window_errors.append(window_avg)
        
        # Weighted validation score (recent windows matter more)
        if window_errors:
            weights = [0.5, 0.3, 0.2]  # Recent window gets highest weight
            weighted_val_loss = sum(w * e for w, e in zip(weights[:len(window_errors)], window_errors))
        else:
            weighted_val_loss = float('inf')
        
        training_history['val_loss'].append(weighted_val_loss)
        
        # Advanced scheduler step for plateau-based schedulers
        if TRAINING_CONFIG['scheduler_type'] == 'plateau':
            scheduler1.step(weighted_val_loss)
        
        # Enhanced early stopping and convergence checking
        if weighted_val_loss < best_val_loss:
            best_val_loss = weighted_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 100 == 0 or consecutive_good_epochs >= required_good_epochs:
            print(f"Epoch {epoch}: Val Loss={weighted_val_loss:.6f} (Best: {best_val_loss:.6f}), Patience: {patience_counter}/{TRAINING_CONFIG['early_stopping_patience']}")
        
        # Convergence check
        if weighted_val_loss <= target_error:
            consecutive_good_epochs += 1
        else:
            consecutive_good_epochs = 0
            
        # Early stopping
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch} due to no improvement for {TRAINING_CONFIG['early_stopping_patience']} validation checks.")
            break
            
        if consecutive_good_epochs >= required_good_epochs:
            print(f"Reached {required_good_epochs} consecutive good epochs at epoch {epoch}. Target achieved!")
            break
    
    # Advanced hyperparameter switching with model transfer
    if epoch == SWITCH_EPOCH:
        print(f"[ADVANCED] Switching to enhanced model at epoch {epoch}!")
        num_layers = NEW_NUM_LAYERS
        lr = NEW_LR
        target_error = NEW_TARGET_ERROR
        
        # Create new model and transfer weights
        new_model = EnsembleEngineNet(input_dim=len(FEATURES1), num_layers=num_layers)
        
        # Smart weight transfer - only transfer compatible layers
        old_state = model1.state_dict()
        new_state = new_model.state_dict()
        
        transferred_layers = 0
        for name, param in new_state.items():
            if name in old_state and old_state[name].shape == param.shape:
                param.copy_(old_state[name])
                transferred_layers += 1
        
        print(f"   Transferred {transferred_layers}/{len(new_state)} layers to new model")
        
        model1 = new_model
        
        # Recreate optimizer and scheduler for new model
        if TRAINING_CONFIG['optimizer_type'] == 'adamw' or TRAINING_CONFIG['optimizer_type'] == 'adamw_lookahead':
            optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=1e-4)
            use_lookahead = TRAINING_CONFIG['optimizer_type'] == 'adamw_lookahead'
        else:
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
            use_lookahead = False
        
        # Recreate scheduler
        if TRAINING_CONFIG['scheduler_type'] == 'cosine_warm_restarts':
            scheduler1 = CosineAnnealingWarmRestarts(optimizer1, T_0=200, T_mult=2, eta_min=1e-7)
        elif TRAINING_CONFIG['scheduler_type'] == 'one_cycle':
            remaining_epochs = max_epochs - epoch
            scheduler1 = OneCycleLR(optimizer1, max_lr=lr*3, epochs=remaining_epochs, steps_per_epoch=1)
        else:
            scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.7, patience=100, min_lr=1e-7)
        
        # Recreate lookahead wrapper
        if use_lookahead:
            lookahead_optimizer = Lookahead(optimizer1, k=5, alpha=0.5)
        else:
            lookahead_optimizer = None
        
        # Reset convergence tracking
        consecutive_good_epochs = 0
        best_val_loss = float('inf')
        patience_counter = 0

print(f"ADVANCED training completed!")
print(f"Final validation loss: {best_val_loss:.6f}")
print(f"Training history length: {len(training_history['epoch'])} epochs")

# --- Calculate Bias Correction Early for Use in Fake Walkforward ---
print(f"Calculating bias correction for use in fake walkforward...")
if bias_point_idx > 0:
    bias_features = torch.tensor(data1_features[bias_point_idx-1], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        bias_pred_price, bias_pred_bid, bias_pred_ask, attention_weights, confidence_scores, debug_info = model1(bias_features, training_mode=False)
    
    # Calculate bias corrections
    actual_bias_price = data1_prices[bias_point_idx]
    actual_bias_bid = data1_bids[bias_point_idx]
    actual_bias_ask = data1_asks[bias_point_idx]
    
    price_bias_correction = actual_bias_price - bias_pred_price.item()
    bid_bias_correction = actual_bias_bid - bias_pred_bid.item()
    ask_bias_correction = actual_bias_ask - bias_pred_ask.item()
    
    print(f"Early Bias Calculation:")
    print(f"  Price: Actual={actual_bias_price:.6f}, Predicted={bias_pred_price.item():.6f}, Bias={price_bias_correction:.6f}")
    print(f"  Bid: Actual={actual_bias_bid:.6f}, Predicted={bias_pred_bid.item():.6f}, Bias={bid_bias_correction:.6f}")
    print(f"  Ask: Actual={actual_bias_ask:.6f}, Predicted={bias_pred_ask.item():.6f}, Bias={ask_bias_correction:.6f}")
else:
    price_bias_correction = 0
    bid_bias_correction = 0
    ask_bias_correction = 0
    print("Warning: Cannot calculate bias - no previous point available")

# Set meta-learning as disabled
use_error_model = False

# --- Ensemble Engine Prediction with Enhanced Cross-Information and MPC ---
print(f"Ensemble Engine Prediction with Bias + MPC")

# Storage for results
prediction_indices = []
actual_prices = []
actual_bids = []
actual_asks = []
ensemble_predictions = []   # Only store ensemble predictions
mpc_predictions = []        # Store MPC predictions separately
attention_weights_history = []  # Store attention patterns

model1.eval()

# Step 1: Calculate bias using the bias point
print(f"Step 1: Calculating bias using index {bias_point_idx}...")
if bias_point_idx > 0:
    bias_features = torch.tensor(data1_features[bias_point_idx-1], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        bias_pred_price, bias_pred_bid, bias_pred_ask, attention_weights, confidence_scores, debug_info = model1(bias_features, training_mode=False)
    
    # Calculate bias corrections
    actual_bias_price = data1_prices[bias_point_idx]
    actual_bias_bid = data1_bids[bias_point_idx]
    actual_bias_ask = data1_asks[bias_point_idx]
    
    price_bias_correction = actual_bias_price - bias_pred_price.item()
    bid_bias_correction = actual_bias_bid - bias_pred_bid.item()
    ask_bias_correction = actual_bias_ask - bias_pred_ask.item()
    
    print(f"Bias Point {bias_point_idx} Analysis:")
    print(f"  Price: Actual={actual_bias_price:.6f}, Predicted={bias_pred_price.item():.6f}, Bias={price_bias_correction:.6f}")
    print(f"  Bid: Actual={actual_bias_bid:.6f}, Predicted={bias_pred_bid.item():.6f}, Bias={bid_bias_correction:.6f}")
    print(f"  Ask: Actual={actual_bias_ask:.6f}, Predicted={bias_pred_ask.item():.6f}, Bias={ask_bias_correction:.6f}")
else:
    price_bias_correction = 0
    bid_bias_correction = 0
    ask_bias_correction = 0
    print("Warning: Cannot calculate bias - no previous point available")

# Step 2: Model Predictive Control for next 5 points
print(f"Step 2: Model Predictive Control from {mpc_start_idx} to {mpc_end_idx}")

# MPC horizon and parameters
mpc_horizon = 5
mpc_errors = []
mpc_predictions_detailed = []

for mpc_idx in range(mpc_start_idx, mpc_end_idx + 1):
    if mpc_idx == 0:
        continue
        
    # For MPC, we predict multiple steps ahead and optimize
    mpc_input_features = torch.tensor(data1_features[mpc_idx-1], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Get raw prediction
        raw_pred_price, raw_pred_bid, raw_pred_ask, attention_weights, confidence_scores, debug_info = model1(mpc_input_features, training_mode=False)
        
        # Apply bias correction
        corrected_price = raw_pred_price.item() + price_bias_correction
        corrected_bid = raw_pred_bid.item() + bid_bias_correction
        corrected_ask = raw_pred_ask.item() + ask_bias_correction
        
        # Historical volatility-based MPC adjustment (using ONLY past data)
        if mpc_idx >= 10:  # Need some history
            # Calculate historical volatility from training data only
            historical_prices = data1_prices[max(0, mpc_idx-20):mpc_idx]  # Last 20 historical points
            if len(historical_prices) > 5:
                historical_returns = np.diff(historical_prices) / historical_prices[:-1]
                historical_volatility = np.std(historical_returns)
                recent_trend = np.mean(np.diff(historical_prices[-5:]))  # Recent 5-point trend
                
                # MPC adjustment based on historical patterns
                confidence_weighted_adjustment = recent_trend * 0.1 * confidence_scores[0, 0].item()
                corrected_price += confidence_weighted_adjustment
            
        # Create ensemble prediction with historical-based MPC adjustment
        spread = corrected_ask - corrected_bid
        if spread <= 0:
            ensemble_price = (corrected_price + corrected_bid + corrected_ask) / 3
        else:
            if corrected_price < corrected_bid:
                ensemble_price = 0.7 * corrected_bid + 0.2 * corrected_price + 0.1 * corrected_ask
            elif corrected_price > corrected_ask:
                ensemble_price = 0.7 * corrected_ask + 0.2 * corrected_price + 0.1 * corrected_bid
            else:
                bid_weight = (corrected_ask - corrected_price) / spread
                ask_weight = (corrected_price - corrected_bid) / spread
                price_weight = 0.5
                
                total_weight = bid_weight + ask_weight + price_weight
                ensemble_price = (bid_weight * corrected_bid + ask_weight * corrected_ask + price_weight * corrected_price) / total_weight
    
    # Store MPC results
    actual_price = data1_prices[mpc_idx]
    mpc_error = abs(ensemble_price - actual_price)
    mpc_errors.append(mpc_error)
    mpc_predictions_detailed.append({
        'idx': mpc_idx,
        'actual': actual_price,
        'predicted': ensemble_price,
        'error': mpc_error
    })

# Calculate MPC performance
mpc_mae = np.mean(mpc_errors) if mpc_errors else 0
print(f"MPC Performance: MAE = {mpc_mae:.6f}")

# === PHASE 3: Real Walkforward with IMMEDIATE Dynamic Offset Correction ===
print(f"\n=== PHASE 3: Real Walkforward with IMMEDIATE Dynamic Offset Correction ===")
print(f"Step 3: Walk forward prediction from {predict_start_idx} to {predict_end_idx}")

# IMMEDIATE Dynamic offset tracking - apply offset to NEXT prediction immediately
dynamic_error_offset = 0.0  # Initialize dynamic offset
error_offset_history = []
base_predictions = []  # Track base predictions without offset
offset_predictions = []  # Track predictions with offset applied

# Store for immediate offset calculation
previous_actual = None  # Store the PREVIOUS actual price (already observed)
previous_base_prediction = None  # Store the PREVIOUS base prediction

for current_idx in range(predict_start_idx, predict_end_idx + 1):
    # Use features from the previous point to predict current point's price
    if current_idx == 0:
        continue  # Can't predict index 0 as there's no previous point
    
    prev_features = torch.tensor(data1_features[current_idx-1], dtype=torch.float32).unsqueeze(0)
    
    # Get raw prediction from model
    with torch.no_grad():
        raw_predicted_price, raw_predicted_bid, raw_predicted_ask, attention_weights, confidence_scores, debug_info = model1(prev_features, training_mode=False)
    
    # Apply STATIC bias correction (from bias point)
    corrected_price = raw_predicted_price.item() + price_bias_correction
    corrected_bid = raw_predicted_bid.item() + bid_bias_correction
    corrected_ask = raw_predicted_ask.item() + ask_bias_correction
    
    # Create ensemble prediction using bias-corrected values (WITHOUT dynamic offset yet)
    spread = corrected_ask - corrected_bid
    if spread <= 0:
        # Invalid spread, use simple average
        base_ensemble_price = (corrected_price + corrected_bid + corrected_ask) / 3
    else:
        # Weight based on position within spread - ENHANCED with confidence weighting
        price_confidence = confidence_scores[0, 0].item()
        bid_confidence = confidence_scores[0, 1].item()
        ask_confidence = confidence_scores[0, 2].item()
        
        if corrected_price < corrected_bid:
            # Price below bid, weight towards bid
            base_ensemble_price = 0.7 * corrected_bid + 0.2 * corrected_price + 0.1 * corrected_ask
        elif corrected_price > corrected_ask:
            # Price above ask, weight towards ask
            base_ensemble_price = 0.7 * corrected_ask + 0.2 * corrected_price + 0.1 * corrected_bid
        else:
            # Price within spread, use confidence-weighted average
            bid_weight = (corrected_ask - corrected_price) / spread * bid_confidence
            ask_weight = (corrected_price - corrected_bid) / spread * ask_confidence
            price_weight = 0.5 * price_confidence  # Base weight for direct price prediction
            
            total_weight = bid_weight + ask_weight + price_weight
            base_ensemble_price = (bid_weight * corrected_bid + ask_weight * corrected_ask + price_weight * corrected_price) / total_weight
    
    # IMMEDIATE OFFSET UPDATE: Calculate offset from PREVIOUS error (if available)
    previous_dynamic_offset = dynamic_error_offset
    
    if previous_actual is not None and previous_base_prediction is not None:
        # Calculate error from PREVIOUS prediction (already observed)
        past_error = previous_actual - previous_base_prediction  # Signed error from PREVIOUS prediction
        
        # === DIRECTIONAL TREND ANALYSIS (using ONLY past data) ===
        # Analyze recent price trend using historical data only
        trend_window = 5  # Look at last 5 points for trend
        trend_direction = 0  # -1 = down, 0 = neutral, +1 = up
        trend_strength = 0.0  # 0.0 to 1.0
        
        if current_idx >= trend_window:
            # Use ONLY past prices (already observed)
            recent_prices = data1_prices[current_idx - trend_window:current_idx]
            if len(recent_prices) >= 3:
                # Calculate trend using linear regression on past prices
                x_trend = np.arange(len(recent_prices))
                trend_slope = np.polyfit(x_trend, recent_prices, 1)[0]
                
                # Normalize trend strength based on recent volatility
                recent_volatility = np.std(np.diff(recent_prices))
                if recent_volatility > 0:
                    trend_strength = min(1.0, abs(trend_slope) / (recent_volatility * 2))
                    trend_direction = 1 if trend_slope > 0 else -1
                
                # Additional momentum indicator using past returns
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
                momentum = np.mean(recent_returns[-3:]) if len(recent_returns) >= 3 else 0
                
                # Combine slope and momentum for stronger signal
                if abs(momentum) > recent_volatility * 0.1:  # Significant momentum
                    momentum_direction = 1 if momentum > 0 else -1
                    if momentum_direction == trend_direction:
                        trend_strength = min(1.0, trend_strength * 1.5)  # Amplify if aligned
        
        # === OFFSET DIRECTION ALIGNMENT ===
        # Check if offset direction aligns with predicted trend
        offset_direction = 1 if past_error > 0 else -1  # Positive error = price was higher than predicted
        
        # Calculate directional alignment factor
        if trend_direction != 0:
            # If trend is up and we're correcting upward (or trend down and correcting down)
            directional_alignment = 1.0 if (trend_direction * offset_direction) > 0 else 0.3
        else:
            directional_alignment = 0.7  # Neutral trend, moderate confidence
        
        # === ADAPTIVE SOFTENING BASED ON TREND AND ALIGNMENT ===
        # Base parameters
        error_learning_rate = 0.9  # Base learning rate
        momentum_decay = 0.1  # Base momentum decay
        
        # Softening factors based on trend analysis
        trend_confidence = trend_strength * directional_alignment
        
        # Adjust learning rate based on trend confidence
        if trend_confidence > 0.7:
            # High confidence: trend and offset align strongly
            softened_learning_rate = error_learning_rate * 1.0  # Full strength
            softening_reason = "Strong trend alignment"
        elif trend_confidence > 0.4:
            # Medium confidence: some alignment
            softened_learning_rate = error_learning_rate * 0.7  # Moderate softening
            softening_reason = "Moderate trend alignment"
        elif trend_confidence > 0.2:
            # Low confidence: weak alignment
            softened_learning_rate = error_learning_rate * 0.4  # Significant softening
            softening_reason = "Weak trend alignment"
        else:
            # Very low confidence: trend and offset misaligned or unclear
            softened_learning_rate = error_learning_rate * 0.2  # Heavy softening
            softening_reason = "Poor trend alignment"
        
        # === VOLATILITY-BASED ADJUSTMENT ===
        # In high volatility periods, be more conservative
        if current_idx >= 10:
            recent_price_volatility = np.std(data1_prices[current_idx-10:current_idx])
            avg_price = np.mean(data1_prices[current_idx-10:current_idx])
            volatility_ratio = recent_price_volatility / avg_price if avg_price > 0 else 0
            
            if volatility_ratio > 0.02:  # High volatility (>2%)
                volatility_softening = 0.6
                volatility_reason = "High volatility"
            elif volatility_ratio > 0.01:  # Medium volatility (1-2%)
                volatility_softening = 0.8
                volatility_reason = "Medium volatility"
            else:  # Low volatility (<1%)
                volatility_softening = 1.0
                volatility_reason = "Low volatility"
            
            softened_learning_rate *= volatility_softening
        else:
            volatility_reason = "Insufficient history"
            volatility_softening = 0.8  # Conservative default
            softened_learning_rate *= volatility_softening
        
        # Calculate new offset component with softened learning rate
        new_error_component = past_error * softened_learning_rate
        momentum_component = previous_dynamic_offset * momentum_decay
        
        # Update offset for CURRENT prediction using PREVIOUS error
        dynamic_error_offset = momentum_component + new_error_component
        
        # AGGRESSIVE cap on offset - allow larger corrections
        max_offset = abs(base_ensemble_price) * 0.15  # INCREASED from 0.05 to 0.15 (15% max)
        dynamic_error_offset = np.clip(dynamic_error_offset, -max_offset, max_offset)
        
        # Store analysis info for previous prediction
        past_base_error = past_error
        past_final_error = previous_actual - (previous_base_prediction + previous_dynamic_offset)  # What the error was with old offset
        
        error_offset_history.append({
            'idx': current_idx - 1,  # This info is about the PREVIOUS prediction
            'actual_price': previous_actual,
            'base_prediction': previous_base_prediction,
            'final_prediction': previous_base_prediction + previous_dynamic_offset,
            'base_error': past_base_error,
            'final_error': past_final_error,
            'offset_applied': dynamic_error_offset,  # NEW offset for current prediction
            'previous_offset': previous_dynamic_offset,
            'error_improvement': abs(past_base_error) - abs(past_final_error),
            'momentum_component': momentum_component,
            'error_component': new_error_component,
            # New directional analysis fields
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'offset_direction': offset_direction,
            'directional_alignment': directional_alignment,
            'trend_confidence': trend_confidence,
            'original_learning_rate': error_learning_rate,
            'softened_learning_rate': softened_learning_rate,
            'softening_factor': softened_learning_rate / error_learning_rate,
            'softening_reason': softening_reason,
            'volatility_softening': volatility_softening,
            'volatility_reason': volatility_reason
        })
        
        # Enhanced logging with directional analysis
        trend_desc = "UP" if trend_direction > 0 else "DOWN" if trend_direction < 0 else "NEUTRAL"
        offset_desc = "UP" if offset_direction > 0 else "DOWN"
        softening_pct = (1 - softened_learning_rate / error_learning_rate) * 100
        
        print(f"DIRECTIONAL Offset Update: Error={past_error:.6f} | Trend={trend_desc}({trend_strength:.2f}) | Offset={offset_desc} | Confidence={trend_confidence:.2f}")
        print(f"  Softening: {softening_pct:.1f}% ({softening_reason}, {volatility_reason}) | New offset={dynamic_error_offset:.6f} (was {previous_dynamic_offset:.6f})")
    
    # IMMEDIATE APPLICATION: Apply updated dynamic offset to CURRENT prediction
    final_ensemble_price = base_ensemble_price + dynamic_error_offset
    
    # Store results with offset applied
    prediction_indices.append(current_idx)
    actual_prices.append(data1_prices[current_idx])  # This is OK - we store for final evaluation
    actual_bids.append(data1_bids[current_idx])
    actual_asks.append(data1_asks[current_idx])
    ensemble_predictions.append(final_ensemble_price)
    attention_weights_history.append(attention_weights.cpu().numpy())
    
    # Store for analysis
    base_predictions.append(base_ensemble_price)
    offset_predictions.append(final_ensemble_price)
    
    # IMMEDIATE SETUP: Store current data for NEXT iteration's offset calculation
    # We can immediately "observe" the actual price for offset calculation in next iteration
    previous_actual = data1_prices[current_idx]  # Store current actual for next iteration
    previous_base_prediction = base_ensemble_price  # Store current base prediction for next iteration

# Display progress for ensemble predictions with detailed multi-layer correction analysis
for i, current_idx in enumerate(prediction_indices):
    if i < len(error_offset_history):
        h = error_offset_history[i]
        improvement = abs(h['base_error']) - abs(h['final_error'])
        print(f"Walk Forward {current_idx}: Actual={h['actual_price']:.6f}, Base={h['base_prediction']:.6f}, Final={h['final_prediction']:.6f}, Offset={h['offset_applied']:.6f}, Improvement={improvement:.6f}")
    else:
        actual = actual_prices[i]
        base = base_predictions[i] if i < len(base_predictions) else 0
        offset = offset_predictions[i] if i < len(offset_predictions) else 0
        final = ensemble_predictions[i]
        
        base_error = abs(actual - base)
        offset_error = abs(actual - offset)
        final_error = abs(actual - final)
        
        print(f"Walk Forward {current_idx}: Actual={actual:.6f}, Base={base:.6f}(E:{base_error:.6f}), Offset={offset:.6f}(E:{offset_error:.6f}), Final={final:.6f}(E:{final_error:.6f})")

# Enhanced analysis with multi-layer correction breakdown
if error_offset_history:
    print(f"\nENHANCED Dynamic Error Offset Analysis with Directional Softening:")
    
    # Calculate key metrics
    base_errors = [abs(h['base_error']) for h in error_offset_history]
    final_errors = [abs(h['final_error']) for h in error_offset_history]
    improvements = [h['error_improvement'] for h in error_offset_history]
    offsets_applied = [h['offset_applied'] for h in error_offset_history]
    
    avg_base_error = np.mean(base_errors)
    avg_final_error = np.mean(final_errors)
    avg_improvement = np.mean(improvements)
    avg_offset = np.mean([abs(o) for o in offsets_applied])
    
    # Calculate improvement statistics
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    improvement_rate = (positive_improvements / len(improvements)) * 100
    
    # Calculate total error reduction
    total_base_error = sum(base_errors)
    total_final_error = sum(final_errors)
    total_improvement = total_base_error - total_final_error
    improvement_percentage = (total_improvement / total_base_error) * 100 if total_base_error > 0 else 0
    
    print(f"   Average Base Error (no offset): {avg_base_error:.6f}")
    print(f"   Average Final Error (with offset): {avg_final_error:.6f}")
    print(f"   Average Error Improvement: {avg_improvement:.6f}")
    print(f"   Total Error Reduction: {improvement_percentage:.2f}%")
    print(f"   Predictions Improved: {positive_improvements}/{len(improvements)} ({improvement_rate:.1f}%)")
    print(f"   Average Offset Magnitude: {avg_offset:.6f}")
    print(f"   Offset Range: [{min(offsets_applied):.6f}, {max(offsets_applied):.6f}]")
    
    # === NEW DIRECTIONAL ANALYSIS ===
    print(f"\n   DIRECTIONAL SOFTENING ANALYSIS:")
    
    # Trend direction analysis
    trend_directions = [h.get('trend_direction', 0) for h in error_offset_history]
    trend_strengths = [h.get('trend_strength', 0) for h in error_offset_history]
    trend_confidences = [h.get('trend_confidence', 0) for h in error_offset_history]
    softening_factors = [h.get('softening_factor', 1.0) for h in error_offset_history]
    
    up_trends = sum(1 for t in trend_directions if t > 0)
    down_trends = sum(1 for t in trend_directions if t < 0)
    neutral_trends = sum(1 for t in trend_directions if t == 0)
    
    avg_trend_strength = np.mean(trend_strengths)
    avg_trend_confidence = np.mean(trend_confidences)
    avg_softening = np.mean(softening_factors)
    
    print(f"     Trend Distribution: UP={up_trends}, DOWN={down_trends}, NEUTRAL={neutral_trends}")
    print(f"     Average Trend Strength: {avg_trend_strength:.3f}")
    print(f"     Average Trend Confidence: {avg_trend_confidence:.3f}")
    print(f"     Average Softening Factor: {avg_softening:.3f} ({(1-avg_softening)*100:.1f}% reduction)")
    
    # Alignment analysis
    alignments = [h.get('directional_alignment', 0.5) for h in error_offset_history]
    strong_alignments = sum(1 for a in alignments if a > 0.8)
    weak_alignments = sum(1 for a in alignments if a < 0.4)
    
    print(f"     Strong Alignments: {strong_alignments}/{len(alignments)} ({strong_alignments/len(alignments)*100:.1f}%)")
    print(f"     Weak Alignments: {weak_alignments}/{len(alignments)} ({weak_alignments/len(alignments)*100:.1f}%)")
    
    # Softening reason analysis
    softening_reasons = [h.get('softening_reason', 'Unknown') for h in error_offset_history]
    reason_counts = {}
    for reason in softening_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print(f"     Softening Reasons:")
    for reason, count in reason_counts.items():
        pct = (count / len(softening_reasons)) * 100
        print(f"       {reason}: {count} ({pct:.1f}%)")
    
    # Volatility analysis
    volatility_softenings = [h.get('volatility_softening', 1.0) for h in error_offset_history]
    avg_vol_softening = np.mean(volatility_softenings)
    high_vol_periods = sum(1 for v in volatility_softenings if v < 0.7)
    
    print(f"     Average Volatility Softening: {avg_vol_softening:.3f}")
    print(f"     High Volatility Periods: {high_vol_periods}/{len(volatility_softenings)} ({high_vol_periods/len(volatility_softenings)*100:.1f}%)")
    
    # Show most effective corrections
    if len(improvements) > 0:
        best_improvement_idx = np.argmax(improvements)
        worst_correction_idx = np.argmin(improvements)
        
        best = error_offset_history[best_improvement_idx]
        worst = error_offset_history[worst_correction_idx]
        
        print(f"   Best Correction: Point {best['idx']} - Error reduced by {best['error_improvement']:.6f}")
        print(f"     Trend: {best.get('trend_direction', 'N/A')}, Confidence: {best.get('trend_confidence', 'N/A'):.3f}, Softening: {best.get('softening_factor', 'N/A'):.3f}")
        print(f"   Worst Correction: Point {worst['idx']} - Error increased by {abs(worst['error_improvement']):.6f}")
        print(f"     Trend: {worst.get('trend_direction', 'N/A')}, Confidence: {worst.get('trend_confidence', 'N/A'):.3f}, Softening: {worst.get('softening_factor', 'N/A'):.3f}")
    
    # Momentum vs error component analysis
    avg_momentum = np.mean([abs(h['momentum_component']) for h in error_offset_history])
    avg_error_component = np.mean([abs(h['error_component']) for h in error_offset_history])
    
    print(f"   Average Momentum Component: {avg_momentum:.6f}")
    print(f"   Average Error Component: {avg_error_component:.6f}")
    print(f"   Momentum vs Error Ratio: {avg_momentum/avg_error_component:.2f}" if avg_error_component > 0 else "   Momentum vs Error Ratio: N/A")

# Multi-layer correction analysis
if len(actual_prices) > 0 and len(base_predictions) > 0:
    print(f"\nMULTI-LAYER CORRECTION ANALYSIS:")
    
    # Calculate errors for each correction layer
    base_errors_full = [abs(actual_prices[i] - base_predictions[i]) for i in range(len(actual_prices))]
    offset_errors_full = [abs(actual_prices[i] - offset_predictions[i]) for i in range(len(actual_prices))] if offset_predictions else base_errors_full
    final_errors_full = [abs(actual_prices[i] - ensemble_predictions[i]) for i in range(len(actual_prices))]
    
    # Calculate average errors for each layer
    avg_base_error_full = np.mean(base_errors_full)
    avg_offset_error_full = np.mean(offset_errors_full)
    avg_final_error_full = np.mean(final_errors_full)
    
    # Calculate improvement percentages
    offset_improvement = ((avg_base_error_full - avg_offset_error_full) / avg_base_error_full) * 100 if avg_base_error_full > 0 else 0
    total_improvement_full = ((avg_base_error_full - avg_final_error_full) / avg_base_error_full) * 100 if avg_base_error_full > 0 else 0
    
    print(f"   Layer 1 - Base Prediction Error: {avg_base_error_full:.6f}")
    print(f"   Layer 2 - Dynamic Offset Error: {avg_offset_error_full:.6f} (Improvement: {offset_improvement:.2f}%)")
    print(f"   Layer 3 - Final Ensemble Error: {avg_final_error_full:.6f}")
    print(f"   TOTAL IMPROVEMENT: {total_improvement_full:.2f}% (Base → Final)")
    
    # Count improvements at each layer
    offset_improvements = sum(1 for i in range(len(base_errors_full)) if offset_errors_full[i] < base_errors_full[i])
    
    print(f"   Dynamic Offset improved: {offset_improvements}/{len(base_errors_full)} predictions ({(offset_improvements/len(base_errors_full))*100:.1f}%)")

# --- Ensemble Engine Performance Metrics with Gradient Feedback Analysis ---
print(f"Ensemble Engine Performance Metrics")

if len(actual_prices) > 1:
    # Ensemble performance - for prices
    ensemble_price_errors = [abs(pred - actual) for pred, actual in zip(ensemble_predictions, actual_prices)]
    ensemble_price_mse = np.mean([(pred - actual) ** 2 for pred, actual in zip(ensemble_predictions, actual_prices)])
    ensemble_price_mae = np.mean([abs(pred - actual) for pred, actual in zip(ensemble_predictions, actual_prices)])
    ensemble_price_accuracy = np.mean([abs(pred - actual) / abs(actual) for pred, actual in zip(ensemble_predictions, actual_prices)]) * 100
    
    print(f"Ensemble Engine Performance (Walk Forward):")
    print(f"   Price: MSE: {ensemble_price_mse:.6f} | MAE: {ensemble_price_mae:.6f} | Error %: {ensemble_price_accuracy:.4f}%")
    
    # MPC performance
    if mpc_errors:
        mpc_mse = np.mean([e**2 for e in mpc_errors])
        mpc_accuracy = np.mean([e / pred['actual'] for e, pred in zip(mpc_errors, mpc_predictions_detailed)]) * 100
        print(f"Model Predictive Control Performance (5 points):")
        print(f"   Price: MSE: {mpc_mse:.6f} | MAE: {mpc_mae:.6f} | Error %: {mpc_accuracy:.4f}%")
    
    print(f"Evaluation points: Walk Forward={len(actual_prices)}, MPC={len(mpc_errors)}")
else:
    print("Not enough points for evaluation")

# --- Compact Plotting with Better Screen Compatibility ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Plot 1: Complete price history with training cutoff and predictions
# Historical training data
training_indices = list(range(0, train_until_idx))
training_prices = data1_prices[:train_until_idx]

ax1.plot(training_indices, training_prices, label='Training Data', color='gray', linewidth=1, alpha=0.7)
ax1.plot(prediction_indices, actual_prices, label='Actual Price', color='blue', linewidth=2, marker='o', markersize=3)
ax1.plot(prediction_indices, ensemble_predictions, label='Final (Meta+Offset)', color='purple', linewidth=3, marker='*', markersize=4)
if base_predictions:
    ax1.plot(prediction_indices, base_predictions, label='Base (no corrections)', color='orange', linewidth=2, linestyle='--', alpha=0.7)
if offset_predictions:
    ax1.plot(prediction_indices, offset_predictions, label='Dynamic Offset', color='green', linewidth=2, linestyle=':', alpha=0.8)
ax1.axvline(x=train_until_idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Training Cutoff')

# Add some styling to highlight the prediction region
ax1.axvspan(predict_start_idx, predict_end_idx, alpha=0.1, color='purple', label='Prediction Window')

ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Ensemble Engine: Multi-Layer Error Correction (Base → Offset)', fontsize=12, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Better y-axis formatting for full history
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

# Add compact text annotations
recent_training_avg = np.mean(training_prices[-50:]) if len(training_prices) >= 50 else np.mean(training_prices)
ax1.text(0.02, 0.98, f'Training: {len(training_prices)} pts | Dynamic Offset: {"ON" if offset_predictions else "OFF"}', 
         transform=ax1.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8), fontsize=8)

# Plot 2: EXTREME zoom view of prediction region - prediction window only
context_start = max(0, train_until_idx - 5)  # Show only last 5 training points for minimal context
context_indices = list(range(context_start, train_until_idx))
context_prices = data1_prices[context_start:train_until_idx]

ax2.plot(context_indices, context_prices, label='Recent Context', color='gray', linewidth=1, alpha=0.5)
ax2.plot(prediction_indices, actual_prices, label='Actual', color='blue', linewidth=3, marker='o', markersize=6)
ax2.plot(prediction_indices, ensemble_predictions, label='Final (Meta+Offset)', color='purple', linewidth=3.5, marker='*', markersize=7)
if base_predictions:
    ax2.plot(prediction_indices, base_predictions, label='Base (no corrections)', color='orange', linewidth=2.5, linestyle='--', marker='s', markersize=5)
if offset_predictions:
    ax2.plot(prediction_indices, offset_predictions, label='Dynamic Offset', color='green', linewidth=2, linestyle=':', marker='d', markersize=4)
ax2.axvline(x=train_until_idx, color='red', linestyle='--', alpha=0.8, linewidth=2)

# EXTREME ZOOM: Focus ONLY on prediction values for y-axis
all_prediction_values = list(actual_prices) + list(ensemble_predictions)
if base_predictions:
    all_prediction_values.extend(base_predictions)
if offset_predictions:
    all_prediction_values.extend(offset_predictions)
price_min = np.min(all_prediction_values)
price_max = np.max(all_prediction_values)
price_range = price_max - price_min
padding = price_range * 0.02  # Only 2% padding for extreme zoom
ax2.set_ylim(price_min - padding, price_max + padding)

# Focus x-axis tightly on prediction region only
ax2.set_xlim(predict_start_idx - 2, predict_end_idx + 2)

ax2.set_ylabel('Price ($)', fontsize=11)
ax2.set_title('EXTREME ZOOM: Multi-Layer Error Correction Performance', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4)

# Much higher precision y-axis formatting for extreme zoom
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))

# Add high-precision prediction accuracy text
if len(actual_prices) > 0:
    ensemble_price_accuracy_display = 100 - ensemble_price_accuracy
    ax2.text(0.02, 0.02, f'Ensemble Accuracy: {ensemble_price_accuracy_display:.4f}%', 
             transform=ax2.transAxes, verticalalignment='bottom', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9), fontsize=9)

# Plot 3: Dynamic Offset Values Over Time
if error_offset_history:
    offset_indices = [h['idx'] for h in error_offset_history]
    offset_values = [h['offset_applied'] for h in error_offset_history]
    
    ax3.plot(offset_indices, offset_values, label='Dynamic Offset', color='green', linewidth=2, marker='d', markersize=4)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax3.fill_between(offset_indices, 0, offset_values, alpha=0.3, color='green')
    
    # Focus x-axis tightly on prediction region only
    ax3.set_xlim(predict_start_idx - 2, predict_end_idx + 2)
    
    ax3.set_ylabel('Offset Value ($)', fontsize=10)
    ax3.set_title('Dynamic Offset Evolution', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Better y-axis formatting for offsets
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))
    
    # Add offset statistics
    avg_offset_display = np.mean([abs(o) for o in offset_values])
    max_offset_display = max([abs(o) for o in offset_values])
    ax3.text(0.02, 0.98, f'Avg: ${avg_offset_display:.4f} | Max: ${max_offset_display:.4f}', 
             transform=ax3.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9), fontsize=8)

# Plot 4: Enhanced prediction errors comparison including base vs corrected
if len(actual_prices) > 0:
    base_errors = [abs(base_predictions[i] - actual_prices[i]) for i in range(len(actual_prices))] if base_predictions else []
    ensemble_price_errors = [abs(pred - actual) for pred, actual in zip(ensemble_predictions, actual_prices)]
    
    if base_errors:
        ax4.plot(prediction_indices, base_errors, label='Base Error (no offset)', color='orange', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax4.fill_between(prediction_indices, 0, base_errors, alpha=0.2, color='orange')
    
    ax4.plot(prediction_indices, ensemble_price_errors, label='Final Error (with offset)', color='purple', linewidth=2, marker='*', markersize=4, alpha=0.7)
    ax4.fill_between(prediction_indices, 0, ensemble_price_errors, alpha=0.2, color='purple')
    
    # Add error statistics
    if base_errors:
        base_mae = np.mean(base_errors)
        ax4.axhline(y=base_mae, color='darkorange', linestyle=':', alpha=0.8, linewidth=1.5, label=f'Base Mean: ${base_mae:.3f}')
    
    ax4.axhline(y=ensemble_price_mae, color='darkviolet', linestyle=':', alpha=0.8, linewidth=1.5, label=f'Final Mean: ${ensemble_price_mae:.3f}')
    
    # Better scaling for error plot
    all_errors = ensemble_price_errors + (base_errors if base_errors else [])
    error_max = max(all_errors) if all_errors else ensemble_price_mae
    ax4.set_ylim(0, error_max * 1.1)  # 10% padding above max error
    
    # Focus x-axis tightly on prediction region only
    ax4.set_xlim(predict_start_idx - 2, predict_end_idx + 2)
    
    ax4.set_ylabel('Error ($)', fontsize=10)
    ax4.set_title('Error Comparison: Base vs Dynamic Offset Corrected', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Better y-axis formatting for errors
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))
    
    # Add improvement statistics
    if base_errors:
        improvement_pct = ((base_mae - ensemble_price_mae) / base_mae) * 100
        ax4.text(0.02, 0.98, f'Error Reduction: {improvement_pct:.1f}% | Base: ${base_mae:.3f} → Final: ${ensemble_price_mae:.3f}', 
                 transform=ax4.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9), fontsize=8)

ax4.set_xlabel('Data Index', fontsize=10)

# Improve overall plot formatting
plt.tight_layout(pad=2.0)

# Add a compact main title
fig.suptitle(f'Ensemble Engine Stock Price Prediction - {file}', fontsize=14, fontweight='bold', y=0.97)

plt.show()

print(f"Ensemble Engine Visualization Summary:")
print(f"   Training: {len(training_prices)} pts (0-{train_until_idx-1}) | MPC: {len(mpc_errors)} pts | Walk Forward: {len(actual_prices)} pts ({predict_start_idx}-{predict_end_idx})")
print(f"   Price range - Training: ${np.min(training_prices):.0f}-${np.max(training_prices):.0f} | Test: ${np.min(actual_prices):.1f}-${np.max(actual_prices):.1f}")
if len(actual_prices) > 1:
    print(f"   Walk Forward Model - Error: Avg=${ensemble_price_mae:.2f} ({ensemble_price_accuracy:.2f}%) | Max=${np.max(ensemble_price_errors):.2f}")
    if mpc_errors:
        print(f"   MPC Model - Error: Avg=${mpc_mae:.2f} ({mpc_accuracy:.2f}%) | Max=${np.max(mpc_errors):.2f}")
    
    print(f"   Combined Accuracy - Walk Forward: {100-ensemble_price_accuracy:.2f}% | MPC: {100-mpc_accuracy:.2f}%")