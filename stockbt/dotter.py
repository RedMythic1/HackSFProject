import os
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to datasets directory
DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')

# List all CSV files in the datasets directory
csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]

if len(csv_files) < 2:
    raise Exception('Not enough CSV files in datasets directory to select two random datasets.')

# Select two random CSV files
file1, file2 = random.sample(csv_files, 2)
path1 = os.path.join(DATASETS_DIR, file1)
path2 = os.path.join(DATASETS_DIR, file2)

# Helper to extract Close values from a CSV file
def get_close_vector(csv_path):
    close_values = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                close_values.append(float(row['Close']))
            except (KeyError, ValueError):
                continue
    return close_values

# Get close vectors
v1 = get_close_vector(path1)
v2 = get_close_vector(path2)

# Truncate to the shortest length
min_len = min(len(v1), len(v2))
v1 = v1[:min_len]
v2 = v2[:min_len]

# Convert to numpy arrays for norm calculation
v1_np = np.array(v1)
v2_np = np.array(v2)

# Calculate magnitudes (L2 norms)
magnitude_a = np.linalg.norm(v1_np)
magnitude_b = np.linalg.norm(v2_np)

# --- PROFILE MENU ---
print("Select a profile:")
print("1. Normal (just cross/divide)")
print("2. Volatile (moderate angle, squeeze, noise)")
print("3. Deadly (high angle, squeeze, noise)")
print("4. Flat (angle=0, squeeze=1, no noise)")
print("5. Flat Erratic (angle=0, high squeeze, high noise)")
profile_choice = input("Enter 1-5: ").strip()

profile_map = {
    '1': 'normal',
    '2': 'volatile',
    '3': 'deadly',
    '4': 'flat',
    '5': 'flat_erratic'
}
profile = profile_map.get(profile_choice, 'normal')

neg_mod = input("Negative modifier? (y/n): ").strip().lower() == 'y'

# Set parameters based on profile
if profile == 'normal':
    angle = 0
    squeeze = 1
    noise_level = 0
    base_amplitude = 0
elif profile == 'volatile':
    angle = 30
    squeeze = 2
    noise_level = 50
    base_amplitude = 4
elif profile == 'deadly':
    angle = 45
    squeeze = 3
    noise_level = 80
    base_amplitude = 5
elif profile == 'flat':
    angle = 0
    squeeze = 1
    noise_level = 0
    base_amplitude = 0
elif profile == 'flat_erratic':
    angle = 0
    squeeze = 3
    noise_level = 80
    base_amplitude = 5
else:
    angle = 0
    squeeze = 1
    noise_level = 0
    base_amplitude = 0

if neg_mod:
    angle = -angle

# --- PIPELINE ---
# Step 1: Cross the two vectors (element-wise multiplication)
crossed = v1_np * v2_np

# Step 2: Normalize
normalization_factor = (magnitude_a + magnitude_b) ** 0.25 if (magnitude_a + magnitude_b) != 0 else 1.0
crossed_norm = crossed / normalization_factor

# Step 3: Rotate
desired_angle_rad = np.deg2rad(angle)
x_vals = np.arange(len(crossed_norm))
y_vals = crossed_norm
x_start, x_end = 0, len(crossed_norm) - 1
y_start, y_end = crossed_norm[0], crossed_norm[-1]
current_angle = np.arctan2(y_end - y_start, x_end - x_start)
rotation = desired_angle_rad - current_angle
x0, y0 = x_start, y_start
x_rot = x0 + (x_vals - x0) * np.cos(rotation) - (y_vals - y0) * np.sin(rotation)
y_rot = y0 + (x_vals - x0) * np.sin(rotation) + (y_vals - y0) * np.cos(rotation)

# Step 4: Interpolate to ensure one y per integer x
def simple_linear_interp(x, y, x_new):
    y_new = np.empty_like(x_new, dtype=float)
    n = len(x)
    for i, xi in enumerate(x_new):
        if xi <= x[0]:
            y_new[i] = y[0]
        elif xi >= x[-1]:
            y_new[i] = y[-1]
        else:
            j = np.searchsorted(x, xi) - 1
            x0, x1 = x[j], x[j+1]
            y0, y1 = y[j], y[j+1]
            t = (xi - x0) / (x1 - x0)
            y_new[i] = y0 + t * (y1 - y0)
    return y_new
sort_idx = np.argsort(x_rot)
x_rot_sorted = x_rot[sort_idx]
y_rot_sorted = y_rot[sort_idx]
y_rot_interp = simple_linear_interp(x_rot_sorted, y_rot_sorted, np.arange(len(crossed_norm)))

# Step 5: Fit a line to the rotated, interpolated data
def fit_line(x, y):
    m = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
    b = y[0]
    return m, b
m_fit, b_fit = fit_line(np.arange(len(y_rot_interp)), y_rot_interp)
line_fit = m_fit * np.arange(len(y_rot_interp)) + b_fit
print(f"Equation of the best fit line after rotation: y = {m_fit:.4f}x + {b_fit:.4f}")

# Step 6: Squeeze
y_squeezed = line_fit + squeeze * (y_rot_interp - line_fit)

# Step 7: Dynamic noise (always applied last)
def random_fourier_noise(x, num_terms=10, amplitude=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(x)
    noise = np.zeros(n)
    L = n
    for k in range(1, num_terms + 1):
        a = np.random.randn() * amplitude / np.sqrt(k)
        b = np.random.randn() * amplitude / np.sqrt(k)
        noise += a * np.sin(2 * np.pi * k * x / L) + b * np.cos(2 * np.pi * k * x / L)
    return noise

def local_volatility(y, window=10):
    diffs = np.abs(np.diff(y, prepend=y[0]))
    vol = np.convolve(diffs, np.ones(window)/window, mode='same')
    return vol

vol = local_volatility(y_squeezed, window=10)
vol_norm = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)
n = np.clip(noise_level, 0, 100) / 100.0
weights = (1 - vol_norm) ** (1 - n) + n * 0.5
x = np.arange(len(y_squeezed))
num_terms = max(1, int(5 + 25 * n))
noise = random_fourier_noise(x, num_terms=num_terms, amplitude=base_amplitude)
noise_scaled = noise * weights
y_noisy = y_squeezed + noise_scaled

# Plot and print Riemann sum for the noisy series
plt.figure(figsize=(12, 6))
plt.plot(x, y_noisy, label=f'Profile: {profile}, Neg: {neg_mod}, Angle: {angle}, Squeeze: {squeeze}, Noise: {noise_level}, Amp: {base_amplitude}')
plt.plot(x, line_fit, 'r--', label='Best Fit Line')
plt.title(f'Profile: {profile} | {file1} x {file2}')
plt.xlabel('Price Point (Index)')
plt.ylabel('Final Value')
plt.legend()
plt.gca().set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.show()

riemann_sum = np.sum(y_noisy - line_fit)
print(f'Riemann sum (integral) of final - best fit line: {riemann_sum:.4f}')
