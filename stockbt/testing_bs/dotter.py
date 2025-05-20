import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

# Convert to numpy arrays
v1_np = np.array(v1)
v2_np = np.array(v2)

# Element-wise product
product = v1_np * v2_np

# Cleanup function to limit point-to-point changes to ±20%
def cleanup_series(series, max_pct_change=0.2):
    cleaned = np.array(series).copy()
    for i in range(1, len(cleaned)):
        prev = cleaned[i-1]
        max_up = prev * (1 + max_pct_change)
        max_down = prev * (1 - max_pct_change)
        if cleaned[i] > max_up:
            cleaned[i] = max_up
        elif cleaned[i] < max_down:
            cleaned[i] = max_down
    return cleaned

# Apply cleanup to the transformed vector
transformed_clean = cleanup_series(product)

# --- PROFILE MENU ---
print("Select a profile:")
print("1. Down Volatile")
print("2. Down Non-Volatile")
print("3. Up Volatile")
print("4. Up Non-Volatile")
print("5. Flat Volatile")
print("6. Flat Non-Volatile")
print("7. Full (combine two datasets with separate profiles)")
profile_choice = input("Enter 1-7: ").strip()

profile_map = {
    '1': ('down', 2.0),
    '2': ('down', 0.5),
    '3': ('up', 2.0),
    '4': ('up', 0.5),
    '5': ('flat', 2.0),
    '6': ('flat', 0.5)
}

if profile_choice == '7':
    print("First half profile:")
    first_choice = input("Enter 1-6: ").strip()
    trend1, g1 = profile_map.get(first_choice, ('flat', 1.0))
    print("Second half profile:")
    second_choice = input("Enter 1-6: ").strip()
    trend2, g2 = profile_map.get(second_choice, ('flat', 1.0))

    # Split the data in half
    mid = len(transformed_clean) // 2
    data1 = transformed_clean[:mid]
    data2 = transformed_clean[mid:]

    # Helper to process a segment
    def process_segment(data, trend, g):
        startpoint = data[0]
        endpoint = data[-1]
        ratio = startpoint / endpoint if endpoint != 0 else 1.0
        trend_for_calc = 'up' if trend == 'down' else trend
        if trend_for_calc == 'flat':
            trend_div = ratio
        elif trend_for_calc == 'up':
            trend_div = random.randint(5, 40) * ratio
        else:
            trend_div = ratio
        trend_series = data / trend_div if trend_div != 0 else data
        x_vals = np.arange(len(trend_series))
        if trend_for_calc == 'flat':
            avg_val = (trend_series[0] + trend_series[-1]) / 2
            line_vals = np.full_like(trend_series, avg_val)
        else:
            y_start = trend_series[0]
            y_end = trend_series[-1]
            x_start = 0
            x_end = len(trend_series) - 1
            m = (y_end - y_start) / (x_end - x_start) if x_end != x_start else 0
            b = y_start
            line_vals = m * x_vals + b
        bounced = ((trend_series - line_vals) * g) + line_vals
        if trend == 'down':
            # Mirror over midpoint
            mid = len(bounced) // 2
            bounced = np.concatenate([bounced[mid:][::-1], bounced[:mid][::-1]]) if len(bounced) % 2 == 0 else np.concatenate([bounced[mid+1:][::-1], [bounced[mid]], bounced[:mid][::-1]])
        return bounced

    bounced1 = process_segment(data1, trend1, g1)
    bounced2 = process_segment(data2, trend2, g2)
    offset = bounced1[-1] - bounced2[0]
    bounced2_offset = bounced2 + offset
    combined = np.concatenate([bounced1, bounced2_offset])
    bounced = combined
    trend = f'full ({trend1} {g1} + {trend2} {g2})'
    label = f'Profile: full | 1st: {trend1} {"Volatile" if g1 > 1 else "Non-Volatile"} | 2nd: {trend2} {"Volatile" if g2 > 1 else "Non-Volatile"} | {file1} x {file2}'
else:
    trend, g = profile_map.get(profile_choice, ('flat', 1.0))
    # --- PIPELINE ---
    # Apply trend
    startpoint = transformed_clean[0]
    endpoint = transformed_clean[-1]
    ratio = startpoint / endpoint if endpoint != 0 else 1.0
    # For 'down' profiles, generate as 'up' first
    trend_for_calc = 'up' if trend == 'down' else trend
    if trend_for_calc == 'flat':
        trend_div = ratio
    elif trend_for_calc == 'up':
        trend_div = random.randint(5, 40) * ratio
    else:
        trend_div = ratio
    trend_series = transformed_clean / trend_div if trend_div != 0 else transformed_clean
    # Apply bounce
    g = float(g)
    x_vals = np.arange(len(trend_series))
    if trend_for_calc == 'flat':
        avg_val = (trend_series[0] + trend_series[-1]) / 2
        line_vals = np.full_like(trend_series, avg_val)
    else:
        y_start = trend_series[0]
        y_end = trend_series[-1]
        x_start = 0
        x_end = len(trend_series) - 1
        m = (y_end - y_start) / (x_end - x_start) if x_end != x_start else 0
        b = y_start
        line_vals = m * x_vals + b
    bounced = ((trend_series - line_vals) * g) + line_vals
    # Mirror for 'down' profiles
    def mirror_over_midpoint(arr):
        mid = len(arr) // 2
        return np.concatenate([arr[mid:][::-1], arr[:mid][::-1]]) if len(arr) % 2 == 0 else np.concatenate([arr[mid+1:][::-1], [arr[mid]], arr[:mid][::-1]])
    if trend == 'down':
        bounced = mirror_over_midpoint(bounced)
    label = f'Profile: {trend} {"Volatile" if g > 1 else "Non-Volatile"} | {file1} x {file2}'

# Take the absolute value before plotting and saving
bounced = np.abs(bounced)

# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(bounced, label=label)
plt.title(f'{label}')
plt.xlabel('Index')
plt.ylabel('Final Value')
plt.legend()
plt.tight_layout()
plt.show()

# --- CSV OUTPUT ---
output_dir = 'stockbt/generated_data'
os.makedirs(output_dir, exist_ok=True)

existing = [f for f in os.listdir(output_dir) if f.startswith('genstock') and f.endswith('.csv')]
nums = [int(f[len('genstock'):-4]) for f in existing if f[len('genstock'):-4].isdigit()]
next_num = max(nums) + 1 if nums else 1
output_file = os.path.join(output_dir, f'genstock{next_num}.csv')

start_date = datetime.strptime('1976-01-22', '%Y-%m-%d')
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(bounced))]

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Close'])
    for date, price in zip(dates, bounced):
        writer.writerow([date, price])

print(f'Generated CSV: {output_file}')
