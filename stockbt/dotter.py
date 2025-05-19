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

# --- PLOTTING ---
plt.figure(figsize=(12, 6))
plt.plot(product, label=f'{file1} x {file2}')
plt.title(f'Element-wise Product\n{file1} x {file2}')
plt.xlabel('Index')
plt.ylabel('Product')
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
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(product))]

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Close'])
    for date, price in zip(dates, product):
        writer.writerow([date, price])

print(f'Generated CSV: {output_file}')
