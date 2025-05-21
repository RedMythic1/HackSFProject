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

def run_walk_forward_training_segment(
    model, optimizer, loss_fn, data_array,
    target_train_until_idx_inclusive, 
    current_trained_idx_exclusive, 
    history_X_live, history_y_live, 
    schedule_sampling_prob,
    epochs_per_step=50,
    label_prefix=""
):
    loop_start_idx = current_trained_idx_exclusive 
    loop_end_idx_exclusive = target_train_until_idx_inclusive + 1

    if loop_start_idx >= loop_end_idx_exclusive:
        print(f"{label_prefix}No new walk-forward training needed (target index {target_train_until_idx_inclusive}).")
        return current_trained_idx_exclusive 

    print(f"{label_prefix}Running walk-forward training from data index {loop_start_idx-1} up to {target_train_until_idx_inclusive}...")

    for i in range(loop_start_idx, loop_end_idx_exclusive):
        if i >= len(data_array) - 1:
            print(f"{label_prefix}Reached end of data for training pairs at index {i-1} (history_X will contain up to data[{i-1}]). Max data index is {len(data_array)-1}.")
            return i 

        X_train_tensor = torch.stack(history_X_live)
        y_train_tensor = torch.stack([torch.tensor(vec) for vec in history_y_live])

        for epoch in range(epochs_per_step):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            target = torch.cat([
                y_train_tensor[-1, 0:9],
                torch.tensor([y_train_tensor[-1, 9]])
            ]).float()
            loss = loss_fn(pred[-1], target)
            loss.backward()
            optimizer.step()
            if epoch == epochs_per_step -1 and i % max(1, (loop_end_idx_exclusive - loop_start_idx)//5) == 0 :
                 print(f"  {label_prefix}Train Step for data[{i}] (loop {i-loop_start_idx+1}/{loop_end_idx_exclusive-loop_start_idx}), Epoch {epoch+1}/{epochs_per_step}, Loss: {loss.item():.6f}")
        
        use_pred = random.random() < schedule_sampling_prob
        if use_pred:
            model.eval()
            with torch.no_grad():
                pred_for_data_i = model(X_train_tensor)[-1].cpu().numpy()
            history_X_live.append(torch.tensor(pred_for_data_i))
            history_y_live.append(pred_for_data_i) 
        else:
            history_X_live.append(torch.tensor(data_array[i]))
            history_y_live.append(data_array[i+1])
            
    new_trained_idx_exclusive = target_train_until_idx_inclusive + 1
    print(f"{label_prefix}Finished walk-forward training. Model effectively trained with data up to index {target_train_until_idx_inclusive}. Next training step would be based on data[{new_trained_idx_exclusive-1}].")
    return new_trained_idx_exclusive

def run_autoregressive_prediction_segment(
    model, data_array,
    seed_data_until_idx_inclusive, 
    predict_from_idx_inclusive,
    predict_until_idx_inclusive,
    label_prefix=""
):
    num_steps_to_predict = predict_until_idx_inclusive - predict_from_idx_inclusive + 1

    if num_steps_to_predict <= 0:
        print(f"{label_prefix}No steps to predict autoregressively (from {predict_from_idx_inclusive} to {predict_until_idx_inclusive}).")
        # Ensure plot is not generated for no predictions
        # plt.figure() 
        # plt.text(0.5, 0.5, "No predictions.", ha='center', va='center')
        # plt.title(f'{label_prefix}Autoregressive Prediction: No Steps')
        # plt.show()
        return np.array([]), np.array([])

    print(f"{label_prefix}Running autoregressive prediction for {num_steps_to_predict} steps (indices {predict_from_idx_inclusive} to {predict_until_idx_inclusive}).")
    if seed_data_until_idx_inclusive >= 0:
      print(f"{label_prefix}Seeding with actual data up to index {seed_data_until_idx_inclusive}.")
    else:
      print(f"{label_prefix}No actual data seeding (prediction starts from index 0 or history is empty).")


    auto_history = [torch.tensor(data_array[j]) for j in range(predict_from_idx_inclusive)] # seed with data[0]...data[predict_from_idx-1]

    predicted_vectors_list = []
    real_vectors_list = []

    for i in range(predict_from_idx_inclusive, predict_until_idx_inclusive + 1):
        if not auto_history: # Should only happen if predict_from_idx_inclusive is 0
             if i == 0 and predict_from_idx_inclusive == 0: # Predicting the very first point
                 # For the very first point, the model needs some form of initial "empty" or "start" input.
                 # This part of the model architecture (DeepSelfAttention) might need adjustment
                 # if it strictly requires a non-empty sequence.
                 # For now, let's assume if auto_history is empty, it's an issue or needs a special seed.
                 # A simple workaround could be to seed with a zero tensor if i=0 and history is empty.
                 # However, the loop for auto_history above `range(predict_from_idx_inclusive)` means if predict_from_idx_inclusive is 0, history is empty.
                 # Let's adjust seeding slightly: auto_history seeds with data[0]...data[predict_from_idx_inclusive-1]
                 # If predict_from_idx_inclusive is 0, this means auto_history is empty.
                 # The model expects a sequence. If it's empty, torch.stack will fail.
                 # We must ensure auto_history has at least one element if the model needs it.
                 # This implies a true autoregressive model cannot predict the very first point without a seed.
                 # For now, this function assumes `predict_from_idx_inclusive` implies `data_array[predict_from_idx_inclusive-1]` exists.
                 # Or, that the model can handle an empty sequence start (unlikely for this attention model).
                 # The fix: if predict_from_idx_inclusive is 0, this loop should not run or auto_history needs a special seed.
                 # Given current structure, predict_from_idx_inclusive=0 means auto_history is empty, stack will fail.
                 # This needs to be handled by ensuring predict_from_idx_inclusive is always >= 1 or model changes.
                 # For now, we rely on `num_steps_to_predict > 0` check.
                 # If predict_from_idx_inclusive is 0, auto_history should be explicitly seeded if needed, or the model needs a start token.
                 # Let's assume predict_from_idx_inclusive >= 0.
                 # If auto_history is empty and i=0, it's a special case.
                 # Current model structure: model(X_train) where X_train is torch.stack(history_X). If history_X has one item, X_train is (1, d_model).
                 # If auto_history is empty, we can't stack.
                 # Let's assume we always have seed data if predict_from_idx_inclusive > 0.
                 # If predict_from_idx_inclusive == 0, this is predicting the very first point.
                 # The problem is how the model is called: `model(torch.stack(auto_history))`.
                 # If auto_history is empty (predicting index 0), this fails.
                 # This implies the model can't predict index 0 without some seed.
                 # A practical solution: if predicting index 0, skip or use a predefined seed.
                 # For this implementation, let's assume predict_from_idx_inclusive means we have data[0]...data[predict_from_idx_inclusive-1] available as seed
                 # So auto_history will contain those. If predict_from_idx_inclusive = 0, then auto_history is empty.
                 print(f"CRITICAL ERROR: {label_prefix} Auto_history is empty when trying to predict index {i}. This setup is unsupported for the current model. Predictions will fail.")
                 break # Cannot proceed
        
        X_auto = torch.stack(auto_history)
        model.eval()
        with torch.no_grad():
            pred_vec_output = model(X_auto)
            pred_vec = pred_vec_output[-1].cpu().numpy()

        next_vec = np.array(pred_vec)
        
        auto_history.append(torch.tensor(next_vec))
        predicted_vectors_list.append(next_vec)
        real_vectors_list.append(data_array[i])
    
    if not predicted_vectors_list: # If loop didn't run or broke early
        print(f"{label_prefix}No valid predictions were generated.")
        return np.array([]), np.array([])

    predicted_vectors_np = np.array(predicted_vectors_list)
    real_vectors_np = np.array(real_vectors_list)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(predicted_vectors_np)), predicted_vectors_np[:, 9], label='Predicted Price', color='red')
    plt.plot(range(len(real_vectors_np)), real_vectors_np[:, 9], label='Actual Price', color='black', linestyle='--')
    
    title = f'{label_prefix}Autoregressive Prediction: {len(predicted_vectors_np)} Steps (Orig. Indices {predict_from_idx_inclusive} to {predict_from_idx_inclusive + len(predicted_vectors_np) - 1})'
    plt.xlabel(f'Step Number (Total {len(predicted_vectors_np)})')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False) # Changed to non-blocking

    final_pred_p = predicted_vectors_np[-1]
    final_real_p = real_vectors_np[-1]
    final_error_p = np.abs(final_pred_p - final_real_p)
    
    mae_all_feats = np.mean(np.abs(predicted_vectors_np - real_vectors_np), axis=0)
    mae_total = np.mean(mae_all_feats)
    # Ensure index 9 is valid for mae_all_feats (it has 10 elements for 10 features)
    mae_price = mae_all_feats[9] if len(mae_all_feats) > 9 else np.nan
    
    safe_real_prices = real_vectors_np[:, 9].copy()
    safe_real_prices[safe_real_prices == 0] = 1e-8 # Avoid division by zero for MAPE
    mape_price = np.mean(np.abs((real_vectors_np[:, 9] - predicted_vectors_np[:, 9]) / safe_real_prices)) * 100

    print(f"\n{label_prefix}--- Prediction Summary ---")
    print(f"Predicted for original indices: {predict_from_idx_inclusive} to {predict_from_idx_inclusive + len(predicted_vectors_np) - 1}")
    print(f"Last Predicted Vector: {final_pred_p}")
    print(f"Corresponding Actual Vector: {final_real_p}")
    print(f"Abs Error (last point): {final_error_p}")
    print(f"Mean Abs Error (all features, all steps): {mae_total:.6f}")
    print(f"Mean Abs Error (Price feature 9, all steps): {mae_price:.6f}")
    print(f"Mean Abs Percentage Error (Price feature 9, all steps): {mape_price:.2f}%")

    return predicted_vectors_np, real_vectors_np

print(f"Found {len(files)} CSV files: {files}")
# Model printout removed as it's less relevant now with dynamic output dim based on features
# print(f"Model: DeepSelfAttention(d_model=10, num_layers=1, output_dim=10)") 

# Initialize model and optimizer ONCE
# Assuming d_model matches the number of features (10 in this case)
model = DeepSelfAttention(d_model=10, num_layers=1) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

SCHEDULED_SAMPLING_PROB = 0.2
EPOCHS_PER_TRAIN_STEP = 50 # Original was 50 epochs per data point in walk-forward

cols = [ # Define column order once
    "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price",
    "PriceChange", "BuyVolChange", "SellVolChange", "BidPriceChange", "AskPriceChange", "Price"
]

# --- SEQUENTIAL FILE TRAINING PHASE (on files before the last one) ---
for file_idx, file in enumerate(files[:-1]):
    print(f"\n=== Training on file {file_idx+1}/{len(files)-1}: {file} ===")
    df_seq = pd.read_csv(os.path.join(data_dir, file))
    df_seq["PriceChange"] = df_seq["Price"].diff().fillna(0)
    df_seq["BuyVolChange"] = df_seq["Buy_Vol"].diff().fillna(0)
    df_seq["SellVolChange"] = df_seq["Sell_Vol"].diff().fillna(0)
    df_seq["BidPriceChange"] = df_seq["Bid_Price"].diff().fillna(0)
    df_seq["AskPriceChange"] = df_seq["Ask_Price"].diff().fillna(0)
    
    data_seq = df_seq[cols].values.astype(np.float32)

    if len(data_seq) < 2:
        print(f"  Skipping file {file} due to insufficient data (less than 2 rows).")
        continue

    history_X_seq = [torch.tensor(data_seq[0])]
    history_y_seq = [data_seq[1]] # Store as numpy array, convert later
    
    # Train on this sequential file
    # Loop from index 1 up to len(data_seq)-2 to make pairs (data[i], data[i+1])
    # This means history_X_seq will contain up to data_seq[len(data_seq)-2]
    # The last training step uses i = len(data_seq)-2.
    # history_X_seq gets data_seq[i], history_y_seq gets data_seq[i+1]
    # So, effectively trains on data up to index len(data_seq)-2.
    
    # The loop should go up to the point where data_seq[i] and data_seq[i+1] are valid.
    # If i goes up to len(data_seq)-2, then data_seq[i+1] is data_seq[len(data_seq)-1]. Valid.
    for i in range(1, len(data_seq)-1): 
        X_train = torch.stack(history_X_seq)
        y_train_targets = [torch.tensor(vec) for vec in history_y_seq]
        y_train = torch.stack(y_train_targets)
        
        for epoch in range(EPOCHS_PER_TRAIN_STEP):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            target = torch.cat([
                y_train[-1, 0:9], 
                torch.tensor([y_train[-1, 9]])
            ]).float()
            loss = loss_fn(pred[-1], target)
            loss.backward()
            optimizer.step()
            if epoch == EPOCHS_PER_TRAIN_STEP-1 and i % max(1, (len(data_seq)-2)//5) == 0:
                print(f"  File {file_idx+1}, Step for data[{i}] (of {len(data_seq)-2}), Epoch {epoch+1}/{EPOCHS_PER_TRAIN_STEP}, Loss: {loss.item():.6f}")

        use_pred = random.random() < SCHEDULED_SAMPLING_PROB
        if use_pred:
            model.eval()
            with torch.no_grad():
                pred_for_data_i = model(X_train)[-1].cpu().numpy()
            history_X_seq.append(torch.tensor(pred_for_data_i))
            history_y_seq.append(pred_for_data_i) 
        else:
            history_X_seq.append(torch.tensor(data_seq[i]))
            history_y_seq.append(data_seq[i+1])
            
    print(f"  Finished training on {file}. Model updated.")

# --- LAST FILE PROCESSING ---
if not files:
    print("No CSV files found in the directory.")
    # exit() # Or handle appropriately
elif len(files) == 0:
    print("No CSV files found.")
else:
    last_file = files[-1]
    print(f"\n=== Processing Last File: {last_file} ===")
    df_last = pd.read_csv(os.path.join(data_dir, last_file))
    df_last["PriceChange"] = df_last["Price"].diff().fillna(0)
    df_last["BuyVolChange"] = df_last["Buy_Vol"].diff().fillna(0)
    df_last["SellVolChange"] = df_last["Sell_Vol"].diff().fillna(0)
    df_last["BidPriceChange"] = df_last["Bid_Price"].diff().fillna(0)
    df_last["AskPriceChange"] = df_last["Ask_Price"].diff().fillna(0)
    data_last = df_last[cols].values.astype(np.float32)
    N_last = data_last.shape[0]

    if N_last < 2:
        print(f"Last file {last_file} has insufficient data (less than 2 rows). Cannot proceed with tasks.")
        # exit() # Or handle
    else:
        user_target_point_one_based = -1
        while True:
            try:
                # Calculate dynamic example values for the prompt
                example_target_point = N_last // 2 if N_last > 1 else N_last # Or any other sensible example like 374 if N_last is large enough
                if N_last <= 0: example_target_point = 1 # default if N_last is 0 or 1
                else: example_target_point = min(N_last, max(1, N_last // 2)) # Ensure it's within 1 to N_last
                
                fixed_cutoff_example = max(0, N_last - 50)

                prompt_msg = (f"Enter the target data point number (1 to {N_last}) for {last_file}.\n"
                              f"Example: If you enter {example_target_point}, predictions will aim to include data point {example_target_point} (index {example_target_point-1}).\n"
                              f"Two prediction tasks will be run based on this:\n"
                              f"1. Predict 50 steps ending at point {example_target_point}.\n"
                              f"2. Predict from a fixed cutoff (index {fixed_cutoff_example}) up to point {example_target_point}.\n"
                              f"Your choice: ")
                target_day_str = input(prompt_msg)
                user_target_point_one_based = int(target_day_str)
                if 1 <= user_target_point_one_based <= N_last:
                    break
                else:
                    print(f"Please enter a number between 1 and {N_last}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        target_idx_inclusive = user_target_point_one_based - 1 # Convert to 0-based index

        # Shared state for walk-forward training on the last file
        # Initialized as if we are about to process data_last[1] using data_last[0]
        # history_X stores inputs, history_y stores targets
        last_file_history_X = [torch.tensor(data_last[0])] 
        last_file_history_y = [data_last[1]] # Store as numpy array
        # current_train_idx_last_file is the *next data index i* to be potentially incorporated into history_X
        # after a training step. If it's 1, means history_X has data[0], history_y has data[1].
        # Training loop `for i in range(current_train_idx_last_file, ...)` will process data[i] and data[i+1].
        current_train_idx_last_file = 1 


        # --- Task 1: Predict 50 points ending at target_idx_inclusive ---
        print(f"\n=== TASK 1: Predict 50 steps leading up to user-specified point {user_target_point_one_based} (index {target_idx_inclusive}) ===")
        task1_pred_end_idx = target_idx_inclusive
        task1_num_steps_target = 50
        # Start prediction from this index (inclusive)
        task1_pred_start_idx = max(0, task1_pred_end_idx - task1_num_steps_target + 1)
        # Walk-forward training should cover data up to one before prediction starts
        task1_train_until_idx = task1_pred_start_idx - 1

        if task1_train_until_idx >= 0 :
            # current_train_idx_last_file is 1 initially.
            # target_train_until_idx_inclusive wants history_X to include this index eventually.
            current_train_idx_last_file = run_walk_forward_training_segment(
                model, optimizer, loss_fn, data_last,
                target_train_until_idx_inclusive=task1_train_until_idx,
                current_trained_idx_exclusive=current_train_idx_last_file,
                history_X_live=last_file_history_X,
                history_y_live=last_file_history_y,
                schedule_sampling_prob=SCHEDULED_SAMPLING_PROB,
                epochs_per_step=EPOCHS_PER_TRAIN_STEP,
                label_prefix="[Task 1 Setup] "
            )
        else:
            print("[Task 1 Setup] No prior walk-forward training on this file needed for this task (prediction starts at or before index 0).")

        run_autoregressive_prediction_segment(
            model, data_last,
            # Seed autoregression with data up to one before prediction starts
            seed_data_until_idx_inclusive = task1_pred_start_idx - 1, 
            predict_from_idx_inclusive = task1_pred_start_idx,
            predict_until_idx_inclusive = task1_pred_end_idx,
            label_prefix="[Task 1] "
        )

        # --- Task 2: Predict from N_last-50 up to target_idx_inclusive ---
        print(f"\n=== TASK 2: Predict from fixed cutoff (index {max(0, N_last - 50)}) up to user-specified point {user_target_point_one_based} (index {target_idx_inclusive}) ===")
        task2_pred_start_idx = max(0, N_last - 50)
        task2_pred_end_idx = target_idx_inclusive 
        task2_train_until_idx = task2_pred_start_idx - 1

        if task2_pred_start_idx > task2_pred_end_idx:
            print("[Task 2] Skipping prediction: start index for prediction is after target end index.")
        else:
            # current_train_idx_last_file reflects training done for Task 1.
            # Check if more training is needed. Model needs to be trained up to task2_train_until_idx.
            # current_train_idx_last_file is exclusive, so model is trained up to current_train_idx_last_file - 2 for features and current_train_idx_last_file -1 for targets used in history_y.
            # More simply: history_X contains data up to index (current_train_idx_last_file - 1).
            # We need history_X to contain data up to task2_train_until_idx.
            if task2_train_until_idx >= (current_train_idx_last_file -1) : # If target training index is beyond current history
                 if task2_train_until_idx < 0 : # Should generally not happen if N_last-50 > 0
                     print(f"[Task 2 Setup] No training needed as task2_train_until_idx is {task2_train_until_idx}.")
                 else:
                    current_train_idx_last_file = run_walk_forward_training_segment(
                        model, optimizer, loss_fn, data_last,
                        target_train_until_idx_inclusive=task2_train_until_idx,
                        current_trained_idx_exclusive=current_train_idx_last_file,
                        history_X_live=last_file_history_X,
                        history_y_live=last_file_history_y,
                        schedule_sampling_prob=SCHEDULED_SAMPLING_PROB,
                        epochs_per_step=EPOCHS_PER_TRAIN_STEP,
                        label_prefix="[Task 2 Setup] "
                    )
            else:
                 print(f"[Task 2 Setup] Model already trained sufficiently (history includes up to index {current_train_idx_last_file -1}) for this task (needs training for history up to {task2_train_until_idx}).")

            run_autoregressive_prediction_segment(
                model, data_last,
                seed_data_until_idx_inclusive = task2_pred_start_idx - 1,
                predict_from_idx_inclusive = task2_pred_start_idx,
                predict_until_idx_inclusive = task2_pred_end_idx,
                label_prefix="[Task 2] "
            )
plt.show() # Show all plots at the end if any were non-blocking
print("\nAll tasks complete.")
