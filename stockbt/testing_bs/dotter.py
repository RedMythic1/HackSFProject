import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from functools import lru_cache
from copy import deepcopy
import time

# Create folders for visualization
os.makedirs('trees', exist_ok=True)
os.makedirs('test_images', exist_ok=True)

# Directory containing all CSV files
data_dir = "stockbt/testing_bs/data_folder"
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

# FAST DEV MODE: reduced data, epochs, and optimization steps for speed
POPULATION_SIZE = 3
GENETIC_GENERATIONS = 3
PATTERN_SEARCH_ITERATIONS = 3

# Define attention block with extra layers
class DeepSelfAttention(nn.Module):
    def __init__(self, d_model, num_layers=6):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.final_weight = nn.Parameter(torch.randn(d_model, 1))  # 6d weight vector
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
        out = out @ self.final_weight
        return out.squeeze(-1)

# Optimization constants
INITIAL_LEARNING_RATE = 0.02
MIN_LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.05
MOMENTUM = 0.8
DECAY_RATE = 0.9
GRADIENT_STEPS = 20
EPSILON = 0.0001
PATIENCE = 3

# Genetic algorithm parameters
POPULATION_SIZE = 10
GENETIC_GENERATIONS = 10
MUTATION_RANGE = 0.2

# Pattern search parameters
INITIAL_STEP_SIZE = 0.05
MIN_STEP_SIZE = 0.0001
STEP_REDUCTION_FACTOR = 0.5
PATTERN_SEARCH_ITERATIONS = 15
RANDOM_RESTART_COUNT = 3

# Define bounds for hyperparameters
HYPERPARAM_BOUNDS = {
    'num_layers': (1, 50),           # Integer values
    'learning_rate': (0.0001, 0.05), # Float values
    'epochs_per_step': (1, 5),      # Integer values
}

def round_to_nearest(value, precision=0.0001):
    """Round a value to the nearest precision."""
    if isinstance(value, int):
        return value
    return round(value / precision) * precision

@lru_cache(maxsize=128)
def evaluate_model_cached(train_files_tuple, val_file, param_tuple):
    """Cached version of model evaluation that accepts immutable parameters"""
    hyperparams = {
        'num_layers': int(param_tuple[0]),
        'learning_rate': param_tuple[1],
        'epochs_per_step': int(param_tuple[2])
    }
    
    return evaluate_hyperparams(hyperparams, list(train_files_tuple), val_file)

def train_and_validate(model, optimizer, loss_fn, epochs, train_files, val_file):
    """Train on training files and evaluate on validation file"""
    print(f"  Training on {len(train_files)} files, will validate on {val_file}")
    
    for file_idx, file in enumerate(train_files):
        print(f"    Training file {file_idx+1}/{len(train_files)}: {file}")
        df = pd.read_csv(os.path.join(data_dir, file))
        data = df[["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]].values.astype(np.float32)
        data = data[:40]  # FAST: only use first 40 rows
        print(f"      Data shape: {data.shape}, training on {len(data)-2} steps")
        
        history_X = [torch.tensor(data[0])]
        history_y = [data[1, 0]]
        
        train_losses = []
        for i in range(1, len(data)-1):
            X_train = torch.stack(history_X)
            y_train = torch.tensor(history_y)
            
            epoch_losses = []
            early_stop_count = 0
            last_loss = None
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_train)
                loss = loss_fn(pred[-1:], y_train[-1:])
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                # Early stopping: if loss < 1e-3 for 2 consecutive epochs
                if loss.item() < 1e-3:
                    if last_loss is not None and last_loss < 1e-3:
                        early_stop_count += 1
                    else:
                        early_stop_count = 1
                    if early_stop_count >= 2:
                        print(f"        Early stopping at epoch {epoch+1} (loss < 1e-3)")
                        break
                last_loss = loss.item()
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_epoch_loss)
            
            # Print progress every 20% of the way through
            if i % max(1, (len(data)-2) // 5) == 0:
                print(f"      Step {i}/{len(data)-2}, Avg epoch loss: {avg_epoch_loss:.6f}")
            
            history_X.append(torch.tensor(data[i]))
            history_y = list(y_train.numpy()) + [data[i+1, 0]]
        
        print(f"      Completed training on {file} with avg loss: {sum(train_losses)/len(train_losses):.6f}")
    
    # Validate on val_file
    print(f"  Validating on: {val_file}")
    df = pd.read_csv(os.path.join(data_dir, val_file))
    data = df[["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]].values.astype(np.float32)
    data = data[:40]  # FAST: only use first 40 rows
    history_X = [torch.tensor(data[0])]
    history_y = [data[1, 0]]
    val_losses = []
    
    print(f"    Validation data shape: {data.shape}, validating on {len(data)-2} steps")
    for i in range(1, len(data)-1):
        X_train = torch.stack(history_X)
        y_train = torch.tensor(history_y)
        model.eval()
        with torch.no_grad():
            X_pred = torch.stack(history_X + [torch.tensor(data[i])])
            pred_next = model(X_pred)[-1].item()
            squared_error = (pred_next - data[i+1, 0])**2
            val_losses.append(squared_error)
        
        # Print progress every 20% of the way through
        if i % max(1, (len(data)-2) // 5) == 0:
            print(f"    Validation step {i}/{len(data)-2}, Squared error: {squared_error:.6f}")
            print(f"      Predicted: {pred_next:.4f}, Actual: {data[i+1, 0]:.4f}")
        
        history_X.append(torch.tensor(data[i]))
        history_y = list(y_train.numpy()) + [data[i+1, 0]]
    
    mean_val_loss = np.mean(val_losses)
    print(f"  Validation complete. MSE Loss: {mean_val_loss:.6f}")
    return mean_val_loss

def evaluate_hyperparams(hyperparams, train_files, val_file):
    """Evaluate hyperparameters by training and validating a model.
       Returns the validation MSE loss (lower is better).
    """
    print(f"\nEvaluating hyperparameters: {hyperparams}")
    
    # Create a new model with these hyperparameters
    model = DeepSelfAttention(d_model=6, num_layers=hyperparams['num_layers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    loss_fn = nn.MSELoss()
    epochs = hyperparams['epochs_per_step']
    
    print(f"  Model architecture: {model}")
    print(f"  Optimizer: Adam(lr={hyperparams['learning_rate']})")
    print(f"  Loss function: MSE")
    print(f"  Epochs per step: {epochs}")
    
    # Train and validate
    val_loss = train_and_validate(model, optimizer, loss_fn, epochs, train_files, val_file)
    print(f"  Evaluation complete. Validation loss: {val_loss:.6f}")
    return val_loss

def generate_random_hyperparams(bounds):
    """Generate random hyperparameters within bounds."""
    hyperparams = {
        'num_layers': random.randint(bounds['num_layers'][0], bounds['num_layers'][1]),
        'learning_rate': round_to_nearest(random.uniform(bounds['learning_rate'][0], bounds['learning_rate'][1])),
        'epochs_per_step': random.randint(bounds['epochs_per_step'][0], bounds['epochs_per_step'][1])
    }
    print(f"Generated random hyperparameters: {hyperparams}")
    return hyperparams

def initialize_population(bounds, size=POPULATION_SIZE):
    """Initialize a population of hyperparameter sets."""
    population = []
    
    # Include a default hyperparameters set
    center_hyperparams = {
        'num_layers': 10,
        'learning_rate': 0.001,
        'epochs_per_step': 10
    }
    population.append(center_hyperparams)
    
    # Generate random individuals for the rest of the population
    for _ in range(size - 1):
        population.append(generate_random_hyperparams(bounds))
    
    return population

def evaluate_population(train_files, val_file, population):
    """Evaluate fitness (negative validation loss) of each set of hyperparameters."""
    fitness_scores = []
    train_files_tuple = tuple(train_files)  # Convert to tuple for caching
    
    print(f"\nEvaluating population of {len(population)} hyperparameter sets...")
    for idx, hyperparams in enumerate(population):
        print(f"\nEvaluating hyperparameter set {idx+1}/{len(population)}:")
        
        # Create param tuple for caching
        param_tuple = (
            hyperparams['num_layers'],
            hyperparams['learning_rate'],
            hyperparams['epochs_per_step']
        )
        
        # Get negative loss (higher is better)
        print(f"  Starting evaluation for: {hyperparams}")
        start_time = time.time()
        loss = evaluate_model_cached(train_files_tuple, val_file, param_tuple)
        end_time = time.time()
        fitness = -loss  # We want to maximize fitness (minimize loss)
        fitness_scores.append(fitness)
        
        print(f"  Evaluation completed in {end_time - start_time:.2f} seconds")
        print(f"  Validation loss: {loss:.6f}, Fitness score: {fitness:.6f}")
    
    print("\nPopulation evaluation complete!")
    best_idx = np.argmax(fitness_scores)
    print(f"Best hyperparameters in this population: {population[best_idx]}")
    print(f"Best fitness score: {fitness_scores[best_idx]:.6f} (loss: {-fitness_scores[best_idx]:.6f})")
    
    return fitness_scores

def generate_offspring(best_hyperparams, bounds, size=POPULATION_SIZE):
    """Generate new population based on the best hyperparameters with random variations."""
    offspring = [best_hyperparams.copy()]  # Keep the best hyperparameters unchanged
    
    # Generate variations of the best hyperparameters
    for _ in range(size - 1):
        child = {}
        for key, val in best_hyperparams.items():
            # Get parameter bounds
            min_val, max_val = bounds[key]
            range_size = max_val - min_val
            
            # Add random variation based on parameter range
            variation = random.uniform(-MUTATION_RANGE, MUTATION_RANGE) * range_size
            if key in ['num_layers', 'epochs_per_step']:
                new_val = int(val + variation)
            else:
                new_val = val + variation
            
            # Ensure value stays within bounds
            new_val = max(min_val, min(max_val, new_val))
            
            # Round to precision for float values
            if key == 'learning_rate':
                new_val = round_to_nearest(new_val)
            
            child[key] = new_val
        
        offspring.append(child)
    
    return offspring

def run_simple_genetic_algorithm(train_files, val_file, bounds, generations=GENETIC_GENERATIONS):
    """Run genetic algorithm to find optimal hyperparameters."""
    print("\nStarting Genetic Algorithm for hyperparameter optimization...")
    
    # Initialize population
    population = initialize_population(bounds)
    
    all_generations = []
    all_fitness_scores = []
    best_hyperparams = None
    best_fitness = float('-inf')
    
    for generation in range(generations):
        # Evaluate current population
        fitness_scores = evaluate_population(train_files, val_file, population)
        
        # Store current generation and fitness scores
        all_generations.append(deepcopy(population))
        all_fitness_scores.append(fitness_scores)
        
        # Find the best individual in this generation
        current_best_idx = np.argmax(fitness_scores)
        current_best_hyperparams = population[current_best_idx]
        current_best_fitness = fitness_scores[current_best_idx]
        current_best_loss = -current_best_fitness
        
        # Update overall best if improved
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_hyperparams = deepcopy(current_best_hyperparams)
        
        # Print generation stats
        print(f"\nGeneration {generation + 1}/{generations}:")
        print(f"Best validation loss: {current_best_loss:.6f}, Avg loss: {-np.mean(fitness_scores):.6f}")
        print(f"Best hyperparameters: {current_best_hyperparams}")
        
        # Create next generation based on the best individual (except for last iteration)
        if generation < generations - 1:
            population = generate_offspring(current_best_hyperparams, bounds)
    
    print("\nGenetic Algorithm completed:")
    print(f"Best validation loss found: {-best_fitness:.6f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    
    return best_hyperparams, -best_fitness, (all_generations, all_fitness_scores)

def calculate_numerical_gradient(train_files, val_file, hyperparams, epsilon=EPSILON):
    """Calculate numerical gradient for each hyperparameter using finite differences."""
    gradients = {}
    train_files_tuple = tuple(train_files)
    
    # Calculate base loss
    param_tuple = (
        hyperparams['num_layers'],
        hyperparams['learning_rate'],
        hyperparams['epochs_per_step']
    )
    base_loss = evaluate_model_cached(train_files_tuple, val_file, param_tuple)
    
    # Calculate gradient for each parameter
    for param_name in hyperparams:
        # Adjust epsilon based on parameter value
        param_value = hyperparams[param_name]
        adaptive_epsilon = max(epsilon, abs(param_value * 0.01))  # At least 1% of parameter value
        
        # For integer parameters, use at least 1
        if param_name in ['num_layers', 'epochs_per_step']:
            adaptive_epsilon = max(1, adaptive_epsilon)
        
        # Create copy with increased parameter
        hyperparams_plus = hyperparams.copy()
        hyperparams_plus[param_name] += adaptive_epsilon
        
        # Calculate loss with increased parameter
        param_tuple_plus = (
            hyperparams_plus['num_layers'],
            hyperparams_plus['learning_rate'],
            hyperparams_plus['epochs_per_step']
        )
        loss_plus = evaluate_model_cached(train_files_tuple, val_file, param_tuple_plus)
        
        # Calculate gradient (negative since we want to minimize loss)
        gradient = -(loss_plus - base_loss) / adaptive_epsilon
        gradients[param_name] = gradient
    
    return gradients

def pattern_search_optimization(train_files, val_file, initial_hyperparams, bounds, max_iterations=PATTERN_SEARCH_ITERATIONS):
    """Apply pattern search optimization to find optimal hyperparameters."""
    print("\nStarting Pattern Search for hyperparameter optimization...")
    
    # Initial evaluation
    train_files_tuple = tuple(train_files)
    param_tuple = (
        initial_hyperparams['num_layers'],
        initial_hyperparams['learning_rate'],
        initial_hyperparams['epochs_per_step']
    )
    initial_loss = evaluate_model_cached(train_files_tuple, val_file, param_tuple)
    
    current_hyperparams = initial_hyperparams.copy()
    best_hyperparams = initial_hyperparams.copy()
    best_loss = initial_loss
    
    # Calculate initial step sizes based on parameter ranges
    step_sizes = {}
    for param_name, (min_val, max_val) in bounds.items():
        range_size = max_val - min_val
        step_sizes[param_name] = range_size * INITIAL_STEP_SIZE
        # Ensure integer parameters have integer steps
        if param_name in ['num_layers', 'epochs_per_step']:
            step_sizes[param_name] = max(1, int(step_sizes[param_name]))
    
    print(f"\nStarting pattern search from: {initial_hyperparams}")
    print(f"Initial validation loss: {best_loss:.6f}")
    
    history = [(0, best_loss, best_hyperparams.copy())]
    iteration = 0
    restart_count = 0
    no_improvement_count = 0
    
    while iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Try each coordinate direction
        for param_name in current_hyperparams:
            # Try positive direction
            test_hyperparams = current_hyperparams.copy()
            test_hyperparams[param_name] += step_sizes[param_name]
            
            # Ensure within bounds
            min_val, max_val = bounds[param_name]
            test_hyperparams[param_name] = max(min_val, min(max_val, test_hyperparams[param_name]))
            
            # Round appropriately
            if param_name == 'learning_rate':
                test_hyperparams[param_name] = round_to_nearest(test_hyperparams[param_name])
            elif param_name in ['num_layers', 'epochs_per_step']:
                test_hyperparams[param_name] = int(test_hyperparams[param_name])
            
            # Calculate loss
            test_param_tuple = (
                test_hyperparams['num_layers'],
                test_hyperparams['learning_rate'],
                test_hyperparams['epochs_per_step']
            )
            test_loss = evaluate_model_cached(train_files_tuple, val_file, test_param_tuple)
            
            # If improved, update current hyperparameters
            if test_loss < best_loss:
                current_hyperparams = test_hyperparams.copy()
                best_hyperparams = test_hyperparams.copy()
                improvement = best_loss - test_loss
                best_loss = test_loss
                improved = True
                print(f"Iteration {iteration}, {param_name}+: New best loss: {best_loss:.6f} (-{improvement:.6f})")
                continue
            
            # Try negative direction
            test_hyperparams = current_hyperparams.copy()
            test_hyperparams[param_name] -= step_sizes[param_name]
            
            # Ensure within bounds
            test_hyperparams[param_name] = max(min_val, min(max_val, test_hyperparams[param_name]))
            
            # Round appropriately
            if param_name == 'learning_rate':
                test_hyperparams[param_name] = round_to_nearest(test_hyperparams[param_name])
            elif param_name in ['num_layers', 'epochs_per_step']:
                test_hyperparams[param_name] = int(test_hyperparams[param_name])
            
            # Calculate loss
            test_param_tuple = (
                test_hyperparams['num_layers'],
                test_hyperparams['learning_rate'],
                test_hyperparams['epochs_per_step']
            )
            test_loss = evaluate_model_cached(train_files_tuple, val_file, test_param_tuple)
            
            # If improved, update current hyperparameters
            if test_loss < best_loss:
                current_hyperparams = test_hyperparams.copy()
                best_hyperparams = test_hyperparams.copy()
                improvement = best_loss - test_loss
                best_loss = test_loss
                improved = True
                print(f"Iteration {iteration}, {param_name}-: New best loss: {best_loss:.6f} (-{improvement:.6f})")
        
        # Track history
        history.append((iteration, best_loss, best_hyperparams.copy()))
        
        # If no improvement in any direction, reduce step size
        if not improved:
            no_improvement_count += 1
            
            # Check if we should do a random restart
            smallest_step = min(step_sizes.values())
            if (smallest_step <= MIN_STEP_SIZE or 
                (param_name in ['num_layers', 'epochs_per_step'] and smallest_step <= 1)) and restart_count < RANDOM_RESTART_COUNT:
                restart_count += 1
                print(f"\nNo further improvement with current step size. Random restart ({restart_count}/{RANDOM_RESTART_COUNT})...")
                
                # Keep best hyperparameters but add random perturbation
                current_hyperparams = best_hyperparams.copy()
                for param_name in current_hyperparams:
                    min_val, max_val = bounds[param_name]
                    range_size = max_val - min_val
                    # More substantial perturbation for restart
                    perturbation = random.uniform(-0.2, 0.2) * range_size
                    if param_name in ['num_layers', 'epochs_per_step']:
                        perturbation = int(perturbation)
                        # Ensure at least ±1 change for integer parameters
                        if perturbation == 0:
                            perturbation = random.choice([-1, 1])
                    current_hyperparams[param_name] += perturbation
                    current_hyperparams[param_name] = max(min_val, min(max_val, current_hyperparams[param_name]))
                    if param_name == 'learning_rate':
                        current_hyperparams[param_name] = round_to_nearest(current_hyperparams[param_name])
                
                # Reset step sizes
                for param_name in step_sizes:
                    range_size = bounds[param_name][1] - bounds[param_name][0]
                    step_sizes[param_name] = range_size * INITIAL_STEP_SIZE
                    if param_name in ['num_layers', 'epochs_per_step']:
                        step_sizes[param_name] = max(1, int(step_sizes[param_name]))
                
                print(f"New starting point: {current_hyperparams}")
                no_improvement_count = 0
            elif smallest_step <= MIN_STEP_SIZE or (param_name in ['num_layers', 'epochs_per_step'] and smallest_step <= 1):
                # If we've exhausted all restarts and step size is too small, terminate
                print(f"\nSearch converged - minimum step size reached after {iteration} iterations.")
                break
            else:
                # Reduce step sizes
                for param_name in step_sizes:
                    step_sizes[param_name] *= STEP_REDUCTION_FACTOR
                    if param_name in ['num_layers', 'epochs_per_step']:
                        step_sizes[param_name] = max(1, int(step_sizes[param_name]))
                
                print(f"Iteration {iteration}: No improvement, reducing step sizes - {', '.join([f'{k}={v}' for k, v in step_sizes.items()])}")
        else:
            no_improvement_count = 0
        
        # If we've hit max iterations, terminate
        if iteration >= max_iterations:
            print(f"\nReached maximum iterations: {max_iterations}")
            break
    
    print(f"\nPattern search completed after {iteration} iterations:")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    
    return best_hyperparams, best_loss, history

def create_visualization(ga_data, ga_best, ps_history, filename="hyperparam_optimization.png"):
    """Create a visualization of the optimization process."""
    all_generations, all_fitness_scores = ga_data
    
    plt.figure(figsize=(20, 12))
    
    # Extract data
    ga_generations = list(range(1, len(all_generations) + 1))
    ga_best_losses = [-max(scores) for scores in all_fitness_scores]  # Convert fitness back to loss
    
    ps_steps = [len(ga_generations) + step for step, _, _ in ps_history]
    ps_losses = [loss for _, loss, _ in ps_history]
    
    # Combine into full optimization timeline
    all_steps = ga_generations + ps_steps
    all_losses = ga_best_losses + ps_losses
    
    # Plot combined loss progress
    plt.subplot(2, 1, 1)
    # Genetic algorithm phase
    plt.plot(ga_generations, ga_best_losses, 'r-o', linewidth=2, label='Genetic Algorithm')
    # Pattern search phase
    plt.plot(ps_steps, ps_losses, 'b-o', linewidth=2, label='Pattern Search')
    
    # Add vertical line to separate phases
    plt.axvline(x=len(ga_generations), color='gray', linestyle='--', alpha=0.7)
    
    # Annotate transition point
    ga_final_loss = ga_best_losses[-1]
    plt.scatter(len(ga_generations), ga_final_loss, color='purple', s=100, zorder=5)
    plt.annotate(f'GA→PS: {ga_final_loss:.6f}', 
                 xy=(len(ga_generations), ga_final_loss),
                 xytext=(len(ga_generations) + 0.5, ga_final_loss),
                 fontsize=10,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Annotate best loss
    best_idx = np.argmin(all_losses)
    best_step = all_steps[best_idx]
    best_loss = all_losses[best_idx]
    plt.scatter(best_step, best_loss, color='gold', s=120, zorder=6)
    plt.annotate(f'Best: {best_loss:.6f}', 
                 xy=(best_step, best_loss),
                 xytext=(best_step + 0.5, best_loss),
                 fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.title('Hybrid Optimization Process (Genetic Algorithm + Pattern Search)', fontsize=14)
    plt.xlabel('Optimization Step', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot hyperparameter trajectories
    plt.subplot(2, 1, 2)
    
    # Get hyperparameter names
    param_names = list(ga_best.keys())
    
    # Extract GA best hyperparameter progression
    ga_params_history = {}
    for param_name in param_names:
        ga_params_history[param_name] = []
        for gen_idx in range(len(all_generations)):
            # Find the best individual in each generation
            best_idx = np.argmax(all_fitness_scores[gen_idx])
            best_individual = all_generations[gen_idx][best_idx]
            ga_params_history[param_name].append(best_individual[param_name])
    
    # Extract PS hyperparameter history
    ps_params = [param for _, _, param in ps_history]
    
    # Plot hyperparameter trajectories
    for param_name in param_names:
        # GA phase
        plt.plot(ga_generations, ga_params_history[param_name], 'o-', linewidth=1.5, 
                 label=f'{param_name} (GA)', alpha=0.7)
        
        # PS phase
        ps_param_values = [p[param_name] for p in ps_params]
        plt.plot(ps_steps, ps_param_values, 'o-', linewidth=1.5, 
                 label=f'{param_name} (PS)')
    
    # Add vertical line to separate phases
    plt.axvline(x=len(ga_generations), color='gray', linestyle='--', alpha=0.7)
    
    plt.title('Hyperparameter Trajectories During Optimization', fontsize=14)
    plt.xlabel('Optimization Step', fontsize=12)
    plt.ylabel('Hyperparameter Value', fontsize=12)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('trees', filename), dpi=300, bbox_inches='tight')
    plt.show()

def run_hybrid_optimization(train_files, val_file, bounds=HYPERPARAM_BOUNDS):
    """Run hybrid optimization (GA + Pattern Search) to find optimal hyperparameters."""
    print(f"\n{'=' * 80}")
    print(f"STARTING HYBRID OPTIMIZATION FOR NEURAL NETWORK HYPERPARAMETERS")
    print(f"{'=' * 80}")
    print(f"\nOptimizing the following hyperparameters:")
    for param, (min_val, max_val) in bounds.items():
        print(f"  {param}: range [{min_val}, {max_val}]")
    
    # Phase 1: Genetic Algorithm
    print(f"\n{'=' * 80}")
    print(f"PHASE 1: GENETIC ALGORITHM OPTIMIZATION")
    print(f"{'=' * 80}")
    print(f"Strategy: Evolve a population of {POPULATION_SIZE} hyperparameter sets for {GENETIC_GENERATIONS} generations")
    print(f"Parameters: Mutation range: {MUTATION_RANGE}, Selection: top performing sets")
    
    start_time = time.time()
    ga_best, ga_best_loss, ga_data = run_simple_genetic_algorithm(
        train_files, val_file, bounds, generations=GENETIC_GENERATIONS)
    ga_time = time.time() - start_time
    print(f"\nGenetic algorithm completed in {ga_time:.2f} seconds")
    print(f"GA found best hyperparameters: {ga_best}")
    print(f"GA best validation loss: {ga_best_loss:.6f}")
    
    # Phase 2: Pattern Search
    print(f"\n{'=' * 80}")
    print(f"PHASE 2: PATTERN SEARCH OPTIMIZATION")
    print(f"{'=' * 80}")
    print(f"Strategy: Fine-tune the best GA solution with coordinate descent")
    print(f"Parameters: Initial step size: {INITIAL_STEP_SIZE}, Min step size: {MIN_STEP_SIZE}")
    print(f"            Step reduction factor: {STEP_REDUCTION_FACTOR}, Max iterations: {PATTERN_SEARCH_ITERATIONS}")
    print(f"            Random restarts: {RANDOM_RESTART_COUNT}")
    
    start_time = time.time()
    final_hyperparams, final_loss, ps_history = pattern_search_optimization(
        train_files, val_file, ga_best, bounds, max_iterations=PATTERN_SEARCH_ITERATIONS)
    ps_time = time.time() - start_time
    print(f"\nPattern search completed in {ps_time:.2f} seconds")
    
    # Create visualization
    print("\nCreating visualization of the optimization process...")
    create_visualization(ga_data, ga_best, ps_history, "neural_network_hyperparams.png")
    print("Visualization saved to trees/neural_network_hyperparams.png")
    
    # Calculate improvement
    improvement = ga_best_loss - final_loss
    improvement_pct = (improvement / ga_best_loss) * 100 if ga_best_loss != 0 else float('inf')
    
    print(f"\n{'=' * 80}")
    print(f"HYBRID OPTIMIZATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Total optimization time: {ga_time + ps_time:.2f} seconds")
    print(f"Initial GA best validation loss: {ga_best_loss:.6f}")
    print(f"Final validation loss: {final_loss:.6f}")
    print(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
    print(f"\nInitial GA best hyperparameters: {ga_best}")
    print(f"\nFinal optimized hyperparameters: {final_hyperparams}")
    
    return final_hyperparams, final_loss

# Split files into train and validation sets
train_files = files[:-1]
val_file = files[-1]

# Run hybrid optimization to find the best hyperparameters
print(f"\nFound {len(files)} CSV files for sequential training.")
print(f"Using {len(train_files)} files for training and {val_file} for validation.")

best_hyperparams, best_loss = run_hybrid_optimization(train_files, val_file)
print("\nBest hyperparameters found:", best_hyperparams)

# Now train and plot with the best hyperparameters
model = DeepSelfAttention(d_model=6, num_layers=best_hyperparams['num_layers'])
optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
loss_fn = nn.MSELoss()
epochs = best_hyperparams['epochs_per_step']

print("\n--- FINAL TRAINING WITH BEST HYPERPARAMETERS ---")
print(f"Model architecture: DeepSelfAttention(d_model=6, num_layers={best_hyperparams['num_layers']})")
print(f"Optimizer: Adam(lr={best_hyperparams['learning_rate']})")
print(f"Loss function: MSE")
print(f"Epochs per step: {epochs}")

plt.figure(figsize=(14, 7))
for idx, file in enumerate(train_files):
    print(f"\n=== Training on file {idx+1}/{len(train_files)}: {file} ===")
    df = pd.read_csv(os.path.join(data_dir, file))
    cols = ["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]
    data = df[cols].values.astype(np.float32)
    data = data[:40]  # FAST: only use first 40 rows
    print(f"Data shape: {data.shape}, Will train on {len(data)-2} steps")

    history_X = [torch.tensor(data[0])]
    history_y = [data[1, 0]]
    predictions = []
    actuals = []

    for i in range(1, len(data)-1):
        X_train = torch.stack(history_X)
        y_train = torch.tensor(history_y)
        
        # Track losses for all epochs
        epoch_losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred[-1:], y_train[-1:])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        if i % 10 == 0:
            print(f"  Step {i}/{len(data)-2}, X shape: {X_train.shape}, Final loss: {epoch_losses[-1]:.6f}")
        
        model.eval()
        with torch.no_grad():
            X_pred = torch.stack(history_X + [torch.tensor(data[i])])
            pred_next = model(X_pred)[-1].item()
            if i % 10 == 0:
                print(f"    Predicted: {pred_next:.4f}, Actual: {data[i+1, 0]:.4f}, Error: {abs(pred_next - data[i+1, 0]):.4f}")
        
        predictions.append(pred_next)
        actuals.append(data[i+1, 0])
        history_X.append(torch.tensor(data[i]))
        history_y = list(y_train.numpy()) + [data[i+1, 0]]

    print(f"  Finished training on {file}.")
    print(f"  MSE on predictions: {np.mean([(p-a)**2 for p, a in zip(predictions, actuals)]):.6f}")
    print(f"  First 5 predictions: {predictions[:5]}")
    print(f"  First 5 actuals: {actuals[:5]}")
    print(f"  Last 5 predictions: {predictions[-5:]}")
    print(f"  Last 5 actuals: {actuals[-5:]}")
    
    print(f"  Overlaying predictions and actuals on plot.")
    plt.plot(range(2, len(predictions)+2), predictions, alpha=0.4, label=f'Train Predicted ({file})' if idx==0 else None, color='tab:orange')
    plt.plot(range(2, len(actuals)+2), actuals, alpha=0.2, label=f'Train Actual ({file})' if idx==0 else None, color='tab:blue')

# Now train on the last file up to the last data point, then predict the last value
print(f"\n=== Training (walk-forward) on last file: {val_file} ===")
df = pd.read_csv(os.path.join(data_dir, val_file))
cols = ["Price", "Buy_Vol", "Bid_Price", "Sell_Vol", "Ask_Price", "Change_From_Previous"]
data = df[cols].values.astype(np.float32)
data = data[:40]  # FAST: only use first 40 rows
print(f"Data shape: {data.shape}, Will train on {len(data)-2} steps")

history_X = [torch.tensor(data[0])]
history_y = [data[1, 0]]
test_predictions = []
test_actuals = []

for i in range(1, len(data)-1):
    X_train = torch.stack(history_X)
    y_train = torch.tensor(history_y)
    
    # Track losses for all epochs
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred[-1:], y_train[-1:])
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        X_pred = torch.stack(history_X + [torch.tensor(data[i])])
        pred_next = model(X_pred)[-1].item()
    
    test_predictions.append(pred_next)
    test_actuals.append(data[i+1, 0])
    history_X.append(torch.tensor(data[i]))
    history_y = list(y_train.numpy()) + [data[i+1, 0]]
    
    if i % 10 == 0:
        print(f"  Step {i}/{len(data)-2}, X shape: {X_train.shape}")
        print(f"    Predicted: {pred_next:.4f}, Actual: {data[i+1, 0]:.4f}, Error: {abs(pred_next - data[i+1, 0]):.4f}")
        print(f"    Final loss: {epoch_losses[-1]:.6f}")

print(f"\nTesting complete on {val_file}.")
print(f"MSE on test predictions: {np.mean([(p-a)**2 for p, a in zip(test_predictions, test_actuals)]):.6f}")
print(f"First 5 test predictions: {test_predictions[:5]}")
print(f"First 5 test actuals: {test_actuals[:5]}")
print(f"Last 5 test predictions: {test_predictions[-5:]}")
print(f"Last 5 test actuals: {test_actuals[-5:]}")

print(f"\nOverlaying test predictions and actuals for {val_file}.")
plt.plot(range(2, len(test_predictions)+2), test_predictions, label=f'Test Predicted ({val_file})', color='red', linewidth=2)
plt.plot(range(2, len(test_actuals)+2), test_actuals, label=f'Test Actual ({val_file})', color='black', linewidth=2, linestyle='--')

        plt.xlabel('Time Step')
        plt.ylabel('Price')
plt.title(f'Training with Optimized Hyperparameters: {best_hyperparams}')
        plt.legend()
        plt.tight_layout()
plt.savefig(os.path.join('trees', 'final_model_performance.png'), dpi=300)
print(f"Plot saved to trees/final_model_performance.png")
        plt.show()

if len(test_predictions) > 0:
    print(f"\nLast predicted value for {val_file}: {test_predictions[-1]:.4f}")
    print(f"Actual last value for {val_file}: {test_actuals[-1]:.4f}")
    print(f"Prediction error: {abs(test_predictions[-1] - test_actuals[-1]):.4f}")
    print(f"Prediction error percentage: {100 * abs(test_predictions[-1] - test_actuals[-1]) / test_actuals[-1]:.2f}%") 
else:
    print(f"\nNo predictions made for {val_file}.")

print("\nHyperparameter optimization and training complete!")
print(f"Final model hyperparameters: {best_hyperparams}")
print(f"Final validation loss: {best_loss:.6f}")
print("\nModel performance metrics:")
print(f"Training files: {train_files}")
print(f"Test file: {val_file}")
print(f"Test MSE: {np.mean([(p-a)**2 for p, a in zip(test_predictions, test_actuals)]):.6f}")
print(f"Test MAE: {np.mean([abs(p-a) for p, a in zip(test_predictions, test_actuals)]):.6f}")
print(f"Test RMSE: {np.sqrt(np.mean([(p-a)**2 for p, a in zip(test_predictions, test_actuals)])):.6f}")
