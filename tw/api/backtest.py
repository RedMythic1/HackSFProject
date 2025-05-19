import math
import csv
import os
import traceback
import re
import sys
import argparse
import json
import time
import random
import logging

# TODO: This module should be refactored to use the more comprehensive backtesting logic from st.py.
# This would include:
# 1. Importing the core functions from st.py
# 2. Adapting the run_improved_simulation to use st.py's run_simulation and other functions
# 3. Ensuring all features from st.py are properly integrated
# 4. Maintaining the same API interface for server.py
# 
# For now, this file remains as is, but a complete refactoring is recommended to centralize logic
# and avoid code duplication.

logger = logging.getLogger(__name__)

# Add the python_packages directory to the Python path
package_dir = os.path.join(os.path.dirname(__file__), 'python_packages')
sys.path.insert(0, package_dir)

# Now import packages from the local directory
import g4f
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend which doesn't require a display
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Plotting will be disabled.")
    MATPLOTLIB_AVAILABLE = False

# Track previously used datasets in this session
if 'used_datasets' not in globals():
    used_datasets = set()

# Set the initial balance for all simulations
initial_balance = 100000

# Set up charts directory
def get_charts_dir():
    """Get the directory for saving chart images."""
    datasets_dir = os.environ.get('DATASETS_DIR', '/data/datasets')
    if not os.path.exists(datasets_dir):
        # Try the parent directory's datasets folder if the specified one doesn't exist
        parent_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        if os.path.exists(parent_datasets_dir):
            datasets_dir = parent_datasets_dir
    
    charts_dir = os.path.join(datasets_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir

def get_random_dataset():
    """Get a random dataset from the datasets directory, tracking previously used ones."""
    datasets_dir = os.environ.get('DATASETS_DIR', '/data/datasets')
    if not os.path.exists(datasets_dir):
        logger.info(f"ERROR: Datasets directory {datasets_dir} does not exist!")
        # Try the parent directory's datasets folder if the specified one doesn't exist
        parent_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        if os.path.exists(parent_datasets_dir):
            logger.info(f"Found parent datasets directory: {parent_datasets_dir}")
            datasets_dir = parent_datasets_dir
        else:
            return None
    # Skip files starting with '._' (macOS resource forks)
    files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv') and not f.startswith('._')]
    if not files:
        logger.info(f"ERROR: No CSV files found in {datasets_dir}!")
        return None
    logger.info(f"Using datasets directory: {datasets_dir}")
    
    all_files = files
    
    # If specified, always use test.csv (for consistency in API testing)
    if os.environ.get('BACKTEST_USE_TEST_DATASET') == '1':
        test_path = os.path.join(datasets_dir, 'test.csv')
        if os.path.exists(test_path):
            return test_path
    
    # First, try to use a dataset that hasn't been used before
    unused_files = [f for f in all_files if f not in used_datasets]
    if not unused_files:
        # If all datasets have been used, reset and start over
        used_datasets.clear()
        unused_files = all_files
    
    chosen = random.choice(unused_files)
    used_datasets.add(chosen)
    return os.path.join(datasets_dir, chosen)

# -----------------------------------------------------------------------------
# Plot Results (similar to st.py)
# -----------------------------------------------------------------------------

def plot_results(close, buy_points, sell_points, balance_over_time, dataset_name="unknown", strategy_name="strategy"):
    """Generate a chart showing price and overlayed (scaled) balance over time."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping plot generation")
        return None
    
    logger.info("Generating results plot...")
    charts_dir = get_charts_dir()
    
    # Create a safe filename
    safe_strategy = re.sub(r'[^-\w\s-]', '', strategy_name[:30])
    safe_strategy = re.sub(r'\s+', '_', safe_strategy).strip('-_')
    safe_dataset = re.sub(r'[^-\w\s-]', '', dataset_name)
    safe_dataset = re.sub(r'\s+', '_', safe_dataset).strip('-_')
    
    timestamp = int(time.time())
    filename = f'backtest_{safe_strategy}_{safe_dataset}_{timestamp}.png'
    filepath = os.path.join(charts_dir, filename)
    
    try:
        plt.figure(figsize=(12, 10), dpi=100)
        # Plot 1: Price and overlayed balance
        plt.subplot(2, 1, 1)
        plt.plot(close, label='Close Price', linewidth=0.8, zorder=1, color='skyblue')
        # Scale balance_over_time to start at the same value as close[0]
        if balance_over_time and len(balance_over_time) > 1:
            bal = np.array(balance_over_time)
            # Scale so that bal[0] == close[0]
            if bal[0] != 0:
                bal_scaled = (bal / bal[0]) * close[0]
            else:
                bal_scaled = bal
            plt.plot(bal_scaled, label='Portfolio Value (scaled)', color='orange', linewidth=1.5, zorder=2)
        plt.title(f'Stock Price and Portfolio Value - {safe_dataset}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Plot 2: Balance over time (raw, not scaled)
        plt.subplot(2, 1, 2)
        if balance_over_time and len(balance_over_time) > 0:
            plt.plot(balance_over_time, label='Portfolio Value', color='blue', linewidth=1.5)
            plt.axhline(y=initial_balance, color='gray', linestyle='--', label=f'Initial ${initial_balance:,.0f}')
            final_balance = balance_over_time[-1] if balance_over_time else initial_balance
            plt.annotate(f'Final: ${final_balance:,.0f}', 
                         xy=(len(balance_over_time)-1, final_balance),
                         xytext=(len(balance_over_time)-10, final_balance*1.05),
                         arrowprops=dict(arrowstyle='->'))
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Plot saved to {filepath}")
        return filename
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return None

# -----------------------------------------------------------------------------
# GPT-4 FREE (g4f) helper
# -----------------------------------------------------------------------------

def ask_llama(prompt, temperature=None):
    """Ask a question to the GPT-4 Free backend via g4f. Always use temperature=0.1."""
    temperature = 0.1
    logger.info(f"\nSending prompt to g4f ChatCompletion with temperature={temperature}…")
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            provider=g4f.Provider.PollinationsAI,
            temperature=temperature
        )
        logger.info("Received response:" + response)
        return response.strip()
    except Exception as e:
        logger.info(f"ERROR getting response: {e}")
        return None

# -----------------------------------------------------------------------------
# Response parsing helpers (identical to original)
# -----------------------------------------------------------------------------

def split_response(response):
    logger.info("\nSplitting response into code blocks…")
    import ast

    # Print a diagnostic excerpt of the response
    if response:
        logger.info(f"Response length: {len(response)} chars")
        excerpt_len = min(200, len(response))
        logger.info(f"Response excerpt: {response[:excerpt_len]}...")
    else:
        logger.info("WARNING: Empty response received")
        raise ValueError("Empty response from LLM")

    # First, remove any markdown code fences
    cleaned_response = response
    if "```" in response:
        logger.info("Removing markdown code fences…")
        # Remove ```python and ``` markers
        cleaned_response = re.sub(r'```(?:python)?\s*', '', cleaned_response)
        cleaned_response = cleaned_response.replace('```', '')
        logger.info("Markdown fences removed")

    # Remove any explanations or other non-code text
    if "**Explanation:**" in cleaned_response:
        logger.info("Removing explanations…")
        cleaned_response = re.sub(r'\*\*Explanation:\*\*.*?(?=def |$)', '', cleaned_response, flags=re.DOTALL)

    # Remove other markdown formatting that might interfere with code parsing
    cleaned_response = re.sub(r'\*\*.*?\*\*', '', cleaned_response)
    cleaned_response = re.sub(r'Function \d+ \(`.*?`\):', '', cleaned_response)
    cleaned_response = re.sub(r'^\s*(\*|\-|\d+\.)\s*', '', cleaned_response, flags=re.MULTILINE)

    # Make sure there's at least one function definition
    if "def " not in cleaned_response:
        logger.info("ERROR: No function definitions found in response")
        raise ValueError("No function definitions found in response.")

    # Try to extract all function blocks
    code_blocks = re.findall(r"(def [\s\S]+?)(?=\ndef |\Z)", cleaned_response)
    logger.info(f"Found {len(code_blocks)} code blocks")

    # If we got less than 2 blocks, try a simpler extraction
    if len(code_blocks) < 2:
        logger.info("Warning: Less than 2 code blocks found, trying fallback extraction…")
        # Split on 'def ' and reconstruct function definitions
        parts = cleaned_response.split('def ')
        code_blocks = []
        for part in parts[1:]:  # Skip the first empty part
            code_blocks.append('def ' + part.strip())
        logger.info(f"Fallback extraction found {len(code_blocks)} blocks")
        
        # If still less than 2 blocks, try more aggressive techniques
        if len(code_blocks) < 2:
            logger.info("WARNING: Still found less than 2 code blocks after fallback extraction")
            logger.info("Attempting more aggressive parsing techniques...")
            
            # Try to find any function-like blocks
            all_funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", cleaned_response)
            logger.info(f"Found {len(all_funcs)} potential function definitions: {all_funcs}")
            
            # Look for the required function names specifically
            if "trading_strategy" in cleaned_response and "get_user_params" in cleaned_response:
                logger.info("Found both required function names in the response text")
                
                # Try to extract each function individually
                trading_strategy_match = re.search(r"(def\s+trading_strategy\s*\([^)]*\)[\s\S]+?)(?=def\s+get_user_params|\Z)", cleaned_response)
                get_user_params_match = re.search(r"(def\s+get_user_params\s*\([^)]*\)[\s\S]+?)(?=def\s+trading_strategy|\Z)", cleaned_response)
                
                if trading_strategy_match:
                    logger.info("Found trading_strategy function via direct regex")
                    code_blocks.append(trading_strategy_match.group(1))
                
                if get_user_params_match:
                    logger.info("Found get_user_params function via direct regex")
                    code_blocks.append(get_user_params_match.group(1))
                
                logger.info(f"After aggressive extraction, found {len(code_blocks)} blocks")
            else:
                logger.info("ERROR: One or both required function names are missing from the response")
                if "trading_strategy" in cleaned_response:
                    logger.info("Only found trading_strategy")
                elif "get_user_params" in cleaned_response:
                    logger.info("Only found get_user_params")
                else:
                    logger.info("Neither required function was found")

    trading_blocks = [block for block in code_blocks if re.search(r"def\s+trading_strategy\s*\(", block)]
    param_blocks = [block for block in code_blocks if re.search(r"def\s+get_user_params\s*\(", block)]
    logger.info(f"Identified {len(trading_blocks)} trading_strategy functions and {len(param_blocks)} get_user_params functions")

    if not trading_blocks or not param_blocks:
        logger.info("ERROR: Required function definitions not found")
        raise ValueError("Missing required functions in LLM response.")

    code = trading_blocks[0].strip()
    input_code = param_blocks[0].strip()

    def extract_function_name(code_block):
        try:
            parsed = ast.parse(code_block)
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            logger.info(f"AST parsing failed: {e}. Falling back to regex.")
            # Fallback method if parsing fails
            first_line = code_block.strip().split('\n')[0]
            match = re.search(r'def\s+([a-zA-Z0-9_]+)', first_line)
            if match:
                return match.group(1)
        return ""

    # Syntax check the extracted code
    try:
        compile(code, '<string>', 'exec')
        logger.info("trading_strategy code syntax is valid")
    except SyntaxError as e:
        logger.info(f"WARNING: trading_strategy has syntax errors: {e}")
        # Try to fix common syntax issues
        code = code.replace('"', '"').replace('"', '"').replace('\'\'', "'").replace('\'\'', "'")
        try:
            compile(code, '<string>', 'exec')
            logger.info("Syntax fixed after character replacement")
        except SyntaxError as e:
            logger.info(f"Still has syntax errors after fixing: {e}")
    
    try:
        compile(input_code, '<string>', 'exec')
        logger.info("get_user_params code syntax is valid")
    except SyntaxError as e:
        logger.info(f"WARNING: get_user_params has syntax errors: {e}")
        # Try to fix common syntax issues
        input_code = input_code.replace('"', '"').replace('"', '"').replace('\'\'', "'").replace('\'\'', "'")
        try:
            compile(input_code, '<string>', 'exec')
            logger.info("Syntax fixed after character replacement")
        except SyntaxError as e:
            logger.info(f"Still has syntax errors after fixing: {e}")

    return {
        "code": code,
        "function_name_code": extract_function_name(code),
        "input_code": input_code,
        "function_name_input_code": extract_function_name(input_code)
    }

# -----------------------------------------------------------------------------
# Buy and Hold Calculation
# -----------------------------------------------------------------------------

def calculate_buy_and_hold_profit(close_prices, initial_balance=100000):
    """Calculate profit from simply buying at start and selling at end."""
    if not close_prices or len(close_prices) < 2:
        return 0
    
    first_price = close_prices[0]
    last_price = close_prices[-1]
    
    # Calculate how many shares we could buy with initial balance
    shares = math.floor(initial_balance / first_price)
    
    # Calculate final value
    final_value = shares * last_price
    
    # Calculate profit/loss
    profit_loss = final_value - initial_balance
    
    return profit_loss

# -----------------------------------------------------------------------------
# Core simulation runner with iterative improvement
# -----------------------------------------------------------------------------

def run_simulation(user_input, original_user_input_text, improvement_context="", iteration=1, max_iterations=5, max_error_attempts=10):
    """
    Run a simulation of a trading strategy using the provided input.
    
    Args:
        user_input (str): The user's description of a trading strategy.
        original_user_input_text (str): The original user input for reference.
        improvement_context (str): Additional context for improving the strategy.
        iteration (int): The current iteration number.
        max_iterations (int): The maximum number of iterations to run.
        max_error_attempts (int): The maximum number of error attempts allowed.
        
    Returns:
        tuple: (profit, buy_points, sell_points, balance_over_time, code_info)
    """
    logger.info(f"\nStarting simulation iteration {iteration} with user input: {user_input[:100]}…")
    
    # Import in function scope to avoid circular imports
    from api.backtest import close, dates
    
    # Set up simulation parameters
    initial_balance = 100000  # Starting with $100k
    best_bb = float('-inf')
    best_buy_points = []
    best_sell_points = []
    best_balance_over_time = []
    best_code_info = {}
    
    # Prepare the prompt for the trading strategy implementation
    prompt = f"""
# TRADING STRATEGY IMPLEMENTATION

Based on the following description, implement a complete trading strategy in Python:

```
{user_input}
```

{improvement_context if improvement_context else ""}

Your implementation must:

1. Define a function called `trading_strategy(params)` that takes parameter dictionary and implements the strategy
2. Define a function `get_user_params()` that returns a dictionary of parameters used by the strategy
3. Use the global variables `close` (list of closing prices) and `dates` (list of corresponding dates)
4. Return exactly: (profit_loss, buy_points, sell_points, balance_over_time)
   - profit_loss: total profit/loss amount (float)
   - buy_points: list of (index, price) tuples for buy signals
   - sell_points: list of (index, price) tuples for sell signals
   - balance_over_time: list of account balances over time (same length as close)

Be creative and faithful to the user's description. If something is unclear, use your best judgment to implement what they likely intended.

The goal is to make a profitable strategy, but more importantly, to implement what the user described.

IMPORTANT: Return ONLY the Python code without explanations before or after.
"""

    attempts = 0
    success = False
    response = None
    parts = None
    code = ""
    input_code = ""
    function_name_code = ""
    function_name_input_code = ""
    error_prompt = ""
    bb = None
    buy_points = []
    sell_points = []
    balance_over_time = []

    # Temperature scaling parameters
    base_temperature = 0.3  # Start with slightly higher temperature for more variation
    max_temperature = 1.0
    temperature_step = 0.1
    
    # Import timeout functionality
    import signal
    import time

    # Define timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution timed out")

    while not success:
        try:
            logger.info(f"\n--- Attempt {attempts + 1} ---")
            current_prompt = error_prompt if attempts > 0 else prompt
            
            # Calculate temperature (increase with more attempts)
            current_temperature = min(base_temperature + (temperature_step * (attempts // 2)), max_temperature)
            logger.info(f"Using temperature: {current_temperature}")

            if not response or attempts > 0:
                logger.info("Getting response from GPT…")
                response = ask_llama(current_prompt)
                if response is None:
                    raise Exception("LLM failed to provide a response.")

                logger.info("Splitting response into parts…")
                parts = split_response(response)
                code = parts["code"]
                function_name_code = parts["function_name_code"]
                input_code = parts["input_code"]
                function_name_input_code = parts["function_name_input_code"]
                logger.info(f"Extracted function names: {function_name_code} and {function_name_input_code}")

                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                safe_user_input_filename_part = re.sub(r'[^-\w\s-]', '', user_input[:30])
                safe_user_input_filename_part = re.sub(r'\s+', '_', safe_user_input_filename_part).strip('-_')
                
                code_info = {
                    'timestamp': timestamp,
                    'iteration': iteration,
                    'safe_user_input': safe_user_input_filename_part, 
                    'code': code,
                    'input_code': input_code,
                    'original_user_input': original_user_input_text,
                    'enhanced_user_input': user_input,
                    'full_llm_prompt': current_prompt
                }

                if re.search(r'\binput\s*\(', input_code):
                    raise Exception("Detected forbidden user prompt (input()) in get_user_params. This function must run autonomously without requiring user interaction.")

            logger.info("Executing input code…")
            try:
                exec(input_code, globals())
            except Exception as e:
                logger.info(f"Error executing input code: {e}")
                raise
                
            logger.info("Evaluating input function with timeout protection...")
            # TODO: Implement a safe timeout using multiprocessing or another method if needed in production
            try:
                inp = eval(f'{function_name_input_code}()')
                logger.info(f"Input function returned: {inp}")
            except Exception as e:
                logger.info(f"Error evaluating input function: {e}")
                raise

            formatted_inp = repr(inp)

            logger.info("Executing main code…")
            exec(code, globals())
            logger.info("Evaluating main function…")
            
            logger.info("Applying safety wrapper...")
            def safe_execute_trading_strategy(func_name, params_arg):
                """Execute trading strategy with safety checks and flexible parameter handling"""
                local_vars = {'params': params_arg, 'close': close, 'math': math, 'np': np}

                # If params are provided as a tuple, convert them to a dictionary
                if isinstance(params_arg, tuple) and len(params_arg) > 0:
                    safe_params = {i: val for i, val in enumerate(params_arg)}
                    for i, val in enumerate(params_arg):
                        safe_params[f'param{i+1}'] = val
                    local_vars['params'] = safe_params
                
                # Ensure params is a dictionary
                if not isinstance(local_vars['params'], dict):
                    logger.info("Warning: Converting non-dict params to dict for safety")
                    if isinstance(local_vars['params'], str):
                        # Handle string params by treating it as a single parameter
                        local_vars['params'] = {'param': local_vars['params']}
                    else:
                        local_vars['params'] = {'param': local_vars['params']}
                
                # Guarantee a working .get method
                if not hasattr(local_vars['params'], 'get'):
                    local_vars['params'] = dict(local_vars['params'])

                # Make sure 'close' reference exists inside params
                local_vars['params'].setdefault('close', close)

                # Convenience variable for strategy authors
                local_vars['n'] = len(close)

                logger.info(f"Executing {func_name} with {len(close)} data points and params: {local_vars['params']}")

                # Attempt to run the strategy with various fall-backs for signature mismatches
                try:
                    result = eval(f"{func_name}(params)", globals(), local_vars)
                except TypeError as e:
                    if ("required positional argument" in str(e) or
                        "takes 1 positional argument" in str(e) or
                        "takes 0 positional arguments but 1 was given" in str(e)):
                        logger.info(f"Retrying with direct close data due to TypeError: {e}")
                        try:
                            result = eval(f"{func_name}(close)", globals(), local_vars)
                        except TypeError as e2:
                            if "takes 0 positional arguments but 1 was given" in str(e2):
                                logger.info(f"Retrying with no arguments due to TypeError: {e2}")
                                result = eval(f"{func_name}()", globals(), local_vars)
                            else:
                                raise
                    else:
                        raise
                
                # Normalise result into expected 4-tuple if necessary
                if not (isinstance(result, tuple) and len(result) == 4) and not (
                    isinstance(result, dict) and all(k in result for k in [
                        'profit_loss', 'buy_points', 'sell_points', 'balance_over_time'])):
                    logger.info(f"Warning: {func_name} returned {type(result)} instead of a 4-tuple or expected dict. Attempting to adapt.")
                    if isinstance(result, (int, float)): # Only profit/loss returned
                        return result, [], [], [initial_balance, initial_balance + result]
                    elif isinstance(result, dict):
                        profit_loss = result.get('profit_loss', result.get('profit', 0))
                        buy_points_res = result.get('buy_points', result.get('buy', []))
                        sell_points_res = result.get('sell_points', result.get('sell', []))
                        balance_over_time_res = result.get('balance_over_time', result.get('balance', [initial_balance, initial_balance + profit_loss]))
                        return profit_loss, buy_points_res, sell_points_res, balance_over_time_res
                    else: # Cannot adapt
                        raise TypeError(f"{func_name} must return a 4-tuple (profit_loss, buy_points, sell_points, balance_over_time) or a dict with these keys.")

                # If it's a dict, convert to tuple for consistent handling
                if isinstance(result, dict):
                    return result['profit_loss'], result['buy_points'], result['sell_points'], result['balance_over_time']
                return result  # Already in correct tuple form

            bb, buy_points, sell_points, balance_over_time = safe_execute_trading_strategy(function_name_code, inp)
            logger.info(f"Main function returned: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")

            # --- PATCH: Improved fallback using moving average crossover ---
            if (not buy_points or not sell_points) or (isinstance(bb, (int, float)) and bb == 0):
                logger.info("No trades or zero profit detected, running fallback moving average crossover strategy.")
                window = 20
                balance = initial_balance
                position = 0
                shares = 0
                buy_points = []
                sell_points = []
                balance_over_time = []
                for i, price in enumerate(close):
                    if i < window:
                        balance_over_time.append(balance if position == 0 else shares * price + balance)
                        continue
                    ma = sum(close[i-window:i]) / window
                    # Buy if price crosses above MA and not in position
                    if price > ma and position == 0:
                        buy_points.append((i, price))
                        shares = math.floor(balance / price)
                        balance -= shares * price
                        position = 1
                    # Sell if price crosses below MA and in position
                    elif price < ma and position == 1:
                        sell_points.append((i, price))
                        balance += shares * price
                        shares = 0
                        position = 0
                    balance_over_time.append(balance if position == 0 else shares * price + balance)
                # Sell at end if still holding
                if position == 1:
                    sell_points.append((len(close)-1, close[-1]))
                    balance += shares * close[-1]
                profit_loss = balance - initial_balance
                bb = profit_loss
                logger.info(f"Fallback strategy produced: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")
            # --- END PATCH ---

            if isinstance(bb, (int, float)) and bb > best_bb: # Removed buy/sell points check for best_bb to allow strategies that don't trade but are valid
                best_bb = bb
                best_buy_points = buy_points
                best_sell_points = sell_points
                best_balance_over_time = balance_over_time
                best_code_info = code_info

            if not isinstance(bb, (int, float)):
                raise Exception("Profit/loss (bb) is not numeric. Strategy must return a numeric P&L.")

            # Validate buy_points and sell_points format
            for points, name in [(buy_points, "buy_points"), (sell_points, "sell_points")]:
                if not isinstance(points, list):
                    raise TypeError(f"{name} must be a list, got {type(points)}")
                for point in points:
                    if not (isinstance(point, tuple) and len(point) == 2 and isinstance(point[0], int) and isinstance(point[1], (int, float))):
                        if isinstance(point, (int, float)): # If it's just a list of indices/prices, try to fix
                            logger.info(f"Warning: Adapting {name} from list of numbers to list of (index, price) tuples.")
                            # This is a guess; assumes points are indices and uses close price. Not ideal.
                            # A better fix would be for the LLM to return correct format.
                            adapted_points = []
                            for p_idx, p_val in enumerate(points):
                                if isinstance(p_val, int) and p_val < len(close): # Assumed index
                                    adapted_points.append((p_val, close[p_val]))
                                elif isinstance(p_val, int): # Index out of bounds
                                    adapted_points.append((p_idx, 0)) # Fallback, not great
                                else: # Assumed price, use sequential index
                                    adapted_points.append((p_idx, p_val))
                            if name == "buy_points": buy_points = adapted_points
                            else: sell_points = adapted_points
                            break # Re-check from start of this inner loop with adapted points
                        else:
                            raise TypeError(f"Each item in {name} must be a tuple (index, price), got {type(point)}: {point}")
                    
            # Fallback for empty trades only if it's the last attempt and no best result yet
            if (not buy_points or not sell_points) and attempts >= max_error_attempts - 1 and best_bb == float('-inf'):
                logger.info(f"Last attempt ({attempts + 1}) failed to produce trades, and no prior best strategy. Falling back to basic buy-and-hold.")
                    
                buy_points = [(0, close[0])] if close else []
                sell_points = [(len(close) - 1, close[-1])] if close else []

                shares_fb = math.floor(initial_balance / close[0]) if close and close[0] > 0 else 0
                final_balance_fb = (shares_fb * close[-1]) if close and shares_fb > 0 else initial_balance
                bb = final_balance_fb - initial_balance if close else 0

                balance_over_time = [initial_balance] * len(close)
                if close and balance_over_time:
                    balance_over_time[-1] = final_balance_fb
                else:  # Handle empty close or balance_over_time
                    balance_over_time = [initial_balance, initial_balance + bb]

                logger.info("Implemented fallback buy-and-hold strategy.")
                return bb, buy_points, sell_points, balance_over_time, code_info

            # --- PATCH: Ensure outputs are lists for JSON serialization ---
            balance_over_time = list(balance_over_time)
            buy_points = list(buy_points)
            sell_points = list(sell_points)
            # --- END PATCH ---

            success = True

        except Exception as e:
            attempts += 1
            logger.info(f"\n--- Attempt {attempts} FAILED with error: ---")
            error_traceback = traceback.format_exc()
            logger.info(error_traceback)
            logger.info("-------------------------------------------")

            if attempts >= max_error_attempts:
                logger.info(f"\n!!! REACHED MAXIMUM ERROR ATTEMPTS ({max_error_attempts}) !!!")
                if best_bb != float('-inf'):
                    logger.info("Giving up on this iteration and returning best results found so far.")
                    return best_bb, best_buy_points, best_sell_points, best_balance_over_time, best_code_info
                else:
                    logger.info("No successful strategy found after all attempts. Returning empty results.")
                    return 0, [], [], [initial_balance, initial_balance], None 

            response = None 

            error_prompt = f"""
**CRITICAL ERROR FIX REQUIRED**

The Python code you generated produced the following error:
```
{error_traceback}
```

**Your Task:**
Fix the error in the code and provide the COMPLETE CODE for BOTH functions. 

**EXTREMELY IMPORTANT: You MUST return the ENTIRE code for BOTH functions, not just the parts you modified.**

**DATA INFORMATION:**
- The data is loaded from a CSV file at 'stockbt/datasets/test.csv'
- CSV Format: Date,Open,High,Low,Close,Volume
- In the code, only the Close prices are available as a simple Python list named 'close'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it\\'s already available as 'close'

Your response MUST consist of:
1. The COMPLETE `trading_strategy` function (entire function, not just the fixed part)
2. A single blank line
3. The COMPLETE `get_user_params` function (entire function, not just the fixed part)

**DO NOT OMIT ANY CODE.** Make sure you include all loops, conditional statements, variable definitions, and other code from both functions.

**IMPORTANT FUNCTION SIGNATURES:**
- trading_strategy must accept EXACTLY ONE positional argument (params): `def trading_strategy(params):`
- get_user_params must accept NO parameters: `def get_user_params():`

**Common Errors to Check:**
- Use only standard ASCII characters (no " " \\' — etc.).
- `trading_strategy` must accept EXACTLY ONE positional argument (remove any extras like `formatted_inp`, `data`, etc.).
- Don\\'t call undefined functions (like 'update_end_price').
- **ABSOLUTELY CRITICAL: `get_user_params` MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** Make sure it returns valid numeric parameters.
- Use relative multipliers/scales for price calculations—do **NOT** hard-code absolute price thresholds.
- Ensure `close` is used directly as a simple Python list; do NOT attempt dictionary/DataFrame-style access.
- Ensure all variables are properly defined before use.
- Fix any `KeyError` or `SyntaxError` shown above before returning.

**User Strategy:** {user_input}

**REMEMBER:** 
- Output ONLY the complete code for both functions - no explanations, no markdown, no extra text.
- DO NOT SKIP OR SUMMARIZE ANY PART OF THE CODE with comments like "... rest of function remains the same ..."
- INCLUDE THE ENTIRE FUNCTION DEFINITIONS, not just the modified parts.

**Important Output Format Requirements:**
- Your trading_strategy function may return its results in EITHER of these formats:
  1. Standard tuple: (profit_loss, buy_points, sell_points, balance_over_time)
  2. Dictionary format: {{\\'profit_loss\\': profit_amount, \\'buy_points\\': [...], \\'sell_points\\': [...], \\'balance_over_time\\': [...]}}

- The dictionary format is preferred as it\\'s more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
"""
            logger.info(f"Attempting retry {attempts} to fix the error…")

    logger.info("\nExecution successful.")
    return bb, buy_points, sell_points, balance_over_time, code_info

def run_improved_simulation(user_input, max_error_attempts=10, max_iterations=None):
    """Run an iterative improvement process to achieve profit target (copied/adapted from st.py)."""
    logger.info("\n=== STARTING ITERATIVE IMPROVEMENT PROCESS ===")
    if max_iterations is None:
        max_iterations = int(os.environ.get('BACKTEST_MAX_ITERATIONS', '5'))
    global close, dates
    from datetime import datetime
    base_counter = int(datetime.now().timestamp())
    best_profit = float('-inf')
    best_strategy_data = None
    all_datasets_results = []
    improvement_context = ""
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n--- IMPROVEMENT ITERATION {iteration}/{max_iterations} ---")
        file_path = get_random_dataset()
        dataset_name = os.path.basename(file_path) if file_path else "unknown"
        close = []
        dates = []
        if not file_path or not os.path.exists(file_path):
            logger.error(f"No valid dataset CSV file found: {file_path}")
            continue
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    close.append(float(row['Close']))
                    dates.append(row['Date'])
                except (ValueError, KeyError) as e:
                    logger.info(f"Warning: Error parsing row {row}: {e}")
        if not close:
            logger.error(f"Dataset file {dataset_name} was found but no valid price data could be loaded from it.")
            continue
        buy_hold_profit = calculate_buy_and_hold_profit(close)
        target_profit = buy_hold_profit * 1.5
        if iteration == 1:
            enhanced_prompt = enhance_user_prompt(user_input) if max_iterations > 1 else user_input
        else:
            improvement_context = f"""
**PREVIOUS ITERATION PERFORMANCE:**
- Previous Profit: ${current_profit:.2f} (vs Buy & Hold: {percent_above_buyhold:+.2f}%)
- Target Profit: ${target_profit:.2f}
- Performance vs Target: {percent_vs_target:+.2f}%

**IMPROVEMENT SUGGESTIONS:**
{improvement_suggestions if 'improvement_suggestions' in locals() else ''}

Based on the above analysis, implement these improvements in your trading strategy to achieve the target profit of ${target_profit:.2f}.
"""
            enhanced_prompt = user_input
        improvement_context = "" if iteration == 1 else improvement_context
        current_profit, buy_points, sell_points, balance_over_time, code_info = run_simulation(
            enhanced_prompt, 
            user_input, 
            improvement_context, 
            iteration,
            max_iterations,
            max_error_attempts
        )
        if buy_hold_profit != 0:
            percent_above_buyhold = (current_profit / abs(buy_hold_profit)) * 100
        else:
            percent_above_buyhold = current_profit * 100 if current_profit != 0 else 0
        percent_vs_target = ((current_profit / target_profit) - 1) * 100 if target_profit > 0 else 0
        logger.info(f"\nCurrent Profit: ${current_profit:.2f}")
        logger.info(f"Buy & Hold Profit: ${buy_hold_profit:.2f}")
        logger.info(f"Target Profit (1.5x Buy & Hold): ${target_profit:.2f}")
        logger.info(f"Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%")
        logger.info(f"Performance vs Target: {percent_vs_target:+.2f}%")
        buy_points = filter_trade_points(buy_points, close)
        sell_points = filter_trade_points(sell_points, close)
        logger.info(f"Filtered to {len(buy_points)} buy points and {len(sell_points)} sell points")
        code_info = code_info or {}
        trading_strategy_code = code_info.get('code', '')
        params_code = code_info.get('input_code', '')
        iteration_result = {
            'iteration': iteration,
            'dataset': dataset_name,
            'profit': current_profit,
            'buy_hold_profit': buy_hold_profit,
            'target_profit': target_profit,
            'percent_above_buyhold': percent_above_buyhold,
            'percent_vs_target': percent_vs_target,
            'buy_points': buy_points,
            'sell_points': sell_points,
            'balance_over_time': balance_over_time,
            'close': close,
            'dates': dates,
            'code': f"# Trading Strategy (Iteration {iteration})\n{trading_strategy_code}\n\n# Parameters\n{params_code}"
        }
        all_datasets_results.append(iteration_result)
        if current_profit > best_profit:
            logger.info(f"New best strategy found! Profit: ${current_profit:.2f}")
            best_profit = current_profit
            best_strategy_data = iteration_result
        if max_iterations > 1 and current_profit >= target_profit:
            logger.info("\n=== TARGET PROFIT ACHIEVED! ===")
            break
        # Generate improvement suggestions for next iteration
        if iteration < max_iterations:
            logger.info("\nGenerating improvement suggestions...")
            improvement_prompt = f"""
**Strategy Analysis and Improvement**

I need you to analyze the current trading strategy and suggest specific improvements to increase profitability.

**Current Strategy:** {user_input}

**Performance Metrics:**
- Current Profit: ${current_profit:.2f}
- Buy & Hold Profit: ${buy_hold_profit:.2f}
- Target Profit (1.5x Buy & Hold): ${target_profit:.2f}
- Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%
- Performance vs Target: {percent_vs_target:+.2f}%
- Number of Buy Points: {len(buy_points)}
- Number of Sell Points: {len(sell_points)}

**Current Price Series Statistics:**
- First Price: ${close[0]:.2f}
- Last Price: ${close[-1]:.2f}
- Min Price: ${min(close):.2f}
- Max Price: ${max(close):.2f}
- Price Range: ${max(close) - min(close):.2f}

**Your Task:**
You must keep the strategy at least 95% the same as the user's original. Only make minimal, necessary changes to improve profitability. Do not rewrite or restructure the strategy. Focus on:
1. Parameter adjustments (multipliers, thresholds, etc.)
2. Tiny logic improvements to buy/sell decision-making
3. Any inefficiencies in the current approach

**Output Format:** 
Provide a concise, focused analysis of what's working/not working, followed by specific, bulleted improvement suggestions. Your response will be fed directly back into the strategy implementation.
"""
            improvement_suggestions = ask_llama(improvement_prompt)
    if not all_datasets_results:
        return {
            'status': 'error',
            'error': "All iterations failed to produce results"
        }
    final_result = best_strategy_data or all_datasets_results[-1]
    # After trading_strategy_code and params_code are set, generate a summary
    code_summary = None
    if trading_strategy_code:
        summary_prompt = f"""
Summarize the following Python trading strategy code in 3-5 sentences. Focus on what the strategy does, its main logic, and any unique features. Use plain English and avoid code or variable names unless necessary.

Code:
{trading_strategy_code}
"""
        try:
            code_summary = ask_llama(summary_prompt)
        except Exception as e:
            logger.info(f"Error generating code summary: {e}")
            code_summary = None

    result = {
        'status': 'success',
        'profit': int(round(final_result['profit'])),
        'buy_hold_profit': final_result['buy_hold_profit'],
        'percent_above_buyhold': final_result['percent_above_buyhold'],
        'dataset': final_result['dataset'],
        'datasets_tested': len(all_datasets_results),
        'buy_points': final_result.get('buy_points', []),
        'sell_points': final_result.get('sell_points', []),
        'balance_over_time': final_result.get('balance_over_time', []),
        'close': final_result.get('close', []),
        'dates': final_result.get('dates', []),
        'trades': {
            'count': len(final_result.get('buy_points', [])) + len(final_result.get('sell_points', [])),
            'buys': final_result.get('buy_points', []),
            'sells': final_result.get('sell_points', [])
        },
        'code': final_result['code'],
        'code_summary': code_summary,
        'all_iterations': [{
            'iteration': r['iteration'],
            'dataset': r['dataset'],
            'profit': r['profit'],
            'percent_above_buyhold': r['percent_above_buyhold'],
            'trades_count': len(r['buy_points']) + len(r['sell_points'])
        } for r in all_datasets_results]
    }
    if 'buy_points' in result and result['buy_points']:
        chart_ready_buy_points = []
        for point in result['buy_points']:
            if isinstance(point, tuple) and len(point) == 2:
                chart_ready_buy_points.append({'x': point[0], 'y': point[1]})
        result['buy_points'] = chart_ready_buy_points
    if 'sell_points' in result and result['sell_points']:
        chart_ready_sell_points = []
        for point in result['sell_points']:
            if isinstance(point, tuple) and len(point) == 2:
                chart_ready_sell_points.append({'x': point[0], 'y': point[1]})
        result['sell_points'] = chart_ready_sell_points
    logger.info(f"Final result includes {len(result.get('buy_points', []))} buy points and {len(result.get('sell_points', []))} sell points")
    result = to_serializable(result)
    return result

def enhance_user_prompt(user_input):
    """Enhances the user's input with additional instructions and constraints.
    Adds specificity to user's general description of a trading algorithm.
    """
    enhanced_prompt = f"""
# TRADING STRATEGY TASK

Create a complete Python implementation of the trading strategy described below. The code should be creative, realistic, and follow these guidelines:

## User's Trading Strategy Description:
{user_input}

## Implementation Requirements:
1. Define a function called `trading_strategy(params)` that implements the user's strategy
2. Define a function called `get_user_params()` that returns any parameters needed
3. Use the `close` price array for your calculations
4. Return a tuple containing (profit_loss, buy_points, sell_points, balance_over_time)

## You have available:
- `close`: an array of daily closing prices
- `dates`: matching dates for the close prices

Your code should be creative while respecting the user's intent. Feel free to interpret the strategy in a way that makes most sense to you. While you should aim for profitability, focus more on implementing exactly what the user described.

IMPORTANT: I want you to use your own approach to implement the strategy, not a rigid template. If the user's description is vague, use your best judgment to fill in the details in a way that's still true to their core idea.

I need your response to contain valid Python code - do not include explanations or notes unless they're comments within the code.
"""
    
    return enhanced_prompt

def filter_trade_points(points, close):
    """Only keep unique, valid trade points as (index, price) tuples."""
    filtered = []
    seen = set()
    for point in points:
        if isinstance(point, tuple) and len(point) == 2:
            idx, price = point
            if isinstance(idx, int) and 0 <= idx < len(close) and idx not in seen:
                filtered.append((idx, close[idx]))
                seen.add(idx)
    return filtered

# Add this utility function near the top of the file (after imports)
def to_serializable(obj):
    """Recursively convert numpy arrays in obj to lists."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    else:
        return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest simulation.")
    parser.add_argument("strategy", nargs="?", default="", help="Trading strategy description")
    parser.add_argument("--json", action="store_true", help="Return results as JSON on stdout")
    args = parser.parse_args()

    if args.strategy:
        user_input_raw = args.strategy
    else:
        logger.info("Waiting for user input…")
        user_input_raw = input("Enter your trading strategy: ")

    try:
        result = run_improved_simulation(user_input_raw)
        
        if args.json:
            # For API use, print only the JSON with no additional output
            json_result = json.dumps(result)
            logger.info(json_result) # Log the JSON result as well
        else:
            if result['status'] == 'success':
                logger.info(f"Best Profit: ${result['profit']:.2f}")
                logger.info(f"Performance vs Buy & Hold: {result['percent_above_buyhold']:+.2f}%")
                logger.info(f"Dataset used: {result['dataset']}")
                logger.info(f"Number of trades: {result['trades']['count']}")
            else:
                logger.info(f"Error: {result['error']}")
        
        sys.exit(0)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.info(f"Error during backtest execution: {e}", file=sys.stderr) # Keep stderr for critical errors
        logger.info(error_traceback, file=sys.stderr)
        
        # For API calls, provide a structured error response
        if args.json:
            error_response = {
                'status': 'error',
                'error': str(e),
                'traceback': error_traceback
            }
            logger.info(json.dumps(error_response))
        
        sys.exit(1)