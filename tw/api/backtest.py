import math
import csv
import os
import traceback
import re
import g4f
import argparse
import json
import sys
import numpy as np
import time
import random

# Define a log file path
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'backtest_live.log')

# Clear the log file at the start of the script
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

def log_message(message):
    """Logs a message to both the console and the log file."""
    print(message) # Keep original console output
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(str(message) + '\n')
    except Exception as e:
        print(f"Error writing to log file: {e}") # Log errors to console

# Track previously used datasets in this session
if 'used_datasets' not in globals():
    used_datasets = set()

# Set the initial balance for all simulations
initial_balance = 100000

def get_random_dataset():
    """Get a random dataset from the datasets directory, tracking previously used ones."""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    all_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    # If specified, always use test.csv (for consistency in API testing)
    if os.environ.get('BACKTEST_USE_TEST_DATASET') == '1':
        return os.path.join(datasets_dir, 'test.csv')
    
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
# GPT-4 FREE (g4f) helper
# -----------------------------------------------------------------------------

def ask_llama(prompt, temperature=0.7):
    """Ask a question to the GPT-4 Free backend via g4f."""
    log_message(f"\nSending prompt to g4f ChatCompletion with temperature={temperature}…")
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            provider=g4f.Provider.PollinationsAI,
            temperature=temperature
        )
        # g4f returns plain text (no dict like OpenAI)
        log_message("Received response:" + response)
        return response.strip()
    except Exception as e:
        log_message(f"ERROR getting response: {e}")
        return None

# -----------------------------------------------------------------------------
# Response parsing helpers (identical to original)
# -----------------------------------------------------------------------------

def split_response(response):
    log_message("\nSplitting response into code blocks…")
    import ast

    # Print a diagnostic excerpt of the response
    if response:
        log_message(f"Response length: {len(response)} chars")
        excerpt_len = min(200, len(response))
        log_message(f"Response excerpt: {response[:excerpt_len]}...")
    else:
        log_message("WARNING: Empty response received")
        raise ValueError("Empty response from LLM")

    # First, remove any markdown code fences
    cleaned_response = response
    if "```" in response:
        log_message("Removing markdown code fences…")
        # Remove ```python and ``` markers
        cleaned_response = re.sub(r'```(?:python)?\s*', '', cleaned_response)
        cleaned_response = cleaned_response.replace('```', '')
        log_message("Markdown fences removed")

    # Remove any explanations or other non-code text
    if "**Explanation:**" in cleaned_response:
        log_message("Removing explanations…")
        cleaned_response = re.sub(r'\*\*Explanation:\*\*.*?(?=def |$)', '', cleaned_response, flags=re.DOTALL)

    # Remove other markdown formatting that might interfere with code parsing
    cleaned_response = re.sub(r'\*\*.*?\*\*', '', cleaned_response)
    cleaned_response = re.sub(r'Function \d+ \(`.*?`\):', '', cleaned_response)
    cleaned_response = re.sub(r'^\s*(\*|\-|\d+\.)\s*', '', cleaned_response, flags=re.MULTILINE)

    # Make sure there's at least one function definition
    if "def " not in cleaned_response:
        log_message("ERROR: No function definitions found in response")
        raise ValueError("No function definitions found in response.")

    # Try to extract all function blocks
    code_blocks = re.findall(r"(def [\s\S]+?)(?=\ndef |\Z)", cleaned_response)
    log_message(f"Found {len(code_blocks)} code blocks")

    # If we got less than 2 blocks, try a simpler extraction
    if len(code_blocks) < 2:
        log_message("Warning: Less than 2 code blocks found, trying fallback extraction…")
        # Split on 'def ' and reconstruct function definitions
        parts = cleaned_response.split('def ')
        code_blocks = []
        for part in parts[1:]:  # Skip the first empty part
            code_blocks.append('def ' + part.strip())
        log_message(f"Fallback extraction found {len(code_blocks)} blocks")
        
        # If still less than 2 blocks, try more aggressive techniques
        if len(code_blocks) < 2:
            log_message("WARNING: Still found less than 2 code blocks after fallback extraction")
            log_message("Attempting more aggressive parsing techniques...")
            
            # Try to find any function-like blocks
            all_funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", cleaned_response)
            log_message(f"Found {len(all_funcs)} potential function definitions: {all_funcs}")
            
            # Look for the required function names specifically
            if "trading_strategy" in cleaned_response and "get_user_params" in cleaned_response:
                log_message("Found both required function names in the response text")
                
                # Try to extract each function individually
                trading_strategy_match = re.search(r"(def\s+trading_strategy\s*\([^)]*\)[\s\S]+?)(?=def\s+get_user_params|\Z)", cleaned_response)
                get_user_params_match = re.search(r"(def\s+get_user_params\s*\([^)]*\)[\s\S]+?)(?=def\s+trading_strategy|\Z)", cleaned_response)
                
                if trading_strategy_match:
                    log_message("Found trading_strategy function via direct regex")
                    code_blocks.append(trading_strategy_match.group(1))
                
                if get_user_params_match:
                    log_message("Found get_user_params function via direct regex")
                    code_blocks.append(get_user_params_match.group(1))
                
                log_message(f"After aggressive extraction, found {len(code_blocks)} blocks")
            else:
                log_message("ERROR: One or both required function names are missing from the response")
                if "trading_strategy" in cleaned_response:
                    log_message("Only found trading_strategy")
                elif "get_user_params" in cleaned_response:
                    log_message("Only found get_user_params")
                else:
                    log_message("Neither required function was found")

    trading_blocks = [block for block in code_blocks if re.search(r"def\s+trading_strategy\s*\(", block)]
    param_blocks = [block for block in code_blocks if re.search(r"def\s+get_user_params\s*\(", block)]
    log_message(f"Identified {len(trading_blocks)} trading_strategy functions and {len(param_blocks)} get_user_params functions")

    if not trading_blocks or not param_blocks:
        log_message("ERROR: Required function definitions not found")
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
            log_message(f"AST parsing failed: {e}. Falling back to regex.")
            # Fallback method if parsing fails
            first_line = code_block.strip().split('\n')[0]
            match = re.search(r'def\s+([a-zA-Z0-9_]+)', first_line)
            if match:
                return match.group(1)
        return ""

    # Syntax check the extracted code
    try:
        compile(code, '<string>', 'exec')
        log_message("trading_strategy code syntax is valid")
    except SyntaxError as e:
        log_message(f"WARNING: trading_strategy has syntax errors: {e}")
        # Try to fix common syntax issues
        code = code.replace('"', '"').replace('"', '"').replace('\'\'', "'").replace('\'\'', "'")
        try:
            compile(code, '<string>', 'exec')
            log_message("Syntax fixed after character replacement")
        except SyntaxError as e:
            log_message(f"Still has syntax errors after fixing: {e}")
    
    try:
        compile(input_code, '<string>', 'exec')
        log_message("get_user_params code syntax is valid")
    except SyntaxError as e:
        log_message(f"WARNING: get_user_params has syntax errors: {e}")
        # Try to fix common syntax issues
        input_code = input_code.replace('"', '"').replace('"', '"').replace('\'\'', "'").replace('\'\'', "'")
        try:
            compile(input_code, '<string>', 'exec')
            log_message("Syntax fixed after character replacement")
        except SyntaxError as e:
            log_message(f"Still has syntax errors after fixing: {e}")

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
    log_message(f"\nStarting simulation iteration {iteration} with user input: {user_input[:100]}…")

    # Construct the prompt with improvement context if provided
    if improvement_context:
        prompt = f"""
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions based on the user\\'s strategy, following the strict rules below.

**User Strategy:** {user_input}

**IMPROVEMENT CONTEXT:** 
{improvement_context}

**DATA INFORMATION:**
- The data is loaded from a CSV file at 'stockbt/datasets/test.csv'
- CSV Format: Date,Open,High,Low,Close,Volume
- In the code, only the Close prices are available as a simple Python list named 'close'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it\\'s already available as 'close'

**CODE TEMPLATE GUIDELINES:**

1. Use a TRADING_STRATEGY function that accepts PARAMS dict
2. Iterate over CLOSE prices: for i, price_today in enumerate(close)
3. Create BUY_POINTS and SELL_POINTS lists for trades
4. Return (profit_loss, buy_points, sell_points, balance_over_time)
5. Replace YOUR_BUY_CONDITION and YOUR_SELL_CONDITION with actual logic
6. Use params.get('key', default) to safely access parameters
7. NO input() calls or user prompting in any function
8. Use try/except blocks to handle potential errors

**Critical Output Requirements:**

1.  **Format:**
    *   Your entire response MUST be ONLY the Python code for the two functions.
    *   **ABSOLUTELY NO MARKDOWN:** Do not use ```python, ```, or any other markdown formatting.
    *   **NO Explanations** outside of the code itself: Do not add any prose before or after the functions.
    *   **Limited Comments Allowed:** Inside the functions, you MAY include concise comments or docstrings that explain the purpose of scale/ratio parameters, but avoid any other extraneous commentary.
    *   Separate the two functions with exactly ONE blank line.

2.  **Function 1: `trading_strategy`**
    *   Must be named exactly `trading_strategy`.
    *   Must accept **EXACTLY ONE** positional argument (e.g., `params`), which will be the output (tuple or dict) from `get_user_params`. Do **NOT** add extra parameters (such as `formatted_inp`, `data`, etc.).
    *   Must implement the user\\'s trading strategy using the provided `params`.
    *   When determining buy/sell prices, you **MUST NOT** rely on hard-coded or absolute price values. Instead, compute them as a **multiplier of the current `price_today`** or other relative measures derived from `params` (e.g., `buy_price = price_today * params["buy_mult"]`). A minimum execution price of $0.01 will be enforced.
    *   Define necessary loop variables such as `i`, `price_today`, and any multiplier-derived `buy_price`/`sell_price` consistent with the strategy (note that `close` is available in the execution context).
    *   Must include the **MANDATORY TRADING LOGIC** block verbatim within its primary execution loop or logic.
    *   Must include realistic market execution features such as slippage, bid-ask spread, commissions/fees, and liquidity constraints.
    *   Must **return** exactly `balance - initial_balance, buy_points, sell_points, balance_over_time` in that order.

3.  **Function 2: `get_user_params`**
    *   Must be named exactly `get_user_params`.
    *   **ABSOLUTELY CRITICAL: THIS FUNCTION MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** It must automatically determine or iteratively search for optimized numeric parameters relevant to the strategy.
    *   Must **return** the chosen numeric parameters as a single tuple or dictionary.
    *   Include realistic market execution parameters (slippage_pct, spread_pct, commission_per_share, min_commission, max_position_pct).
    *   **DO NOT** read files (e.g., CSVs) within this function.
    *   **DO NOT** hard-code absolute price thresholds; parameters should be relative scales/ratios or other dimensionless values.

4.  **Environment:**
    *   Assume `math` module is pre-imported and available. Do **NOT** add import statements.
    *   Assume a Python list named `close` containing the price data is available in the execution scope of `trading_strategy`.

5.  **Simplified Data Structure:** The price series is available as the global Python list named `close`. It\\'s a simple list of floating-point values. Access elements with standard indexing: `close[i]` or iterate with `for i, price_today in enumerate(close):`. No pandas/DataFrame code is needed.

6.  **Character Set:** Your entire code must use ONLY standard ASCII characters. Avoid typographic quotes (\\' \\' \\" \\" ) or long dashes (—). Use straight quotes (\\' \\") and hyphens (-) instead.

7.  **Data Access:** The price series is available as the global list named `close`. **Do NOT** attempt to access data via dictionaries or DataFrames. Simply use `close[i]` or iterate with `for i, price_today in enumerate(close):`.

8.  **Key-Error Safety:** Your code should never raise `KeyError`. Reference only variables you explicitly define.

**MANDATORY TRADING LOGIC (Include and adapt this block inside `trading_strategy`):**
```python
initial_balance = {initial_balance}
balance = initial_balance
shares = 0
buy_points = []
sell_points = []
balance_over_time = [balance] # Record initial balance

# Market execution parameters (realistic market conditions)
slippage_pct = params.get('slippage_pct', 0.001)  # 0.1% slippage by default
spread_pct = params.get('spread_pct', 0.0005)  # 0.05% half-spread by default
commission_per_share = params.get('commission_per_share', 0.005)  # $0.005 per share
min_commission_per_trade = params.get('min_commission', 1.0)  # Minimum $1.00 per trade
max_position_pct_of_volume = params.get('max_position_pct', 0.01)  # Maximum 1% of volume

# --- Start of logic needing integration with your strategy loop ---
# You need a loop here (e.g., for i, price_today in enumerate(close):)
# Inside the loop, calculate buy/sell signals based on user_input strategy and params.

# Example Buy Logic (adapt to your strategy):
# if YOUR_BUY_CONDITION and shares == 0: # Only buy if no shares are held
#   # Apply bid-ask spread (buy at ask price = higher)
#   base_price = price_today * (1 + spread_pct)
#   # Apply slippage (price moves against you when executing)
#   buy_price_with_spread = base_price * (1 + slippage_pct)
#   # Ensure minimum price
#   buy_price = max(0.01, buy_price_with_spread * params.get('buy_price_multiplier', 0.99))
#   
#   # Apply liquidity constraint (limit position size)
#   daily_volume = params.get('avg_volume', 100000)  # Default or from params
#   max_shares_by_volume = int(daily_volume * max_position_pct_of_volume)
#   
#   # Calculate shares to buy considering price and liquidity
#   max_shares_by_cash = math.floor(balance / buy_price) if buy_price > 0 else 0
#   shares_to_buy = min(max_shares_by_cash, max_shares_by_volume)
#   
#   if shares_to_buy > 0:
#       # Calculate commission
#       commission = max(min_commission_per_trade, shares_to_buy * commission_per_share)
#       # Execute trade with commission
#       trade_cost = (shares_to_buy * buy_price) + commission
#       balance -= trade_cost
#       shares += shares_to_buy
#       buy_points.append((i, price_today)) # Record buy point (index, price at trade)
#       print(f"Bought [shares_to_buy] shares at [buy_price:.2f] with [commission:.2f] commission on day [i] (price: [price_today:.2f])")

# Example Sell Logic (adapt to your strategy):
# elif YOUR_SELL_CONDITION and shares > 0: # Only sell if shares are held
#   # Apply bid-ask spread (sell at bid price = lower)
#   base_price = price_today * (1 - spread_pct)
#   # Apply slippage (price moves against you when executing)
#   sell_price_with_spread = base_price * (1 - slippage_pct)
#   # Ensure minimum price
#   sell_price = max(0.01, sell_price_with_spread * params.get('sell_price_multiplier', 1.01))
#   
#   if shares > 0: # Check if shares are held before selling
#       # Calculate commission
#       commission = max(min_commission_per_trade, shares * commission_per_share)
#       # Execute trade with commission
#       sale_proceeds = (shares * sell_price) - commission
#       balance += sale_proceeds
#       sell_points.append((i, price_today)) # Record sell point (index, price at trade)
#       print(f"Sold [shares] shares at [sell_price:.2f] with [commission:.2f] commission on day [i] (price: [price_today:.2f[])")
#       shares = 0

# At the end of each iteration in your loop (after potential buy/sell):
# balance_over_time.append(balance + (shares * price_today if shares > 0 else 0)) # Append current total value

# After the loop, if shares are still held, liquidate them at the last known price
if shares > 0:
    # Apply bid-ask spread and slippage for final liquidation
    base_price = max(0.01, close[-1] if close else 0.01) * (1 - spread_pct)
    last_price = max(0.01, base_price * (1 - slippage_pct))
    # Calculate commission for final sale
    commission = max(min_commission_per_trade, shares * commission_per_share)
    # Execute final liquidation with commission
    sale_proceeds = (shares * last_price) - commission
    balance += sale_proceeds
    sell_points.append((len(close) -1, close[-1] if close else 0.01)) # Record final auto-sell
    print(f"Final liquidation: Sold [shares] shares at [last_price:.2f] with [commission:.2f] commission")
    shares = 0
balance_over_time.append(balance) # Append final balance state
return balance - initial_balance, buy_points, sell_points, balance_over_time
```
**(Note:** The MANDATORY TRADING LOGIC above is a template. You **must** integrate it correctly within the `trading_strategy` function\\'s loop, defining variables like `i`, `price_today`, `buy_price`, `sell_price` according to the user\\'s strategy and the parameters from `get_user_params`. The comments indicate where your strategy-specific logic needs to fit.)

**Final Check:** Ensure your output is only the two raw Python function definitions separated by a single blank line. No markdown, no comments outside the functions, no extra text.

**Important Output Format Requirements:**
- Your trading_strategy function may return its results in EITHER of these formats:
  1. Standard tuple: (profit_loss, buy_points, sell_points, balance_over_time)
  2. Dictionary format: {{\\'profit_loss\\': profit_amount, \\'buy_points\\': [...], \\'sell_points\\': [...], \\'balance_over_time\\': [...]}}

- The dictionary format is preferred as it\\'s more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
"""
    else:
        prompt = f"""
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions based on the user\\'s strategy, following the strict rules below.

**User Strategy:** {user_input}

**DATA INFORMATION:**
- The data is loaded from a CSV file at 'stockbt/datasets/test.csv'
- CSV Format: Date,Open,High,Low,Close,Volume
- In the code, only the Close prices are available as a simple Python list named 'close'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it\\'s already available as 'close'

**CODE TEMPLATE GUIDELINES:**

1. Use a TRADING_STRATEGY function that accepts PARAMS dict
2. Iterate over CLOSE prices: for i, price_today in enumerate(close)
3. Create BUY_POINTS and SELL_POINTS lists for trades
4. Return (profit_loss, buy_points, sell_points, balance_over_time)
5. Replace YOUR_BUY_CONDITION and YOUR_SELL_CONDITION with actual logic
6. Use params.get('key', default) to safely access parameters
7. NO input() calls or user prompting in any function
8. Use try/except blocks to handle potential errors

**Critical Output Requirements:**

1.  **Format:**
    *   Your entire response MUST be ONLY the Python code for the two functions.
    *   **ABSOLUTELY NO MARKDOWN:** Do not use ```python, ```, or any other markdown formatting.
    *   **NO Explanations** outside of the code itself: Do not add any prose before or after the functions.
    *   **Limited Comments Allowed:** Inside the functions, you MAY include concise comments or docstrings that explain the purpose of scale/ratio parameters, but avoid any other extraneous commentary.
    *   Separate the two functions with exactly ONE blank line.

2.  **Function 1: `trading_strategy`**
    *   Must be named exactly `trading_strategy`.
    *   Must accept **EXACTLY ONE** positional argument (e.g., `params`), which will be the output (tuple or dict) from `get_user_params`. Do **NOT** add extra parameters (such as `formatted_inp`, `data`, etc.).
    *   Must implement the user\\'s trading strategy using the provided `params`.
    *   When determining buy/sell prices, you **MUST NOT** rely on hard-coded or absolute price values. Instead, compute them as a **multiplier of the current `price_today`** or other relative measures derived from `params` (e.g., `buy_price = price_today * params["buy_mult"]`). A minimum execution price of $0.01 will be enforced.
    *   Define necessary loop variables such as `i`, `price_today`, and any multiplier-derived `buy_price`/`sell_price` consistent with the strategy (note that `close` is available in the execution context).
    *   Must include the **MANDATORY TRADING LOGIC** block verbatim within its primary execution loop or logic.
    *   Must include realistic market execution features such as slippage, bid-ask spread, commissions/fees, and liquidity constraints.
    *   Must **return** exactly `balance - initial_balance, buy_points, sell_points, balance_over_time` in that order.

3.  **Function 2: `get_user_params`**
    *   Must be named exactly `get_user_params`.
    *   **ABSOLUTELY CRITICAL: THIS FUNCTION MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** It must automatically determine or iteratively search for optimized numeric parameters relevant to the strategy.
    *   Must **return** the chosen numeric parameters as a single tuple or dictionary.
    *   Include realistic market execution parameters (slippage_pct, spread_pct, commission_per_share, min_commission, max_position_pct).
    *   **DO NOT** read files (e.g., CSVs) within this function.
    *   **DO NOT** hard-code absolute price thresholds; parameters should be relative scales/ratios or other dimensionless values.

4.  **Environment:**
    *   Assume `math` module is pre-imported and available. Do **NOT** add import statements.
    *   Assume a Python list named `close` containing the price data is available in the execution scope of `trading_strategy`.

5.  **Simplified Data Structure:** The price series is available as the global Python list named `close`. It\\'s a simple list of floating-point values. Access elements with standard indexing: `close[i]` or iterate with `for i, price_today in enumerate(close):`. No pandas/DataFrame code is needed.

6.  **Character Set:** Your entire code must use ONLY standard ASCII characters. Avoid typographic quotes (\\' \\' \\" \\" ) or long dashes (—). Use straight quotes (\\' \\") and hyphens (-) instead.

7.  **Data Access:** The price series is available as the global list named `close`. **Do NOT** attempt to access data via dictionaries or DataFrames. Simply use `close[i]` or iterate with `for i, price_today in enumerate(close):`.

8.  **Key-Error Safety:** Your code should never raise `KeyError`. Reference only variables you explicitly define.

**MANDATORY TRADING LOGIC (Include and adapt this block inside `trading_strategy`):**
```python
initial_balance = {initial_balance}
balance = initial_balance
shares = 0
buy_points = []
sell_points = []
balance_over_time = [balance] # Record initial balance

# Market execution parameters (realistic market conditions)
slippage_pct = params.get('slippage_pct', 0.001)  # 0.1% slippage by default
spread_pct = params.get('spread_pct', 0.0005)  # 0.05% half-spread by default
commission_per_share = params.get('commission_per_share', 0.005)  # $0.005 per share
min_commission_per_trade = params.get('min_commission', 1.0)  # Minimum $1.00 per trade
max_position_pct_of_volume = params.get('max_position_pct', 0.01)  # Maximum 1% of volume

# --- Start of logic needing integration with your strategy loop ---
# You need a loop here (e.g., for i, price_today in enumerate(close):)
# Inside the loop, calculate buy/sell signals based on user_input strategy and params.

# Example Buy Logic (adapt to your strategy):
# if YOUR_BUY_CONDITION and shares == 0: # Only buy if no shares are held
#   # Apply bid-ask spread (buy at ask price = higher)
#   base_price = price_today * (1 + spread_pct)
#   # Apply slippage (price moves against you when executing)
#   buy_price_with_spread = base_price * (1 + slippage_pct)
#   # Ensure minimum price
#   buy_price = max(0.01, buy_price_with_spread * params.get('buy_price_multiplier', 0.99))
#
#   # Apply liquidity constraint (limit position size)
#   daily_volume = params.get('avg_volume', 100000)  # Default or from params
#   max_shares_by_volume = int(daily_volume * max_position_pct_of_volume)
#   
#   # Calculate shares to buy considering price and liquidity
#   max_shares_by_cash = math.floor(balance / buy_price) if buy_price > 0 else 0
#   shares_to_buy = min(max_shares_by_cash, max_shares_by_volume)
#   
#   if shares_to_buy > 0:
#       # Calculate commission
#       commission = max(min_commission_per_trade, shares_to_buy * commission_per_share)
#       # Execute trade with commission
#       trade_cost = (shares_to_buy * buy_price) + commission
#       balance -= trade_cost
#       shares += shares_to_buy
#       buy_points.append((i, price_today)) # Record buy point (index, price at trade)
#       print(f"Bought [shares_to_buy] shares at [buy_price:.2f] with [commission:.2f] commission on day [i] (price: [price_today:.2f]")

# Example Sell Logic (adapt to your strategy):
# elif YOUR_SELL_CONDITION and shares > 0: # Only sell if shares are held
#   # Apply bid-ask spread (sell at bid price = lower)
#   base_price = price_today * (1 - spread_pct)
#   # Apply slippage (price moves against you when executing)
#   sell_price_with_spread = base_price * (1 - slippage_pct)
#   # Ensure minimum price
#   sell_price = max(0.01, sell_price_with_spread * params.get('sell_price_multiplier', 1.01))
#   
#   if shares > 0: # Check if shares are held before selling
#       # Calculate commission
#       commission = max(min_commission_per_trade, shares * commission_per_share)
#       # Execute trade with commission
#       sale_proceeds = (shares * sell_price) - commission
#       balance += sale_proceeds
#       sell_points.append((i, price_today)) # Record sell point (index, price at trade)
#       print(f"Sold [shares] shares at [sell_price:.2f] with [commission:.2f] commission on day [i] (price: [price_today:.2f])")
#       shares = 0

# At the end of each iteration in your loop (after potential buy/sell):
# balance_over_time.append(balance + (shares * price_today if shares > 0 else 0)) # Append current total value

# After the loop, if shares are still held, liquidate them at the last known price
if shares > 0:
    # Apply bid-ask spread and slippage for final liquidation
    base_price = max(0.01, close[-1] if close else 0.01) * (1 - spread_pct)
    last_price = max(0.01, base_price * (1 - slippage_pct))
    # Calculate commission for final sale
    commission = max(min_commission_per_trade, shares * commission_per_share)
    # Execute final liquidation with commission
    sale_proceeds = (shares * last_price) - commission
    balance += sale_proceeds
    sell_points.append((len(close) -1, close[-1] if close else 0.01)) # Record final auto-sell
    print(f"Final liquidation: Sold [shares] shares at [last_price:.2f] with [commission:.2f] commission")
    shares = 0
balance_over_time.append(balance) # Append final balance state
return balance - initial_balance, buy_points, sell_points, balance_over_time
```
**(Note:** The MANDATORY TRADING LOGIC above is a template. You **must** integrate it correctly within the `trading_strategy` function\\'s loop, defining variables like `i`, `price_today`, `buy_price`, `sell_price` according to the user\\'s strategy and the parameters from `get_user_params`. The comments indicate where your strategy-specific logic needs to fit.)

**Final Check:** Ensure your output is only the two raw Python function definitions separated by a single blank line. No markdown, no comments outside the functions, no extra text.

**Important Output Format Requirements:**
- Your trading_strategy function may return its results in EITHER of these formats:
  1. Standard tuple: (profit_loss, buy_points, sell_points, balance_over_time)
  2. Dictionary format: {{\\'profit_loss\\': profit_amount, \\'buy_points\\': [...], \\'sell_points\\': [...], \\'balance_over_time\\': [...]}}

- The dictionary format is preferred as it\\'s more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
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

    # Track best results during attempts
    best_bb = float('-inf')
    best_buy_points = []
    best_sell_points = []
    best_balance_over_time = []
    best_code_info = None

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
            log_message(f"\n--- Attempt {attempts + 1} ---")
            current_prompt = error_prompt if attempts > 0 else prompt
            
            # Calculate temperature (increase with more attempts)
            current_temperature = min(base_temperature + (temperature_step * (attempts // 2)), max_temperature)
            log_message(f"Using temperature: {current_temperature}")

            if not response or attempts > 0:
                log_message("Getting response from GPT…")
                response = ask_llama(current_prompt, temperature=current_temperature)
                if response is None:
                    raise Exception("LLM failed to provide a response.")

                log_message("Splitting response into parts…")
                parts = split_response(response)
                code = parts["code"]
                function_name_code = parts["function_name_code"]
                input_code = parts["input_code"]
                function_name_input_code = parts["function_name_input_code"]
                log_message(f"Extracted function names: {function_name_code} and {function_name_input_code}")

                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                safe_user_input_filename_part = re.sub(r'[^\\w\\s-]', '', user_input[:30])
                safe_user_input_filename_part = re.sub(r'\\s+', '_', safe_user_input_filename_part).strip('-_')
                
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

            log_message("Executing input code…")
            try:
                exec(input_code, globals())
            except Exception as e:
                log_message(f"Error executing input code: {e}")
                raise
                
            log_message("Evaluating input function with timeout protection...")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30) 
            
            try:
                inp = eval(f'{function_name_input_code}()')
                signal.alarm(0)
                log_message(f"Input function returned: {inp}")
            except TimeoutError:
                signal.alarm(0)
                log_message("ERROR: get_user_params function execution timed out after 30 seconds")
                raise Exception("get_user_params function timed out - likely contains an infinite loop or excessive computation.")
            except Exception as e:
                signal.alarm(0)
                log_message(f"Error evaluating input function: {e}")
                raise

            formatted_inp = repr(inp)

            log_message("Executing main code…")
            exec(code, globals())
            log_message("Evaluating main function…")
            
            log_message("Applying safety wrapper...")
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
                        log_message("Warning: Converting non-dict params to dict for safety")
                        local_vars['params'] = {'param': local_vars['params']}
                    
                # Guarantee a working .get method
                    if not hasattr(local_vars['params'], 'get'):
                        local_vars['params'] = dict(local_vars['params'])

                # Make sure 'close' reference exists inside params
                local_vars['params'].setdefault('close', close)

                # Convenience variable for strategy authors
                local_vars['n'] = len(close)

                log_message(f"Executing {func_name} with {len(close)} data points and params: {local_vars['params']}")

                # Attempt to run the strategy with various fall-backs for signature mismatches
                try:
                        result = eval(f"{func_name}(params)", globals(), local_vars)
                except TypeError as e:
                    if ("required positional argument" in str(e) or
                        "takes 1 positional argument" in str(e) or
                        "takes 0 positional arguments but 1 was given" in str(e)):
                        log_message(f"Retrying with direct close data due to TypeError: {e}")
                        try:
                            result = eval(f"{func_name}(close)", globals(), local_vars)
                        except TypeError as e2:
                            if "takes 0 positional arguments but 1 was given" in str(e2):
                                log_message(f"Retrying with no arguments due to TypeError: {e2}")
                                result = eval(f"{func_name}()", globals(), local_vars)
                            else:
                                raise
                        else:
                            raise
                    
                # Normalise result into expected 4-tuple if necessary
                if not (isinstance(result, tuple) and len(result) == 4) and not (
                    isinstance(result, dict) and all(k in result for k in [
                        'profit_loss', 'buy_points', 'sell_points', 'balance_over_time'])):
                    log_message(f"Warning: {func_name} returned {type(result)} instead of a 4-tuple or expected dict. Attempting to adapt.")
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
            log_message(f"Main function returned: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")

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
                            log_message(f"Warning: Adapting {name} from list of numbers to list of (index, price) tuples.")
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
                log_message(f"Last attempt ({attempts + 1}) failed to produce trades, and no prior best strategy. Falling back to basic buy-and-hold.")
                    
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

                log_message("Implemented fallback buy-and-hold strategy.")
                return bb, buy_points, sell_points, balance_over_time, code_info

            success = True

        except Exception as e:
            attempts += 1
            log_message(f"\n--- Attempt {attempts} FAILED with error: ---")
            error_traceback = traceback.format_exc()
            log_message(error_traceback)
            log_message("-------------------------------------------")

            if attempts >= max_error_attempts:
                log_message(f"\n!!! REACHED MAXIMUM ERROR ATTEMPTS ({max_error_attempts}) !!!")
                if best_bb != float('-inf'):
                    log_message("Giving up on this iteration and returning best results found so far.")
                    return best_bb, best_buy_points, best_sell_points, best_balance_over_time, best_code_info
                else:
                    log_message("No successful strategy found after all attempts. Returning empty results.")
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
            log_message(f"Attempting retry {attempts} to fix the error…")

    log_message("\nExecution successful.")
    return bb, buy_points, sell_points, balance_over_time, code_info

def run_improved_simulation(user_input, max_error_attempts=10):
    """Run an iterative improvement process to achieve profit target."""
    log_message("\n=== STARTING ITERATIVE IMPROVEMENT PROCESS ===")
    
    # For API use, limit to just one iteration with fixed dataset
    global close, dates
    
    # Create a base counter for plots
    from datetime import datetime
    base_counter = int(datetime.now().timestamp())
    
    # Select the dataset (use test.csv for consistency)
    file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'test.csv')
    log_message(f"Loading data from {file_path}")
    close = []
    dates = []
    
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    close.append(float(row['Close']))
                    dates.append(row['Date'])
                except (ValueError, KeyError) as e:
                    log_message(f"Warning: Error parsing row {row}: {e}")
        log_message(f"Loaded {len(close)} data points.")
        
        # Calculate buy and hold profit as baseline for this dataset
        buy_hold_profit = calculate_buy_and_hold_profit(close)
        
        # Enhanced prompt is just the user input for API mode
        enhanced_prompt = user_input
        
        # Run single simulation
        current_profit, buy_points, sell_points, balance_over_time, code_info = run_simulation(
            enhanced_prompt, 
            user_input, 
            "", 
            1,
            1,
            max_error_attempts
        )
        
        # Calculate percentage above buy-and-hold
        percent_above_buyhold = ((current_profit / buy_hold_profit) - 1) * 100 if buy_hold_profit > 0 else 0
        
        log_message(f"\nCurrent Profit: ${current_profit:.2f}")
        log_message(f"Buy & Hold Profit: ${buy_hold_profit:.2f}")
        log_message(f"Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%")
        
        # Get code content
        code_info = code_info or {}
        trading_strategy_code = code_info.get('code', '')
        params_code = code_info.get('input_code', '')
        
        # Build the full response
        result = {
            'status': 'success',
            'profit': current_profit,
            'buy_hold_profit': buy_hold_profit,
            'percent_above_buyhold': percent_above_buyhold,
            'dataset': os.path.basename(file_path),
            'buy_points': buy_points,
            'sell_points': sell_points,
            'balance_over_time': balance_over_time,
            'close': close,  # Include close price data for client-side charting
            'dates': dates,  # Include dates for chart labels
            'trades': {
                'count': len(buy_points),
                'buys': buy_points,
                'sells': sell_points
            },
            'code': f"# Trading Strategy\n{trading_strategy_code}\n\n# Parameters\n{params_code}"
        }
        
        return result
    except Exception as e:
        error_traceback = traceback.format_exc()
        log_message(f"Error during simulation: {e}")
        log_message(error_traceback)
        return {
            'status': 'error',
            'error': str(e)
        }

def enhance_user_prompt(original_prompt):
    """Enhance the user's trading strategy prompt using LLM."""
    log_message("\n=== ENHANCING USER PROMPT ===")
    log_message(f"Original prompt: {original_prompt[:100]}...")
    
    enhancement_prompt = f"""
You are an expert trading strategy developer. Your task is to MINIMALLY refine the user's trading strategy while preserving ~90% of the original content.

**Original Strategy:**
{original_prompt}

**Your Task:**
1. Preserve approximately 90% of the original text
2. Only fix grammatical errors and improve clarity
3. Replace vague terms with more precise technical terminology where appropriate
4. Add missing context ONLY when absolutely necessary (e.g., if "mia khalifa method" is mentioned, briefly define what this method entails in trading terms)
5. DO NOT completely restructure or rewrite the strategy

The user values their original ideas - your job is just to make minor improvements while keeping their intent intact.

Provide ONLY the enhanced strategy description. DO NOT include explanations, introductions, or any text outside the improved strategy itself.
"""

    try:
        log_message("Getting enhanced prompt from LLM...")
        enhanced_prompt = ask_llama(enhancement_prompt, temperature=0.5)  # Lower temperature for more conservative edits
        
        if enhanced_prompt and len(enhanced_prompt) > 50:  # Basic validation
            log_message("Successfully enhanced the prompt.")
            log_message(f"Enhanced prompt: {enhanced_prompt[:100]}...")
            return enhanced_prompt
        else:
            log_message("Warning: Enhancement returned empty or very short result. Using original prompt.")
            return original_prompt
    except Exception as e:
        log_message(f"Error enhancing prompt: {e}")
        return original_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest simulation.")
    parser.add_argument("strategy", nargs="?", default="", help="Trading strategy description")
    parser.add_argument("--json", action="store_true", help="Return results as JSON on stdout")
    args = parser.parse_args()

    if args.strategy:
        user_input_raw = args.strategy
    else:
        log_message("Waiting for user input…")
        user_input_raw = input("Enter your trading strategy: ")

    try:
        result = run_improved_simulation(user_input_raw)
        
        if args.json:
            # For API use, print only the JSON with no additional output
            json_result = json.dumps(result)
            log_message(json_result) # Log the JSON result as well
        else:
            if result['status'] == 'success':
                log_message(f"Best Profit: ${result['profit']:.2f}")
                log_message(f"Performance vs Buy & Hold: {result['percent_above_buyhold']:+.2f}%")
                log_message(f"Dataset used: {result['dataset']}")
                log_message(f"Number of trades: {result['trades']['count']}")
            else:
                log_message(f"Error: {result['error']}")
        
        sys.exit(0)
    except Exception as e:
        error_traceback = traceback.format_exc()
        log_message(f"Error during backtest execution: {e}", file=sys.stderr) # Keep stderr for critical errors
        log_message(error_traceback, file=sys.stderr)
        
        # For API calls, provide a structured error response
        if args.json:
            error_response = {
                'status': 'error',
                'error': str(e),
                'traceback': error_traceback
            }
            print(json.dumps(error_response))
        
        sys.exit(1)