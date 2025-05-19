import math
import csv
import matplotlib.pyplot as plt
import os
import traceback
import re
import g4f
import argparse
import json
import sys
import math
import csv
import matplotlib.pyplot as plt
import os
import traceback
import re
import g4f
import argparse
import json
import sys
import numpy as np
import time
# Special functions (gamma, beta, erf, zeta, factorial, etc.)
from scipy import special

# Numerical integration (single, double, triple integrals; ODEs)
from scipy import integrate

# Linear algebra (solving systems, eigenvalues, decompositions)
from scipy import linalg

# Optimization and solving equations
from scipy import optimize

# Fourier transforms (e.g., for signal and number theory work)
from scipy import fft

import random
import wikipediaapi

# Track previously used datasets in this session
if 'used_datasets' not in globals():
    used_datasets = set()

# Set the initial balance for all simulations
initial_balance = 100000

def get_random_dataset():
    datasets_dir = '/Users/avneh/Code/HackSFProject/stockbt/datasets'
    all_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    unused_files = [f for f in all_files if f not in used_datasets]
    if not unused_files:
        used_datasets.clear()
        unused_files = all_files
    chosen = random.choice(unused_files)
    used_datasets.add(chosen)
    return os.path.join(datasets_dir, chosen)

# -----------------------------------------------------------------------------
# GPT-4 FREE (g4f) helperx
# -----------------------------------------------------------------------------

def ask_llama(prompt, temperature=0.7):
    """Ask a question to the GPT-4 Free backend via g4f."""
    print(f"\nSending prompt to g4f ChatCompletion with temperature={temperature}…")
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-o3",
            messages=[{"role": "user", "content": prompt}],
            provider=g4f.Provider.PollinationsAI,
            temperature=temperature
        )
        # g4f returns plain text (no dict like OpenAI)
        print("Received response:" + response)
        return response.strip()
    except Exception as e:
        print(f"ERROR getting response: {e}")
        return None

# -----------------------------------------------------------------------------
# Response parsing helpers (identical to original)
# -----------------------------------------------------------------------------

def split_response(response):
    print("\nSplitting response into code blocks…")
    import ast

    # Print a diagnostic excerpt of the response
    if response:
        print(f"Response length: {len(response)} chars")
        excerpt_len = min(200, len(response))
        print(f"Response excerpt: {response[:excerpt_len]}...")
    else:
        print("WARNING: Empty response received")
        raise ValueError("Empty response from LLM")

    # First, remove any markdown code fences
    cleaned_response = response
    if "```" in response:
        print("Removing markdown code fences…")
        # Remove ```python and ``` markers
        cleaned_response = re.sub(r'```(?:python)?\s*', '', cleaned_response)
        cleaned_response = cleaned_response.replace('```', '')
        print("Markdown fences removed")

    # Remove any explanations or other non-code text
    if "**Explanation:**" in cleaned_response:
        print("Removing explanations…")
        cleaned_response = re.sub(r'\*\*Explanation:\*\*.*?(?=def |$)', '', cleaned_response, flags=re.DOTALL)

    # Remove other markdown formatting that might interfere with code parsing
    cleaned_response = re.sub(r'\*\*.*?\*\*', '', cleaned_response)
    cleaned_response = re.sub(r'Function \d+ \(`.*?`\):', '', cleaned_response)
    cleaned_response = re.sub(r'^\s*(\*|\-|\d+\.)\s*', '', cleaned_response, flags=re.MULTILINE)

    # Make sure there's at least one function definition
    if "def " not in cleaned_response:
        print("ERROR: No function definitions found in response")
        raise ValueError("No function definitions found in response.")

    # Try to extract all function blocks
    code_blocks = re.findall(r"(def [\s\S]+?)(?=\ndef |\Z)", cleaned_response)
    print(f"Found {len(code_blocks)} code blocks")

    # If we got less than 2 blocks, try a simpler extraction
    if len(code_blocks) < 2:
        print("Warning: Less than 2 code blocks found, trying fallback extraction…")
        # Split on 'def ' and reconstruct function definitions
        parts = cleaned_response.split('def ')
        code_blocks = []
        for part in parts[1:]:  # Skip the first empty part
            code_blocks.append('def ' + part.strip())
        print(f"Fallback extraction found {len(code_blocks)} blocks")
        
        # If still less than 2 blocks, try more aggressive techniques
        if len(code_blocks) < 2:
            print("WARNING: Still found less than 2 code blocks after fallback extraction")
            print("Attempting more aggressive parsing techniques...")
            
            # Try to find any function-like blocks
            all_funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", cleaned_response)
            print(f"Found {len(all_funcs)} potential function definitions: {all_funcs}")
            
            # Look for the required function names specifically
            if "trading_strategy" in cleaned_response and "get_user_params" in cleaned_response:
                print("Found both required function names in the response text")
                
                # Try to extract each function individually
                trading_strategy_match = re.search(r"(def\s+trading_strategy\s*\([^)]*\)[\s\S]+?)(?=def\s+get_user_params|\Z)", cleaned_response)
                get_user_params_match = re.search(r"(def\s+get_user_params\s*\([^)]*\)[\s\S]+?)(?=def\s+trading_strategy|\Z)", cleaned_response)
                
                if trading_strategy_match:
                    print("Found trading_strategy function via direct regex")
                    code_blocks.append(trading_strategy_match.group(1))
                
                if get_user_params_match:
                    print("Found get_user_params function via direct regex")
                    code_blocks.append(get_user_params_match.group(1))
                
                print(f"After aggressive extraction, found {len(code_blocks)} blocks")
            else:
                print("ERROR: One or both required function names are missing from the response")
                if "trading_strategy" in cleaned_response:
                    print("Only found trading_strategy")
                elif "get_user_params" in cleaned_response:
                    print("Only found get_user_params")
                else:
                    print("Neither required function was found")

    trading_blocks = [block for block in code_blocks if re.search(r"def\s+trading_strategy\s*\(", block)]
    param_blocks = [block for block in code_blocks if re.search(r"def\s+get_user_params\s*\(", block)]
    print(f"Identified {len(trading_blocks)} trading_strategy functions and {len(param_blocks)} get_user_params functions")

    if not trading_blocks or not param_blocks:
        print("ERROR: Required function definitions not found")
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
            print(f"AST parsing failed: {e}. Falling back to regex.")
            # Fallback method if parsing fails
            first_line = code_block.strip().split('\n')[0]
            match = re.search(r'def\s+([a-zA-Z0-9_]+)', first_line)
            if match:
                return match.group(1)
        return ""

    # Syntax check the extracted code
    try:
        compile(code, '<string>', 'exec')
        print("trading_strategy code syntax is valid")
    except SyntaxError as e:
        print(f"WARNING: trading_strategy has syntax errors: {e}")
        # Try to fix common syntax issues
        code = code.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        try:
            compile(code, '<string>', 'exec')
            print("Syntax fixed after character replacement")
        except SyntaxError as e:
            print(f"Still has syntax errors after fixing: {e}")
    
    try:
        compile(input_code, '<string>', 'exec')
        print("get_user_params code syntax is valid")
    except SyntaxError as e:
        print(f"WARNING: get_user_params has syntax errors: {e}")
        # Try to fix common syntax issues
        input_code = input_code.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        try:
            compile(input_code, '<string>', 'exec')
            print("Syntax fixed after character replacement")
        except SyntaxError as e:
            print(f"Still has syntax errors after fixing: {e}")

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
    print(f"\nStarting simulation iteration {iteration} with user input: {user_input[:100]}…")

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
            print(f"\n--- Attempt {attempts + 1} ---")
            current_prompt = error_prompt if attempts > 0 else prompt
            
            # Calculate temperature (increase with more attempts)
            current_temperature = min(base_temperature + (temperature_step * (attempts // 2)), max_temperature)
            print(f"Using temperature: {current_temperature}")

            if not response or attempts > 0:
                print("Getting response from GPT…")
                response = ask_llama(current_prompt, temperature=current_temperature)
                if response is None:
                    raise Exception("LLM failed to provide a response.")

                print("Splitting response into parts…")
                parts = split_response(response)
                code = parts["code"]
                function_name_code = parts["function_name_code"]
                input_code = parts["input_code"]
                function_name_input_code = parts["function_name_input_code"]
                print(f"Extracted function names: {function_name_code} and {function_name_input_code}")

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

            print("Executing input code…")
            try:
                exec(input_code, globals())
            except Exception as e:
                print(f"Error executing input code: {e}")
                raise
                
            print("Evaluating input function with timeout protection...")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30) 
            
            try:
                inp = eval(f'{function_name_input_code}()')
                signal.alarm(0)
                print(f"Input function returned: {inp}")
            except TimeoutError:
                signal.alarm(0)
                print("ERROR: get_user_params function execution timed out after 30 seconds")
                raise Exception("get_user_params function timed out - likely contains an infinite loop or excessive computation.")
            except Exception as e:
                signal.alarm(0)
                print(f"Error evaluating input function: {e}")
                raise

            formatted_inp = repr(inp)

            print("Executing main code…")
            exec(code, globals())
            print("Evaluating main function…")
            
            print("Applying safety wrapper...")
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
                        print("Warning: Converting non-dict params to dict for safety")
                        local_vars['params'] = {'param': local_vars['params']}
                    
                # Guarantee a working .get method
                    if not hasattr(local_vars['params'], 'get'):
                        local_vars['params'] = dict(local_vars['params'])

                # Make sure 'close' reference exists inside params
                local_vars['params'].setdefault('close', close)

                # Convenience variable for strategy authors
                local_vars['n'] = len(close)

                print(f"Executing {func_name} with {len(close)} data points and params: {local_vars['params']}")

                # Attempt to run the strategy with various fall-backs for signature mismatches
                try:
                        result = eval(f"{func_name}(params)", globals(), local_vars)
                except TypeError as e:
                    if ("required positional argument" in str(e) or
                        "takes 1 positional argument" in str(e) or
                        "takes 0 positional arguments but 1 was given" in str(e)):
                        print(f"Retrying with direct close data due to TypeError: {e}")
                        try:
                            result = eval(f"{func_name}(close)", globals(), local_vars)
                        except TypeError as e2:
                            if "takes 0 positional arguments but 1 was given" in str(e2):
                                print(f"Retrying with no arguments due to TypeError: {e2}")
                                result = eval(f"{func_name}()", globals(), local_vars)
                            else:
                                raise
                        else:
                            raise
                    
                # Normalise result into expected 4-tuple if necessary
                if not (isinstance(result, tuple) and len(result) == 4) and not (
                    isinstance(result, dict) and all(k in result for k in [
                        'profit_loss', 'buy_points', 'sell_points', 'balance_over_time'])):
                    print(f"Warning: {func_name} returned {type(result)} instead of a 4-tuple or expected dict. Attempting to adapt.")
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
            print(f"Main function returned: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")

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
                            print(f"Warning: Adapting {name} from list of numbers to list of (index, price) tuples.")
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
                print(f"Last attempt ({attempts + 1}) failed to produce trades, and no prior best strategy. Falling back to basic buy-and-hold.")
                    
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

                print("Implemented fallback buy-and-hold strategy.")
                return bb, buy_points, sell_points, balance_over_time, code_info

            success = True

        except Exception as e:
            attempts += 1
            print(f"\n--- Attempt {attempts} FAILED with error: ---")
            error_traceback = traceback.format_exc()
            print(error_traceback)
            print("-------------------------------------------")

            if attempts >= max_error_attempts:
                print(f"\n!!! REACHED MAXIMUM ERROR ATTEMPTS ({max_error_attempts}) !!!")
                if best_bb != float('-inf'):
                    print("Giving up on this iteration and returning best results found so far.")
                    return best_bb, best_buy_points, best_sell_points, best_balance_over_time, best_code_info
                else:
                    print("No successful strategy found after all attempts. Returning empty results.")
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
            print(f"Attempting retry {attempts} to fix the error…")

    print("\nExecution successful.")
    return bb, buy_points, sell_points, balance_over_time, code_info

def plot_results(close, buy_points, sell_points, balance_over_time, counter=1, user_prompt=""):
    print("\nGenerating plot…")
    os.makedirs('stockbt/test_images', exist_ok=True)
    safe_prompt = re.sub(r'[^\w\s-]', '', user_prompt)[:30]
    safe_prompt = re.sub(r'\s+', '_', safe_prompt).strip('-_')

    # Optional: Clean close data
    close_arr = np.array(close, dtype=float)
    for i in range(1, len(close_arr)):
        if close_arr[i] == 0 or np.isnan(close_arr[i]):
            close_arr[i] = close_arr[i-1]
    close = close_arr.tolist()

    # Only use indices for plotting bars, always use close[index] for price
    valid_buy_indices = [i for i, _ in buy_points if 0 <= i < len(close)]
    valid_sell_indices = [i for i, _ in sell_points if 0 <= i < len(close)]

    plt.figure(figsize=(12, 10), dpi=150)
    plt.subplot(2, 1, 1)
    plt.plot(close, label='Close Price', linewidth=0.5, zorder=1)

    ymin, ymax = min(close), max(close)
    yrange = ymax - ymin
    plt.ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)

    # Plot green vertical bars for buy points
    for i in valid_buy_indices:
        plt.axvline(x=i, color='green', linestyle='-', linewidth=1, alpha=0.7, zorder=2, label='Buy' if i == valid_buy_indices[0] else "")
    # Plot red vertical bars for sell points
    for i in valid_sell_indices:
        plt.axvline(x=i, color='red', linestyle='-', linewidth=1, alpha=0.7, zorder=2, label='Sell' if i == valid_sell_indices[0] else "")

    plt.title('Stock Close Price with Buy/Sell Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(balance_over_time, label='Balance Over Time', color='blue', linewidth=2)
    plt.title('Balance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${{x:,.0f}}'))
    plt.legend()
    plt.tight_layout()

    filename = f'stockbt/test_images/test_image_{counter}_{safe_prompt}.png'
    plt.savefig(filename, dpi=150)
    print(f"Plot saved as '{filename}'")
    plt.close()

def run_improved_simulation(user_input, max_error_attempts=10):
    """Run an iterative improvement process to achieve profit target."""
    print("\n=== STARTING ITERATIVE IMPROVEMENT PROCESS ===")
    
    # Calculate buy and hold profit as baseline
    # (We will update close and dates each iteration)
    global close, dates
    
    # Create a base counter for plots
    from datetime import datetime
    base_counter = int(datetime.now().timestamp())
    
    # Keep track of best strategy
    best_profit = float('-inf')
    best_strategy_data = None
    
    iteration = 1
    max_iterations = 5
    current_profit = 0
    improvement_context = ""
    
    # Initialize to allow first loop
    buy_hold_profit = 0
    target_profit = float('inf')

    while iteration <= max_iterations and current_profit < target_profit:
        print(f"\n--- IMPROVEMENT ITERATION {iteration}/{max_iterations} ---")
        
        # Select a new random dataset for this iteration
        file_path = get_random_dataset()
        print(f"[ITER {iteration}] Loading data from {file_path}")
        close = []
        dates = []
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    close.append(float(row['Close']))
                    dates.append(row['Date'])
                except (ValueError, KeyError) as e:
                    print(f"Warning: Error parsing row {row}: {e}")
        print(f"Loaded {len(close)} data points for this iteration.")
        
        # Calculate buy and hold profit as baseline for this dataset
        buy_hold_profit = calculate_buy_and_hold_profit(close)
        target_profit = buy_hold_profit * 1.5
        
        # Run simulation with current context
        current_profit, buy_points, sell_points, balance_over_time, code_info = run_simulation(
            enhanced_prompt, 
            user_input_raw, 
            improvement_context, 
            iteration,
            max_iterations,
            max_error_attempts
        )
        
        # Calculate percentage above buy-and-hold
        percent_above_buyhold = ((current_profit / buy_hold_profit) - 1) * 100 if buy_hold_profit > 0 else 0
        percent_vs_target = ((current_profit / target_profit) - 1) * 100 if target_profit > 0 else 0
        
        print(f"\nCurrent Profit: ${current_profit:.2f}")
        print(f"Buy & Hold Profit: ${buy_hold_profit:.2f}")
        print(f"Target Profit (1.5x Buy & Hold): ${target_profit:.2f}")
        print(f"Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%")
        print(f"Performance vs Target: {percent_vs_target:+.2f}%")
        
        # Update best strategy if current one is better
        if current_profit > best_profit:
            print(f"New best strategy found! Profit: ${current_profit:.2f}")
            best_profit = current_profit
            best_strategy_data = {
                'profit': current_profit,
                'buy_points': buy_points,
                'sell_points': sell_points,
                'balance_over_time': balance_over_time,
                'code_info': code_info,
                'iteration': iteration,
                'buy_hold_profit': buy_hold_profit,
                'target_profit': target_profit,
                'percent_above_buyhold': percent_above_buyhold,
                'percent_vs_target': percent_vs_target
            }
        
        if current_profit >= target_profit:
            print("\n=== TARGET PROFIT ACHIEVED! ===")
            break
        
        # If we haven't reached our target and haven't hit max iterations, 
        # ask the LLM to suggest improvements
        if iteration < max_iterations:
            print("\nGenerating improvement suggestions...")
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
            
            if improvement_suggestions:
                improvement_context = f"""
**PREVIOUS ITERATION PERFORMANCE:**
- Previous Profit: ${current_profit:.2f} (vs Buy & Hold: {percent_above_buyhold:+.2f}%)
- Target Profit: ${target_profit:.2f}
- Performance vs Target: {percent_vs_target:+.2f}%

**IMPROVEMENT SUGGESTIONS:**
{improvement_suggestions}

Based on the above analysis, implement these improvements in your trading strategy to achieve the target profit of ${target_profit:.2f}.
"""
            
        iteration += 1
    
    # After all iterations, save only the best strategy's code and image
    if best_strategy_data:
        print(f"\n=== SAVING BEST STRATEGY (Profit: ${best_strategy_data['profit']:.2f}) ===")
        
        # Get percentage metrics from the best strategy
        percent_vs_target = best_strategy_data['percent_vs_target']
        percent_above_buyhold = best_strategy_data['percent_above_buyhold']
        buy_hold_profit = best_strategy_data['buy_hold_profit']
        target_profit = best_strategy_data['target_profit']
        
        print(f"Best Profit is {percent_vs_target:+.2f}% compared to target ({percent_above_buyhold:+.2f}% vs buy & hold)")
        
        # Save the best strategy code
        code_info = best_strategy_data['code_info']
        original_input = code_info.get('original_user_input', 'Original input not available')
        enhanced_input = code_info.get('enhanced_user_input', 'Enhanced input not available')
        full_llm_prompt_for_best = code_info.get('full_llm_prompt', 'Full LLM prompt not available')
        strategy_filename = f"/Users/avneh/Code/HackSFProject/stockbt/generated_code/best_strategy_{code_info['timestamp']}_{code_info['safe_user_input']}.txt"
        
        with open(strategy_filename, 'w') as f:
            f.write(f"# BEST Trading Strategy - Iteration {code_info['iteration']}\n")
            f.write(f"# User Input (Original): {original_input[:500]}...\n")
            f.write(f"# User Input (Enhanced by LLM): {enhanced_input[:500]}...\n")
            f.write(f"# Profit: ${best_strategy_data['profit']:.2f}\n")
            f.write(f"# Buy & Hold Profit: ${buy_hold_profit:.2f}\n")
            f.write(f"# Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%\n")
            f.write(f"# Performance vs Target: {percent_vs_target:+.2f}%\n\n")
            f.write("# --- Full LLM Prompt Used for This Strategy --- #\n")
            f.write(full_llm_prompt_for_best)
            f.write("\n\n# --- Generated Strategy Function --- #\n")
            f.write(code_info['code'])
            f.write("\n\n# --- Generated Parameters Function --- #\n")
            f.write(code_info['input_code'])
        
        print(f"Saved best strategy code to {strategy_filename}")
        
        # Generate and save the best strategy plot
        plot_results(
            close, 
            best_strategy_data['buy_points'], 
            best_strategy_data['sell_points'], 
            best_strategy_data['balance_over_time'],
            counter=base_counter + 1000,
            user_prompt=f"BEST_{code_info['safe_user_input']}"
        )
        
        best_image_path = f"stockbt/test_images/test_image_{base_counter + 1000}_BEST_{code_info['safe_user_input']}.png"
    else:
        strategy_filename = ""
        best_image_path = ""
        percent_vs_target = 0
        percent_above_buyhold = 0
    
    return (best_profit,
            best_strategy_data['buy_points'] if best_strategy_data else [],
            best_strategy_data['sell_points'] if best_strategy_data else [],
            best_strategy_data['balance_over_time'] if best_strategy_data else [],
            strategy_filename,
            best_image_path,
            percent_vs_target,
            percent_above_buyhold if best_strategy_data else 0)

def enhance_user_prompt(original_prompt):
    """Enhance the user's trading strategy prompt using LLM."""
    print("\n=== ENHANCING USER PROMPT ===")
    print(f"Original prompt: {original_prompt[:100]}...")
    
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
        print("Getting enhanced prompt from LLM...")
        enhanced_prompt = ask_llama(enhancement_prompt, temperature=0.5)  # Lower temperature for more conservative edits
        
        if enhanced_prompt and len(enhanced_prompt) > 50:  # Basic validation
            print("Successfully enhanced the prompt.")
            print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
            return enhanced_prompt
        else:
            print("Warning: Enhancement returned empty or very short result. Using original prompt.")
            return original_prompt
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return original_prompt

def print_wikipedia_history_and_url(topic):
    """
    Fetches the 'History' section and full URL for a given Wikipedia topic and prints them.
    """
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')
    page_py = wiki_wiki.page(topic)
    section_history = page_py.section_by_title('History')
    if section_history:
        print("%s - %s" % (section_history.title, section_history.text))
    else:
        print("No 'History' section found for this topic.")
    print(page_py.fullurl)


def describe_key_methods_with_llm(topic):
    """
    Calls the LLM to describe key methods used in the topic, as they would appear in Wikipedia.
    """
    prompt = f"""
Describe the key methods used in {topic} as they would appear in a Wikipedia article. List and briefly explain each method, focusing on their purpose and typical usage.
"""
    description = ask_llama(prompt, temperature=0.3)
    print(description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest simulation.")
    parser.add_argument("strategy", nargs="?", default="", help="Trading strategy description")
    parser.add_argument("--json", action="store_true", help="Return results as JSON on stdout")
    args = parser.parse_args()

    if args.strategy:
        user_input_raw = args.strategy
    else:
        print("Waiting for user input…")
        user_input_raw = input("Enter your trading strategy: ")

    # Enhance the user's prompt conservatively
    enhanced_prompt = enhance_user_prompt(user_input_raw)

    print("Starting simulation with iterative improvement…")
    best_profit, buy_pts, sell_pts, bal_hist, code_path, img_path, percent_vs_target, percent_above_buyhold = run_improved_simulation(enhanced_prompt)
    print("Simulation complete!")

    if args.json:
        summary = {
            "profit": best_profit,
            "code_path": code_path,
            "image_path": img_path,
            "buy_points": buy_pts[:10],  # sample
            "sell_points": sell_pts[:10],  # sample
            "percent_vs_target": percent_vs_target,
            "percent_above_buyhold": percent_above_buyhold
        }
        print(json.dumps(summary))
    else:
        print(f"Best Profit: ${best_profit:.2f}")
        print(f"Performance vs Buy & Hold: {percent_above_buyhold:+.2f}%")
        print(f"Performance vs Target: {percent_vs_target:+.2f}%")
        print(f"Code saved to: {code_path}")
        print(f"Image saved to: {img_path}")
        sys.exit(0) #yeye