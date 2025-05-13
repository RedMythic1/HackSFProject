import math
import csv
import matplotlib.pyplot as plt
import os
import traceback
import re
import g4f

print("Initializing stocker_test.py with g4f (GPT-4 Free)…")

# Create test_images folder if it doesn't exist
os.makedirs('stockbt/test_images', exist_ok=True)
print("Created test_images directory")

file_path = 'stockbt/datasets/test.csv'
print(f"Loading data from {file_path}")
# Load data directly into lists instead of a DataFrame
close = []
dates = []
with open(file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            # Convert to float and add to list
            close.append(float(row['Close']))
            dates.append(row['Date'])
        except (ValueError, KeyError) as e:
            print(f"Warning: Error parsing row {row}: {e}")

print(f"Loaded {len(close)} data points")

initial_balance = 100000
print(f"Initial balance set to ${initial_balance:,.2f}")

# -----------------------------------------------------------------------------
# GPT-4 FREE (g4f) helper
# -----------------------------------------------------------------------------

def ask_llama(prompt, temperature=0.7):
    """Ask a question to the GPT-4 Free backend via g4f."""
    print(f"\nSending prompt to g4f ChatCompletion with temperature={temperature}…")
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o",
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
        except Exception:
            # Fallback method if parsing fails
            first_line = code_block.strip().split('\n')[0]
            match = re.search(r'def\s+([a-zA-Z0-9_]+)', first_line)
            if match:
                return match.group(1)
        return ""

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

def run_simulation(user_input, improvement_context="", iteration=1, max_iterations=5, max_error_attempts=10):
    print(f"\nStarting simulation iteration {iteration} with user input: {user_input[:100]}…")

    # Construct the prompt with improvement context if provided
    if improvement_context:
        prompt = f"""[INST]
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions based on the user's strategy, following the strict rules below.

**User Strategy:** {user_input}

**IMPROVEMENT CONTEXT:** 
{improvement_context}

**DATA INFORMATION:**
- The data is loaded from a CSV file at 'stockbt/datasets/test.csv'
- CSV Format: Date,Open,High,Low,Close,Volume
- In the code, only the Close prices are available as a simple Python list named 'close'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it's already available as 'close'

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
    *   Must implement the user's trading strategy using the provided `params`.
    *   When determining buy/sell prices, you **MUST NOT** rely on hard-coded or absolute price values. Instead, compute them as a **multiplier of the current `price_today`** or other relative measures derived from `params` (e.g., `buy_price = price_today * params["buy_mult"]`).
    *   Define necessary loop variables such as `i`, `price_today`, and any multiplier-derived `buy_price`/`sell_price` consistent with the strategy (note that `close` is available in the execution context).
    *   Must include the **MANDATORY TRADING LOGIC** block verbatim within its primary execution loop or logic.
    *   Must **return** exactly `balance - initial_balance, buy_points, sell_points, balance_over_time` in that order.

3.  **Function 2: `get_user_params`**
    *   Must be named exactly `get_user_params`.
    *   **ABSOLUTELY CRITICAL: THIS FUNCTION MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** It must automatically determine or iteratively search for optimized numeric parameters relevant to the strategy.
    *   Must **return** the chosen numeric parameters as a single tuple or dictionary.
    *   **DO NOT** read files (e.g., CSVs) within this function.
    *   **DO NOT** hard-code absolute price thresholds; parameters should be relative scales/ratios or other dimensionless values.

4.  **Environment:**
    *   Assume `math` module is pre-imported and available. Do **NOT** add import statements.
    *   Assume a Python list named `close` containing the price data is available in the execution scope of `trading_strategy`.

5.  **Simplified Data Structure:** The price series is available as the global Python list named `close`. It's a simple list of floating-point values. Access elements with standard indexing: `close[i]` or iterate with `for i, price_today in enumerate(close):`. No pandas/DataFrame code is needed.

6.  **Character Set:** Your entire code must use ONLY standard ASCII characters. Avoid typographic quotes (' ' " " ) or long dashes (—). Use straight quotes (' ") and hyphens (-) instead.

7.  **Data Access:** The price series is available as the global list named `close`. **Do NOT** attempt to access data via dictionaries or DataFrames. Simply use `close[i]` or iterate with `for i, price_today in enumerate(close):`.

8.  **Key-Error Safety:** Your code should never raise `KeyError`. Reference only variables you explicitly define.

**MANDATORY TRADING LOGIC (Include and adapt this block inside `trading_strategy`):**
```python
initial_balance = 100000
balance = initial_balance
shares = 0
end_price = 0 # This likely needs to be updated based on strategy/loop
buy_points = []
sell_points = []
balance_over_time = [balance]

# --- Start of logic needing integration with your strategy loop ---
# You need a loop here (e.g., for i, price_today in enumerate(close):)
# Inside the loop, calculate buy/sell signals based on user_input strategy and params.
# If buying:
#   buy_price = price_today * params.get('buy_mult', 1)  # Use a multiplier from params rather than absolute price
#   shares_to_buy = math.floor(balance / buy_price)
#   if shares_to_buy > 0:
#       balance -= shares_to_buy * buy_price
#       shares += shares_to_buy
#       buy_points.append((i, price_today))
#       print(f"Bought {{shares_to_buy}} shares at ${{price_today:.2f}}. Balance: ${{balance:.2f}}")
# If selling:
#   if shares > 0:
#       sell_price = price_today * params.get('sell_mult', 1)  # Use a multiplier from params
#       balance += shares * sell_price
#       sell_points.append((i, price_today))
#       print(f"Sold {{shares}} shares at ${{sell_price:.2f}}. Balance: ${{balance:.2f}}")
#       shares = 0
# Update end_price if needed by the strategy for subsequent calculations
# --- End of logic needing integration ---

balance_over_time.append(balance) # Append balance *after* each potential trade in the loop

# After the loop:
if shares > 0: # Check if shares are held at the end
    final_price = close[-1] if close else 0
    balance += shares * final_price
    print(f"Ending balance adjustment: Added value of {{shares}} held shares at ${{final_price:.2f}}")
balance_over_time.append(balance) # Append final balance state
return balance - initial_balance, buy_points, sell_points, balance_over_time
```
**(Note:** The MANDATORY TRADING LOGIC above is a template. You **must** integrate it correctly within the `trading_strategy` function's loop, defining variables like `i`, `price_today`, `buy_price`, `sell_price` according to the user's strategy and the parameters from `get_user_params`. The comments indicate where your strategy-specific logic needs to fit.)

**Final Check:** Ensure your output is only the two raw Python function definitions separated by a single blank line. No markdown, no comments outside the functions, no extra text.

**Important Output Format Requirements:**
- Your trading_strategy function may return its results in EITHER of these formats:
  1. Standard tuple: (profit_loss, buy_points, sell_points, balance_over_time)
  2. Dictionary format: {{'profit_loss': profit_amount, 'buy_points': [...], 'sell_points': [...], 'balance_over_time': [...]}}

- The dictionary format is preferred as it's more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
[/INST]"""
    else:
        prompt = f"""[INST]
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions based on the user's strategy, following the strict rules below.

**User Strategy:** {user_input}

**DATA INFORMATION:**
- The data is loaded from a CSV file at 'stockbt/datasets/test.csv'
- CSV Format: Date,Open,High,Low,Close,Volume
- In the code, only the Close prices are available as a simple Python list named 'close'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it's already available as 'close'

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
    *   Must implement the user's trading strategy using the provided `params`.
    *   When determining buy/sell prices, you **MUST NOT** rely on hard-coded or absolute price values. Instead, compute them as a **multiplier of the current `price_today`** or other relative measures derived from `params` (e.g., `buy_price = price_today * params["buy_mult"]`).
    *   Define necessary loop variables such as `i`, `price_today`, and any multiplier-derived `buy_price`/`sell_price` consistent with the strategy (note that `close` is available in the execution context).
    *   Must include the **MANDATORY TRADING LOGIC** block verbatim within its primary execution loop or logic.
    *   Must **return** exactly `balance - initial_balance, buy_points, sell_points, balance_over_time` in that order.

3.  **Function 2: `get_user_params`**
    *   Must be named exactly `get_user_params`.
    *   **ABSOLUTELY CRITICAL: THIS FUNCTION MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** It must automatically determine or iteratively search for optimized numeric parameters relevant to the strategy.
    *   Must **return** the chosen numeric parameters as a single tuple or dictionary.
    *   **DO NOT** read files (e.g., CSVs) within this function.
    *   **DO NOT** hard-code absolute price thresholds; parameters should be relative scales/ratios or other dimensionless values.

4.  **Environment:**
    *   Assume `math` module is pre-imported and available. Do **NOT** add import statements.
    *   Assume a Python list named `close` containing the price data is available in the execution scope of `trading_strategy`.

5.  **Simplified Data Structure:** The price series is available as the global Python list named `close`. It's a simple list of floating-point values. Access elements with standard indexing: `close[i]` or iterate with `for i, price_today in enumerate(close):`. No pandas/DataFrame code is needed.

6.  **Character Set:** Your entire code must use ONLY standard ASCII characters. Avoid typographic quotes (' ' " " ) or long dashes (—). Use straight quotes (' ") and hyphens (-) instead.

7.  **Data Access:** The price series is available as the global list named `close`. **Do NOT** attempt to access data via dictionaries or DataFrames. Simply use `close[i]` or iterate with `for i, price_today in enumerate(close):`.

8.  **Key-Error Safety:** Your code should never raise `KeyError`. Reference only variables you explicitly define.

**MANDATORY TRADING LOGIC (Include and adapt this block inside `trading_strategy`):**
```python
initial_balance = 100000
balance = initial_balance
shares = 0
end_price = 0 # This likely needs to be updated based on strategy/loop
buy_points = []
sell_points = []
balance_over_time = [balance]

# --- Start of logic needing integration with your strategy loop ---
# You need a loop here (e.g., for i, price_today in enumerate(close):)
# Inside the loop, calculate buy/sell signals based on user_input strategy and params.
# If buying:
#   buy_price = price_today * params.get('buy_mult', 1)  # Use a multiplier from params rather than absolute price
#   shares_to_buy = math.floor(balance / buy_price)
#   if shares_to_buy > 0:
#       balance -= shares_to_buy * buy_price
#       shares += shares_to_buy
#       buy_points.append((i, price_today))
#       print(f"Bought {{shares_to_buy}} shares at ${{price_today:.2f}}. Balance: ${{balance:.2f}}")
# If selling:
#   if shares > 0:
#       sell_price = price_today * params.get('sell_mult', 1)  # Use a multiplier from params
#       balance += shares * sell_price
#       sell_points.append((i, price_today))
#       print(f"Sold {{shares}} shares at ${{sell_price:.2f}}. Balance: ${{balance:.2f}}")
#       shares = 0
# Update end_price if needed by the strategy for subsequent calculations
# --- End of logic needing integration ---

balance_over_time.append(balance) # Append balance *after* each potential trade in the loop

# After the loop:
if shares > 0: # Check if shares are held at the end
    final_price = close[-1] if close else 0
    balance += shares * final_price
    print(f"Ending balance adjustment: Added value of {{shares}} held shares at ${{final_price:.2f}}")
balance_over_time.append(balance) # Append final balance state
return balance - initial_balance, buy_points, sell_points, balance_over_time
```
**(Note:** The MANDATORY TRADING LOGIC above is a template. You **must** integrate it correctly within the `trading_strategy` function's loop, defining variables like `i`, `price_today`, `buy_price`, `sell_price` according to the user's strategy and the parameters from `get_user_params`. The comments indicate where your strategy-specific logic needs to fit.)

**Final Check:** Ensure your output is only the two raw Python function definitions separated by a single blank line. No markdown, no comments outside the functions, no extra text.

**Important Output Format Requirements:**
- Your trading_strategy function may return its results in EITHER of these formats:
  1. Standard tuple: (profit_loss, buy_points, sell_points, balance_over_time)
  2. Dictionary format: {{'profit_loss': profit_amount, 'buy_points': [...], 'sell_points': [...], 'balance_over_time': [...]}}

- The dictionary format is preferred as it's more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
[/INST]"""

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

                # Validate that get_user_params does not attempt to prompt the user.
                if re.search(r'\binput\s*\(', input_code):
                    raise Exception("Detected forbidden user prompt (input()) in get_user_params. This function must run autonomously without requiring user interaction.")

            # Execute the potentially corrected code
            print("Executing input code…")
            exec(input_code, globals())
            print("Evaluating input function…")
            inp = eval(f'{function_name_input_code}()')
            print(f"Input function returned: {inp}")

            formatted_inp = repr(inp)

            print("Executing main code…")
            exec(code, globals())
            print("Evaluating main function…")
            
            # Add a safety wrapper for common trading_strategy errors
            print("Applying safety wrapper...")
            def safe_execute_trading_strategy(func_name, params):
                """Execute trading strategy with extra safeguards against common errors"""
                try:
                    # Create a safer scope for execution
                    local_vars = {'params': params, 'close': close}
                    
                    # Defensive conversion of params (for common type errors)
                    if isinstance(params, tuple) and len(params) > 0:
                        # Try to build a dict from the tuple for safer access
                        safe_params = {}
                        for i, val in enumerate(params):
                            safe_params[i] = val
                            safe_params[f'param{i+1}'] = val
                        local_vars['params'] = safe_params
                    
                    # If params isn't a dict, make it into one with defaults
                    if not isinstance(local_vars['params'], dict):
                        print("Warning: Converting non-dict params to dict for safety")
                        local_vars['params'] = {'param': local_vars['params']}
                    
                    # Make sure params dict has get method (for safety)
                    if not hasattr(local_vars['params'], 'get'):
                        old_params = local_vars['params']
                        local_vars['params'] = dict(old_params)
                    
                    # CRITICAL FIX: Make sure 'close' is also accessible via params dict
                    # This handles the case where the function expects close as params['close']
                    if 'close' not in local_vars['params']:
                        local_vars['params']['close'] = close
                    
                    # Handle common derivative/math issues with helpful variables
                    local_vars['np'] = __import__('numpy') if 'numpy' not in globals() else globals()['numpy']
                    local_vars['math'] = __import__('math')
                    local_vars['n'] = len(close)  # Common shorthand for data length
                    
                    # Execute the function in this safe scope
                    print(f"Executing {func_name} with {len(close)} data points and {len(local_vars['params'])} parameters")
                    try:
                        result = eval(f"{func_name}(params)", globals(), local_vars)
                    except TypeError as e:
                        # If first attempt fails, try passing just the close data directly
                        if "required positional argument" in str(e) or "takes 1 positional argument" in str(e):
                            print(f"Retrying with direct close data: {e}")
                            result = eval(f"{func_name}(close)", globals(), local_vars)
                        else:
                            raise
                    
                    # Validate the results to handle common errors
                    if not isinstance(result, tuple) or len(result) != 4:
                        print(f"Error: {func_name} returned {result}, not a 4-tuple. Fixing...")
                        if isinstance(result, (int, float)):
                            # Only profit/loss returned
                            return result, [], [], [initial_balance, initial_balance + result]
                        elif isinstance(result, dict) and 'balance' in result:
                            # Extract balance from dictionary result
                            bb = result.get('balance', 0) - initial_balance
                            return bb, [], [], [initial_balance, initial_balance + bb]
                        elif isinstance(result, dict) and ('buy_points' in result or 'sell_points' in result or 'buy' in result or 'sell' in result):
                            # Handle dictionary with buy/sell points in various key formats
                            print("Converting dictionary with buy/sell data...")
                            
                            # Extract buy points - check multiple possible keys
                            buy_points = result.get('buy_points', result.get('buy', result.get('buys', result.get('buy_indices', []))))
                            
                            # Extract sell points - check multiple possible keys
                            sell_points = result.get('sell_points', result.get('sell', result.get('sells', result.get('sell_indices', []))))
                            
                            # Make sure points are formatted as (index, price) tuples
                            if buy_points and not isinstance(buy_points[0], tuple):
                                formatted_buy_points = []
                                for idx in buy_points:
                                    if isinstance(idx, int) and 0 <= idx < len(close):
                                        formatted_buy_points.append((idx, close[idx]))
                                buy_points = formatted_buy_points
                            
                            if sell_points and not isinstance(sell_points[0], tuple):
                                formatted_sell_points = []
                                for idx in sell_points:
                                    if isinstance(idx, int) and 0 <= idx < len(close):
                                        formatted_sell_points.append((idx, close[idx]))
                                sell_points = formatted_sell_points
                            
                            # Calculate approximate profit based on trades
                            balance = initial_balance
                            balance_over_time = [balance]
                            shares = 0
                            
                            # Sort points by index for chronological processing
                            buy_indices = [idx for idx, _ in buy_points] if buy_points and isinstance(buy_points[0], tuple) else buy_points
                            sell_indices = [idx for idx, _ in sell_points] if sell_points and isinstance(sell_points[0], tuple) else sell_points
                            
                            all_indices = sorted(set(buy_indices + sell_indices))  # Use set to remove duplicates
                            
                            for idx in all_indices:
                                if idx in buy_indices and shares == 0:
                                    # Buy operation
                                    buy_price = close[idx]
                                    shares_to_buy = math.floor(balance / buy_price)
                                    if shares_to_buy > 0:
                                        balance -= shares_to_buy * buy_price
                                        shares += shares_to_buy
                                elif idx in sell_indices and shares > 0:
                                    # Sell operation
                                    sell_price = close[idx]
                                    balance += shares * sell_price
                                    shares = 0
                                
                                balance_over_time.append(balance + (shares * close[idx] if shares > 0 else 0))
                            
                            # Final liquidation if still holding shares
                            if shares > 0:
                                balance += shares * close[-1]
                                
                            bb = balance - initial_balance
                            
                            # Create standardized dictionary output format
                            standardized_output = {
                                'profit_loss': bb,
                                'buy_points': buy_points,
                                'sell_points': sell_points,
                                'balance_over_time': balance_over_time
                            }
                            
                            # Also return as regular expected tuple for backwards compatibility
                            return bb, buy_points, sell_points, balance_over_time
                        elif isinstance(result, list):
                            # If we got a list of signals, convert to buy/sell points
                            print("Converting signal list to buy/sell points...")
                            buy_points = []
                            sell_points = []
                            balance = initial_balance
                            balance_over_time = [balance]
                            shares = 0
                            
                            for i, signal in enumerate(result):
                                if i >= len(close):
                                    break
                                    
                                if signal > 0 and shares == 0:  # Buy signal
                                    buy_price = close[i]
                                    shares_to_buy = math.floor(balance / buy_price)
                                    if shares_to_buy > 0:
                                        balance -= shares_to_buy * buy_price
                                        shares += shares_to_buy
                                        buy_points.append((i, close[i]))
                                elif signal < 0 and shares > 0:  # Sell signal
                                    sell_price = close[i]
                                    balance += shares * sell_price
                                    sell_points.append((i, close[i]))
                                    shares = 0
                                
                                balance_over_time.append(balance + (shares * close[i] if shares > 0 else 0))
                            
                            # Final liquidation
                            if shares > 0:
                                balance += shares * close[-1]
                                shares = 0
                                
                            bb = balance - initial_balance
                            return bb, buy_points, sell_points, balance_over_time
                        else:
                            raise ValueError(f"Invalid return value: {result}")
                    
                    # Unpack the results
                    bb, buy_points, sell_points, balance_over_time = result
                    
                    # Ensure numeric profit/loss 
                    if not isinstance(bb, (int, float)):
                        print(f"Warning: Converting non-numeric profit {bb} to float")
                        try:
                            bb = float(bb)
                        except:
                            bb = 0.0
                    
                    # Ensure buy_points and sell_points are lists
                    if not isinstance(buy_points, list):
                        print(f"Warning: Converting buy_points to list")
                        buy_points = [buy_points] if buy_points else []
                    
                    if not isinstance(sell_points, list):
                        print(f"Warning: Converting sell_points to list")
                        sell_points = [sell_points] if sell_points else []
                    
                    # Convert tuples to the right format
                    buy_points = [(i if isinstance(i, int) else 0, p) 
                                  for i, p in enumerate(buy_points) 
                                  if not isinstance(buy_points[0], tuple)]
                    
                    sell_points = [(i if isinstance(i, int) else 0, p) 
                                   for i, p in enumerate(sell_points) 
                                   if not isinstance(sell_points[0], tuple)]
                    
                    # Ensure balance_over_time is a list
                    if not isinstance(balance_over_time, list):
                        print(f"Warning: Converting balance_over_time to list")
                        balance_over_time = [initial_balance, initial_balance + bb]
                    
                    return bb, buy_points, sell_points, balance_over_time
                except Exception as e:
                    print(f"Error in safe execution: {e}")
                    traceback.print_exc()
                    # Return a safe fallback to allow the process to continue
                    return 0, [], [], [initial_balance, initial_balance]
            
            # Execute the trading strategy with the safety wrapper
            bb, buy_points, sell_points, balance_over_time = safe_execute_trading_strategy(function_name_code, inp)
            print(f"Main function returned: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")

            # NEW validation: ensure the strategy executed at least one trade and bb is numeric
            if not buy_points or not sell_points:
                error_msg = "Generated strategy did not execute any trades (buy/sell points empty)."
                if attempts < max_error_attempts - 1:  # If we have attempts left
                    error_msg += " Please adjust parameters/logic to ensure trades are executed."
                    raise Exception(error_msg)
                else:  # On last attempt, return a basic buy-and-hold strategy
                    print("Last attempt failed - falling back to basic buy-and-hold strategy")
                    # Implement a basic buy-and-hold strategy
                    buy_points = [(0, close[0])]
                    sell_points = [(len(close)-1, close[-1])]
                    shares = math.floor(initial_balance / close[0])
                    final_balance = shares * close[-1]
                    bb = final_balance - initial_balance
                    balance_over_time = [initial_balance, final_balance]
                    print("Implemented fallback buy-and-hold strategy")
                    return bb, buy_points, sell_points, balance_over_time

            if not isinstance(bb, (int, float)):
                raise Exception("Profit/loss (bb) is not numeric. Strategy must return a numeric P&L.")

            success = True

        except Exception as e:
            attempts += 1
            print(f"\n--- Attempt {attempts} FAILED with error: ---")
            error_traceback = traceback.format_exc()
            print(error_traceback)
            print("-------------------------------------------")

            # Check if we've reached the maximum number of attempts
            if attempts >= max_error_attempts:
                print(f"\n!!! REACHED MAXIMUM ERROR ATTEMPTS ({max_error_attempts}) !!!")
                print("Giving up on this iteration and returning empty results.")
                return 0, [], [], [initial_balance, initial_balance]  # Return empty results

            response = None  # Clear previous response to force LLM call

            # Construct the error prompt for the next attempt
            error_prompt = f"""[INST]
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
- You DO NOT need to load the data yourself - it's already available as 'close'

Your response MUST consist of:
1. The COMPLETE `trading_strategy` function (entire function, not just the fixed part)
2. A single blank line
3. The COMPLETE `get_user_params` function (entire function, not just the fixed part)

**DO NOT OMIT ANY CODE.** Make sure you include all loops, conditional statements, variable definitions, and other code from both functions.

**IMPORTANT FUNCTION SIGNATURES:**
- trading_strategy must accept EXACTLY ONE positional argument (params): `def trading_strategy(params):`
- get_user_params must accept NO parameters: `def get_user_params():`

**Common Errors to Check:**
- Use only standard ASCII characters (no " " ' — etc.).
- `trading_strategy` must accept EXACTLY ONE positional argument (remove any extras like `formatted_inp`, `data`, etc.).
- Don't call undefined functions (like 'update_end_price').
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
  2. Dictionary format: {{'profit_loss': profit_amount, 'buy_points': [...], 'sell_points': [...], 'balance_over_time': [...]}}

- The dictionary format is preferred as it's more explicit. For buy_points and sell_points, you can provide either:
  * A list of (index, price) tuples
  * A list of indices where trades occurred
[/INST]"""
            print(f"Attempting retry {attempts} to fix the error…")

    print("\nExecution successful. Generating plot…")
    
    # Create a unique counter for this plot
    from datetime import datetime
    plot_counter = int(datetime.now().timestamp())
    
    plot_results(close, buy_points, sell_points, balance_over_time, counter=plot_counter, user_prompt=user_input[:50])

    return bb, buy_points, sell_points, balance_over_time

def run_improved_simulation(user_input, max_error_attempts=10):
    """Run an iterative improvement process to achieve profit target."""
    print("\n=== STARTING ITERATIVE IMPROVEMENT PROCESS ===")
    
    # Calculate buy and hold profit as baseline
    buy_hold_profit = calculate_buy_and_hold_profit(close)
    target_profit = buy_hold_profit * 1.5
    
    print(f"Buy & Hold Profit: ${buy_hold_profit:.2f}")
    print(f"Target Profit (1.5x): ${target_profit:.2f}")
    
    iteration = 1
    max_iterations = 5
    current_profit = 0
    improvement_context = ""
    
    # Create a base counter for plots
    from datetime import datetime
    base_counter = int(datetime.now().timestamp())
    
    while iteration <= max_iterations and current_profit < target_profit:
        print(f"\n--- IMPROVEMENT ITERATION {iteration}/{max_iterations} ---")
        
        # Run simulation with current context
        current_profit, buy_points, sell_points, balance_over_time = run_simulation(
            user_input, 
            improvement_context, 
            iteration,
            max_iterations,
            max_error_attempts
        )
        
        print(f"\nCurrent Profit: ${current_profit:.2f}")
        print(f"Target Profit: ${target_profit:.2f}")
        
        # Create a separate plot for this iteration, with the iteration number
        plot_counter = base_counter + iteration
        plot_results(close, buy_points, sell_points, balance_over_time, 
                    counter=plot_counter, 
                    user_prompt=f"iter{iteration}_{user_input[:40]}")
        
        if current_profit >= target_profit:
            print("\n=== TARGET PROFIT ACHIEVED! ===")
            break
        
        # If we haven't reached our target and haven't hit max iterations, 
        # ask the LLM to suggest improvements
        if iteration < max_iterations:
            print("\nGenerating improvement suggestions...")
            improvement_prompt = f"""[INST]
**Strategy Analysis and Improvement**

I need you to analyze the current trading strategy and suggest specific improvements to increase profitability.

**Current Strategy:** {user_input}

**Performance Metrics:**
- Current Profit: ${current_profit:.2f}
- Buy & Hold Profit: ${buy_hold_profit:.2f}
- Target Profit (1.5x Buy & Hold): ${target_profit:.2f}
- Number of Buy Points: {len(buy_points)}
- Number of Sell Points: {len(sell_points)}

**Current Price Series Statistics:**
- First Price: ${close[0]:.2f}
- Last Price: ${close[-1]:.2f}
- Min Price: ${min(close):.2f}
- Max Price: ${max(close):.2f}
- Price Range: ${max(close) - min(close):.2f}

**Your Task:**
Based on this information, suggest 3-5 specific, actionable improvements to the trading strategy to increase profit. Focus on:
1. Parameter adjustments (multipliers, thresholds, etc.)
2. Logic improvements to buy/sell decision-making
3. Any inefficiencies in the current approach

**Output Format:** 
Provide a concise, focused analysis of what's working/not working, followed by specific, bulleted improvement suggestions. Your response will be fed directly back into the strategy implementation.
[/INST]"""
            
            improvement_suggestions = ask_llama(improvement_prompt)
            
            if improvement_suggestions:
                improvement_context = f"""
**PREVIOUS ITERATION PERFORMANCE:**
- Previous Profit: ${current_profit:.2f}
- Target Profit: ${target_profit:.2f}
- Buy & Hold Profit: ${buy_hold_profit:.2f}

**IMPROVEMENT SUGGESTIONS:**
{improvement_suggestions}

Based on the above analysis, implement these improvements in your trading strategy to achieve the target profit of ${target_profit:.2f}.
"""
            
        iteration += 1
    
    return current_profit, buy_points, sell_points, balance_over_time

# -----------------------------------------------------------------------------
# Plotting helper (unchanged)
# -----------------------------------------------------------------------------

def plot_results(close, buy_points, sell_points, balance_over_time, counter=1, user_prompt=""):
    print("\nGenerating plot…")
    plt.figure(figsize=(18, 9), dpi=150)

    print("Plotting price data…")
    plt.subplot(2, 1, 1)
    plt.plot(range(len(close)), close, label='Close Price', linewidth=0.5)

    # Align scatter points directly with their recorded indices
    if buy_points:
        xs, ys = zip(*buy_points)
        plt.scatter(xs, ys, color='green', marker='o', s=10, label='Buy Points')
    if sell_points:
        xs, ys = zip(*sell_points)
        plt.scatter(xs, ys, color='red', marker='o', s=10, label='Sell Points')
    print(f"Plotted {len(buy_points)} buy points and {len(sell_points)} sell points")

    plt.title('Stock Close Price with Buy/Sell Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    print("Plotting balance data…")
    plt.subplot(2, 1, 2)
    
    # Create a list of balance points that only includes points after sales
    balance_after_sales = []
    sale_indices = [i for i, _ in sell_points]
    
    # Add initial balance
    balance_after_sales.append(balance_over_time[0])
    
    # Add balance points after each sale
    for i in range(1, len(balance_over_time)):
        if i-1 in sale_indices:  # If this point is after a sale
            balance_after_sales.append(balance_over_time[i])
    
    # Add final balance if there are held shares
    if len(balance_over_time) > len(balance_after_sales):
        balance_after_sales.append(balance_over_time[-1])
    
    plt.plot(balance_after_sales, label='Balance After Sales', color='blue', linewidth=2)
    plt.title('Balance After Sales')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.legend()

    plt.tight_layout()
    
    # Create a valid filename from the user prompt (remove invalid filename characters)
    safe_prompt = re.sub(r'[^\w\s-]', '', user_prompt)[:30]  # Limit to 30 chars
    safe_prompt = re.sub(r'\s+', '_', safe_prompt).strip('-_')
    
    filename = f'stockbt/test_images/test_image_{counter}_{safe_prompt}.png'
    print(f"Saving plot as '{filename}'…")
    plt.savefig(filename, dpi=150)
    print(f"Plot saved as '{filename}'")

# -----------------------------------------------------------------------------
# Prompt enhancement
# -----------------------------------------------------------------------------

def enhance_user_prompt(original_prompt):
    """Enhance the user's trading strategy prompt using LLM."""
    print("\n=== ENHANCING USER PROMPT ===")
    print(f"Original prompt: {original_prompt[:100]}...")
    
    enhancement_prompt = f"""[INST]
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
[/INST]"""

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

# -----------------------------------------------------------------------------
# Simple CLI entry-point for quick testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nWaiting for user input…")
    user_input = input("Enter your trading strategy: ")
    
    # Enhance the user's prompt
    enhanced_prompt = enhance_user_prompt(user_input)
    
    print("Starting simulation with iterative improvement…")
    print("\nUsing the following enhanced strategy:")
    print("-" * 40)
    print(enhanced_prompt)
    print("-" * 40)
    
    run_improved_simulation(enhanced_prompt)
    print("Simulation complete!") 