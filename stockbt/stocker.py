import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import llama_cpp._internals as _internals
from llama_cpp import Llama
import traceback
import re

print("Initializing stocker.py...")

# Create test_images folder if it doesn't exist
os.makedirs('stockbt/test_images', exist_ok=True)
print("Created test_images directory")

file_path = 'stockbt/datasets/test.csv'
print(f"Loading data from {file_path}")
data = pd.read_csv(file_path)
close = data['Close']
print(f"Loaded {len(close)} data points")

initial_balance = 100000
print(f"Initial balance set to ${initial_balance:,.2f}")

_internals.LlamaSampler.__del__ = lambda self: None
print("Patched Llama sampler")

def get_llama_model():
    """Get a shared Llama model instance"""
    print("Getting Llama model instance...")
    if not hasattr(get_llama_model, "instance") or get_llama_model.instance is None:
        print("No existing model instance found, initializing new one...")
        possible_model_paths = [
            "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "~/llama-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        ]

        model_path = None
        for path in possible_model_paths:
            print(f"Checking for model at: {path}")
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break

        if not model_path:
            print("ERROR: Could not find model file in any location")
            raise FileNotFoundError("Could not find model file")

        print("Initializing Llama model with settings:")
        print("- Context window: 4096")
        print("- Threads: 4")
        print("- GPU layers: 0")
        get_llama_model.instance = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            chat_format="mistral-instruct",
            verbose=False
        )
        print("Model initialized successfully")
    else:
        print("Using existing model instance")

    return get_llama_model.instance

def ask_llama(prompt):
    """Ask a question to the Llama model"""
    print("\nSending prompt to Llama model...")
    try:
        formatted_prompt = f"{prompt}"
        print("Formatted prompt:", formatted_prompt[:100] + "..." if len(formatted_prompt) > 100 else formatted_prompt)
        
        llm = get_llama_model()
        print("Generating response...")
        response = llm(formatted_prompt, max_tokens=16000, temperature=0.1)
        result = response["choices"][0]["text"].strip()
        print("Received response:" + result)
        return result
    except Exception as e:
        print(f"ERROR getting response: {e}")
        return None

def split_response(response):
    print("\nSplitting response into code blocks...")
    import ast

    # First, remove any markdown code fences
    cleaned_response = response
    if "```" in response:
        print("Removing markdown code fences...")
        # Remove ```python and ``` markers
        cleaned_response = re.sub(r'```(?:python)?\s*', '', cleaned_response)
        cleaned_response = cleaned_response.replace('```', '')
        print("Markdown fences removed")
    
    # Remove any explanations or other non-code text
    if "**Explanation:**" in cleaned_response:
        print("Removing explanations...")
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
        print("Warning: Less than 2 code blocks found, trying fallback extraction...")
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
    print("Successfully split response into code and input sections")

    def extract_function_name(code_block):
        try:
            parsed = ast.parse(code_block)
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            print(f"Error extracting function name: {e}")
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

def run_simulation(user_input):
    print(f"\nStarting simulation with user input: {user_input[:100]}...")

    # Initial prompt construction (remains the same)
    prompt = f"""[INST]
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions based on the user's strategy, following the strict rules below.

**User Strategy:** {user_input}

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
    *   Assume `math` and `pandas` (as `pd`) are pre-imported and available. Do **NOT** add import statements.
    *   Assume a pandas Series named `close` containing the price data is available in the execution scope of `trading_strategy`.

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
    final_price = close.iloc[-1] # Get the last closing price
    balance += shares * final_price
    print(f"Ending balance adjustment: Added value of {{shares}} held shares at ${{final_price:.2f}}")
balance_over_time.append(balance) # Append final balance state
return balance - initial_balance, buy_points, sell_points, balance_over_time
```
**(Note:** The MANDATORY TRADING LOGIC above is a template. You **must** integrate it correctly within the `trading_strategy` function's loop, defining variables like `i`, `price_today`, `buy_price`, `sell_price` according to the user's strategy and the parameters from `get_user_params`. The comments indicate where your strategy-specific logic needs to fit.)

**Final Check:** Ensure your output is only the two raw Python function definitions separated by a single blank line. No markdown, no comments outside the functions, no extra text.
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

    while not success:
        try:
            print(f"\n--- Attempt {attempts + 1} ---")
            current_prompt = error_prompt if attempts > 0 else prompt
            
            if not response or attempts > 0: # Get initial response or retry response
                print("Getting response from Llama...")
                response = ask_llama(current_prompt)
                if response is None:
                     raise Exception("LLM failed to provide a response.")
                
                print("Splitting response into parts...")
                parts = split_response(response)
                code = parts["code"]
                function_name_code = parts["function_name_code"]
                input_code = parts["input_code"]
                function_name_input_code = parts["function_name_input_code"]
                print(f"Extracted function names: {function_name_code} and {function_name_input_code}")

                # NEW: Validate that get_user_params does not attempt to prompt the user.
                if re.search(r'\binput\s*\(', input_code):
                    raise Exception("Detected forbidden user prompt (input()) in get_user_code. This function must run autonomously without requiring user interaction.")
            
            # Execute the potentially corrected code
            print("Executing input code...")
            exec(input_code, globals())
            print("Evaluating input function...")
            inp = eval(f'{function_name_input_code}()')
            print(f"Input function returned: {inp}")

            # Properly handle different parameter types for eval
            formatted_inp = repr(inp)

            print("Executing main code...")
            exec(code, globals())
            print("Evaluating main function...")
            bb, buy_points, sell_points, balance_over_time = eval(f'{function_name_code}({formatted_inp})')
            print(f"Main function returned: bb={bb}, {len(buy_points)} buy points, {len(sell_points)} sell points")

            success = True # If we reach here, it worked

        except Exception as e:
            attempts += 1
            print(f"\n--- Attempt {attempts} FAILED with error: ---")
            error_traceback = traceback.format_exc()
            print(error_traceback)
            print("-------------------------------------------")
            
            response = None # Clear previous response to force LLM call

            # Construct the error prompt for the next attempt
            error_prompt = f"""[INST]
**CRITICAL ERROR FIX REQUIRED**

The Python code you generated produced the following error:
```
{error_traceback}
```

**Your Task:**
Fix the error in the code by providing ONLY the two corrected functions. 

**Do NOT include explanatory text or markdown outside of the functions (concise in-function comments are allowed)**. Your response MUST consist of:
1. The fixed `trading_strategy` function
2. A single blank line
3. The fixed `get_user_params` function

**Common Errors to Check:**
- `trading_strategy` must accept EXACTLY ONE positional argument (remove any extras like `formatted_inp`).
- Don't call undefined functions (like 'update_end_price').
- **ABSOLUTELY CRITICAL: `get_user_params` MUST *NEVER* PROMPT THE USER FOR INPUT (e.g., using `input()` or similar functions, printing messages to the console that ask for input). ANY FORM OF USER INTERACTION IS STRICTLY FORBIDDEN.** Make sure it returns valid numeric parameters.
- Use relative multipliers/scales for price calculations—do **NOT** hard-code absolute price thresholds.
- Ensure all variables are properly defined before use.

**User Strategy:** {user_input}

**REMEMBER:** Output ONLY clean code - no explanations, no markdown, no extra text.
[/INST]"""
            print(f"Attempting retry {attempts} to fix the error...")

    if not success:
        print("Simulation failed after all retry attempts.")
        return None, [], [], [] # Return empty results on failure

    # Proceed with plotting if successful
    print("\nExecution successful. Generating plot...")
    plot_results(close, buy_points, sell_points, balance_over_time)

    return bb, buy_points, sell_points, balance_over_time

def plot_results(close, buy_points, sell_points, balance_over_time):
    print("\nGenerating plot...")
    plt.figure(figsize=(18, 9), dpi=150)

    print("Plotting price data...")
    plt.subplot(2, 1, 1)
    plt.plot(close, label='Close Price', linewidth=0.5)

    scaled_buy_points = [(i*19-1, price) for i, price in buy_points]
    scaled_sell_points = [(i*19, price) for i, price in sell_points]
    print(f"Scaled {len(scaled_buy_points)} buy points and {len(scaled_sell_points)} sell points")

    if scaled_buy_points:
        plt.scatter(*zip(*scaled_buy_points), color='green', marker='o', s=7, label='Buy Points')
    if scaled_sell_points:
        plt.scatter(*zip(*scaled_sell_points), color='red', marker='o', s=7, label='Sell Points')

    plt.title('Stock Close Price with Buy/Sell Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    print("Plotting balance data...")
    plt.subplot(2, 1, 2)
    plt.plot(balance_over_time, label='Balance Over Time', color='blue', linewidth=2)
    plt.title('Balance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.legend()

    plt.tight_layout()
    print("Saving plot...")
    plt.savefig('stockbt/test_images/balance_over_time.png', dpi=150)
    print("Plot saved as 'stockbt/test_images/balance_over_time.png'")

print("\nWaiting for user input...")
user_input = input("Enter your trading strategy: ")
print("Starting simulation...")
run_simulation(user_input)
print("Simulation complete!")
