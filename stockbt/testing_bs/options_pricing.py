import math
import csv
import matplotlib.pyplot as plt
import os
import traceback
import re
import numpy as np
import time
import random
from scipy.stats import norm

# Black-Scholes formulas for option pricing
def black_scholes_call(S, K, T, r, sigma):
    """
    Price a European call option using the Black-Scholes formula.
    S: current stock price
    K: strike price
    T: time to expiration (in years)
    r: risk-free interest rate (annualized, decimal)
    sigma: volatility of the underlying asset (annualized, decimal)
    Returns: call option price
    """
    if T <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """
    Price a European put option using the Black-Scholes formula.
    S: current stock price
    K: strike price
    T: time to expiration (in years)
    r: risk-free interest rate (annualized, decimal)
    sigma: volatility of the underlying asset (annualized, decimal)
    Returns: put option price
    """
    if T <= 0:
        return max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

# Define Greeks calculation functions
def delta_call(S, K, T, r, sigma):
    """Calculate delta of a call option"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    """Calculate delta of a put option"""
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    """Calculate gamma (same for calls and puts)"""
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def theta_call(S, K, T, r, sigma):
    """Calculate theta of a call option"""
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)

def theta_put(S, K, T, r, sigma):
    """Calculate theta of a put option"""
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)

def vega(S, K, T, r, sigma):
    """Calculate vega (same for calls and puts)"""
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm.pdf(d1)

# Set the initial balance for all simulations
initial_balance = 100000

# Track previously used datasets in this session
used_datasets = set()

def get_random_dataset():
    """Get a random stock price dataset from the datasets directory"""
    datasets_dir = 'stockbt/datasets'
    all_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    unused_files = [f for f in all_files if f not in used_datasets]
    if not unused_files:
        used_datasets.clear()
        unused_files = all_files
    chosen = random.choice(unused_files)
    used_datasets.add(chosen)
    return os.path.join(datasets_dir, chosen)

def load_stock_data(file_path):
    """Load stock price data from a CSV file"""
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
    return close, dates

def ask_llama(prompt, temperature=0.7):
    print(f"\nSending prompt to g4f ChatCompletion with temperature={temperature}…")
    import g4f
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-o3",
            messages=[{"role": "user", "content": prompt}],
            provider=g4f.Provider.PollinationsAI,
            temperature=temperature
        )
        print("Received response:" + response)
        return response.strip()
    except Exception as e:
        print(f"ERROR getting response: {e}")
        return None

def split_response(response):
    print("\nSplitting response into code blocks…")
    import ast
    if response:
        print(f"Response length: {len(response)} chars")
        excerpt_len = min(200, len(response))
        print(f"Response excerpt: {response[:excerpt_len]}...")
    else:
        print("WARNING: Empty response received")
        raise ValueError("Empty response from LLM")
    cleaned_response = response
    if "```" in response:
        print("Removing markdown code fences…")
        cleaned_response = re.sub(r'```(?:python)?\s*', '', cleaned_response)
        cleaned_response = cleaned_response.replace('```', '')
        print("Markdown fences removed")
    if "**Explanation:**" in cleaned_response:
        print("Removing explanations…")
        cleaned_response = re.sub(r'\*\*Explanation:\*\*.*?(?=def |$)', '', cleaned_response, flags=re.DOTALL)
    cleaned_response = re.sub(r'\*\*.*?\*\*', '', cleaned_response)
    cleaned_response = re.sub(r'Function \d+ \(`.*?`\):', '', cleaned_response)
    cleaned_response = re.sub(r'^\s*(\*|\-|\d+\.)\s*', '', cleaned_response, flags=re.MULTILINE)
    if "def " not in cleaned_response:
        print("ERROR: No function definitions found in response")
        raise ValueError("No function definitions found in response.")
    code_blocks = re.findall(r"(def [\s\S]+?)(?=\ndef |\Z)", cleaned_response)
    print(f"Found {len(code_blocks)} code blocks")
    if len(code_blocks) < 2:
        print("Warning: Less than 2 code blocks found, trying fallback extraction…")
        parts = cleaned_response.split('def ')
        code_blocks = []
        for part in parts[1:]:
            code_blocks.append('def ' + part.strip())
        print(f"Fallback extraction found {len(code_blocks)} blocks")
    strategy_blocks = [block for block in code_blocks if re.search(r"def\s+options_strategy\s*\(", block)]
    param_blocks = [block for block in code_blocks if re.search(r"def\s+get_user_params\s*\(", block)]
    if not strategy_blocks or not param_blocks:
        # fallback: just take first two
        strategy_blocks = code_blocks[:1]
        param_blocks = code_blocks[1:2]
    code = strategy_blocks[0].strip()
    input_code = param_blocks[0].strip()
    def extract_function_name(code_block):
        try:
            parsed = ast.parse(code_block)
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            print(f"AST parsing failed: {e}. Falling back to regex.")
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

def calculate_options_pnl(underlying_prices, option_type, strike, expiry_days, risk_free_rate, volatility, position='long'):
    """Calculate profit/loss for an options strategy based on underlying price movement"""
    pnl = []
    days_to_expiry = []
    option_prices = []
    
    for i, price in enumerate(underlying_prices):
        # Calculate time to expiry in years
        days_remaining = max(0, expiry_days - i)
        T = days_remaining / 365.0
        days_to_expiry.append(days_remaining)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = black_scholes_call(price, strike, T, risk_free_rate, volatility)
        else:  # put
            price = black_scholes_put(price, strike, T, risk_free_rate, volatility)
        
        option_prices.append(price)
        
        # Initial option price (entry price)
        if i == 0:
            initial_price = price
        
        # Calculate P&L based on position
        if position.lower() == 'long':
            current_pnl = price - initial_price
        else:  # short
            current_pnl = initial_price - price
        
        pnl.append(current_pnl)
    
    return pnl, option_prices, days_to_expiry

def plot_options_results(underlying_prices, option_prices, buy_points, sell_points, 
                        portfolio_value_history, days_to_expiry, params):
    """Generate enhanced plots for options trading results"""
    print("\nGenerating options trading plots…")
    os.makedirs('stockbt/test_images', exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Calculate additional metrics
    if params['option_type'].lower() == 'call':
        deltas = [delta_call(underlying_prices[i], 
                           underlying_prices[0] * params['strike_pct'], 
                           max(0, days_to_expiry[i]/365), 
                           params['risk_free_rate'], 
                           params['volatility']) for i in range(len(underlying_prices))]
        thetas = [theta_call(underlying_prices[i], 
                           underlying_prices[0] * params['strike_pct'], 
                           max(0.01, days_to_expiry[i]/365), 
                           params['risk_free_rate'], 
                           params['volatility'])/365 for i in range(len(underlying_prices))]
    else:
        deltas = [delta_put(underlying_prices[i], 
                          underlying_prices[0] * params['strike_pct'], 
                          max(0, days_to_expiry[i]/365), 
                          params['risk_free_rate'], 
                          params['volatility']) for i in range(len(underlying_prices))]
        thetas = [theta_put(underlying_prices[i], 
                          underlying_prices[0] * params['strike_pct'], 
                          max(0.01, days_to_expiry[i]/365), 
                          params['risk_free_rate'], 
                          params['volatility'])/365 for i in range(len(underlying_prices))]
    
    gammas = [gamma(underlying_prices[i], 
                   underlying_prices[0] * params['strike_pct'], 
                   max(0.01, days_to_expiry[i]/365), 
                   params['risk_free_rate'], 
                   params['volatility']) for i in range(len(underlying_prices))]
    
    vegas = [vega(underlying_prices[i], 
                 underlying_prices[0] * params['strike_pct'], 
                 max(0.01, days_to_expiry[i]/365), 
                 params['risk_free_rate'], 
                 params['volatility']) for i in range(len(underlying_prices))]
    
    # Calculate returns for both underlying and option
    underlying_returns = [0]
    option_returns = [0]
    for i in range(1, len(underlying_prices)):
        if underlying_prices[i-1] > 0:
            underlying_returns.append((underlying_prices[i] - underlying_prices[i-1]) / underlying_prices[i-1] * 100)
        else:
            underlying_returns.append(0)
            
        if option_prices[i-1] > 0:
            option_returns.append((option_prices[i] - option_prices[i-1]) / option_prices[i-1] * 100)
        else:
            option_returns.append(0)
    
    # Create subplots with gridspec for better control over sizes
    fig = plt.figure(figsize=(14, 18), dpi=150)
    gs = fig.add_gridspec(5, 2, height_ratios=[2, 2, 1.5, 1.5, 1.5])
    
    # 1. Price Charts (Top Row)
    ax1 = fig.add_subplot(gs[0, 0:2])  # Spans both columns
    
    # Plot underlying price and option price on the same chart with different y-axes
    color1 = 'blue'
    ax1.plot(underlying_prices, color=color1, linewidth=1.2, label='Underlying Price')
    ax1.set_ylabel('Underlying Price ($)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title('Underlying Asset vs Option Price', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add option price on secondary y-axis
    ax1b = ax1.twinx()
    color2 = 'purple'
    ax1b.plot(option_prices, color=color2, linewidth=1.2, label=f"{params['option_type'].capitalize()} Option Price")
    ax1b.set_ylabel('Option Price ($)', color=color2)
    ax1b.tick_params(axis='y', labelcolor=color2)
    
    # Add buy/sell points
    buy_indices = [i for i, _ in buy_points]
    buy_prices = [p for _, p in buy_points]
    sell_indices = [i for i, _ in sell_points]
    sell_prices = [p for _, p in sell_points]
    
    ax1b.scatter(buy_indices, buy_prices, color='green', marker='^', s=100, label='Buy')
    ax1b.scatter(sell_indices, sell_prices, color='red', marker='v', s=100, label='Sell')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add strike price line
    strike_price = underlying_prices[0] * params['strike_pct']
    ax1.axhline(y=strike_price, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.text(len(underlying_prices)*0.02, strike_price*1.02, f"Strike: ${strike_price:.2f}", 
             color='black', alpha=0.7, fontsize=9)
    
    # Highlight the region where option is in-the-money
    if params['option_type'].lower() == 'call':
        for i in range(len(underlying_prices)):
            if underlying_prices[i] > strike_price:
                ax1.axvspan(i, len(underlying_prices), alpha=0.1, color='green')
                break
    else:  # put
        for i in range(len(underlying_prices)):
            if underlying_prices[i] < strike_price:
                ax1.axvspan(i, len(underlying_prices), alpha=0.1, color='green')
                break
    
    # 2. Returns Comparison
    ax2 = fig.add_subplot(gs[1, 0:2])  # Spans both columns
    ax2.plot(underlying_returns, color='blue', alpha=0.7, linewidth=0.8, label='Underlying Returns')
    ax2.plot(option_returns, color='purple', linewidth=1.0, label='Option Returns')
    ax2.set_title('Daily Returns Comparison', fontweight='bold')
    ax2.set_ylabel('Daily Return (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotations for significant return days
    # Find top 3 option return days
    if len(option_returns) > 5:
        top_returns_indices = sorted(range(len(option_returns)), key=lambda i: abs(option_returns[i]), reverse=True)[:3]
        for idx in top_returns_indices:
            if abs(option_returns[idx]) > 10:  # Only annotate significant moves
                ax2.annotate(f"{option_returns[idx]:.1f}%", 
                            xy=(idx, option_returns[idx]),
                            xytext=(idx, option_returns[idx] * (1.1 if option_returns[idx] > 0 else 0.9)),
                            fontsize=8, color='purple')
    
    # 3. Option Greeks
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(deltas, color='blue', linewidth=1.2, label='Delta')
    ax3.set_title('Option Delta (Price Sensitivity)', fontweight='bold')
    ax3.set_ylabel('Delta')
    ax3.grid(True, alpha=0.3)
    # Add horizontal lines at 0, 0.5, and 1 or -1 for reference
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)
    if params['option_type'].lower() == 'call':
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.2)
    else:
        ax3.axhline(y=-1, color='black', linestyle='--', alpha=0.2)
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(gammas, color='green', linewidth=1.2, label='Gamma')
    ax4.set_title('Option Gamma (Delta\'s Rate of Change)', fontweight='bold')
    ax4.set_ylabel('Gamma')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(thetas, color='red', linewidth=1.2, label='Theta')
    ax5.set_title('Option Theta (Time Decay)', fontweight='bold')
    ax5.set_ylabel('Theta ($/day)')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(vegas, color='purple', linewidth=1.2, label='Vega')
    ax6.set_title('Option Vega (Volatility Sensitivity)', fontweight='bold')
    ax6.set_ylabel('Vega')
    ax6.grid(True, alpha=0.3)
    
    # 5. Portfolio Value
    ax8 = fig.add_subplot(gs[4, 0:2])  # Spans both columns
    ax8.plot(portfolio_value_history, color='green', linewidth=1.5)
    
    # Calculate and add a reference line for the initial investment
    initial_value = portfolio_value_history[0]
    ax8.axhline(y=initial_value, color='red', linestyle='--', alpha=0.7)
    ax8.text(len(portfolio_value_history)*0.02, initial_value*1.02, f"Initial: ${initial_value:,.0f}", 
             color='red', alpha=0.7)
    
    # Add final value annotation
    final_value = portfolio_value_history[-1]
    profit_pct = ((final_value - initial_value) / initial_value) * 100
    
    ax8.text(len(portfolio_value_history)*0.6, final_value*0.9, 
             f"Final: ${final_value:,.0f} ({profit_pct:+.1f}%)", 
             color='green' if profit_pct > 0 else 'red',
             fontsize=10, fontweight='bold')
    
    ax8.set_title('Portfolio Value Over Time', fontweight='bold')
    ax8.set_ylabel('Value ($)')
    ax8.grid(True, alpha=0.3)
    ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Improve overall appearance
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    
    # Add a title at the top
    plt.figtext(0.5, 0.97, f"{params['option_type'].upper()} OPTION BACKTEST - Strike: {params['strike_pct']*100:.0f}% of initial, {params['position'].upper()} position", 
                ha='center', fontsize=14, fontweight='bold')
    
    # Save the enhanced chart
    filename = f'stockbt/test_images/options_backtest_{params["option_type"]}_{int(params["strike_pct"]*100)}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Enhanced plot saved as '{filename}'")
    plt.close()

def explain_options_results(underlying_prices, option_prices, buy_points, sell_points, portfolio_value_history, params):
    """Explain options trading results in clear, accessible terms with enhanced data"""
    print("\n" + "="*80)
    print(f"🔍 OPTIONS TRADING ANALYSIS - {params['option_type'].upper()} {params['position'].upper()}")
    print("="*80)
    
    # Find the market movement pattern
    first_price = underlying_prices[0]
    last_price = underlying_prices[-1]
    price_change_pct = ((last_price - first_price) / first_price) * 100
    
    # Get buy and sell information
    buy_day = buy_points[0][0] if buy_points else 0
    buy_price = buy_points[0][1] if buy_points else 0
    sell_day = sell_points[0][0] if sell_points else len(underlying_prices) - 1
    sell_price = sell_points[0][1] if sell_points else 0
    
    # Calculate profit/loss
    initial_value = portfolio_value_history[0]
    final_value = portfolio_value_history[-1]
    profit_loss = final_value - initial_value
    profit_loss_pct = (profit_loss / initial_value) * 100
    
    # Calculate additional metrics
    max_drawdown_pct = 0
    max_portfolio_value = initial_value
    for value in portfolio_value_history:
        max_portfolio_value = max(max_portfolio_value, value)
        drawdown = (max_portfolio_value - value) / max_portfolio_value * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
    
    # Calculate annualized return
    trading_days = len(underlying_prices)
    years = trading_days / 252  # Assuming 252 trading days per year
    annualized_return = ((1 + profit_loss_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # SECTION 1: SUMMARY STATS
    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"{'Initial Investment:':<25} ${initial_value:,.2f}")
    print(f"{'Final Portfolio Value:':<25} ${final_value:,.2f}")
    print(f"{'Total Profit/Loss:':<25} ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)")
    print(f"{'Annualized Return:':<25} {annualized_return:+.2f}% per year")
    print(f"{'Maximum Drawdown:':<25} {max_drawdown_pct:.2f}%")
    print(f"{'Trading Period:':<25} {trading_days} days ({years:.1f} years)")
    
    # SECTION 2: MARKET CONDITIONS
    print("\n📈 MARKET CONDITIONS:")
    print(f"{'Starting Stock Price:':<25} ${first_price:.2f}")
    print(f"{'Ending Stock Price:':<25} ${last_price:.2f}")
    print(f"{'Stock Price Change:':<25} {price_change_pct:+.2f}%")
    
    volatility_realized = np.std([np.log(underlying_prices[i+1]/underlying_prices[i]) for i in range(len(underlying_prices)-1)]) * np.sqrt(252) * 100
    print(f"{'Realized Volatility:':<25} {volatility_realized:.2f}%")
    
    # SECTION 3: OPTION DETAILS
    print("\n🔢 OPTION STRATEGY DETAILS:")
    print(f"{'Option Type:':<25} {params['option_type'].capitalize()}")
    print(f"{'Position:':<25} {params['position'].capitalize()}")
    print(f"{'Strike Price:':<25} ${first_price * params['strike_pct']:.2f} ({params['strike_pct']*100:.0f}% of initial)")
    
    in_the_money_days = 0
    if params['option_type'].lower() == 'call':
        for price in underlying_prices:
            if price > first_price * params['strike_pct']:
                in_the_money_days += 1
        itm_pct = (in_the_money_days / len(underlying_prices)) * 100
        print(f"{'In-the-money:':<25} {in_the_money_days} days ({itm_pct:.1f}% of time)")
    else:  # put
        for price in underlying_prices:
            if price < first_price * params['strike_pct']:
                in_the_money_days += 1
        itm_pct = (in_the_money_days / len(underlying_prices)) * 100
        print(f"{'In-the-money:':<25} {in_the_money_days} days ({itm_pct:.1f}% of time)")
    
    # SECTION 4: TRADE ANALYSIS
    print("\n🛒 TRADE ANALYSIS:")
    
    # Calculate leverage factor
    option_change_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
    leverage_factor = option_change_pct / price_change_pct if price_change_pct != 0 else 0
    
    print(f"{'Entry Day:':<25} Day {buy_day}")
    print(f"{'Exit Day:':<25} Day {sell_day}")
    print(f"{'Holding Period:':<25} {sell_day - buy_day} days")
    print(f"{'Option Entry Price:':<25} ${buy_price:.2f}")
    print(f"{'Option Exit Price:':<25} ${sell_price:.2f}")
    print(f"{'Option Price Change:':<25} {option_change_pct:+.2f}%")
    print(f"{'Leverage Factor:':<25} {abs(leverage_factor):.2f}x underlying")
    
    # SECTION 5: KEY INSIGHTS 
    print("\n💡 KEY INSIGHTS:")
    if option_change_pct > 0:
        if leverage_factor > 5:
            print("✓ High leverage multiplied your returns significantly")
        if in_the_money_days > trading_days * 0.7:
            print("✓ Option spent most of its time in-the-money, increasing its value")
        if price_change_pct > 20:
            print("✓ Strong trend in the underlying stock worked in your favor")
    else:
        if price_change_pct * (1 if params['option_type'].lower() == 'call' else -1) < 0:
            print("✗ Underlying stock moved against your position")
        if in_the_money_days < trading_days * 0.2:
            print("✗ Option spent little time in-the-money, limiting its value")
        if abs(leverage_factor) < 1:
            print("✗ Poor leverage - option underperformed the underlying")
    
    # Time decay effect
    if buy_day + params['expiry_days'] > sell_day:
        time_remaining = params['expiry_days'] - (sell_day - buy_day)
        time_pct = (time_remaining / params['expiry_days']) * 100
        print(f"✓ Sold with {time_remaining} days ({time_pct:.1f}%) of time value remaining")
    else:
        print("✗ Held until or past expiration, losing all time value")
    
    print("\n" + "="*80)

def plot_simplified_options_results(underlying_prices, option_prices, buy_points, sell_points, 
                               portfolio_value_history, days_to_expiry, params):
    """Generate a single, clear graph that shows all the key options trading data"""
    print("\nGenerating simplified options trading visualization…")
    os.makedirs('stockbt/test_images', exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create figure
    plt.figure(figsize=(14, 8), dpi=150)
    
    # Create primary axis for stock price
    ax1 = plt.gca()
    
    # Plot underlying stock price
    ax1.plot(underlying_prices, color='blue', alpha=0.7, linewidth=1.5, label='Underlying Price')
    
    # Calculate strike price
    strike_price = underlying_prices[0] * params['strike_pct']
    
    # Add strike price line
    ax1.axhline(y=strike_price, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
    ax1.text(len(underlying_prices)*0.02, strike_price*1.02, 
             f"Strike: ${strike_price:.2f}", color='black', alpha=0.7, fontsize=10)
    
    # Create secondary axis for option price
    ax2 = ax1.twinx()
    
    # Plot option price
    ax2.plot(option_prices, color='purple', linewidth=1.5, label=f"{params['option_type'].capitalize()} Option")
    
    # Extract buy/sell points
    buy_indices = [i for i, _ in buy_points]
    buy_prices = [p for _, p in buy_points]
    sell_indices = [i for i, _ in sell_points]
    sell_prices = [p for _, p in sell_points]
    
    # Add buy points as green triangles
    for i, (idx, price) in enumerate(buy_points):
        # Draw vertical line from strike price to buy point
        ax1.plot([idx, idx], [strike_price, underlying_prices[idx]], 
                color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Draw buy marker on the stock price line
        ax1.scatter(idx, underlying_prices[idx], color='green', marker='^', s=100, 
                   label='Buy' if i == 0 else "")
        
        # Draw buy marker on the option price line
        ax2.scatter(idx, price, color='green', marker='^', s=100, edgecolor='black')
        
        # Add annotation for option price
        ax2.annotate(f"Buy: ${price:.2f}", 
                    xy=(idx, price),
                    xytext=(idx + 30, price * 1.1),
                    arrowprops=dict(arrowstyle="->", color='green'),
                    color='green', fontweight='bold')
    
    # Add sell points as red triangles
    for i, (idx, price) in enumerate(sell_points):
        # Draw vertical line from strike price to sell point
        ax1.plot([idx, idx], [strike_price, underlying_prices[idx]], 
                color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Draw sell marker on the stock price line
        ax1.scatter(idx, underlying_prices[idx], color='red', marker='v', s=100, 
                   label='Sell' if i == 0 else "")
        
        # Draw sell marker on the option price line
        ax2.scatter(idx, price, color='red', marker='v', s=100, edgecolor='black')
        
        # Add annotation for option price
        ax2.annotate(f"Sell: ${price:.2f}", 
                    xy=(idx, price),
                    xytext=(idx + 30, price * 0.9),
                    arrowprops=dict(arrowstyle="->", color='red'),
                    color='red', fontweight='bold')
    
    # If we have at least one buy and one sell, show the price difference
    if buy_points and sell_points:
        # Connect the buy and sell option prices with an arrow
        buy_idx, buy_price = buy_points[0]
        sell_idx, sell_price = sell_points[0]
        
        # Vertical span showing option holding period
        ax1.axvspan(buy_idx, sell_idx, alpha=0.1, color='blue')
        
        # Draw curved arrow connecting entry to exit prices
        mid_x = (buy_idx + sell_idx) / 2
        if sell_price > buy_price:
            # Profit arrow (green)
            profit = sell_price - buy_price
            profit_pct = (profit / buy_price) * 100
            ax2.annotate(f"+${profit:.2f} (+{profit_pct:.1f}%)", 
                       xy=(sell_idx, sell_price),
                       xytext=(mid_x, max(buy_price, sell_price) * 1.1),
                       arrowprops=dict(arrowstyle="fancy", color='green', connectionstyle="arc3,rad=.3"),
                       color='green', fontweight='bold', fontsize=11)
        else:
            # Loss arrow (red)
            loss = buy_price - sell_price
            loss_pct = (loss / buy_price) * 100
            ax2.annotate(f"-${loss:.2f} (-{loss_pct:.1f}%)", 
                       xy=(sell_idx, sell_price),
                       xytext=(mid_x, max(buy_price, sell_price) * 1.1),
                       arrowprops=dict(arrowstyle="fancy", color='red', connectionstyle="arc3,rad=.3"),
                       color='red', fontweight='bold', fontsize=11)
    
    # Highlight in-the-money regions
    if params['option_type'].lower() == 'call':
        # For calls, highlight where stock price > strike price
        above_strike = False
        for i in range(len(underlying_prices)):
            if not above_strike and underlying_prices[i] > strike_price:
                # Start of in-the-money region
                ax1.axvspan(i, i+1, alpha=0.1, color='green')
                above_strike = True
            elif above_strike and underlying_prices[i] <= strike_price:
                # End of in-the-money region
                above_strike = False
            elif above_strike:
                # Continue in-the-money region
                ax1.axvspan(i, i+1, alpha=0.1, color='green')
    else:
        # For puts, highlight where stock price < strike price
        below_strike = False
        for i in range(len(underlying_prices)):
            if not below_strike and underlying_prices[i] < strike_price:
                # Start of in-the-money region
                ax1.axvspan(i, i+1, alpha=0.1, color='green')
                below_strike = True
            elif below_strike and underlying_prices[i] >= strike_price:
                # End of in-the-money region
                below_strike = False
            elif below_strike:
                # Continue in-the-money region
                ax1.axvspan(i, i+1, alpha=0.1, color='green')
    
    # Add portfolio value as area chart at the bottom
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.08))  # Offset the third y-axis
    portfolio_scaled = [v / max(portfolio_value_history) * max(underlying_prices) * 0.3 for v in portfolio_value_history]
    ax3.fill_between(range(len(portfolio_scaled)), 0, portfolio_scaled, alpha=0.2, color='green')
    ax3.plot(portfolio_scaled, color='green', alpha=0.6, linewidth=1.0, label='Portfolio Value')
    ax3.set_yticks([])  # Hide y-axis for portfolio value
    
    # Add a portfolio value annotation at the right side
    final_value = portfolio_value_history[-1]
    initial_value = portfolio_value_history[0]
    profit_pct = ((final_value - initial_value) / initial_value) * 100
    profit_text = f"Portfolio: ${final_value:,.0f} ({profit_pct:+.1f}%)"
    ax3.text(len(portfolio_value_history) * 0.99, portfolio_scaled[-1], 
            profit_text, color='green' if profit_pct > 0 else 'red', 
            horizontalalignment='right', fontsize=10, fontweight='bold')
    
    # Add option type and position information
    title = f"{params['option_type'].upper()} OPTION ({params['position'].upper()}) - Strike: ${strike_price:.2f}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Labels and legends
    ax1.set_xlabel('Trading Days', fontsize=11)
    ax1.set_ylabel('Underlying Price ($)', color='blue', fontsize=11)
    ax2.set_ylabel('Option Price ($)', color='purple', fontsize=11)
    
    # Set axis colors
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
              loc='upper left', fontsize=10)
    
    # Improve grid and layout
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f'stockbt/test_images/options_single_view_{params["option_type"]}_{int(params["strike_pct"]*100)}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Simplified visualization saved as '{filename}'")
    plt.close()

def plot_storyboard_option_trade(underlying_prices, option_prices, buy_points, sell_points, portfolio_value_history, days_to_expiry, params):
    """A single, creative, story-style visualization for options trades"""
    import matplotlib.patches as patches
    import matplotlib as mpl
    print("\nGenerating creative storyboard visualization…")
    os.makedirs('stockbt/test_images', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Setup
    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)
    
    # Plot underlying price
    ax.plot(underlying_prices, color='#3498db', linewidth=3, label='Stock Price')
    
    # Overlay option price as a thick, semi-transparent line
    ax.plot(option_prices, color='#9b59b6', linewidth=6, alpha=0.3, label=f"{params['option_type'].capitalize()} Option Value")
    
    # Strike price
    strike_price = underlying_prices[0] * params['strike_pct']
    ax.axhline(strike_price, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(5, strike_price*1.05, f'Strike: ${strike_price:.2f}', fontsize=14, color='black', alpha=0.7, fontweight='bold')
    
    # Buy/Sell points
    if buy_points:
        buy_idx, buy_price = buy_points[0]
        ax.scatter(buy_idx, underlying_prices[buy_idx], s=400, marker='^', color='lime', edgecolor='black', zorder=10)
        ax.annotate('BOUGHT HERE! 💸', (buy_idx, underlying_prices[buy_idx]),
                    xytext=(buy_idx+30, underlying_prices[buy_idx]*1.1),
                    fontsize=18, color='green', fontweight='bold', arrowprops=dict(arrowstyle='->', color='green', lw=3))
    if sell_points:
        sell_idx, sell_price = sell_points[0]
        ax.scatter(sell_idx, underlying_prices[sell_idx], s=400, marker='v', color='red', edgecolor='black', zorder=10)
        ax.annotate('SOLD HERE! 🤑', (sell_idx, underlying_prices[sell_idx]),
                    xytext=(sell_idx+30, underlying_prices[sell_idx]*0.9),
                    fontsize=18, color='red', fontweight='bold', arrowprops=dict(arrowstyle='->', color='red', lw=3))
    
    # Draw a big arrow for profit/loss
    if buy_points and sell_points:
        buy_idx, buy_price = buy_points[0]
        sell_idx, sell_price = sell_points[0]
        profit = sell_price - buy_price
        profit_pct = (profit / buy_price) * 100 if buy_price else 0
        color = 'green' if profit > 0 else 'red'
        face = '😃' if profit > 0 else '😢'
        summary = f"You {'MADE' if profit > 0 else 'LOST'} ${abs(profit):.2f} ({profit_pct:+.1f}%) {face}"
        # Arrow
        ax.annotate('', xy=(sell_idx, underlying_prices[sell_idx]), xytext=(buy_idx, underlying_prices[buy_idx]),
                    arrowprops=dict(arrowstyle='-|>', lw=8, color=color, alpha=0.5))
        # Big summary box
        ax.text((buy_idx+sell_idx)/2, max(underlying_prices)*0.95, summary,
                fontsize=28, color=color, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.7'))
    
    # Comic-strip style story at the bottom
    story_y = min(underlying_prices) - (max(underlying_prices)-min(underlying_prices))*0.15
    story = []
    story.append(f"Day 0: Stock at ${underlying_prices[0]:.2f}")
    if buy_points:
        story.append(f"You bought a {params['option_type'].upper()} option for ${buy_price:.2f}")
    if sell_points:
        story.append(f"You sold it for ${sell_price:.2f}")
    if buy_points and sell_points:
        if profit > 0:
            story.append(f"You made money! 🎉")
        else:
            story.append(f"You lost money. 😢")
    else:
        story.append("You held the option to expiry.")
    # Draw the story
    for i, line in enumerate(story):
        ax.text(len(underlying_prices)*0.01, story_y - i*0.04*max(underlying_prices), line, fontsize=18, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Remove all spines except bottom and left
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Labels
    ax.set_xlabel('Trading Days', fontsize=16, fontweight='bold')
    ax.set_ylabel('Stock Price ($)', fontsize=16, fontweight='bold')
    
    # Remove y2 axis, legend, and grid for simplicity
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
    ax.grid(False)
    ax.legend().set_visible(False)
    
    # Save
    filename = f'stockbt/test_images/options_storyboard_{params["option_type"]}_{int(params["strike_pct"]*100)}_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    print(f"Storyboard visualization saved as '{filename}'")
    plt.close()

def run_llm_options_simulation(user_input, max_error_attempts=5):
    print(f"\nStarting LLM-driven options simulation with user input: {user_input[:100]}…")
    # Load data
    file_path = get_random_dataset()
    underlying_prices, dates = load_stock_data(file_path)
    days_to_expiry = [max(0, 30 - i) for i in range(len(underlying_prices))]  # Example, can be improved

    # Compose initial LLM prompt
    prompt = f'''
**Your Role:** You are a specialized Python code generation assistant. Your sole task is to generate *exactly* two Python functions for options trading backtesting.

**User Strategy:** {user_input}

**DATA INFORMATION:**
- The data is loaded from a CSV file containing historical stock prices
- Only the Close prices are available as a simple Python list named 'underlying_prices'
- No DataFrame or pandas is used - data is just a plain Python list of float values
- You DO NOT need to load the data yourself - it's already available as 'underlying_prices'

**OPTIONS INFORMATION:**
- Black-Scholes functions are available for pricing: black_scholes_call and black_scholes_put

**CODE TEMPLATE GUIDELINES:**
1. Create an options_strategy function that accepts params dict
2. Iterate over underlying_prices: for i, price_today in enumerate(underlying_prices)
3. Create buy_points and sell_points lists for option trades
4. Return (profit_loss, buy_points, sell_points, portfolio_value_history)
5. Use params.get('key', default) to safely access parameters
6. NO input() calls or user prompting in any function
7. Use try/except blocks to handle potential errors

**Critical Output Requirements:**
- Your entire response MUST be ONLY the Python code for the two functions.
- NO markdown, explanations, or extra text.
- Separate the two functions with exactly ONE blank line.

**Function 1: options_strategy**
- Must be named exactly options_strategy.
- Must accept exactly one positional argument (params), which will be the output from get_user_params.
- Must implement the user's options trading strategy using the provided params.
- Must use black_scholes_call and black_scholes_put for option pricing.
- Must return exactly profit_loss, buy_points, sell_points, portfolio_value_history in that order.

**Function 2: get_user_params**
- Must be named exactly get_user_params.
- Must NEVER prompt the user for input.
- Must return the chosen parameters as a dictionary.
- Include realistic market execution parameters (slippage_pct, option_commission, etc).

**Environment:**
- Assume math module is pre-imported and available.
- Assume a Python list named underlying_prices containing the price data is available in the execution scope of options_strategy.

**REMEMBER:** Output ONLY the complete code for both functions - no explanations, no markdown, no extra text.
'''

    attempts = 0
    response = None
    error_prompt = ""
    while attempts < max_error_attempts:
        try:
            current_prompt = error_prompt if attempts > 0 else prompt
            response = ask_llama(current_prompt)
            parts = split_response(response)
            code = parts['code']
            input_code = parts['input_code']
            function_name_code = parts['function_name_code']
            function_name_input_code = parts['function_name_input_code']

            exec(input_code, globals())
            params = eval(f'{function_name_input_code}()')
            # --- Robust parameter normalization (from st.py style) ---
            # If params is a tuple or list, convert to dict
            if isinstance(params, (tuple, list)):
                params = {i: val for i, val in enumerate(params)}
            # If not a dict, force to dict
            if not isinstance(params, dict):
                params = {'param': params}
            # Guarantee .get method
            if not hasattr(params, 'get'):
                params = dict(params)
            # Add underlying_prices to params for LLM code that expects it
            params.setdefault('underlying_prices', underlying_prices)
            # Patch missing keys for compatibility
            params.setdefault('option_type', 'call')
            params.setdefault('position', 'long')
            # Accept both strike_pct and strike_offset_pct
            if 'strike_pct' not in params and 'strike_offset_pct' in params:
                params['strike_pct'] = params['strike_offset_pct']
            if 'strike_offset_pct' not in params and 'strike_pct' in params:
                params['strike_offset_pct'] = params['strike_pct']
            params.setdefault('strike_pct', 0.05)
            params.setdefault('strike_offset', params.get('strike_pct', 0.05))
            params.setdefault('expiry_days', params.get('days_to_expiration', params.get('time_to_expiry_days', 30)))
            params.setdefault('days_to_expiry', params.get('expiry_days', 30))
            params.setdefault('risk_free_rate', 0.01)
            params.setdefault('volatility', 0.2)
            params.setdefault('position_size', 1)
            params.setdefault('initial_capital', initial_balance)
            # --- End robust parameter normalization ---
            exec(code, globals())
            globals()['underlying_prices'] = underlying_prices
            profit_loss, buy_points, sell_points, portfolio_value_history = eval(f'{function_name_code}(params)')
            # If we get here, it worked!
            plot_storyboard_option_trade(
                underlying_prices,
                [
                    black_scholes_call(
                        p,
                        p * (1 + params.get('strike_pct', 0.05)),
                        params.get('days_to_expiry', 30) / 252,
                        params.get('risk_free_rate', 0.01),
                        params.get('volatility', 0.2)
                    ) if params.get('option_type', 'call') == 'call'
                    else
                    black_scholes_put(
                        p,
                        p * (1 + params.get('strike_pct', 0.05)),
                        params.get('days_to_expiry', 30) / 252,
                        params.get('risk_free_rate', 0.01),
                        params.get('volatility', 0.2)
                    )
                    for p in underlying_prices
                ],
                buy_points, sell_points, portfolio_value_history, days_to_expiry, params
            )
            print(f"Profit/Loss: {profit_loss}")
            return profit_loss, buy_points, sell_points, portfolio_value_history
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"\n--- Attempt {attempts + 1} FAILED with error: ---")
            print(error_traceback)
            print("-------------------------------------------")
            # Compose error correction prompt for LLM
            error_prompt = f"""
**CRITICAL ERROR FIX REQUIRED**

The Python code you generated produced the following error:
```
{error_traceback}
```

**Your Task:**
Fix the error in the code and provide the COMPLETE CODE for BOTH functions. 

**EXTREMELY IMPORTANT: You MUST return the ENTIRE code for BOTH functions, not just the parts you modified.**

**User Strategy:** {user_input}

**REMEMBER:** 
- Output ONLY the complete code for both functions - no explanations, no markdown, no extra text.
- DO NOT SKIP OR SUMMARIZE ANY PART OF THE CODE with comments like "... rest of function remains the same ..."
- INCLUDE THE ENTIRE FUNCTION DEFINITIONS, not just the modified parts.
"""
            attempts += 1

    print("Max error attempts reached. Returning fallback result.")
    # Fallback: buy-and-hold
    buy_points = [(0, underlying_prices[0])] if underlying_prices else []
    sell_points = [(len(underlying_prices) - 1, underlying_prices[-1])] if underlying_prices else []
    profit_loss = (underlying_prices[-1] - underlying_prices[0]) if underlying_prices else 0
    portfolio_value_history = [initial_balance, initial_balance + profit_loss]
    return profit_loss, buy_points, sell_points, portfolio_value_history

# --- Main entry point ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM-driven options trading simulation.")
    parser.add_argument("strategy", nargs="?", default="", help="Options trading strategy description")
    args = parser.parse_args()
    user_input_raw = args.strategy if args.strategy else input("Enter your options trading strategy: ")
    run_llm_options_simulation(user_input_raw) 