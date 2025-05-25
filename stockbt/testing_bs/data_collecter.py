import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
from datetime import datetime, timedelta
from bisect import bisect_left
import pandas as pd

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# API credentials
API_KEY = "PK6TOP0STK4VO996JZ5Q"
API_SECRET = "PiNdt7ccSjzydlk35W5GOR5q9Ng1tAc6VEiSADeY"

# Stock symbol
SYMBOL = "GDHG"

# Time range
DAYS_BACK = 7  # How many days of data to fetch

# MULTI-INTERVAL CONFIGURATION
# Set to True to generate multiple intervals, False for single interval
GENERATE_MULTIPLE_INTERVALS = False

# Single interval (used when GENERATE_MULTIPLE_INTERVALS = False)
SINGLE_INTERVAL = "5T"

# Multiple intervals (used when GENERATE_MULTIPLE_INTERVALS = True)
INTERVALS_TO_GENERATE = [
    "1T",   # 1 minute
    "5T",   # 5 minutes
    "15T",  # 15 minutes
    "30T",  # 30 minutes
    "1H",   # 1 hour
]

# Advanced options
SAVE_OHLC_DATA = False  # Save separate OHLC files with Open/High/Low/Close/Volume
SHOW_SAMPLE_DATA = False  # Show sample data for each interval
MAX_SAMPLE_ROWS = 5  # How many sample rows to display

# =============================================================================
# DATA COLLECTION AND PROCESSING
# =============================================================================

print(f"Integrated Multi-Interval Data Collector")
print(f"Symbol: {SYMBOL}")
print(f"Days Back: {DAYS_BACK}")

if GENERATE_MULTIPLE_INTERVALS:
    print(f"Mode: Multiple Intervals")
    print(f"Intervals: {INTERVALS_TO_GENERATE}")
else:
    print(f"Mode: Single Interval ({SINGLE_INTERVAL})")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Calculate time range
end = datetime.now()
start = end - timedelta(days=DAYS_BACK)

print(f"Time Range: {start} to {end}")

# Fetch trades (prices) with increased limit
print(f"Fetching market data...")
trades_request = StockTradesRequest(
    symbol_or_symbols=[SYMBOL],
    start=start,
    end=end,
    limit=50000
)

try:
    trades = client.get_stock_trades(trades_request)
    trades = trades[SYMBOL]
    print(f"Fetched {len(trades)} trades")
        
except Exception as e:
    print(f"Error fetching trades: {e}")
    trades = []

# Fetch quotes (bid/ask) with increased limit
quotes_request = StockQuotesRequest(
    symbol_or_symbols=[SYMBOL],
    start=start,
    end=end,
    limit=50000
)

try:
    quotes = client.get_stock_quotes(quotes_request)
    quotes = quotes[SYMBOL]
    print(f"Fetched {len(quotes)} quotes")
        
except Exception as e:
    print(f"Error fetching quotes: {e}")
    quotes = []

# Validation
if len(trades) == 0 or len(quotes) == 0:
    print("Cannot proceed without both trades and quotes data")
    print("Possible issues:")
    print("   - Weekend/holiday (markets closed)")
    print("   - API limitations or rate limits")
    print("   - Network connectivity issues")
    print("   - Invalid symbol")
    exit(1)

# Convert to DataFrames (do this once, reuse for all intervals)
print(f"Converting raw data to DataFrames...")

# Convert trades to DataFrame
trades_data = []
for trade in trades:
    trades_data.append({
        'timestamp': trade.timestamp,
        'price': trade.price,
        'size': trade.size
    })

df_trades = pd.DataFrame(trades_data)
df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
df_trades.set_index('timestamp', inplace=True)

print(f"Raw trades DataFrame: {df_trades.shape}")

# Convert quotes to DataFrame
quotes_data = []
for quote in quotes:
    quotes_data.append({
        'timestamp': quote.timestamp,
        'bid_price': quote.bid_price,
        'ask_price': quote.ask_price,
        'bid_size': quote.bid_size,
        'ask_size': quote.ask_size
    })

df_quotes = pd.DataFrame(quotes_data)
df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'])
df_quotes.set_index('timestamp', inplace=True)

print(f"Raw quotes DataFrame: {df_quotes.shape}")

# Determine which intervals to process
if GENERATE_MULTIPLE_INTERVALS:
    intervals_to_process = INTERVALS_TO_GENERATE
else:
    intervals_to_process = [SINGLE_INTERVAL]

# Create output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_folder')
os.makedirs(DATA_DIR, exist_ok=True)

# Storage for results summary
results_summary = []

# =============================================================================
# PROCESS EACH TIME INTERVAL
# =============================================================================

for interval in intervals_to_process:
    print(f"Processing {interval} intervals...")
    
    try:
        # Resample trades to regular intervals with OHLC
        trades_resampled = df_trades['price'].resample(interval).agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': lambda x: df_trades.loc[x.index, 'size'].sum() if len(x) > 0 else 0
        }).dropna()

        # Resample quotes to regular intervals
        quotes_resampled = df_quotes.resample(interval).agg({
            'bid_price': 'last',
            'ask_price': 'last',
            'bid_size': 'last', 
            'ask_size': 'last'
        }).dropna()

        # Combine resampled data
        combined_resampled = quotes_resampled.join(trades_resampled, how='inner')
        combined_resampled = combined_resampled.dropna()

        if len(combined_resampled) == 0:
            print(f"No data after resampling for {interval}")
            results_summary.append({
                'interval': interval,
                'status': 'Failed - No data',
                'rows': 0,
                'file': None
            })
            continue

        # Create final DataFrame for ensemble_engine.py
        final_data = pd.DataFrame({
            'Bid_Price': combined_resampled['bid_price'],
            'Ask_Price': combined_resampled['ask_price'], 
            'Price': combined_resampled['Close']
        })
        final_data.reset_index(drop=True, inplace=True)

        # Generate filename
        interval_name = interval.replace('T', 'min').replace('H', 'hour')
        main_filename = f"{SYMBOL}_{interval_name}.csv"
        main_filepath = os.path.join(DATA_DIR, main_filename)

        # Save main data file
        final_data.to_csv(main_filepath, index=False)
        print(f"Saved {len(final_data)} intervals to {main_filename}")

        # Data quality analysis
        price_range = (final_data['Price'].min(), final_data['Price'].max())
        avg_spread = (final_data['Ask_Price'] - final_data['Bid_Price']).mean()

        # Store results for summary
        results_summary.append({
            'interval': interval,
            'status': 'Success',
            'rows': len(final_data),
            'file': main_filename,
            'price_range': price_range,
            'avg_spread': avg_spread
        })

    except Exception as e:
        print(f"Error processing {interval}: {e}")
        results_summary.append({
            'interval': interval,
            'status': f'Failed - {str(e)}',
            'rows': 0,
            'file': None
        })

# =============================================================================
# FINAL SUMMARY
# =============================================================================

successful = [r for r in results_summary if r['status'] == 'Success']
failed = [r for r in results_summary if r['status'] != 'Success']

print(f"Successful: {len(successful)}/{len(results_summary)} intervals")

if successful:
    print(f"Generated Files:")
    for result in successful:
        price_min, price_max = result['price_range']
        print(f"   {result['file']}: {result['rows']} rows | ${price_min:.2f}-${price_max:.2f} | Spread: ${result['avg_spread']:.4f}")

print(f"Ready for Ensemble Engine with {len(successful)} interval types")
