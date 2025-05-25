import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from bisect import bisect_left
import pandas as pd

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# API credentials - Updated with your Alpaca keys
API_KEY = "PK6TOP0STK4VO996JZ5Q"
API_SECRET = "PiNdt7ccSjzydlk35W5GOR5q9Ng1tAc6VEiSADeY"

# Stock symbols - Focus on major stocks for better data availability
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX"]

# Time range - Reduced for more reliable data collection
TRADING_DAYS_BACK = 3  # How many trading days of data to fetch

# Single interval configuration - Keep it simple
SINGLE_INTERVAL = "5T"  # 5-minute intervals

# Advanced options
SAVE_OHLC_DATA = False   # Save OHLC files for ensemble engine
SHOW_SAMPLE_DATA = True # Show sample data
MAX_SAMPLE_ROWS = 3     # Sample rows to display

# =============================================================================
# SMART DATE CALCULATION FOR TRADING DAYS
# =============================================================================

def get_last_trading_days(trading_days_back=5):
    """
    Calculate start and end dates that account for weekends and holidays.
    Goes back enough calendar days to capture the requested trading days.
    """
    from datetime import datetime, timedelta
    
    end = datetime.now()
    
    # If it's weekend, go back to Friday
    if end.weekday() == 5:  # Saturday
        end = end - timedelta(days=1)  # Friday
    elif end.weekday() == 6:  # Sunday
        end = end - timedelta(days=2)  # Friday
    
    # Go back enough days to capture trading days
    # Assume worst case: need ~1.4x calendar days to get trading days (accounting for weekends)
    calendar_days_back = max(trading_days_back * 2, 10)  # At least 10 days to be safe
    
    start = end - timedelta(days=calendar_days_back)
    
    return start, end

# =============================================================================
# DATA COLLECTION AND PROCESSING
# =============================================================================

print(f"🚀 Stock Data Collector for Ensemble Engine")
print(f"Symbols: {SYMBOLS}")
print(f"Days Back: {TRADING_DAYS_BACK}")
print(f"Total symbols to process: {len(SYMBOLS)}")
print(f"Interval: {SINGLE_INTERVAL}")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Calculate time range
start, end = get_last_trading_days(TRADING_DAYS_BACK)

print(f"Time Range: {start} to {end}")

# Create output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
os.makedirs(DATA_DIR, exist_ok=True)

# Storage for results
successful_symbols = []
failed_symbols = []

# =============================================================================
# PROCESS EACH SYMBOL
# =============================================================================

for symbol_idx, symbol in enumerate(SYMBOLS):
    print(f"\n{'='*60}")
    print(f"📈 PROCESSING SYMBOL {symbol_idx + 1}/{len(SYMBOLS)}: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Fetch bars data (more reliable than trades)
        print(f"Fetching bar data for {symbol}...")
        bars_request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            start=start,
            end=end,
            timeframe=TimeFrame.Minute
        )
        
        bars = client.get_stock_bars(bars_request)
        
        if hasattr(bars, 'data') and symbol in bars.data:
            symbol_bars = bars.data[symbol]
            print(f"✅ Fetched {len(symbol_bars)} bars for {symbol}")
        elif hasattr(bars, 'data') and len(bars.data) > 0:
            symbol_bars = list(bars.data.values())[0]
            print(f"✅ Fetched {len(symbol_bars)} bars for {symbol} (first available)")
        else:
            print(f"❌ No bars data for {symbol}")
            failed_symbols.append(f"{symbol} (no bars)")
            continue
        
        if len(symbol_bars) == 0:
            print(f"⚠️ Empty data for {symbol} - skipping")
            failed_symbols.append(f"{symbol} (empty data)")
            continue

        # Convert bars to DataFrame
        bars_data = []
        for bar in symbol_bars:
            bars_data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })

        df_bars = pd.DataFrame(bars_data)
        df_bars['timestamp'] = pd.to_datetime(df_bars['timestamp'])
        df_bars.set_index('timestamp', inplace=True)

        print(f"📊 {symbol} - Raw bars: {df_bars.shape}")

        # Resample to desired interval
        print(f"   Resampling {symbol} to {SINGLE_INTERVAL} intervals...")
        
        bars_resampled = df_bars.resample(SINGLE_INTERVAL).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if len(bars_resampled) == 0:
            print(f"   ⚠️ No data after resampling for {symbol}")
            failed_symbols.append(f"{symbol} (no data after resampling)")
            continue

        # Create final DataFrame for ensemble engine
        final_data = pd.DataFrame({
            'Bid_Price': bars_resampled['low'],    # Use low as bid approximation
            'Ask_Price': bars_resampled['high'],   # Use high as ask approximation
            'Price': bars_resampled['close']       # Close price as main price
        })
        final_data.reset_index(drop=True, inplace=True)

        # Generate filename
        interval_name = SINGLE_INTERVAL.replace('T', 'min').replace('H', 'hour')
        main_filename = f"{symbol}_{interval_name}.csv"
        main_filepath = os.path.join(DATA_DIR, main_filename)

        # Save main data file
        final_data.to_csv(main_filepath, index=False)
        print(f"   ✅ Saved {len(final_data)} intervals to {main_filename}")

        # Save OHLC data if requested
        if SAVE_OHLC_DATA:
            ohlc_filename = f"{symbol}_{interval_name}_OHLC.csv"
            ohlc_filepath = os.path.join(DATA_DIR, ohlc_filename)
            bars_resampled.to_csv(ohlc_filepath)
            print(f"   ✅ Saved OHLC data to {ohlc_filename}")

        # Show sample data if requested
        if SHOW_SAMPLE_DATA:
            print(f"   📋 Sample data for {symbol}:")
            print(final_data.head(MAX_SAMPLE_ROWS).to_string())

        # Data quality analysis
        price_range = (final_data['Price'].min(), final_data['Price'].max())
        avg_spread = (final_data['Ask_Price'] - final_data['Bid_Price']).mean()

        print(f"   📊 Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
        print(f"   📊 Average spread: ${avg_spread:.4f}")

        successful_symbols.append(symbol)
        print(f"✅ {symbol} completed successfully")

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        failed_symbols.append(f"{symbol} (error: {str(e)[:50]})")
        continue

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n🎯 DATA COLLECTION COMPLETE!")
print(f"{'='*60}")

print(f"📊 RESULTS:")
print(f"   Total symbols attempted: {len(SYMBOLS)}")
print(f"   Successful symbols: {len(successful_symbols)}")
print(f"   Failed symbols: {len(failed_symbols)}")
print(f"   Success rate: {len(successful_symbols)/len(SYMBOLS)*100:.1f}%")

# Successful symbols summary
if successful_symbols:
    print(f"\n✅ SUCCESSFUL SYMBOLS:")
    for symbol in successful_symbols:
        interval_name = SINGLE_INTERVAL.replace('T', 'min').replace('H', 'hour')
        main_file = f"{symbol}_{interval_name}.csv"
        ohlc_file = f"{symbol}_{interval_name}_OHLC.csv"
        print(f"   {symbol}: {main_file}")
        if SAVE_OHLC_DATA:
            print(f"        {ohlc_file}")

# Failed symbols summary
if failed_symbols:
    print(f"\n❌ FAILED SYMBOLS:")
    for failed_info in failed_symbols:
        print(f"   {failed_info}")

print(f"\n📁 FILES SAVED TO: {DATA_DIR}")
print(f"🚀 Ready for Ensemble Engine with data from {len(successful_symbols)} symbols!")

if len(successful_symbols) == 0:
    print(f"\n⚠️  WARNING: No symbols were successfully processed!")
    print(f"   Possible issues:")
    print(f"     - Weekend/holiday (markets closed)")
    print(f"     - API limitations or rate limits")
    print(f"     - Network connectivity issues")
elif len(successful_symbols) < len(SYMBOLS):
    print(f"\n⚠️  PARTIAL SUCCESS: {len(successful_symbols)}/{len(SYMBOLS)} symbols processed")
else:
    print(f"\n🎉 COMPLETE SUCCESS: All {len(SYMBOLS)} symbols processed!")
    print(f"\nNext steps:")
    print(f"1. Run: python ensemble_engine.py (train your model)")
    print(f"2. Run: python quick_demo.py (see live signals)")
    print(f"3. Run: python alpaca_paper_trader.py (start trading)")
