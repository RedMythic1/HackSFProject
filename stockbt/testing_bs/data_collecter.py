import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
from datetime import datetime, timedelta
from bisect import bisect_left
import pandas as pd

API_KEY = "PK6TOP0STK4VO996JZ5Q"
API_SECRET = "PiNdt7ccSjzydlk35W5GOR5q9Ng1tAc6VEiSADeY"

client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol = "AAPL"

end = datetime.now()
start = end - timedelta(days=0.1)

# Fetch trades (prices)
trades_request = StockTradesRequest(
    symbol_or_symbols=[symbol],
    start=start,
    end=end,
    limit=500
)
trades = client.get_stock_trades(trades_request)
trades = trades[symbol]

# Print the first trade to inspect its structure
print('First trade:', trades[0])

# Fetch quotes (bid/ask)
quotes_request = StockQuotesRequest(
    symbol_or_symbols=[symbol],
    start=start,
    end=end,
    limit=500
)
quotes = client.get_stock_quotes(quotes_request)
quotes = quotes[symbol]

trade_times = [trade.timestamp.timestamp() for trade in trades]
quote_times = [quote.timestamp.timestamp() for quote in quotes]

def find_closest_index(sorted_list, value):
    pos = bisect_left(sorted_list, value)
    if pos == 0:
        return 0
    if pos == len(sorted_list):
        return len(sorted_list) - 1
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    if after - value < value - before:
        return pos
    else:
        return pos - 1

combined_data = []

for trade in trades:
    trade_ts = trade.timestamp.timestamp()
    q_idx = find_closest_index(quote_times, trade_ts)
    quote = quotes[q_idx]
    if abs(quote.timestamp.timestamp() - trade_ts) < 1.0:
        combined_data.append([quote.bid_price, quote.ask_price, trade.price])

print(f"Combined {len(combined_data)} data points [Bid, Ask, Price]:")
for v in combined_data:
    print(v)

# Save to CSV for use in gen_autoregressive.py
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_folder')
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, f"{symbol}.csv")
df = pd.DataFrame(combined_data, columns=["Bid_Price", "Ask_Price", "Price"])
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
