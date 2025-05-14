import json
import os
import csv
from typing import List, Tuple

# Path helpers – always use absolute paths relative to this file so Vercel can find the files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'stockbt', 'datasets', 'test.csv'))


def load_close_prices(limit: int = 5000) -> List[float]:
    """Load closing prices from the default CSV dataset. Return up to `limit` rows for speed."""
    closes: List[float] = []
    try:
        with open(DATASET_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                try:
                    closes.append(float(row["Close"]))
                except (KeyError, ValueError):
                    # Skip invalid rows
                    continue
    except FileNotFoundError:
        # If the dataset is missing, just return an empty array which we will handle later
        pass
    return closes


def simple_backtest(closes: List[float], initial_balance: float = 100_000) -> Tuple[float, List[Tuple[int, float]], List[Tuple[int, float]], List[float]]:
    """A *very* naive backtest: buy on day-0, sell on last day."""
    if not closes:
        return 0.0, [], [], []

    first_price = closes[0]
    last_price = closes[-1]

    shares = int(initial_balance // first_price)
    remaining_cash = initial_balance - (shares * first_price)

    balance_over_time: List[float] = []
    for price in closes:
        balance_over_time.append(remaining_cash + shares * price)

    final_balance = remaining_cash + shares * last_price
    profit_loss = final_balance - initial_balance

    buy_points = [(0, first_price)]
    sell_points = [(len(closes) - 1, last_price)]

    return profit_loss, buy_points, sell_points, balance_over_time


def handler(request, response):  # Vercel Python entry-point
    # Ensure POST only
    if request.method != "POST":
        response.status_code = 405
        return {"error": "Method not allowed. Use POST."}

    try:
        body = request.json() if callable(getattr(request, "json", None)) else json.loads(request.body.decode())
    except Exception:
        body = {}

    strategy = body.get("strategy") if isinstance(body, dict) else None
    if not strategy:
        response.status_code = 400
        return {"error": "Missing 'strategy' in JSON body."}

    # Load data & run a *very* simple backtest – this is placeholder logic you can later replace with something smarter.
    closes = load_close_prices()
    profit_loss, buy_pts, sell_pts, balance_history = simple_backtest(closes)

    # Build response
    result = {
        "profit_loss": profit_loss,
        "buy_points": buy_pts,
        "sell_points": sell_pts,
        "balance_over_time": balance_history,
        # For now we just echo back the strategy as pseudo-code.
        "generated_code": f"# Pseudo-code generated for strategy:\n# {strategy}",
        "chart_url": ""  # Front-end will hide image if this is empty
    }

    response.status_code = 200
    # Vercel automatically serialises dicts to JSON, but be explicit for clarity.
    response.headers["Content-Type"] = "application/json"
    return result 