#!/usr/bin/env python
# Flask server for Fly.io deployment

from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import logging
from datetime import datetime
import glob
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='dist')
app.config['JSON_SORT_KEYS'] = False

# Cache for articles
MEMORY_CACHE = {
    'articles': [],
    'articleDetails': {},
    'summaries': {}
}

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'article_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_id_from_filename(filename):
    """Extracts ID from filename like final_article_ID.json or final_article_ID_with_underscores.json"""
    base = os.path.basename(filename)
    if base.startswith('final_article_') and base.endswith('.json'):
        return base[len('final_article_'):-len('.json')]
    return None

def initialize_article_cache():
    logger.info(f"Attempting to load articles from cache directory: {CACHE_DIR}")
    loaded_count = 0
    file_pattern = os.path.join(CACHE_DIR, 'final_article_*.json')
    article_files = glob.glob(file_pattern)

    if not article_files:
        logger.info(f"No article files found matching pattern {file_pattern}")
        return

    logger.info(f"Found {len(article_files)} potential article files.")

    for filepath in article_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            article_id = extract_id_from_filename(filepath)
            if not article_id:
                logger.warning(f"Could not extract ID from filename: {filepath}")
                continue

            title = data.get('title', 'Untitled Article')
            # If title is still basic, try to get from ID (e.g., 12345_My_Title -> My Title)
            if title == 'Untitled Article' and '_' in article_id:
                try:
                    title_part = article_id.split('_', 1)[1].replace('_', ' ')
                    if title_part: title = title_part
                except IndexError:
                    pass # stick with Untitled
            
            summary = data.get('summary', data.get('subject', ''))
            if not summary and data.get('content'):
                # Basic summary: first non-empty paragraph from content
                paragraphs = [p.strip() for p in data.get('content', '').split('\n\n') if p.strip()]
                if paragraphs:
                    summary = paragraphs[0]
            if not summary: summary = "No summary available."

            content = data.get('content', '')
            link = data.get('link', data.get('url', f'/article/{article_id}')) # Default link to be relative
            
            # Timestamp: from data, or from filename (first part of ID if numeric)
            timestamp_ms = data.get('timestamp') # Assuming it might be in ms or seconds
            if not timestamp_ms:
                try:
                    id_timestamp_part = article_id.split('_')[0]
                    if id_timestamp_part.isdigit():
                        ts = int(id_timestamp_part)
                        # Check if it's seconds or milliseconds (simple heuristic: if it's a common seconds value)
                        if ts > 1000000000 and ts < 2000000000: # Likely seconds since epoch
                             timestamp_ms = ts * 1000
                        else: # Assume ms or needs other handling
                             timestamp_ms = ts 
                except (ValueError, IndexError):
                    pass # Could not parse from ID
            if not timestamp_ms: timestamp_ms = datetime.now().timestamp() * 1000 # Fallback to now


            list_item = {
                'id': article_id,
                'title': title,
                'subject': summary, # Use summary as subject for the list view
                'score': data.get('score', 0),
                'timestamp': timestamp_ms,
                'link': link
            }
            MEMORY_CACHE['articles'].append(list_item)

            detail_item = {
                'title': title,
                'link': link,
                'summary': summary,
                'content': content,
                'timestamp': timestamp_ms
            }
            MEMORY_CACHE['articleDetails'][article_id] = detail_item
            loaded_count += 1
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {filepath}")
        except Exception as e:
            logger.error(f"Error processing article file {filepath}: {e}")
    
    logger.info(f"Successfully loaded {loaded_count} articles into local MEMORY_CACHE.")

# Initialize cache when the application starts
initialize_article_cache()

def load_demo_articles():
    """Generate demo articles when no articles are available"""
    logger.info('Generating demo articles')
    current_time = datetime.now().timestamp() * 1000
    
    # Sample demo articles
    return [
        {
            'id': 'demo1',
            'title': 'Introduction to Machine Learning',
            'subject': 'A comprehensive guide to understanding the basics of Machine Learning and its applications in the modern world.',
            'score': 0,
            'timestamp': current_time - 3600000,  # 1 hour ago
            'link': 'https://example.com/article/1'
        },
        {
            'id': 'demo2',
            'title': 'The Future of Web Development',
            'subject': 'Exploring emerging trends in web development including WebAssembly, Progressive Web Apps, and more.',
            'score': 0,
            'timestamp': current_time - 7200000,  # 2 hours ago
            'link': 'https://example.com/article/2'
        },
        {
            'id': 'demo3',
            'title': 'Blockchain Technology Explained',
            'subject': 'Understanding the fundamentals of blockchain technology and its potential beyond cryptocurrencies.',
            'score': 0,
            'timestamp': current_time - 10800000,  # 3 hours ago
            'link': 'https://example.com/article/3'
        },
        {
            'id': 'demo4',
            'title': 'Artificial Intelligence Ethics',
            'subject': 'Examining the ethical considerations in AI development and implementation in society.',
            'score': 0,
            'timestamp': current_time - 14400000,  # 4 hours ago
            'link': 'https://example.com/article/4'
        },
        {
            'id': 'demo5',
            'title': 'Cloud Computing Fundamentals',
            'subject': 'An overview of cloud computing services, models, and best practices for businesses.',
            'score': 0,
            'timestamp': current_time - 18000000,  # 5 hours ago
            'link': 'https://example.com/article/5'
        }
    ]

def get_demo_article_detail(article_id):
    """Get demo article detail by ID"""
    demo_details = {
        'demo1': {
            'title': 'Introduction to Machine Learning',
            'link': 'https://example.com/article/1',
            'summary': 'Machine Learning is a rapidly growing field at the intersection of computer science and statistics. It focuses on developing algorithms that can learn from and make predictions on data. This article covers the fundamental concepts of Machine Learning, including supervised and unsupervised learning, regression, classification, and neural networks.',
            'content': '# Introduction to Machine Learning\n\nMachine Learning is a rapidly growing field that enables computers to learn from data without being explicitly programmed.\n\n## Supervised Learning\n\nIn supervised learning, algorithms learn from labeled training data to make predictions or decisions.\n\n## Unsupervised Learning\n\nUnsupervised learning algorithms find patterns in unlabeled data.\n\n## Applications\n\nMachine learning is used in many fields including healthcare, finance, and autonomous vehicles.'
        },
        'demo2': {
            'title': 'The Future of Web Development',
            'link': 'https://example.com/article/2',
            'summary': 'Web development is constantly evolving with new technologies and approaches. This article examines the latest trends shaping the future of web development, including WebAssembly, Progressive Web Apps, and JAMstack architecture.',
            'content': '# The Future of Web Development\n\nWeb development continues to evolve at a rapid pace with new technologies emerging regularly.\n\n## WebAssembly\n\nWebAssembly enables high-performance applications in the browser.\n\n## Progressive Web Apps\n\nPWAs combine the best of web and mobile apps.\n\n## JAMstack\n\nJAMstack architecture provides improved performance, security, and developer experience.'
        },
        'demo3': {
            'title': 'Blockchain Technology Explained',
            'link': 'https://example.com/article/3',
            'summary': 'Blockchain is a distributed ledger technology that enables secure, transparent, and immutable record-keeping without central authorities.',
            'content': '# Blockchain Technology Explained\n\nBlockchain is a distributed ledger technology that enables secure and transparent transactions.\n\n## Key Concepts\n\nBlockchain relies on consensus mechanisms, cryptographic hashing, and immutable ledgers.\n\n## Applications\n\nBeyond cryptocurrencies, blockchain has applications in supply chains, voting systems, and more.'
        }
    }
    
    return demo_details.get(article_id, {
        'title': 'Sample Article',
        'link': '#',
        'summary': 'This is a placeholder article summary. The requested article could not be found.',
        'content': '# Sample Article\n\nThis is a placeholder article. The requested article could not be found.'
    })

@app.route('/api/articles', methods=['GET'])
def get_articles():
    """Get processed articles"""
    logger.info("Processing articles endpoint")
    
    # Cache is now initialized at startup. If it's still empty, then load demos.
    if not MEMORY_CACHE['articles']:
        logger.info("MEMORY_CACHE['articles'] is empty after initial load attempt, generating demo articles.")
        demo_articles = load_demo_articles()
        # Also populate articleDetails for demos if we are falling back to them
        MEMORY_CACHE['articles'] = demo_articles 
        for article in demo_articles:
            if article['id'] not in MEMORY_CACHE['articleDetails']:
                 # For demo articles, the detail content needs to be generated/fetched
                 # This part assumes get_demo_article_detail can provide the necessary structure
                 demo_detail = get_demo_article_detail(article['id'])
                 MEMORY_CACHE['articleDetails'][article['id']] = demo_detail
        return jsonify(demo_articles)
    
    logger.info(f"Returning {len(MEMORY_CACHE['articles'])} articles from memory cache")
    return jsonify(MEMORY_CACHE['articles'])

@app.route('/api/article/<article_id>', methods=['GET'])
def get_article(article_id):
    """Get a specific article by ID"""
    logger.info(f"Getting article {article_id}")
    
    # Check if we have the article in memory cache (should be populated by initialize_article_cache)
    if article_id in MEMORY_CACHE['articleDetails']:
        logger.info(f"Returning article {article_id} from memory cache (articleDetails)")
        return jsonify(MEMORY_CACHE['articleDetails'][article_id])
    
    # If not in details, it might be a demo ID not fully populated by a direct call to demo loader.
    # Or it simply doesn't exist.
    # The get_demo_article_detail function handles non-existent demo IDs gracefully.
    logger.info(f"Article {article_id} not in MEMORY_CACHE['articleDetails'], attempting to load as demo.")
    article_detail = get_demo_article_detail(article_id) # This will return a placeholder if not a valid demo ID
    
    # Optionally, cache this demo detail if it was valid and not already cached (though unlikely path if demos loaded correctly)
    if not article_id.startswith("demo") or ("placeholder" not in article_detail.get('summary','').lower()):
         MEMORY_CACHE['articleDetails'][article_id] = article_detail # Cache if it seems like a valid demo lookup

    return jsonify(article_detail)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check request received")
    return jsonify({
        'status': 'ok',
        'message': 'Server is running on Fly.io',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """API endpoint for backtesting trading strategies
    
    Takes a trading strategy description from the user
    Returns buy/sell points, profit, and performance data
    """
    logger.info("Processing backtesting endpoint")
    try:
        # Get strategy description from request body
        data = request.get_json()
        if not data or 'strategy' not in data:
            logger.error("No strategy provided in backtest request")
            return jsonify({"status": "error", "error": "No strategy provided"}), 400
        
        strategy_description = data['strategy']
        logger.info(f"Received backtest request with strategy: {strategy_description[:100]}...")
        
        # Generate simulated stock data (in a real app, you'd fetch real market data)
        # Generate 100 days of simulated stock data with some randomness and a general trend
        days = 100
        base_price = 100.0
        volatility = 0.02
        trend = 0.001
        
        # Generate random walk for stock prices
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(trend, volatility, days)
        price_series = base_price * np.cumprod(1 + daily_returns)
        
        # Generate date labels (last 100 days)
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        dates.reverse()  # Oldest first
        
        # Simple strategy parsing (in a real app, you'd use NLP or a more sophisticated parser)
        # Look for keywords to determine strategy parameters
        
        # Default simple strategy: Buy when price drops by 2%, sell when price rises by 3%
        buy_threshold = -0.02
        sell_threshold = 0.03
        window_size = 5  # Look back period for calculating changes
        
        # Analyze the strategy text to modify parameters
        if "moving average" in strategy_description.lower():
            # Extract the window size for moving average if specified
            import re
            ma_matches = re.findall(r'(\d+)[-\s]day moving average', strategy_description.lower())
            if ma_matches:
                window_size = int(ma_matches[0])
                logger.info(f"Using moving average with window size: {window_size}")
        
        if "buy" in strategy_description.lower() and "below" in strategy_description.lower():
            # Strategy mentions buying below something - make threshold more negative
            buy_threshold = -0.03
        
        if "sell" in strategy_description.lower() and "above" in strategy_description.lower():
            # Strategy mentions selling above something - make threshold more positive
            sell_threshold = 0.04
            
        # Execute backtest with the parameters
        buy_points = []
        sell_points = []
        balance = 10000  # Start with $10,000
        shares = 0
        balance_history = [balance]
        buy_prices = []
        
        # Calculate moving averages for the entire series
        # This is a simple implementation that can be enhanced based on specific strategies
        ma_series = []
        for i in range(days):
            if i < window_size - 1:
                ma_series.append(price_series[i])  # Not enough data for MA yet
            else:
                ma = np.mean(price_series[i-window_size+1:i+1])
                ma_series.append(ma)
        
        # Execute trades based on strategy
        for i in range(1, days):
            price = price_series[i]
            prev_price = price_series[i-1]
            ma = ma_series[i]
            prev_ma = ma_series[i-1]
            
            # Determine if we should buy (price dropped below MA by threshold)
            price_to_ma_ratio = price / ma - 1
            if shares == 0 and price_to_ma_ratio < buy_threshold:
                # Buy condition met
                shares_to_buy = int(balance / price)
                if shares_to_buy > 0:
                    shares = shares_to_buy
                    cost = shares * price
                    balance -= cost
                    buy_points.append([i, price])  # [day index, price]
                    buy_prices.append(price)
                    logger.info(f"BUY: Day {i}, Price: ${price:.2f}, Shares: {shares}, Balance: ${balance:.2f}")
            
            # Determine if we should sell (price rose above MA by threshold)
            elif shares > 0 and price_to_ma_ratio > sell_threshold:
                # Sell condition met
                sale_value = shares * price
                balance += sale_value
                sell_points.append([i, price])  # [day index, price]
                logger.info(f"SELL: Day {i}, Price: ${price:.2f}, Shares: {shares}, Balance: ${balance:.2f}")
                shares = 0
            
            # Update balance history (account for shares held)
            current_value = balance
            if shares > 0:
                current_value += shares * price
            balance_history.append(current_value)
        
        # Calculate final statistics
        initial_value = balance_history[0]
        final_value = balance_history[-1]
        profit_loss = final_value - initial_value
        
        # Generate sample code based on the strategy
        generated_code = f"""# Python trading strategy based on your description:
# "{strategy_description}"

def execute_strategy(price_data, window_size={window_size}):
    buy_signals = []
    sell_signals = []
    holdings = 0
    
    # Calculate moving average
    for i in range(window_size, len(price_data)):
        price = price_data[i]
        ma = sum(price_data[i-window_size:i]) / window_size
        
        # Buy condition: price below MA by {buy_threshold*100}%
        if holdings == 0 and price < ma * (1 + {buy_threshold}):
            buy_signals.append(i)
            holdings = 1
            
        # Sell condition: price above MA by {sell_threshold*100}%
        elif holdings > 0 and price > ma * (1 + {sell_threshold}):
            sell_signals.append(i)
            holdings = 0
            
    return buy_signals, sell_signals
"""

        # Results in the format expected by the frontend
        result = {
            "status": "success",
            "profit_loss": profit_loss,
            "buy_points": buy_points,
            "sell_points": sell_points,
            "balance_over_time": balance_history,
            "generated_code": generated_code,
            "close": price_series.tolist(),  # The price series
            "dates": dates,    # Date labels
            "trades": {
                "count": len(buy_points),
                "buys": buy_points,
                "sells": sell_points
            }
        }
        
        logger.info(f"Backtest completed successfully. Profit/Loss: ${profit_loss:.2f}, Trades: {len(buy_points)}")
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error in backtest endpoint: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    if path == "" or path == "/":
        path = "index.html"
    
    logger.info(f"Serving static file: {path}")
    return send_from_directory('dist', path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Flask server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False) 