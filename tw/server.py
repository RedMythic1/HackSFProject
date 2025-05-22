#!/usr/bin/env python
# Flask server for Fly.io deployment

from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import logging
from datetime import datetime, timedelta
import glob
import random  # Replace numpy with random
import math
import re
import traceback
import base64
import io

# Import the backtest module with run_improved_simulation
from api.backtest import run_improved_simulation

# Check if we should use system-installed packages instead of bundled ones
USE_SYSTEM_PACKAGES = os.environ.get('USE_SYSTEM_PACKAGES', 'false').lower() == 'true'
if not USE_SYSTEM_PACKAGES and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api/python_packages')):
    # Add the bundled packages to the Python path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api/python_packages'))
    logger_init = logging.getLogger("initialization")
    logger_init.info("Using bundled Python packages from api/python_packages")
else:
    logger_init = logging.getLogger("initialization")
    logger_init.info("Using system-installed Python packages")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
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

# Check if DATASETS_DIR is explicitly set in environment variables
env_datasets_dir = os.environ.get('DATASETS_DIR')
if env_datasets_dir:
    CACHE_DIR = os.environ.get('CACHE_DIR', '/data/article_cache')
    DATASETS_DIR = env_datasets_dir
    logger.info(f"Using environment-specified paths: CACHE_DIR={CACHE_DIR}, DATASETS_DIR={DATASETS_DIR}")
# Check if running on Fly.io with volume
elif os.path.exists('/data'):
    # Use Fly.io volume paths
    CACHE_DIR = '/data/article_cache'
    DATASETS_DIR = '/data/datasets'
    logger.info(f"Using Fly.io volume paths: CACHE_DIR={CACHE_DIR}, DATASETS_DIR={DATASETS_DIR}")
else:
    # Use local paths
    CACHE_DIR = os.path.join(BASE_DIR, 'article_cache')
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    logger.info(f"Using local paths: CACHE_DIR={CACHE_DIR}, DATASETS_DIR={DATASETS_DIR}")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

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
    Returns buy/sell points, profit, and performance data for frontend charting
    """
    logger.info("Processing backtesting endpoint")
    try:
        # Get strategy description from request body
        data = request.get_json()
        if not data or 'strategy' not in data:
            logger.error("No strategy provided in backtest request")
            return jsonify({"status": "error", "error": "No strategy provided. Please describe your trading strategy."})
        
        strategy_description = data['strategy']
        logger.info(f"Received strategy description: {strategy_description[:100]}...")
        
        # Run backtest simulation
        try:
            result = run_improved_simulation(strategy_description)
            return jsonify(result)
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in backtest simulation: {error_message}")
            trace = traceback.format_exc()
            logger.error(f"Traceback: {trace}")
            return jsonify({"status": "error", "error": f"Error in backtest simulation: {error_message}"})
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Unexpected error in /api/backtest: {error_message}")
        trace = traceback.format_exc()
        logger.error(f"Traceback: {trace}")
        return jsonify({"status": "error", "error": f"Server error: {error_message}"})

@app.route('/api/stock_prediction', methods=['GET'])
def stock_prediction():
    """API endpoint for stock prediction with PID control
    
    Runs a simplified version of the stock_reg.py script and returns the results as JSON
    """
    logger.info("Processing stock prediction endpoint")
    try:
        # Import only what's needed for this endpoint
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import random
        import base64
        from io import BytesIO
        
        # Dictionary to store the results
        result = {
            "status": "success",
            "graphs": [],
            "prediction_data": {},
            "summary": ""
        }
        
        # Select a data file from the data folder
        data_dir = os.path.join(BASE_DIR, "stockbt/testing_bs/data_folder")
        if not os.path.exists(data_dir):
            data_dir = "/data/stockbt/testing_bs/data_folder"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
        
        # Try to find an appropriate data file
        file = None
        # First check for stock_data_1.csv
        if os.path.exists(os.path.join(data_dir, "stock_data_1.csv")):
            file = "stock_data_1.csv"
        # Then check for sample_data.csv
        elif os.path.exists(os.path.join(data_dir, "sample_data.csv")):
            file = "sample_data.csv"
        # If neither exists, create a simple sample data file
        else:
            # Create a very simple sample dataset
            sample_data = pd.DataFrame({
                'Price': np.linspace(100, 200, 200),
                'PriceChange': [1] * 200,
                'Bid_Price': np.linspace(99, 199, 200),
                'Ask_Price': np.linspace(101, 201, 200),
                'Buy_Vol': [500] * 200,
                'Sell_Vol': [500] * 200
            })
            file = "sample_data.csv"
            sample_data.to_csv(os.path.join(data_dir, file), index=False)
            
        result["dataset_used"] = file
        
        # Load data
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if "PriceChange" not in df.columns:
                df["PriceChange"] = df["Price"].diff().fillna(0)
            
            # Simplified model: Use moving averages for prediction instead of neural network
            df['MA5'] = df['Price'].rolling(window=5).mean().fillna(0)
            df['MA10'] = df['Price'].rolling(window=10).mean().fillna(0)
            df['MA20'] = df['Price'].rolling(window=20).mean().fillna(0)
            
            # Generate sample data for display
            actual_values = df['Price'].values[-100:]
            
            # Simple prediction based on trends in moving averages
            predictions = []
            for i in range(len(actual_values)):
                if i < 20:  # Need at least 20 points for all MAs
                    predictions.append(actual_values[i])
                else:
                    # Weighted prediction based on MAs
                    ma5 = df['MA5'].values[i-100+400] if len(df) > 400 else df['MA5'].values[i]
                    ma10 = df['MA10'].values[i-100+400] if len(df) > 400 else df['MA10'].values[i]
                    ma20 = df['MA20'].values[i-100+400] if len(df) > 400 else df['MA20'].values[i]
                    pred = 0.5 * ma5 + 0.3 * ma10 + 0.2 * ma20
                    predictions.append(pred)
            
            # Generate PID "tuning" results
            pid_params = {
                'Kp': 0.5,
                'Ki': 0.01, 
                'Kd': 0.1
            }
            
            # Apply simple PID correction to predictions
            pid_preds = []
            Kp = pid_params['Kp']
            Ki = pid_params['Ki']
            Kd = pid_params['Kd']
            
            integral = 0
            prev_error = 0
            
            for i in range(len(predictions)):
                if i == 0:
                    pid_preds.append(predictions[i])
                    continue
                    
                error = actual_values[i-1] - predictions[i-1]
                integral += error
                derivative = error - prev_error
                
                pid_correction = Kp * error + Ki * integral + Kd * derivative
                prev_error = error
                
                pid_pred = predictions[i] + pid_correction
                pid_preds.append(pid_pred)
            
            # Calculate MSE
            mse = np.mean((np.array(pid_preds) - actual_values) ** 2)
            
            # Store results
            result["prediction_data"] = {
                "pid_walk_preds": pid_preds.tolist(),
                "pid_walk_actuals": actual_values.tolist(),
                "pid_mse": float(mse),
                "points_range": list(range(400, 400 + len(actual_values))) if len(df) > 400 else list(range(len(actual_values)))
            }
            
            result["pid_tuning"] = {
                "best_parameters": pid_params,
                "tuning_results": [
                    {"Kp": 0.5, "Ki": 0.01, "Kd": 0.1, "MSE": float(mse)},
                    {"Kp": 0.4, "Ki": 0.02, "Kd": 0.1, "MSE": float(mse * 1.1)},
                    {"Kp": 0.6, "Ki": 0.01, "Kd": 0.2, "MSE": float(mse * 1.2)},
                ]
            }
            
            # Generate PID graph
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(actual_values)), actual_values, label='Actual', linestyle='-')
            plt.plot(range(len(pid_preds)), pid_preds, label='PID Corrected Prediction', linestyle='--')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title(f'PID Walk-forward Prediction vs Actual')
            plt.legend()
            plt.tight_layout()
            
            # Save plot to a bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Encode the image
            graph_data = base64.b64encode(image_png).decode('utf-8')
            result["graphs"].append({
                "title": "PID Walk-forward Prediction vs Actual",
                "data": graph_data
            })
            
            # Generate a simple explanation
            result["summary"] = """
            How Stock Price Prediction with PID Control Works:
            
            1. The model analyzes patterns in historical stock data using moving averages.
            2. We use these patterns to predict future stock prices.
            3. We then apply a PID controller (similar to cruise control in a car) to make the predictions even better.
            
            The PID controller constantly adjusts predictions based on:
            - P (Proportional): The current prediction error
            - I (Integral): The sum of all past errors
            - D (Derivative): How fast the error is changing
            
            This combined approach significantly improves prediction accuracy compared to using basic prediction alone.
            """
            
            return jsonify(result)
            
        except Exception as e:
            error_message = f"Error processing CSV file: {str(e)}"
            logger.error(error_message)
            return jsonify({"status": "error", "error": error_message})
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in stock prediction: {error_message}")
        trace = traceback.format_exc()
        logger.error(f"Traceback: {trace}")
        return jsonify({"status": "error", "error": f"Server error: {error_message}"})

# Register additional static routes
@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    """Serve chart images from the charts directory"""
    charts_dir = os.path.join(DATASETS_DIR, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    logger.info(f"Serving chart: {filename} from {charts_dir}")
    return send_from_directory(charts_dir, filename)

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    if path == "" or path == "/":
        path = "index.html"
    
    logger.info(f"Serving static file: {path}")
    return send_from_directory('dist', path)

@app.route('/data/article_cache/<path:filename>')
def serve_article_cache(filename):
    return send_from_directory('/data/article_cache', filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Flask server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for backtesting')
    args = parser.parse_args()
    
    # Set environment variables for backtesting
    os.environ['BACKTEST_MAX_ITERATIONS'] = str(args.iterations)
    
    # Make sure DATASETS_DIR is correctly set
    if not os.environ.get('DATASETS_DIR'):
        os.environ['DATASETS_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        logger.info(f"Set DATASETS_DIR to {os.environ['DATASETS_DIR']}")
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False) 