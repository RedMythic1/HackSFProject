#!/usr/bin/env python
# Flask server for Fly.io deployment

from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import logging
from datetime import datetime
import glob

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
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

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
    
    # Check if we have articles in memory cache
    if len(MEMORY_CACHE['articles']) > 0:
        logger.info(f"Returning {len(MEMORY_CACHE['articles'])} articles from memory cache")
        return jsonify(MEMORY_CACHE['articles'])
    
    # Generate demo articles
    demo_articles = load_demo_articles()
    MEMORY_CACHE['articles'] = demo_articles
    
    return jsonify(demo_articles)

@app.route('/api/article/<article_id>', methods=['GET'])
def get_article(article_id):
    """Get a specific article by ID"""
    logger.info(f"Getting article {article_id}")
    
    # Check if we have the article in memory cache
    if article_id in MEMORY_CACHE['articleDetails']:
        logger.info(f"Returning article {article_id} from memory cache")
        return jsonify(MEMORY_CACHE['articleDetails'][article_id])
    
    # Get demo article detail
    article_detail = get_demo_article_detail(article_id)
    MEMORY_CACHE['articleDetails'][article_id] = article_detail
    
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