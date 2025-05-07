import logging
import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import glob
import tempfile
import requests
from bs4 import BeautifulSoup
import threading
import time
import asyncio
import hashlib

# Set up logging to file only (avoid console output which causes I/O errors)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
    ]
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')

# Ensure cache directory exists
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
    logger.info(f"Cache directory created/verified at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Error creating cache directory: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/check-cache', methods=['GET'])
def check_cache():
    """Check if articles are cached"""
    try:
        # Look for summary_*.json files in the cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir):
            cache_dir = CACHE_DIR  # Use the global cache dir if local not found
            
        # Count all summary_*.json files
        summary_files = glob.glob(os.path.join(cache_dir, 'summary_*.json'))
        article_count = len(summary_files)
        
        # Check parent directory if no files found
        if article_count == 0 and os.path.exists(CACHE_DIR):
            summary_files = glob.glob(os.path.join(CACHE_DIR, 'summary_*.json'))
            article_count = len(summary_files)
        
        logger.info(f"Found {article_count} cached article summaries")
        
        return jsonify({
            "status": "success",
            "message": "Cache check successful",
            "cached": article_count > 0,
            "article_count": article_count
        })
    except Exception as e:
        logger.error(f"Exception checking cache: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/cache-articles', methods=['GET'])
def cache_articles():
    """Run ansys.py with --cache-only flag to generate questions and cache all articles"""
    try:
        logger.info("Starting article caching and question generation...")
        
        # Get the path to ansys.py (it might be in the same directory or parent directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ansys_path = os.path.join(script_dir, 'ansys.py')
        
        # If not found in the current directory, check the parent
        if not os.path.exists(ansys_path):
            ansys_path = os.path.join(os.path.dirname(script_dir), 'ansys.py')
        
        if not os.path.exists(ansys_path):
            logger.error(f"ansys.py not found in either {script_dir} or its parent directory")
            return jsonify({
                "status": "error",
                "message": f"ansys.py not found. Please make sure it's located in the correct directory."
            }), 404
        
        logger.info(f"Found ansys.py at: {ansys_path}")
        
        # Run ansys.py with --cache-only flag in a separate thread
        def run_ansys_cache():
            try:
                logger.info(f"Running: python {ansys_path} --cache-only")
                process = subprocess.Popen(
                    [sys.executable, ansys_path, "--cache-only"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"ansys.py failed with return code {process.returncode}")
                    logger.error(f"stderr: {stderr}")
                else:
                    logger.info("ansys.py completed successfully")
                    logger.info(f"stdout: {stdout}")
            except Exception as e:
                logger.error(f"Exception running ansys.py: {e}")
                logger.error(traceback.format_exc())
        
        # Start the thread
        thread = threading.Thread(target=run_ansys_cache)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Started caching articles and generating questions in the background."
        })
    except Exception as e:
        logger.error(f"Exception in cache_articles: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/run-ansys', methods=['POST'])
def run_ansys():
    """Process interests and run analysis"""
    try:
        data = request.json
        email = data.get('email', '')
        interests = data.get('interests', '')
        
        if not interests:
            return jsonify({"status": "error", "message": "No interests provided"}), 400
            
        if not email:
            return jsonify({"status": "error", "message": "No email provided"}), 400
        
        logger.info(f"Received email: {email}")
        logger.info(f"Received interests: {interests}")
        
        # Save the email to a file for future use
        user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_data')
        os.makedirs(user_data_dir, exist_ok=True)
        user_file = os.path.join(user_data_dir, f'{email}.txt')
        
        # Save the user's interests
        with open(user_file, 'w') as f:
            f.write(f"Email: {email}\n")
            f.write(f"Interests: {interests}\n")
        
        return jsonify({
            "status": "success",
            "message": "Request received and saved. Analysis will be performed in the background.",
            "user_email": email
        })
            
    except Exception as e:
        logger.error(f"Exception in run_ansys: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting server (simplified version - skipping cache initialization)...")
        app.run(debug=True, port=5001, host='0.0.0.0')
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 