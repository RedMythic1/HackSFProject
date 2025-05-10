import logging
import os
import sys
import json
import traceback
from flask import Flask, request, jsonify, make_response
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
import re
import shutil
import uuid
import datetime
import concurrent.futures

# Add this line to import ansys module - handle cases where it might be in different locations
try:
    # Try importing from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ansys
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported ansys module from parent directory")
except ImportError:
    # If that fails, log an error
    logger = logging.getLogger(__name__)
    logger.error("Could not import ansys module from parent directory")
    ansys = None

# Set up logging to file only (avoid console output which causes I/O errors)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
    ]
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')
FINAL_ARTICLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_articles')
HTML_DIR = os.path.join(FINAL_ARTICLES_DIR, 'html')
# Add constant for the processed articles tracking file
PROCESSED_ARTICLES_FILE = os.path.join(CACHE_DIR, 'processed_articles.json')

# Ensure directories exist
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
    logger.info(f"Final articles directory created: {HTML_DIR}")
except Exception as e:
    logger.error(f"Error creating directories: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.urandom(24)  # Add secret key for cookie signing

# --- Progress Tracking Globals ---
PROGRESS_STORE = {}
PROGRESS_LOCK = threading.Lock()
# --- End Progress Tracking Globals ---

def get_ansys_path():
    """Get the path to ansys.py in the parent directory"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ansys_path = os.path.join(parent_dir, 'ansys.py')
    if not os.path.exists(ansys_path):
        raise FileNotFoundError(f"ansys.py not found in parent directory: {parent_dir}")
    return ansys_path

def _log_scoring_to_new_terminal(log_messages):
    """
    WHAT THIS DOES:
    Displays all the article scoring log messages in a clear, readable format.
    
    Parameters:
    - log_messages: A list of log message strings to display
    
    HOW IT WORKS:
    Instead of opening a new terminal window (which would be confusing),
    this just prints all the messages to the current console with nice formatting.
    """
    # Skip if there are no messages to log
    if not log_messages:
        return
    
    # Join all messages into one block of text
    log_content = "\n".join(log_messages)
    
    # Print with nice formatting
    print("\n\033[1;35m===== DETAILED SCORING LOG =====\033[0m")
    print(log_content)
    print("\033[1;35m================================\033[0m\n")
    
    # Also log to the logger for persistent records
    logger.debug(f"Article scoring log:\n{log_content}")

# Add a helper function to get processed article IDs
def get_processed_article_ids():
    """Get a list of already processed article IDs"""
    processed_ids = set()
    
    # Check if we have a tracking file
    if os.path.exists(PROCESSED_ARTICLES_FILE):
        try:
            with open(PROCESSED_ARTICLES_FILE, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                processed_ids = set(processed_data.get('processed_ids', []))
                logger.info(f"Loaded {len(processed_ids)} processed article IDs from tracking file")
        except Exception as e:
            logger.error(f"Error loading processed articles file: {e}")
    
    # Also check for existing final article files
    try:
        final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
        for article_path in final_articles:
            # Extract the ID from the filename
            filename = os.path.basename(article_path)
            article_id = filename.replace('final_article_', '').replace('.json', '')
            processed_ids.add(article_id)
    except Exception as e:
        logger.error(f"Error scanning for existing final articles: {e}")
    
    logger.info(f"Found total of {len(processed_ids)} processed article IDs")
    return processed_ids

def update_processed_article_ids(new_ids):
    """Update the list of processed article IDs"""
    processed_ids = get_processed_article_ids()
    processed_ids.update(new_ids)
    
    try:
        os.makedirs(os.path.dirname(PROCESSED_ARTICLES_FILE), exist_ok=True)
        with open(PROCESSED_ARTICLES_FILE, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Updated processed articles tracking file with {len(processed_ids)} IDs")
    except Exception as e:
        logger.error(f"Error updating processed articles file: {e}")

# Add this helper function to clean title format
def normalize_article_title(title):
    """Clean and normalize article titles by removing arrow notations and redundant text"""
    # Remove arrow notation (-> text) from titles
    if "->" in title:
        title = title.split("->")[0].strip()
    return title

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
        
        # Count all final_article_*.json files
        final_articles = glob.glob(os.path.join(cache_dir, 'final_article_*.json'))
        final_article_count = len(final_articles)
        
        # Check parent directory if no files found
        if final_article_count == 0 and os.path.exists(CACHE_DIR):
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
            final_article_count = len(final_articles)
        
        # Count unique articles based on title
        unique_titles = set()
        valid_article_count = 0
        
        if final_article_count > 0:
            for article_path in final_articles:
                try:
                    with open(article_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    content = data.get('content', '')
                    if content:
                        title = content.splitlines()[0] if content else 'Unknown Title'
                        if title.startswith('# '):
                            title = title[2:]  # Remove Markdown heading marker
                        
                        # Normalize title to remove arrow notation
                        title = normalize_article_title(title)
                        unique_titles.add(title)
                        valid_article_count += 1
                        
                        # Extract the filename and id from the path
                        filename = os.path.basename(article_path)
                        article_id = filename.replace('final_article_', '').replace('.json', '')
                        
                        # Generate missing HTML files
                        html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{article_id}.html")
                        
                        # Only create HTML file if it doesn't exist
                        if not os.path.exists(html_path):
                            try:
                                # Convert to HTML
                                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
            font-size: 1.8em;
        }}
        h3 {{
            color: #16a085;
            font-size: 1.4em;
        }}
        h4 {{
            color: #c0392b;
            font-size: 1.2em;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            background: #f5f5f5;
            border-left: 5px solid #3498db;
            padding: 10px 20px;
            margin: 20px 0;
        }}
        code {{
            background: #eee;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    {content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")}
</body>
</html>"""
                                
                                # Save HTML version
                                with open(html_path, 'w', encoding='utf-8') as html_file:
                                    html_file.write(html_content)
                                logger.info(f"Generated HTML file: {html_path}")
                            except Exception as e:
                                logger.error(f"Error creating HTML file {html_path}: {e}")
                                
                except Exception as e:
                    logger.error(f"Error reading article {article_path}: {e}")
                    # Delete the malformed file
                    try:
                        os.remove(article_path)
                        logger.info(f"Deleted malformed article file: {article_path}")
                    except Exception as remove_error:
                        logger.error(f"Could not delete malformed file {article_path}: {remove_error}")
        
        unique_article_count = len(unique_titles)
        
        logger.info(f"Found {article_count} cached article summaries and {valid_article_count} valid cached final articles ({unique_article_count} unique)")
        
        return jsonify({
            "status": "success",
            "message": "Cache check successful",
            "cached": article_count > 0 or valid_article_count > 0,
            "article_count": article_count,
            "final_article_count": unique_article_count,  # Return unique count
            "valid_article_count": valid_article_count    # Also return valid count
        })
    except Exception as e:
        logger.error(f"Exception checking cache: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/get-final-articles', methods=['GET'])
def get_final_articles():
    """Get the list of cached final articles"""
    try:
        # Look for final_article_*.json files in the cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir):
            cache_dir = CACHE_DIR  # Use the global cache dir if local not found
            
        # Find all final_article_*.json files
        final_articles = glob.glob(os.path.join(cache_dir, 'final_article_*.json'))
        
        # Check parent directory if no files found
        if len(final_articles) == 0 and os.path.exists(CACHE_DIR):
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
        
        # Extract and load article data
        article_data = []
        invalid_files = []
        
        for article_path in final_articles:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract filename
                filename = os.path.basename(article_path)
                
                # Extract timestamp
                timestamp = filename.replace('final_article_', '').replace('.json', '')
                
                # Get the first line as the title
                content = data.get('content', '')
                if not content:
                    logger.warning(f"Article has no content: {article_path}")
                    invalid_files.append(article_path)
                    continue
                    
                title = content.splitlines()[0] if content else 'Unknown Title'
                if title.startswith('# '):
                    title = title[2:]  # Remove Markdown heading marker
                
                # Normalize title
                title = normalize_article_title(title)
                
                article_data.append({
                    'id': timestamp,
                    'title': title,
                    'timestamp': data.get('timestamp', 0),
                    'filename': filename
                })
            except Exception as e:
                logger.error(f"Error loading article {article_path}: {e}")
                invalid_files.append(article_path)
        
        # Clean up invalid files
        for invalid_file in invalid_files:
            try:
                os.remove(invalid_file)
                logger.info(f"Deleted invalid article file: {invalid_file}")
            except Exception as remove_error:
                logger.error(f"Could not delete invalid file {invalid_file}: {remove_error}")
        
        # Sort by timestamp (newest first)
        article_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove duplicates based on title (keeping the newest version of each article)
        unique_titles = set()
        unique_articles = []
        for article in article_data:
            title = article['title']
            if title not in unique_titles:
                unique_titles.add(title)
                unique_articles.append(article)
        
        logger.info(f"Found {len(article_data)} cached final articles, {len(unique_articles)} unique, removed {len(invalid_files)} invalid files")
        
        return jsonify({
            "status": "success",
            "message": "Final articles retrieved successfully",
            "articles": unique_articles,
            "total_count": len(article_data),
            "unique_count": len(unique_titles),
            "invalid_count": len(invalid_files)
        })
    except Exception as e:
        logger.error(f"Exception getting final articles: {e}")
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
        
        # Get the path to ansys.py in parent directory
        try:
            ansys_path = get_ansys_path()
        except FileNotFoundError as e:
            logger.error(str(e))
            return jsonify({
                "status": "error",
                "message": f"ansys.py not found in parent directory. Please make sure it's located in the correct directory."
            }), 404
        
        logger.info(f"Found ansys.py at: {ansys_path}")
        
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Verified cache directory exists at: {CACHE_DIR}")
        
        # Get already processed article IDs
        processed_ids = get_processed_article_ids()
        # Create a processed IDs file that ansys.py can use
        processed_ids_file = os.path.join(tempfile.gettempdir(), 'processed_article_ids.json')
        with open(processed_ids_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Created temporary processed IDs file with {len(processed_ids)} IDs at {processed_ids_file}")
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal with enhanced error handling - use environment variable
            # Always set ANSYS_NO_SCORE=1 to ensure no scoring happens during caching
            terminal_cmd = f"ANSYS_NO_SCORE=1 ANSYS_PROCESSED_IDS_FILE={processed_ids_file} {sys.executable} {ansys_path} --cache-only"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {os.path.dirname(ansys_path)} && echo 'Running: {terminal_cmd}' && {terminal_cmd} || echo '\\nERROR: Command failed with exit code $?'"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Article Caching"
            end tell
            '''
            
            # Run the AppleScript
            process = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            if process.returncode != 0:
                logger.error(f"Failed to open Terminal: {process.stderr}")
                return jsonify({
                    "status": "error",
                    "message": f"Failed to start caching process: {process.stderr}"
                }), 500
                
            logger.info("Opened new Terminal window to run the command")
            
            # Immediately check the cache to report status
            time.sleep(1)  # Give it a moment to start
            
            # Count files in cache directory
            summary_files = glob.glob(os.path.join(CACHE_DIR, 'summary_*.json'))
            
            return jsonify({
                "status": "success",
                "message": f"Started caching articles in a new terminal window. Skipping {len(processed_ids)} already processed articles. Please check the terminal for progress.",
                "cache_dir": CACHE_DIR,
                "current_cache_count": len(summary_files),
                "skipped_articles": len(processed_ids)
            })
        else:
            # For non-macOS platforms, run in the background as before
            def run_ansys_cache():
                try:
                    logger.info(f"Running: python {ansys_path} --cache-only with processed IDs file: {processed_ids_file}")
                    
                    # Set environment variable for the subprocess
                    env = os.environ.copy()
                    env["ANSYS_PROCESSED_IDS_FILE"] = processed_ids_file
                    env["ANSYS_NO_SCORE"] = "1"  # Always set this to ensure no scoring happens
                    
                    process = subprocess.Popen(
                        [sys.executable, ansys_path, "--cache-only"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.path.dirname(ansys_path),  # Set correct working directory
                        env=env  # Pass the environment variables
                    )
                    stdout, stderr = process.communicate()
 
                    if process.returncode != 0:
                        logger.error(f"ansys.py failed with return code {process.returncode}")
                        logger.error(f"stderr: {stderr}")
                    else:
                        logger.info("ansys.py completed successfully")
                        logger.info(f"stdout: {stdout}")
                        
                        # Verify cache files were created
                        summary_files = glob.glob(os.path.join(CACHE_DIR, 'summary_*.json'))
                        logger.info(f"Found {len(summary_files)} summary files in cache after completion")
                        
                        # Update the processed articles tracking file with any new articles
                        new_final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
                        new_ids = set()
                        for article_path in new_final_articles:
                            filename = os.path.basename(article_path)
                            article_id = filename.replace('final_article_', '').replace('.json', '')
                            new_ids.add(article_id)
                        
                        update_processed_article_ids(new_ids)
 
                except Exception as e:
                    logger.error(f"Exception running ansys.py: {e}")
                    logger.error(traceback.format_exc())
            
            # Start the thread
            thread = threading.Thread(target=run_ansys_cache)
            thread.daemon = True
            thread.start()
            
            # Count files in cache directory
            summary_files = glob.glob(os.path.join(CACHE_DIR, 'summary_*.json'))
            
            return jsonify({
                "status": "success",
                "message": f"Started caching articles and generating questions in the background. Skipping {len(processed_ids)} already processed articles.",
                "cache_dir": CACHE_DIR,
                "current_cache_count": len(summary_files),
                "skipped_articles": len(processed_ids)
            })
        
    except Exception as e:
        logger.error(f"Exception in cache_articles: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/generate-questions', methods=['GET'])
def generate_questions():
    """Run ansys.py with --questions-only flag to generate questions for all articles"""
    try:
        logger.info("Starting question generation...")
        
        # Get the path to ansys.py in parent directory
        try:
            ansys_path = get_ansys_path()
        except FileNotFoundError as e:
            logger.error(str(e))
            return jsonify({
                "status": "error",
                "message": f"ansys.py not found in parent directory. Please make sure it's located in the correct directory."
            }), 404
        
        logger.info(f"Found ansys.py at: {ansys_path}")
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal
            terminal_cmd = f"{sys.executable} {ansys_path} --questions-only"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {os.path.dirname(ansys_path)} && echo 'Running: {terminal_cmd}' && {terminal_cmd} || echo '\\nERROR: Command failed with exit code $?'"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Question Generation"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info("Opened new Terminal window to run the command")
            
            return jsonify({
                "status": "success",
                "message": f"Started question generation in a new terminal window. Please check the terminal for progress.",
                "skipped_articles": 0
            })
        else:
            # For non-macOS platforms, run in the background as before
            # Run ansys.py with full processing in a separate thread
            def run_ansys_full_processing():
                try:
                    logger.info(f"Running ansys.py with predefined interests for full processing, skipping {len(processed_ids)} articles")
                    
                    # Create a command that feeds the interests to ansys.py and uses environment variable
                    env = os.environ.copy()
                    env["ANSYS_PROCESSED_IDS_FILE"] = processed_ids_file
                    env["ANSYS_NO_SCORE"] = "1"  # Always ensure scoring is skipped
                    
                    # Use shell=True to allow piping
                    cmd = f'cat {input_file} | {sys.executable} {ansys_path}'
                    logger.info(f"Executing: {cmd} with ANSYS_PROCESSED_IDS_FILE={processed_ids_file}")
                    
                    # Use shell=True to allow piping
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env  # Pass the environment variables, including skip-scoring
                    )
                    stdout, stderr = process.communicate()
                    
                    # Log results
                    if process.returncode != 0:
                        logger.error(f"ansys.py failed with return code {process.returncode}")
                        logger.error(f"stderr: {stderr}")
                    else:
                        logger.info("ansys.py full processing completed successfully")
                        logger.info(f"stdout: {stdout}")
                        
                        # Cache the generated final articles
                        try:
                            # Look for generated HTML files
                            html_files = glob.glob('tech_deep_dive_*.html')
                            
                            # Track new articles processed
                            new_ids = set()
                            
                            for html_file in html_files:
                                try:
                                    # Extract the timestamp from the filename
                                    timestamp = html_file.replace('tech_deep_dive_', '').replace('.html', '')
                                    timestamp = timestamp.split('_')[0]  # Get just the timestamp part
                                    
                                    # Extract content by reading the HTML file
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    
                                    # Extract the content by parsing the HTML 
                                    soup = BeautifulSoup(html_content, 'html.parser')
                                    title = soup.title.string if soup.title else "Untitled Article"
                                    body_content = soup.body.get_text('\n\n') if soup.body else ""
                                    
                                    # Create markdown-style content from HTML
                                    content = f"# {title}\n\n{body_content}"
                                    
                                    # Create a cache path for this final article
                                    cache_path = os.path.join(CACHE_DIR, f"final_article_{timestamp}.json")
                                    
                                    # Copy the HTML file to the HTML directory
                                    html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{timestamp}.html")
                                    try:
                                        shutil.copy2(html_file, html_path)
                                        logger.info(f"Copied HTML file to {html_path}")
                                    except Exception as e:
                                        logger.error(f"Error copying HTML file to {html_path}: {e}")
                                    
                                    # Cache the content
                                    try:
                                        with open(cache_path, 'w', encoding='utf-8') as f:
                                            json.dump({
                                                'content': content,
                                                'timestamp': int(time.time())
                                            }, f)
                                        logger.info(f"Cached final article to {cache_path}")
                                        new_ids.add(timestamp)
                                    except Exception as e:
                                        logger.error(f"Error caching final article: {e}")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing HTML file {html_file}: {e}")
                            
                            # Update the processed articles tracking file
                            update_processed_article_ids(new_ids)
                            logger.info(f"Added {len(new_ids)} new article IDs to processed tracking file")
                            
                        except Exception as e:
                            logger.error(f"Error processing generated articles: {e}")
                        
                except Exception as e:
                    logger.error(f"Exception running ansys.py with full processing: {e}")
                    logger.error(traceback.format_exc())
            
            # Start the thread
            thread = threading.Thread(target=run_ansys_full_processing)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "status": "success",
                "message": f"Started full article processing with question and answer generation in the background. Skipping {len(processed_ids)} already processed articles.",
                "skipped_articles": len(processed_ids)
            })
        
    except Exception as e:
        logger.error(f"Exception in generate_questions: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/run-ansys', methods=['POST'])
def run_ansys():
    """Run ansys.py with user interests to generate questions and answers"""
    try:
        # Get user interests from request
        data = request.get_json()
        if not data or 'interests' not in data:
            return jsonify({
                "status": "error",
                "message": "No interests provided in request"
            }), 400
            
        interests = data['interests']
        logger.info(f"Received interests: {interests}")
        
        # Save user data
        save_user_data(interests)
        
        # Get the path to ansys.py in parent directory
        try:
            ansys_path = get_ansys_path()
        except FileNotFoundError as e:
            logger.error(str(e))
            return jsonify({
                "status": "error",
                "message": f"ansys.py not found in parent directory. Please make sure it's located in the correct directory."
            }), 404
        
        logger.info(f"Found ansys.py at: {ansys_path}")
        
        # Get already processed article IDs
        processed_ids = get_processed_article_ids()
        # Create a processed IDs file that ansys.py can use
        processed_ids_file = os.path.join(tempfile.gettempdir(), 'processed_article_ids.json')
        with open(processed_ids_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Created temporary processed IDs file with {len(processed_ids)} IDs at {processed_ids_file}")
        
        # Create a temporary file with user interests
        input_file = os.path.join(tempfile.gettempdir(), 'ansys_input.txt')
        with open(input_file, 'w') as f:
            f.write(interests + '\n')
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal
            terminal_cmd = f"ANSYS_PROCESSED_IDS_FILE={processed_ids_file} cat {input_file} | {sys.executable} {ansys_path}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {os.path.dirname(ansys_path)} && echo 'Running: {terminal_cmd}' && {terminal_cmd} || echo '\\nERROR: Command failed with exit code $?'"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Processing"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info("Opened new Terminal window to run the command")
            
            return jsonify({
                "status": "success",
                "message": "Started processing in a new terminal window. Please check the terminal for progress.",
                "skipped_articles": 0
            })
        else:
            # For non-macOS platforms, run in the background as before
            # Run ansys.py with full processing in a separate thread
            def run_ansys_full_processing():
                try:
                    logger.info(f"Running ansys.py with predefined interests for full processing, skipping {len(processed_ids)} articles")
                    
                    # Create a command that feeds the interests to ansys.py and uses environment variable
                    env = os.environ.copy()
                    env["ANSYS_PROCESSED_IDS_FILE"] = processed_ids_file
                    env["ANSYS_NO_SCORE"] = "1"  # Always ensure scoring is skipped
                    
                    # Use shell=True to allow piping
                    cmd = f'cat {input_file} | {sys.executable} {ansys_path}'
                    logger.info(f"Executing: {cmd} with ANSYS_PROCESSED_IDS_FILE={processed_ids_file}")
                    
                    # Use shell=True to allow piping
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env  # Pass the environment variables, including skip-scoring
                    )
                    stdout, stderr = process.communicate()
                    
                    # Log results
                    if process.returncode != 0:
                        logger.error(f"ansys.py failed with return code {process.returncode}")
                        logger.error(f"stderr: {stderr}")
                    else:
                        logger.info("ansys.py full processing completed successfully")
                        logger.info(f"stdout: {stdout}")
                        
                        # Cache the generated final articles
                        try:
                            # Look for generated HTML files
                            html_files = glob.glob('tech_deep_dive_*.html')
                            
                            # Track new articles processed
                            new_ids = set()
                            
                            for html_file in html_files:
                                try:
                                    # Extract the timestamp from the filename
                                    timestamp = html_file.replace('tech_deep_dive_', '').replace('.html', '')
                                    timestamp = timestamp.split('_')[0]  # Get just the timestamp part
                                    
                                    # Extract content by reading the HTML file
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    
                                    # Extract the content by parsing the HTML 
                                    soup = BeautifulSoup(html_content, 'html.parser')
                                    title = soup.title.string if soup.title else "Untitled Article"
                                    body_content = soup.body.get_text('\n\n') if soup.body else ""
                                    
                                    # Create markdown-style content from HTML
                                    content = f"# {title}\n\n{body_content}"
                                    
                                    # Create a cache path for this final article
                                    cache_path = os.path.join(CACHE_DIR, f"final_article_{timestamp}.json")
                                    
                                    # Copy the HTML file to the HTML directory
                                    html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{timestamp}.html")
                                    try:
                                        shutil.copy2(html_file, html_path)
                                        logger.info(f"Copied HTML file to {html_path}")
                                    except Exception as e:
                                        logger.error(f"Error copying HTML file to {html_path}: {e}")
                                    
                                    # Cache the content
                                    try:
                                        with open(cache_path, 'w', encoding='utf-8') as f:
                                            json.dump({
                                                'content': content,
                                                'timestamp': int(time.time())
                                            }, f)
                                        logger.info(f"Cached final article to {cache_path}")
                                        new_ids.add(timestamp)
                                    except Exception as e:
                                        logger.error(f"Error caching final article: {e}")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing HTML file {html_file}: {e}")
                            
                            # Update the processed articles tracking file
                            update_processed_article_ids(new_ids)
                            logger.info(f"Added {len(new_ids)} new article IDs to processed tracking file")
                            
                        except Exception as e:
                            logger.error(f"Error processing generated articles: {e}")
                        
                except Exception as e:
                    logger.error(f"Exception running ansys.py with full processing: {e}")
                    logger.error(traceback.format_exc())
            
            # Start the thread
            thread = threading.Thread(target=run_ansys_full_processing)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "status": "success",
                "message": f"Started full article processing with question and answer generation in the background. Skipping {len(processed_ids)} already processed articles.",
                "skipped_articles": len(processed_ids)
            })
        
    except Exception as e:
        logger.error(f"Exception in run_ansys: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"}), 500

@app.route('/get-final-article/<article_id>', methods=['GET'])
def get_final_article(article_id):
    """Get the content of a specific final article by ID"""
    try:
        # Look for the article in the cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir):
            cache_dir = CACHE_DIR  # Use the global cache dir if local not found
            
        # Create the expected filename
        filename = f"final_article_{article_id}.json"
        article_path = os.path.join(cache_dir, filename)
        
        # If not found, check parent directory
        if not os.path.exists(article_path) and os.path.exists(CACHE_DIR):
            article_path = os.path.join(CACHE_DIR, filename)
            
        if not os.path.exists(article_path):
            logger.error(f"Article with ID {article_id} not found")
            return jsonify({
                "status": "error",
                "message": f"Article with ID {article_id} not found"
            }), 404
            
        # Load the article content
        with open(article_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        content = data.get('content', '')
        
        # Get the title (first line)
        title = content.splitlines()[0] if content else 'Unknown Title'
        if title.startswith('# '):
            title = title[2:]  # Remove Markdown heading marker
        
        # Normalize title
        title = normalize_article_title(title)
        
        # Extract or generate a summary
        summary = extract_article_summary(content)
            
        logger.info(f"Retrieved article: {title}")
        
        return jsonify({
            "status": "success",
            "message": "Article retrieved successfully",
            "article": {
                "id": article_id,
                "title": title,
                "content": content,
                "summary": summary,
                "timestamp": data.get('timestamp', 0)
            }
        })
    except Exception as e:
        logger.error(f"Exception getting article {article_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Exception: {str(e)}"
        }), 500

def extract_article_summary(content):
    """Extract or generate a summary from article content"""
    try:
        # First try to find a section explicitly labeled as "Summary"
        summary_pattern = re.compile(r'## Summary\s+([\s\S]+?)(?=##|$)')
        match = summary_pattern.search(content)
        if match:
            return match.group(1).strip()
        
        # If no explicit summary section, generate one from the beginning of the article
        lines = content.splitlines()
        
        # Skip the title if present
        start_idx = 0
        if lines and lines[0].startswith('# '):
            start_idx = 1
            
        # Collect text for summary (up to ~500 characters)
        summary_text = ""
        current_length = 0
        target_length = 500
        
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            # Skip headings and empty lines
            if line.startswith('#') or not line:
                continue
                
            # Add this line to the summary
            summary_text += line + " "
            current_length += len(line)
            
            # Stop if we've reached target length
            if current_length >= target_length:
                summary_text += "..."
                break
                
        return summary_text.strip() or "No summary available."
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Summary unavailable due to an error."

def _update_progress(task_id, status_message, percentage, current_log_messages=None):
    """
    WHAT THIS DOES:
    Updates the progress information for a running article matching task
    
    Parameters:
    - task_id: The unique ID of the task being updated
    - status_message: A short message describing the current status (e.g., "Loading articles...")
    - percentage: A number from 0-100 indicating task completion percentage
    - current_log_messages: Optional list of log messages to store
    
    HOW IT WORKS:
    This function safely updates a global dictionary (PROGRESS_STORE) that tracks all running tasks.
    The frontend checks this progress information to show status updates to the user.
    """
    # Use a lock to safely update the shared progress data
    with PROGRESS_LOCK:
        # Initialize progress tracking for this task if it doesn't exist yet
        if task_id not in PROGRESS_STORE:
            PROGRESS_STORE[task_id] = {} # Should be initialized before worker starts
        
        # Update the status information
        PROGRESS_STORE[task_id]['status'] = status_message
        PROGRESS_STORE[task_id]['percentage'] = percentage
        PROGRESS_STORE[task_id]['timestamp'] = time.time()
        
        # Optionally store recent log messages
        if current_log_messages is not None: # Optional: update full log if needed for polling
             PROGRESS_STORE[task_id]['log_snippet'] = current_log_messages[-3:] # Store last 3 messages as snippet

def _execute_best_article_match_task(task_id, user_interests):
    try:
        # Initialize LLM model with proper error handling
        llm = None
        try:
            llm = ansys.get_llama_model()
            if llm is None:
                raise RuntimeError("Failed to initialize LLM model")
        except Exception as e:
            error_msg = f"Error initializing LLM model: {str(e)}"
            logger.error(error_msg)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error",
                    "message": "Failed to initialize LLM model",
                    "details": str(e)
                }
                PROGRESS_STORE[task_id]['completed'] = True
            return

        # Load articles from cache
        articles_content_list = []
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir):
            cache_dir = CACHE_DIR

        final_articles = glob.glob(os.path.join(cache_dir, 'final_article_*.json'))
        if len(final_articles) == 0 and os.path.exists(CACHE_DIR):
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))

        for article_path in final_articles:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = data.get('content', '')
                if content:
                    title = content.splitlines()[0] if content else 'Unknown Title'
                    if title.startswith('# '):
                        title = title[2:]
                    articles_content_list.append({
                        'id': os.path.basename(article_path).replace('final_article_', '').replace('.json', ''),
                        'title': title,
                        'content': content
                    })
            except Exception as e:
                logger.error(f"Error loading article {article_path}: {e}")

        if not articles_content_list:
            error_msg = "No articles found to match"
            logger.error(error_msg)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error",
                    "message": error_msg
                }
                PROGRESS_STORE[task_id]['completed'] = True
            return

        # Format articles for the prompt
        articles_detail_parts = []
        for i, article in enumerate(articles_content_list):
            title = article['title']
            content = article['content']
            # Limit content length to avoid context overflow
            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"
            articles_detail_parts.append(f"Article {i+1}:\nTitle: {title}\nSummary:\n{content}\n")

        articles_detail_text = "\n\n".join(articles_detail_parts)

        # Prepare the prompt
        prompt_text = "[INST] I need to find which article best matches the user's interests. "
        prompt_text += "Please review the following articles carefully. For each article, provide an alignment score from 0 to 100, "
        prompt_text += "where 100 means a perfect alignment with the user's stated interests, and 0 means no alignment.\n\n"
        prompt_text += f"User interests: {user_interests}\n\n"
        prompt_text += f"Available articles:\n{articles_detail_text}\n\n"
        prompt_text += "Based on the user's interests and the provided titles and summaries, evaluate each article.\n"
        prompt_text += "Format your response as a list of numbers only, one per article, corresponding to the order above.\n"
        prompt_text += "Example:\n"
        prompt_text += "1. [score for Article 1]\n"
        prompt_text += "2. [score for Article 2]\n...\n\n"
        prompt_text += "Provide only the scores. No other text or explanation is needed. [/INST]"

        # Send prompt to LLM with proper error handling
        try:
            response = llm(prompt_text, max_tokens=1024, temperature=0.5)
            response_text = response["choices"][0]["text"].strip()
        except Exception as e:
            error_msg = f"Error getting LLM response: {str(e)}"
            logger.error(error_msg)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error",
                    "message": "Failed to get LLM response",
                    "details": str(e)
                }
                PROGRESS_STORE[task_id]['completed'] = True
            return

        # Process the response
        try:
            score_lines = response_text.split("\n")
            for i, line in enumerate(score_lines):
                if i < len(articles_content_list):
                    score_match = re.search(r'(\d+)', line)
                    if score_match:
                        score = int(score_match.group(1))
                        articles_content_list[i]['match_score'] = score
                    else:
                        articles_content_list[i]['match_score'] = 0
        except Exception as e:
            error_msg = f"Error processing LLM response: {str(e)}"
            logger.error(error_msg)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error",
                    "message": "Failed to process LLM response",
                    "details": str(e)
                }
                PROGRESS_STORE[task_id]['completed'] = True
            return

        # Sort articles by score
        articles_content_list.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        # Get best match
        best_match = articles_content_list[0] if articles_content_list else None
        if not best_match:
            error_msg = "No articles found to match"
            logger.error(error_msg)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error",
                    "message": error_msg
                }
                PROGRESS_STORE[task_id]['completed'] = True
            return

        # Return success result
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id]['final_result'] = {
                "status": "success",
                "message": "Found best matching article",
                "article": best_match
            }
            PROGRESS_STORE[task_id]['completed'] = True

    except Exception as e:
        error_msg = f"Unexpected error in article matching: {str(e)}"
        logger.error(error_msg)
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id]['final_result'] = {
                "status": "error",
                "message": "Unexpected error occurred",
                "details": str(e)
            }
            PROGRESS_STORE[task_id]['completed'] = True

def generate_match_explanation(article, user_interests):
    """Generate an explanation of why an article matched the user's interests"""
    try:
        # Extract article title
        title = article['title']
        
        # Extract user interests as a list
        interests = [interest.strip().lower() for interest in user_interests.split(',')]
        
        # Generate explanation based on match score
        score = article.get('match_score', 0)
        
        if score >= 85:
            strength = "excellent"
        elif score >= 70:
            strength = "strong"
        elif score >= 50:
            strength = "good"
        else:
            strength = "moderate"
            
        # Check for keyword matches in title and content
        content_lower = article.get('content', '').lower()
        matching_interests = []
        
        for interest in interests:
            if interest in title.lower() or interest in content_lower:
                matching_interests.append(interest)
                
        # Generate the explanation
        if matching_interests:
            interests_text = ", ".join(matching_interests)
            explanation = f"This article has a {strength} match with your interests in {interests_text}. "
            
            if score >= 70:
                explanation += "The content provides in-depth coverage of these topics."
            elif score >= 50:
                explanation += "The article covers these topics in reasonable detail."
            else:
                explanation += "The article touches on these topics."
        else:
            explanation = f"This article has a {strength} conceptual alignment with your stated interests. "
            explanation += "While not explicitly mentioning your exact terms, the content covers related concepts and ideas."
            
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating match explanation: {e}")
        return f"This article received a match score of {article.get('match_score', 0)}/100 based on your interests."

@app.route('/get-best-article-match', methods=['POST'])
def get_best_article_match_start():
    """
    WHAT THIS DOES:
    1. Takes the user's interests (e.g., "math", "technology")
    2. Starts a background process to find the best matching article
    3. Returns a task ID that the frontend can use to check progress
    
    HOW TO USE:
    - Send a POST request with {"interests": "your interests here"}
    - You'll get back a task_id to track progress
    - Use the /get-match-progress/<task_id> endpoint to check if it's done
    """
    try:
        # Step 1: Get the user's interests from the request
        data = request.json
        user_interests = data.get('interests', '')

        # Make sure interests were provided
        if not user_interests:
            return jsonify({"status": "error", "message": "No interests provided"}), 400
        
        # Step 2: Create a unique ID for this task
        task_id = uuid.uuid4().hex
        
        # Step 3: Initialize progress tracking for this task
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id] = {
                'status': 'Task initiated. Waiting for worker to start.',
                'percentage': 0,
                'messages': [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id} created for interests: '{user_interests}'"],
                'completed': False,
                'final_result': None,
                'timestamp': time.time()
            }
        
        # Step 4: Run the article matching process in a new terminal or background thread
        if sys.platform == 'darwin':  # If on Mac
            # Create temp files needed for the process
            input_file = os.path.join(tempfile.gettempdir(), f'article_match_interests_{task_id}.txt')
            with open(input_file, 'w') as f:
                f.write(f"{user_interests}\n")
                
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create a Python script that will run in the terminal
            terminal_script = os.path.join(tempfile.gettempdir(), f'run_article_match_{task_id}.py')
            with open(terminal_script, 'w') as f:
                script_content = '''
import sys
import os
import json
import time
import traceback
import glob
import re

# Define task_id at the beginning of the script
task_id = "''' + task_id + '''"

try:
    # Get the script directory
    script_dir = "''' + script_dir + '''"
    sys.path.append(script_dir)
    sys.path.append(os.path.dirname(script_dir))

    # Import necessary modules
    import ansys
    
    def log_with_color(message, color_code="1;37"):
        """Print colored text to the terminal."""
        print("\\033[" + color_code + "m" + message + "\\033[0m")
    
    # Print header
    log_with_color("===== ARTICLE RATING PROCESS =====", "1;36")
    
    # Load interests
    input_file = "''' + input_file + '''"
    with open(input_file, "r") as f:
        interests = f.read().strip()
    
    log_with_color("Analyzing your interests: " + interests, "1;33")
    
    # Initialize LLM model with 32K context window
    log_with_color("Loading LLM model with 32K context window...", "1;36")
    
    # Use the standard model initialization but with larger context window
    try:
        # First check if ansys has a direct way to get the model path
        temp_model = ansys.get_llama_model()
        if hasattr(temp_model, 'model_path'):
            model_path = temp_model.model_path
            log_with_color(f"Found model path from existing model: {model_path}", "1;37")
        else:
            # Try to find model path from environment variable
            model_path = os.environ.get('LLAMA_MODEL_PATH')
            if not model_path:
                # Look in common locations
                possible_locations = [
                    os.path.expanduser('~/llama-models'),
                    os.path.expanduser('~/Downloads'),
                    os.path.join(os.path.dirname(script_dir), 'models')
                ]
                log_with_color(f"Searching for Llama model in common locations", "1;35")
                for loc in possible_locations:
                    if os.path.exists(loc):
                        for root, dirs, files in os.walk(loc):
                            for file in files:
                                if file.endswith('.gguf'):
                                    model_path = os.path.join(root, file)
                                    log_with_color(f"Found potential model: {model_path}", "1;37")
                                    break
                            if model_path:
                                break
                    if model_path:
                        break
                        
        # If we still haven't found a model, use the standard one but it might fail
        if not model_path:
            log_with_color("Could not find model path, using standard model", "1;33")
            llm = ansys.get_llama_model()
        else:
            # Initialize with larger context window
            from llama_cpp import Llama
            log_with_color(f"Initializing Llama model with 32K context window from: {model_path}", "1;36")
            llm = Llama(
                model_path=model_path,
                n_ctx=32768,  # 32K context window
                n_gpu_layers=-1,  # Use all layers on GPU if available
                verbose=False
            )
            log_with_color("Successfully initialized Llama model with 32K context window", "1;32")
    except Exception as e:
        log_with_color(f"Error initializing custom LLM: {e}", "1;31")
        log_with_color("Falling back to standard model", "1;33")
        llm = ansys.get_llama_model()
    
    if llm is None:
        log_with_color("Error: Failed to initialize LLM model", "1;31")
        sys.exit(1)
    
    log_with_color("LLM model initialized successfully", "1;32")
    
    # Try multiple potential cache locations
    cache_locations = [
        os.path.join(script_dir, ".cache"),
        os.path.join(os.path.dirname(script_dir), ".cache"),
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), ".cache"),
        os.path.join(script_dir, "../.cache"),
        os.path.join(script_dir, "..")
    ]
    
    # Search for final article files in all possible locations
    final_articles = []
    article_files_found = []
    
    for cache_dir in cache_locations:
        if os.path.exists(cache_dir):
            log_with_color(f"Checking cache directory: {cache_dir}", "1;35")
            # Try using glob pattern to find final article files
            pattern = os.path.join(cache_dir, "final_article_*.json")
            found_files = glob.glob(pattern)
            article_files_found.extend(found_files)
    
    log_with_color(f"Found {len(article_files_found)} article files in cache directories", "1;35")
    
    if not article_files_found:
        # As a last resort, search the entire project directory
        project_dir = os.path.dirname(script_dir)
        log_with_color(f"Searching entire project directory: {project_dir}", "1;33")
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if file.startswith("final_article_") and file.endswith(".json"):
                    article_files_found.append(os.path.join(root, file))
        log_with_color(f"Found {len(article_files_found)} article files after deep search", "1;35")
    
    # Process found files
    for article_path in article_files_found:
        try:
            with open(article_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            article_id = os.path.basename(article_path).replace("final_article_", "").replace(".json", "")
            content = data.get("content", "")
            title = content.splitlines()[0] if content else "Unknown Title"
            if title.startswith("# "):
                title = title[2:]
            final_articles.append({"id": article_id, "title": title, "content": content})
            log_with_color(f"Loaded article: {title}", "1;37")
        except Exception as e:
            log_with_color(f"Error loading article {article_path}: {e}", "1;31")
    
    log_with_color(f"Loaded {len(final_articles)} articles for rating", "1;36")
    
    if not final_articles:
        log_with_color("No articles found to rate. Make sure to cache articles first.", "1;31")
        log_with_color("Use the 'Cache Articles' button in the application before rating.", "1;33")
        log_with_color("\\nPress Enter to close this window...", "1;37")
        input()
        sys.exit(1)
    
    # Limit articles to prevent context overflow
    MAX_ARTICLES = 10
    if len(final_articles) > MAX_ARTICLES:
        log_with_color(f"Too many articles ({len(final_articles)}). Limiting to {MAX_ARTICLES} for processing.", "1;33")
        final_articles = final_articles[:MAX_ARTICLES]
    
    # Calculate approximate token limit per article to stay within context window
    # Reserve 4000 tokens for the prompt structure and responses
    available_tokens = 24000 - 4000  # Use 24K as default context even if we tried for 32K
    tokens_per_article = available_tokens // len(final_articles)
    
    # Approximately 4 chars per token for English text
    chars_per_article = tokens_per_article * 4
    
    log_with_color(f"Processing {len(final_articles)} articles with ~{tokens_per_article} tokens (~{chars_per_article} chars) per article", "1;36")
    
    # Prepare prompt - use a simplified approach for better results
    articles_detail_parts = []
    for i, article in enumerate(final_articles):
        title = article["title"]
        content = article["content"]
        if len(content) > chars_per_article:
            # Get first 30% and last 70% of allowed length to capture beginning and important parts
            first_part_len = int(chars_per_article * 0.3)
            second_part_len = chars_per_article - first_part_len
            content = content[:first_part_len] + "... [middle content truncated] ..." + content[-second_part_len:]
            log_with_color(f"Truncated article {i+1} from {len(article['content'])} to {len(content)} chars", "1;35")
            
        articles_detail_parts.append(f"Article {i+1}:\\nTitle: {title}\\nSummary:\\n{content}\\n")
    
    articles_detail_text = "\\n\\n".join(articles_detail_parts)
    
    # Create the simplest possible prompt to get better results
    prompt_text = f"[INST]\\nUser interests: {interests}\\n\\n"
    prompt_text += "For each article below, rate how well it matches these interests on a scale of 0-100.\\n"
    prompt_text += "Output format MUST be: Article X: Y where X is article number and Y is score.\\n\\n"
    
    # Add each article with minimal formatting
    for i, article in enumerate(final_articles):
        title = article["title"]
        # Only include titles to make it simpler
        prompt_text += f"Article {i+1}: {title}\\n"
    
    prompt_text += "\\nRate each article in order from 1 to {len(final_articles)} using ONLY this format: Article X: Y\\n[/INST]"
    
    log_with_color(f"Prompt prepared with approximately {len(prompt_text) // 4} tokens", "1;36")
    log_with_color("Sending request to LLM for article scoring...", "1;36")
    
    # Attempt up to 3 times to get scores
    max_attempts = 3
    success = False
    response_text = ""
    
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log_with_color(f"Retry attempt {attempt}/{max_attempts}...", "1;33")
            
        try:
            # Score articles
            response = llm(prompt_text, max_tokens=1024, temperature=0.5)
            response_text = response["choices"][0]["text"].strip()
            
            # Log the raw response for debugging
            log_with_color("\\nRaw LLM response (attempt " + str(attempt) + "):", "1;35")
            log_with_color(response_text, "1;37")
            
            # Check if we got something usable
            if re.search(r'Article \d+: \d+', response_text):
                success = True
                break
            else:
                log_with_color("Response doesn't match expected format. Retrying...", "1;33")
        except Exception as e:
            log_with_color(f"Error during LLM scoring (attempt {attempt}): {e}", "1;31")
    
    log_with_color("\\nParsing scores...", "1;36")
    
    # Parse scores using a simple pattern
    lines = response_text.split('\\n')
    article_scores = {}
    issue_reasons = {}
    
    # Try to extract scores from each line
    for line in lines:
        match = re.search(r'Article (\d+): (\d+)', line)
        if match:
            try:
                article_num = int(match.group(1))
                score = int(match.group(2))
                if 1 <= article_num <= len(final_articles) and 0 <= score <= 100:
                    article_scores[article_num] = score
            except:
                pass
    
    # Check for missing articles and note why they're missing
    for i in range(1, len(final_articles) + 1):
        if i not in article_scores:
            issue_reasons[i] = "No valid score found in LLM response"
    
    # Apply scores to articles
    log_with_color("Article scores:", "1;36")
    for i, article in enumerate(final_articles):
        article_index = i + 1
        if article_index in article_scores:
            score = article_scores[article_index]
            article["match_score"] = score
            log_with_color(f"{article_index}. '{article['title']}': {score}/100", "1;32" if score > 50 else "1;37")
        else:
            article["match_score"] = 0
            reason = issue_reasons.get(article_index, "Unknown scoring issue")
            log_with_color(f"{article_index}. '{article['title']}': NO SCORE - {reason}", "1;31")
    
    # Sort by score - articles with no score will be at the bottom (score 0)
    final_articles.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Check if we have at least one article with a valid score
    valid_scores = [a for a in final_articles if a.get("match_score", 0) > 0]
    
    if valid_scores:
        best_match = valid_scores[0]
        log_with_color("\\n✓ BEST MATCH: '{best_match['title']}' (Score: {best_match.get('match_score', 'N/A')})", "1;32")
        
        # Update task progress in PROGRESS_STORE
        progress_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
        with open(progress_file, "w") as f:
            json.dump({"status": "success", "message": "Found best matching article", 
                      "article": {"id": best_match["id"], "title": best_match["title"], 
                                 "match_score": best_match.get("match_score", 0),
                                 "content": best_match.get("content", "")}}, f)
        log_with_color("\\nResults saved. You can close this window and return to the app.", "1;36")
    else:
        # If we have no valid scores, use a simple keyword matching fallback
        log_with_color("\\nNO VALID SCORES FOUND. Falling back to keyword-based matching.", "1;33")
        
        # Extract keywords from interests
        interest_keywords = [k.strip().lower() for k in interests.split(",")]
        log_with_color(f"Using keywords: {interest_keywords}", "1;35")
        
        # Score articles based on keyword frequency
        for article in final_articles:
            score = 0
            content = article["content"].lower()
            title = article["title"].lower()
            
            # Check each keyword
            for keyword in interest_keywords:
                if keyword in title:
                    # Keywords in title are more valuable
                    score += 20
                # Count occurrences in content (with cap per keyword)
                occurrences = content.count(keyword) 
                score += min(occurrences * 5, 30)  # Cap at 30 points per keyword
            
            # Normalize score to 0-100
            article["match_score"] = min(score, 100)
            log_with_color(f"'{article['title']}': Assigned fallback score {article['match_score']}/100", "1;37")
        
        # Sort again with new scores
        final_articles.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        # Check if we now have valid scores
        if final_articles and final_articles[0].get("match_score", 0) > 0:
            best_match = final_articles[0]
            log_with_color("\\n✓ BEST MATCH (FALLBACK): '{best_match['title']}' (Score: {best_match.get('match_score', 'N/A')})", "1;32")
            
            # Update task progress in PROGRESS_STORE
            progress_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
            with open(progress_file, "w") as f:
                json.dump({"status": "success", "message": "Found best matching article (using fallback scoring)", 
                          "article": {"id": best_match["id"], "title": best_match["title"], 
                                     "match_score": best_match.get("match_score", 0),
                                     "content": best_match.get("content", "")}}, f)
            log_with_color("\\nResults saved using fallback scoring method. You can close this window and return to the app.", "1;36")
        else:
            # If we still have no valid scores, report the issue
            log_with_color("\\nUnable to match articles with interests, even with fallback method.", "1;31")
            log_with_color("Raw LLM response was:", "1;35")
            log_with_color(response_text, "1;37")
            
            # Return an error response - task_id is now defined at the top of script
            progress_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
            with open(progress_file, "w") as f:
                json.dump({"status": "error", "message": "Could not match articles with your interests."}, f)
            
    # Keep terminal open for user to read
    log_with_color("\\nPress Enter to close this window...", "1;37")
    input()
    
except Exception as e:
    print("\\033[1;31mCritical error: " + str(e) + "\\033[0m")
    print(traceback.format_exc())
    print("\\nPress Enter to close this window...")
    input()
'''
                f.write(script_content)
            
            # Step 5A: Open a new terminal window and run the script (Mac-specific)
            terminal_cmd = f"{sys.executable} {terminal_script}"
            apple_script = f'''
            tell application "Terminal"
                activate
                do script "{terminal_cmd}"
                set position of front window to {{100, 100}}
                set bounds of front window to {{100, 100, 800, 600}}
            end tell
            '''
            process = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            
            # If opening terminal fails, fall back to background thread
            if process.returncode != 0:
                logger.error(f"Failed to open Terminal: {process.stderr}")
                logger.info(f"Falling back to background task for {task_id}")
                thread = threading.Thread(target=_execute_best_article_match_task, args=(task_id, user_interests))
                thread.daemon = True
                thread.start()
            else:
                logger.info(f"Opened new Terminal window for article matching task {task_id}")
                
                # Start a monitoring thread to check for results
                def monitor_terminal_results():
                    """Check for results from the terminal process"""
                    result_file = os.path.join(tempfile.gettempdir(), f"article_match_result_{task_id}.json")
                    max_wait_time = 300  # 5 minutes max wait
                    start_time = time.time()
                    
                    while time.time() - start_time < max_wait_time:
                        if os.path.exists(result_file):
                            try:
                                with open(result_file, 'r') as f:
                                    result = json.load(f)
                                
                                with PROGRESS_LOCK:
                                    PROGRESS_STORE[task_id]['final_result'] = result
                                    PROGRESS_STORE[task_id]['completed'] = True
                                    PROGRESS_STORE[task_id]['status'] = "Match found. Process complete."
                                    PROGRESS_STORE[task_id]['percentage'] = 100
                                
                                logger.info(f"Terminal process completed with result: {result}")
                                
                                # Clean up
                                try:
                                    os.remove(result_file)
                                    os.remove(input_file)
                                    os.remove(terminal_script)
                                except:
                                    pass
                                
                                return
                            except Exception as e:
                                logger.error(f"Error reading terminal results: {e}")
                        
                        time.sleep(2)
                    
                    # If we get here, the terminal process didn't complete in time
                    with PROGRESS_LOCK:
                        if not PROGRESS_STORE[task_id].get('completed', False):
                            PROGRESS_STORE[task_id]['final_result'] = {
                                "status": "error", 
                                "message": "Terminal process didn't complete in time"
                            }
                            PROGRESS_STORE[task_id]['completed'] = True
                            PROGRESS_STORE[task_id]['status'] = "Timeout waiting for terminal process"
                            PROGRESS_STORE[task_id]['percentage'] = 100
                
                # Start the monitoring thread
                monitoring_thread = threading.Thread(target=monitor_terminal_results)
                monitoring_thread.daemon = True
                monitoring_thread.start()
        else:
            # Step 5B: For non-Mac systems, run in background thread
            logger.info(f"Starting background task {task_id} for interests: {user_interests}")
            thread = threading.Thread(target=_execute_best_article_match_task, args=(task_id, user_interests))
            thread.daemon = True
            thread.start()
        
        # Step 6: Return task ID to client so they can check progress
        return jsonify({
            "task_id": task_id, 
            "status": "pending", 
            "message": "Article matching process initiated."
        }), 202  # 202 = Accepted (processing started)
        
    except Exception as e:
        # Log any errors that occur
        logger.error(f"Error starting article matching task: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"Exception starting task: {str(e)}"
        }), 500  # 500 = Internal Server Error

@app.route('/get-match-progress/<task_id>', methods=['GET'])
def get_match_progress(task_id):
    """
    WHAT THIS DOES:
    Checks the progress of an article matching task that was started earlier
    
    HOW TO USE:
    - Call this endpoint with the task_id you received from get-best-article-match
    - Keep calling it until 'completed' is true
    - When completed, the 'final_result' will contain the best article match
    """
    # Look up the task in our progress store
    with PROGRESS_LOCK:
        task_info = PROGRESS_STORE.get(task_id)

    # Return error if task doesn't exist
    if not task_info:
        return jsonify({
            "status": "error", 
            "message": "Task ID not found or expired."
        }), 404  # 404 = Not Found
    
    # Return the current progress information
    return jsonify(task_info)

def copy_articles_to_public():
    """
    WHAT THIS DOES: 
    Copies all article HTML files from final_articles/html to the public directory
    so they can be accessed directly by the frontend.
    """
    try:
        # Define source and destination directories
        source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_articles', 'html')
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public', 'articles')
        
        # Create the destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Get a list of all HTML files in the source directory
        html_files = glob.glob(os.path.join(source_dir, '*.html'))
        
        # Count files for logging
        copied_count = 0
        skipped_count = 0
        
        # Copy each file
        for html_file in html_files:
            filename = os.path.basename(html_file)
            dest_file = os.path.join(dest_dir, filename)
            
            # Check if destination file exists and is newer than source
            if os.path.exists(dest_file) and os.path.getmtime(dest_file) >= os.path.getmtime(html_file):
                skipped_count += 1
                continue
                
            # Copy the file
            shutil.copy2(html_file, dest_file)
            copied_count += 1
            
        logger.info(f"Copied {copied_count} articles to public directory, skipped {skipped_count} unchanged files")
        return copied_count, skipped_count
        
    except Exception as e:
        logger.error(f"Error copying articles to public directory: {e}")
        logger.error(traceback.format_exc())
        return 0, 0

@app.route('/copy-articles-to-public', methods=['GET'])
def copy_articles_endpoint():
    """
    WHAT THIS DOES:
    Endpoint to manually trigger copying of article HTML files to the public directory.
    
    HOW TO USE:
    Send a GET request to /copy-articles-to-public
    """
    try:
        copied, skipped = copy_articles_to_public()
        return jsonify({
            "status": "success",
            "message": f"Article copy operation completed successfully. Copied {copied} files, skipped {skipped} unchanged files.",
            "copied": copied,
            "skipped": skipped
        })
    except Exception as e:
        logger.error(f"Exception in copy_articles_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Exception copying articles: {str(e)}"}), 500

@app.route('/verify-email', methods=['POST'])
def verify_email():
    """Verify email and set cookie"""
    try:
        logger.info("/verify-email endpoint called")
        data = request.get_json()
        logger.info(f"/verify-email received data: {data}")
        email = data.get('email', '').strip() if data else ''
        
        if not email:
            return jsonify({
                "status": "error",
                "message": "Email is required"
            }), 400
            
        # Improved email validation: require at least 2 characters for TLD
        email_regex = r'^[^\s@]+@[^\s@]+\.[a-zA-Z0-9]{2,}$'
        if not re.match(email_regex, email):
            return jsonify({
                "status": "error",
                "message": "Invalid email format. Please use a valid email address (e.g. user@example.com)."
            }), 400
            
        # Create response with success message
        response = make_response(jsonify({
            "status": "success",
            "message": "Email verified successfully"
        }))
        
        # Set cookie that expires in 30 days
        is_local = request.host.startswith('localhost') or request.host.startswith('127.0.0.1')
        response.set_cookie(
            'verified_email',
            email,
            max_age=30*24*60*60,  # 30 days in seconds
            httponly=True,  # Prevent JavaScript access
            secure=not is_local,    # Only send over HTTPS unless local
            samesite='Lax'  # Protect against CSRF
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error verifying email: {e}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred while verifying email: {str(e)}"
        }), 500

@app.route('/check-email-verification', methods=['GET'])
def check_email_verification():
    """Check if email has been verified via cookie"""
    try:
        email = request.cookies.get('verified_email')
        if email:
            return jsonify({
                "status": "success",
                "verified": True,
                "email": email
            })
        else:
            return jsonify({
                "status": "success",
                "verified": False
            })
    except Exception as e:
        logger.error(f"Error checking email verification: {e}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while checking email verification"
        }), 500

@app.route('/process-articles', methods=['GET'])
def process_articles():
    """Run ansys.py to generate questions and create articles from cached content without requiring user interests"""
    try:
        logger.info("Starting article processing: generating questions, answers, and creating articles...")
        
        # Get the path to ansys.py in parent directory
        try:
            ansys_path = get_ansys_path()
        except FileNotFoundError as e:
            logger.error(str(e))
            return jsonify({
                "status": "error",
                "message": f"ansys.py not found in parent directory. Please make sure it's located in the correct directory."
            }), 404
        
        logger.info(f"Found ansys.py at: {ansys_path}")
        
        # Get already processed article IDs
        processed_ids = get_processed_article_ids()
        # Create a processed IDs file that ansys.py can use
        processed_ids_file = os.path.join(tempfile.gettempdir(), 'processed_article_ids.json')
        with open(processed_ids_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Created temporary processed IDs file with {len(processed_ids)} IDs at {processed_ids_file}")
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal with appropriate flags
            # Use default interests so it doesn't prompt for input, but still processes articles
            terminal_cmd = f"ANSYS_PROCESSED_IDS_FILE={processed_ids_file} echo 'technology, programming, science' | {sys.executable} {ansys_path}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {os.path.dirname(ansys_path)} && echo 'Running: {terminal_cmd}' && {terminal_cmd} || echo '\\nERROR: Command failed with exit code $?'"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Article Processing"
            end tell
            '''
            
            # Run the AppleScript
            process = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            if process.returncode != 0:
                logger.error(f"Failed to open Terminal: {process.stderr}")
                return jsonify({
                    "status": "error",
                    "message": f"Failed to start processing: {process.stderr}"
                }), 500
                
            logger.info("Opened new Terminal window to run the command")
            
            return jsonify({
                "status": "success",
                "message": f"Started article processing in a new terminal window. Skipping {len(processed_ids)} already processed articles. Please check the terminal for progress.",
                "skipped_articles": len(processed_ids)
            })
        else:
            # For non-macOS platforms, run in the background
            def run_ansys_processing():
                try:
                    logger.info(f"Running ansys.py for article processing with processed IDs file: {processed_ids_file}")
                    
                    # Set environment variable for the subprocess
                    env = os.environ.copy()
                    env["ANSYS_PROCESSED_IDS_FILE"] = processed_ids_file
                    
                    # Use echo to provide default interests so it doesn't prompt for input
                    process = subprocess.Popen(
                        f"echo 'technology, programming, science' | {sys.executable} {ansys_path}",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.path.dirname(ansys_path),
                        env=env
                    )
                    stdout, stderr = process.communicate()
 
                    if process.returncode != 0:
                        logger.error(f"ansys.py failed with return code {process.returncode}")
                        logger.error(f"stderr: {stderr}")
                    else:
                        logger.info("ansys.py completed successfully")
                        logger.info(f"stdout: {stdout}")
                        
                        # Update the processed articles tracking file with any new articles
                        new_final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
                        new_ids = set()
                        for article_path in new_final_articles:
                            filename = os.path.basename(article_path)
                            article_id = filename.replace('final_article_', '').replace('.json', '')
                            new_ids.add(article_id)
                        
                        update_processed_article_ids(new_ids)
 
                except Exception as e:
                    logger.error(f"Exception running ansys.py: {e}")
                    logger.error(traceback.format_exc())
            
            # Start the thread
            thread = threading.Thread(target=run_ansys_processing)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "status": "success",
                "message": f"Started article processing in the background. Skipping {len(processed_ids)} already processed articles.",
                "skipped_articles": len(processed_ids)
            })
        
    except Exception as e:
        logger.error(f"Exception in process_articles: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"Exception: {str(e)}"
        }), 500

if __name__ == '__main__':
    try:
        # Sync article files on startup
        logger.info("Syncing article files on startup...")
        try:
            # Copy articles to public directory
            copied, skipped = copy_articles_to_public()
            logger.info(f"Copied {copied} articles to public directory, skipped {skipped} unchanged files")
        except Exception as e:
            logger.error(f"Error copying articles to public directory: {e}")
            logger.error(traceback.format_exc())
            
        # Generate summaries for existing articles if needed
        logger.info("Checking for articles that need summaries...")
        try:
            # Find all final article files
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
            if len(final_articles) == 0 and os.path.exists(CACHE_DIR):
                final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
                
            # Count how many articles need summaries
            articles_updated = 0
            articles_with_summaries = 0
            
            # Check each article and add a summary section if it doesn't have one
            for article_path in final_articles:
                try:
                    # Read the article
                    with open(article_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get the content
                    content = data.get('content', '')
                    if not content:
                        continue
                        
                    # Check if it already has a Summary section
                    if '## Summary' not in content:
                        # No explicit summary section, generate one
                        summary = extract_article_summary(content)
                        
                        # Add the summary after the title
                        lines = content.splitlines()
                        if lines and lines[0].startswith('# '):
                            # Insert after the title
                            new_content = lines[0] + '\n\n## Summary\n\n' + summary + '\n\n' + '\n'.join(lines[1:])
                            
                            # Update the article
                            data['content'] = new_content
                            with open(article_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f)
                                
                            # Update markdown file if it exists
                            filename = os.path.basename(article_path)
                            article_id = filename.replace('final_article_', '').replace('.json', '')
                            markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
                            if os.path.exists(markdown_path):
                                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                                    md_file.write(new_content)
                                    
                            # Update HTML file if it exists
                            html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{article_id}.html")
                            if os.path.exists(html_path):
                                try:
                                    # Read the HTML file
                                    with open(html_path, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                        
                                    # Insert the summary after the title (this is a very basic approach)
                                    # A better approach would be to parse the HTML and insert properly
                                    title_end = html_content.find('</h1>')
                                    if title_end > 0:
                                        new_html = html_content[:title_end + 5] + f'<h2>Summary</h2><p>{summary}</p>' + html_content[title_end + 5:]
                                        
                                        # Write the updated HTML
                                        with open(html_path, 'w', encoding='utf-8') as f:
                                            f.write(new_html)
                                except Exception as html_err:
                                    logger.error(f"Error updating HTML file {html_path}: {html_err}")
                                    
                            articles_updated += 1
                    else:
                        articles_with_summaries += 1
                        
                except Exception as e:
                    logger.error(f"Error processing article {article_path}: {e}")
                    
            logger.info(f"Added summaries to {articles_updated} articles, {articles_with_summaries} already had summaries")
        except Exception as e:
            logger.error(f"Error generating summaries for articles: {e}")
            logger.error(traceback.format_exc())
            
        logger.info("Starting server (Flask development server)...")
        app.run(debug=True, port=5001, host='0.0.0.0') # debug=True can cause threads to run twice in some cases, be mindful
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 