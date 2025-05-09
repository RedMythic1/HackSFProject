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
import re
import shutil
import uuid

# Add this line to import ansys module - handle cases where it might be in different locations
try:
    # Try directly importing from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ansys
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported ansys module")
except ImportError:
    # If that fails, try to find ansys.py and import it
    logger = logging.getLogger(__name__)
    logger.warning("Could not import ansys directly, will attempt dynamic import when needed")
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
CACHE_DIR = "/Users/avneh/Code/HackSFProject/.cache"
FINAL_ARTICLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_articles')
MARKDOWN_DIR = os.path.join(FINAL_ARTICLES_DIR, 'markdown')
HTML_DIR = os.path.join(FINAL_ARTICLES_DIR, 'html')
# Add constant for the processed articles tracking file
PROCESSED_ARTICLES_FILE = os.path.join(CACHE_DIR, 'processed_articles.json')

# Ensure directories exist
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
    logger.info(f"Final articles directories created: {MARKDOWN_DIR} and {HTML_DIR}")
except Exception as e:
    logger.error(f"Error creating directories: {e}")

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

# --- Progress Tracking Globals ---
PROGRESS_STORE = {}
PROGRESS_LOCK = threading.Lock()
# --- End Progress Tracking Globals ---

def _log_scoring_to_new_terminal(log_messages):
    """Log article scoring messages to the main terminal instead of creating a new terminal window."""
    if not log_messages:
        return
    
    log_content = "\n".join(log_messages)
    
    # Always log to main terminal
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

@app.route('/check-cache', methods=['GET'])
def check_cache():
    """Check if articles are cached"""
    try:
        # Look for summary_*.json files in the cache directory
        summary_files = glob.glob(os.path.join(CACHE_DIR, 'summary_*.json'))
        article_count = len(summary_files)
        
        # Count all final_article_*.json files
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
                        unique_titles.add(title)
                        valid_article_count += 1
                        
                        # Extract the filename and id from the path
                        filename = os.path.basename(article_path)
                        article_id = filename.replace('final_article_', '').replace('.json', '')
                        
                        # Generate missing markdown and HTML files
                        markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
                        html_path = os.path.join(HTML_DIR, f"tech_deep_dive_{article_id}.html")
                        
                        # Only create files if they don't exist
                        if not os.path.exists(markdown_path):
                            try:
                                # Save markdown version
                                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                                    md_file.write(content)
                                logger.info(f"Generated missing markdown file: {markdown_path}")
                            except Exception as e:
                                logger.error(f"Error creating markdown file {markdown_path}: {e}")
                        
                        # Create HTML version if it doesn't exist
                        if not os.path.exists(html_path):
                            try:
                                # Convert markdown to HTML
                                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")}
</body>
</html>"""
                                
                                # Save HTML version
                                with open(html_path, 'w', encoding='utf-8') as html_file:
                                    html_file.write(html_content)
                                logger.info(f"Generated missing HTML file: {html_path}")
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
        # Find all final_article_*.json files
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
    """Run ansys.py with pre-defined interests to force full processing including question generation"""
    try:
        logger.info("Starting full question and answer generation for all articles...")
        
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
        
        # Get already processed article IDs
        processed_ids = get_processed_article_ids()
        # Create a processed IDs file that ansys.py can use
        processed_ids_file = os.path.join(tempfile.gettempdir(), 'processed_article_ids.json')
        with open(processed_ids_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Created temporary processed IDs file with {len(processed_ids)} IDs at {processed_ids_file}")
        
        # Create a temporary file with predefined inputs to feed to ansys.py
        input_file = os.path.join(tempfile.gettempdir(), 'ansys_input.txt')
        with open(input_file, 'w') as f:
            # Add a broad range of interests to ensure most articles are processed
            f.write("technology, programming, science, AI, finance, health, politics, education\n")
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal - use environment variable
            # Always ensure ANSYS_NO_SCORE=1 is set to skip scoring
            terminal_cmd = f"ANSYS_NO_SCORE=1 ANSYS_PROCESSED_IDS_FILE={processed_ids_file} cat {input_file} | {sys.executable} {ansys_path}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {script_dir} && echo 'Running: {terminal_cmd}' && {terminal_cmd}"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Full Processing"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info("Opened new Terminal window to run the command")
            
            return jsonify({
                "status": "success",
                "message": f"Started full article processing in a new terminal window. Skipping {len(processed_ids)} already processed articles. Please check the terminal for progress.",
                "skipped_articles": len(processed_ids)
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
                            # Look for generated markdown files
                            article_files = glob.glob('tech_deep_dive_*.md')
                            
                            # Track new articles processed
                            new_ids = set()
                            
                            for article_file in article_files:
                                # Read the content
                                with open(article_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    
                                # Extract the timestamp from the filename
                                timestamp = article_file.replace('tech_deep_dive_', '').replace('.md', '')
                                
                                # Create a cache path for this final article
                                cache_path = os.path.join(CACHE_DIR, f"final_article_{timestamp}.json")
                                
                                # Copy the markdown file to the markdown directory
                                markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{timestamp}.md")
                                try:
                                    shutil.copy2(article_file, markdown_path)
                                    logger.info(f"Copied markdown file to {markdown_path}")
                                except Exception as e:
                                    logger.error(f"Error copying markdown file to {markdown_path}: {e}")
                                
                                # Look for corresponding HTML file
                                html_file = f"tech_deep_dive_{timestamp}.html"
                                if os.path.exists(html_file):
                                    # Copy the HTML file to the html directory
                                    html_path = os.path.join(HTML_DIR, html_file)
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
                                    logger.info(f"Cached final article {article_file} to {cache_path}")
                                    new_ids.add(timestamp)
                                except Exception as e:
                                    logger.error(f"Error caching final article {article_file}: {e}")
                                    
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
        
        # Get the path to ansys.py
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
            
        # Get already processed article IDs
        processed_ids = get_processed_article_ids()
        # Create a processed IDs file that ansys.py can use
        processed_ids_file = os.path.join(tempfile.gettempdir(), f'processed_article_ids_{email.replace("@", "_at_")}.json')
        with open(processed_ids_file, 'w', encoding='utf-8') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        logger.info(f"Created temporary processed IDs file with {len(processed_ids)} IDs for {email} at {processed_ids_file}")
            
        # Create a temporary file with the user's interests
        input_file = os.path.join(tempfile.gettempdir(), f'ansys_input_{email.replace("@", "_at_")}.txt')
        with open(input_file, 'w') as f:
            f.write(f"{interests}\n")
            
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal - use environment variable WITHOUT ANSYS_NO_SCORE
            # to ensure scoring happens when user submits interests
            terminal_cmd = f"ANSYS_PROCESSED_IDS_FILE={processed_ids_file} cat {input_file} | {sys.executable} {ansys_path}"
            
            # Create a unique title with the user's email
            terminal_title = f"ANSYS Analysis for {email}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {script_dir} && echo 'Running analysis for: {email}' && echo 'Interests: {interests}' && {terminal_cmd} && echo 'Copying generated files to final_articles directories...' && find . -name 'tech_deep_dive_*.md' -exec cp {{}} {MARKDOWN_DIR}/ \\; && find . -name 'tech_deep_dive_*.html' -exec cp {{}} {HTML_DIR}/ \\;"
                set position of front window to {{100, 100}}
                set custom title of front window to "{terminal_title}"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info(f"Opened new Terminal window to run analysis for {email}")
            
            return jsonify({
                "status": "success",
                "message": f"Started analysis in a new terminal window. Skipping {len(processed_ids)} already processed articles. Please check the terminal for progress.",
                "user_email": email,
                "skipped_articles": len(processed_ids)
            })
        else:
            # For non-macOS platforms, just acknowledge the request
            # The actual processing would be done separately
            return jsonify({
                "status": "success",
                "message": f"Request received and saved. Analysis will be performed in the background. Skipping {len(processed_ids)} already processed articles.",
                "user_email": email,
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
        # Create the expected filename
        filename = f"final_article_{article_id}.json"
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
            
        logger.info(f"Retrieved article: {title}")
        
        return jsonify({
            "status": "success",
            "message": "Article retrieved successfully",
            "article": {
                "id": article_id,
                "title": title,
                "content": content,
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

@app.route('/article-html/<article_id>', methods=['GET'])
def serve_article_html(article_id):
    """Serve the HTML version of an article directly"""
    try:
        # Construct the HTML file path
        html_filename = f"tech_deep_dive_{article_id}.html"
        html_path = os.path.join(HTML_DIR, html_filename)
        
        logger.info(f"Attempting to serve HTML article from: {html_path}")
        
        # Check if HTML file exists
        if not os.path.exists(html_path):
            logger.error(f"HTML article with ID {article_id} not found at: {html_path}")
            
            # Try to generate it from the markdown if possible
            markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
            logger.info(f"Looking for markdown file at: {markdown_path}")
            
            if os.path.exists(markdown_path):
                logger.info(f"Found markdown at {markdown_path}, generating HTML on-demand")
                try:
                    # Read markdown content
                    with open(markdown_path, 'r', encoding='utf-8') as md_file:
                        content = md_file.read()
                    logger.info(f"Successfully read markdown file with {len(content)} characters")
                        
                    # Extract title from first line
                    title = content.splitlines()[0] if content else 'Unknown Title'
                    if title.startswith('# '):
                        title = title[2:]  # Remove Markdown heading marker
                    logger.info(f"Extracted title: {title}")
                    
                    # Convert markdown to HTML - using a more robust approach
                    try:
                        # Try to use python-markdown if available
                        import markdown
                        html_body = markdown.markdown(content)
                        logger.info("Used python-markdown to convert content")
                    except ImportError:
                        # Fall back to simple replacement
                        logger.info("python-markdown not available, using simple replacement")
                        
                        # Process markdown links
                        def convert_links(text):
                            # Find Markdown links [text](url)
                            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                            return re.sub(link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
                        
                        # Clean up heading levels
                        def fix_headings(text):
                            # Look for standalone ### or ## not at beginning of line
                            text = re.sub(r'(?<!\n)(\s+)(#{2,3})(\s+)', r'\1<h\2></h\2>\3', text)
                            return text
                            
                        # Improved conversion with paragraph handling and link processing
                        paragraphs = content.split("\n\n")
                        html_parts = []
                        
                        for p in paragraphs:
                            # Process different element types
                            p = convert_links(p)
                            p = fix_headings(p)
                            
                            if p.startswith("# "):
                                html_parts.append(f'<section class="main-heading"><h1>{p[2:]}</h1></section>')
                            elif p.startswith("## "):
                                html_parts.append(f'<section class="section-heading"><h2>{p[3:]}</h2></section>')
                            elif p.startswith("### "):
                                html_parts.append(f'<section class="subsection-heading"><h3>{p[4:]}</h3></section>')
                            elif p.startswith("#### "):
                                html_parts.append(f'<section class="subsubsection-heading"><h4>{p[5:]}</h4></section>')
                            elif p.startswith("**Source"):
                                # Special handling for source boxes
                                html_parts.append(f'<div class="source-box">{p}</div>')
                            elif p.startswith("- "):
                                # Convert to unordered list
                                items = p.split("\n")
                                list_items = "".join([f"<li>{convert_links(item[2:])}</li>" for item in items if item.startswith("- ")])
                                html_parts.append(f'<div class="list-container"><ul>{list_items}</ul></div>')
                            elif p.startswith("1. "):
                                # Convert to ordered list
                                items = p.split("\n")
                                list_items = ""
                                for item in items:
                                    if re.match(r"^\d+\.\s", item):
                                        item_text = item.split(". ", 1)[1]
                                        list_items += f"<li>{convert_links(item_text)}</li>"
                                html_parts.append(f'<div class="list-container"><ol>{list_items}</ol></div>')
                            elif p.strip():
                                if "**Source" in p:  # Handle inline source references
                                    html_parts.append(f'<div class="source-box"><p>{p}</p></div>')
                                else:
                                    html_parts.append(f'<div class="paragraph-container"><p>{p}</p></div>')
                        
                        html_body = "\n".join(html_parts)
                    
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {html_body}
</body>
</html>"""
                    
                    # Save HTML version
                    try:
                        os.makedirs(os.path.dirname(html_path), exist_ok=True)
                        with open(html_path, 'w', encoding='utf-8') as html_file:
                            html_file.write(html_content)
                        logger.info(f"Successfully generated and saved HTML file: {html_path}")
                    except Exception as write_error:
                        logger.error(f"Error writing HTML file: {write_error}")
                        return f"<html><body><h1>Error</h1><p>Could not write HTML file: {str(write_error)}</p></body></html>", 500
                except Exception as e:
                    logger.error(f"Error generating HTML file on-demand: {e}")
                    return f"<html><body><h1>Error</h1><p>Could not generate HTML for article ID {article_id}. Error: {str(e)}</p></body></html>", 500
            else:
                # Try to find content in the cache
                cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
                if not os.path.exists(cache_dir):
                    cache_dir = CACHE_DIR
                
                logger.info(f"Markdown not found, looking for content in cache at: {cache_dir}")
                cache_path = os.path.join(cache_dir, f"final_article_{article_id}.json")
                
                if os.path.exists(cache_path):
                    logger.info(f"Found cache file: {cache_path}")
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        content = data.get('content', '')
                        
                        if content:
                            logger.info(f"Successfully loaded content from cache with {len(content)} characters")
                            
                            # Extract title
                            title = content.splitlines()[0] if content else 'Unknown Title'
                            if title.startswith('# '):
                                title = title[2:]
                            
                            # Convert markdown to HTML
                            try:
                                # Try to use python-markdown if available
                                import markdown
                                html_body = markdown.markdown(content)
                                logger.info("Used python-markdown to convert content")
                            except ImportError:
                                # Fall back to simple replacement
                                logger.info("python-markdown not available, using simple replacement")
                                html_body = content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")
                            
                            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {html_body}
</body>
</html>"""
                            
                            # Save both markdown and HTML files for future use
                            try:
                                # Save markdown first
                                os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
                                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                                    md_file.write(content)
                                logger.info(f"Generated markdown file from cache: {markdown_path}")
                                
                                # Then save HTML
                                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                                with open(html_path, 'w', encoding='utf-8') as html_file:
                                    html_file.write(html_content)
                                logger.info(f"Generated HTML file from cache content: {html_path}")
                                
                                # Serve the HTML content
                                return html_content
                            except Exception as write_error:
                                logger.error(f"Error writing files from cache content: {write_error}")
                                return f"<html><body><h1>Error</h1><p>Could not write files: {str(write_error)}</p></body></html>", 500
                        else:
                            logger.error("Cache file has no content")
                    except Exception as e:
                        logger.error(f"Error processing cache file: {e}")
                else:
                    logger.error(f"No cache file found at: {cache_path}")
                
                return f"<html><body><h1>Article Not Found</h1><p>HTML or markdown not found for article ID {article_id}</p></body></html>", 404
        
        # If we get here, either the HTML file exists or we just created it
        if os.path.exists(html_path):
            logger.info(f"Serving existing HTML file: {html_path}")
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        else:
            logger.error(f"HTML file still not found: {html_path}")
            return f"<html><body><h1>Article Not Found</h1><p>HTML file could not be found or created for article ID {article_id}</p></body></html>", 404
            
    except Exception as e:
        logger.error(f"Exception serving HTML article {article_id}: {e}")
        logger.error(traceback.format_exc())
        return f"<html><body><h1>Error</h1><p>An error occurred: {str(e)}</p></body></html>", 500

def _update_progress(task_id, status_message, percentage, current_log_messages=None):
    with PROGRESS_LOCK:
        if task_id not in PROGRESS_STORE:
            PROGRESS_STORE[task_id] = {} # Should be initialized before worker starts
        
        PROGRESS_STORE[task_id]['status'] = status_message
        PROGRESS_STORE[task_id]['percentage'] = percentage
        PROGRESS_STORE[task_id]['timestamp'] = time.time()
        if current_log_messages is not None: # Optional: update full log if needed for polling
             PROGRESS_STORE[task_id]['log_snippet'] = current_log_messages[-3:] # Store last 3 messages as snippet

def _execute_best_article_match_task(task_id, user_interests):
    global ansys # Make sure we're using the global ansys, potentially imported dynamically
    scoring_log_messages = []
    current_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
    
    _update_progress(task_id, "Task started: Initializing...", 5, scoring_log_messages)

    # Print directly to terminal
    print("\n\033[1;36m===== ARTICLE MATCHING PROCESS =====\033[0m")
    print(f"\033[1;33mAnalyzing your interests: {user_interests}\033[0m\n")
    
    def print_progress_bar(percentage, status_message):
        bar_length = 50
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r\033[1;32m[{bar}] {percentage}%\033[0m \033[1;34m{status_message}\033[0m", end='', flush=True)
    
    def log_with_color(message, color_code="1;37"):
        """Print colored text to the terminal."""
        print(f"\033[{color_code}m{message}\033[0m")
    
    print_progress_bar(5, "Task started: Initializing...")

    try:
        scoring_log_messages.append(f"[{current_time()}] Worker task {task_id} started for interests: '{user_interests}'")
        
        if not user_interests:
            # This case should ideally be caught before starting the thread.
            # If it happens, update progress and log, then exit thread.
            msg = "ERROR: No interests provided to worker task."
            print(f"\n\033[1;31m{msg}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] {msg}")
            _update_progress(task_id, msg, 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "No interests provided"}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return

        _update_progress(task_id, "Loading articles from cache...", 10, scoring_log_messages)
        print_progress_bar(10, "Loading articles from cache...")
        
        cache_dir_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir_worker):
            cache_dir_worker = CACHE_DIR
            
        final_articles_paths = glob.glob(os.path.join(cache_dir_worker, 'final_article_*.json'))
        if len(final_articles_paths) == 0 and os.path.exists(CACHE_DIR):
            final_articles_paths = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
        
        scoring_log_messages.append(f"[{current_time()}] Found {len(final_articles_paths)} potential final_article_*.json files.")
        print(f"\n\033[1;36mFound {len(final_articles_paths)} articles in cache\033[0m")
            
        if not final_articles_paths:
            msg = "ERROR: No articles found in cache for worker task."
            print(f"\n\033[1;31m{msg}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] {msg}")
            _update_progress(task_id, msg, 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "No articles found in cache."}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
            
        articles_content_list = []
        for article_path in final_articles_paths:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                filename = os.path.basename(article_path)
                article_id = filename.replace('final_article_', '').replace('.json', '')
                content = data.get('content', '')
                title = content.splitlines()[0] if content else 'Unknown Title'
                if title.startswith('# '): title = title[2:]
                html_filename = f"tech_deep_dive_{article_id}.html"
                html_path_full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_articles', 'html', html_filename)
                has_html = os.path.exists(html_path_full)
                articles_content_list.append({
                    'id': article_id, 'title': title, 'content': content,
                    'html_path': html_path_full if has_html else None, 'has_html': has_html
                })
            except Exception as e:
                logger.error(f"Task {task_id}: Error loading article {article_path}: {e}")
                print(f"\033[1;31mError loading article {os.path.basename(article_path)}: {e}\033[0m")
                scoring_log_messages.append(f"[{current_time()}] WARNING: Error loading article {article_path}: {e}")
        
        _update_progress(task_id, f"Loaded {len(articles_content_list)} articles.", 20, scoring_log_messages)
        print_progress_bar(20, f"Loaded {len(articles_content_list)} articles.")
        scoring_log_messages.append(f"[{current_time()}] Successfully loaded {len(articles_content_list)} articles for scoring.")

        if not articles_content_list:
            msg = "ERROR: Could not load any article content after attempting to read files for worker task."
            print(f"\n\033[1;31m{msg}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] {msg}")
            _update_progress(task_id, msg, 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Could not load any article content."}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
        
        top_matches_list = []
        
        if ansys is None:
            scoring_log_messages.append(f"[{current_time()}] ansys module is None, attempting dynamic import in worker.")
            _update_progress(task_id, "Importing AI module...", 25, scoring_log_messages)
            print_progress_bar(25, "Importing AI module...")
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                ansys_path_worker = os.path.join(script_dir, 'ansys.py')
                if not os.path.exists(ansys_path_worker): ansys_path_worker = os.path.join(os.path.dirname(script_dir), 'ansys.py')
                
                if os.path.exists(ansys_path_worker):
                    scoring_log_messages.append(f"[{current_time()}] Found ansys.py at: {ansys_path_worker}, importing dynamically.")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("ansys", ansys_path_worker)
                    ansys_module_local = importlib.util.module_from_spec(spec) # Use a local name for the module in thread
                    spec.loader.exec_module(ansys_module_local)
                    ansys = ansys_module_local # Assign to global if successful, though direct use of ansys_module_local is safer in thread
                    scoring_log_messages.append(f"[{current_time()}] ansys module imported successfully in worker.")
                    _update_progress(task_id, "AI module imported.", 30, scoring_log_messages)
                    print_progress_bar(30, "AI module imported.")
                else:
                    msg = "ERROR: ansys.py not found in worker. LLM matching unavailable."
                    print(f"\n\033[1;31m{msg}\033[0m")
                    scoring_log_messages.append(f"[{current_time()}] {msg}")
                    _update_progress(task_id, msg, 100, scoring_log_messages)
                    with PROGRESS_LOCK:
                        PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Cannot score articles: ansys.py script not found."}
                        PROGRESS_STORE[task_id]['completed'] = True
                    _log_scoring_to_new_terminal(scoring_log_messages)
                    return
            except Exception as e:
                msg = f"ERROR: Error importing ansys module in worker: {e}"
                print(f"\n\033[1;31m{msg}\033[0m")
                scoring_log_messages.append(f"[{current_time()}] {msg}")
                logger.error(f"Task {task_id}: {msg}")
                logger.error(traceback.format_exc())
                _update_progress(task_id, msg, 100, scoring_log_messages)
                with PROGRESS_LOCK:
                    PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Cannot score articles: Error importing ansys module."}
                    PROGRESS_STORE[task_id]['completed'] = True
                _log_scoring_to_new_terminal(scoring_log_messages)
                return
        
        if ansys is None:
            msg = "ERROR: ansys module is still None after import attempt in worker."
            print(f"\n\033[1;31m{msg}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] {msg}")
            _update_progress(task_id, msg, 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Cannot score articles: ansys module is not available."}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
            
        # LLM scoring block
        try:
            scoring_log_messages.append(f"[{current_time()}] Attempting to initialize LLM model via ansys.get_llama_model() in worker.")
            _update_progress(task_id, "Initializing LLM model...", 35, scoring_log_messages)
            print_progress_bar(35, "Initializing LLM model...")
            print("\n\033[1;36mLoading LLM model...\033[0m")
            llm = ansys.get_llama_model() 
            
            if llm is None:
                msg = "CRITICAL ERROR: LLM model is None after ansys.get_llama_model() in worker."
                print(f"\n\033[1;31m{msg}\033[0m")
                scoring_log_messages.append(f"[{current_time()}] {msg}")
                _update_progress(task_id, msg, 100, scoring_log_messages)
                logger.error(f"Task {task_id}: {msg}")
                with PROGRESS_LOCK:
                    PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Cannot score articles: LLM model initialization failed (model is None)."}
                    PROGRESS_STORE[task_id]['completed'] = True
                _log_scoring_to_new_terminal(scoring_log_messages)
                return

            _update_progress(task_id, "LLM model initialized. Preparing prompt...", 40, scoring_log_messages)
            print_progress_bar(40, "LLM model initialized. Preparing prompt...")
            scoring_log_messages.append(f"[{current_time()}] LLM model initialized successfully in worker.")
            
            articles_detail_parts = []
            for i, article_data_item in enumerate(articles_content_list):
                title = article_data_item['title']
                individual_article_content_limit = 7500 
                summary_for_prompt = article_data_item['content']
                if len(summary_for_prompt) > individual_article_content_limit:
                    summary_for_prompt = summary_for_prompt[:individual_article_content_limit] + "... [truncated]"
                articles_detail_parts.append(f"Article {i+1}:\\nTitle: {title}\\nSummary:\\n{summary_for_prompt}\\n")
            
            articles_detail_text = "\\n\\n".join(articles_detail_parts)
            scoring_log_messages.append(f"[{current_time()}] Prepared details for {len(articles_content_list)} articles for the prompt.")
            _update_progress(task_id, "Prompt prepared. Sending to LLM...", 50, scoring_log_messages)
            print_progress_bar(50, "Prompt prepared. Sending to LLM...")
            
            prompt_text = "[INST] I need to find which article best matches the user's interests. "
            prompt_text += "Please review the following articles carefully. For each article, provide an alignment score from 0 to 100, "
            prompt_text += "where 100 means a perfect alignment with the user's stated interests, and 0 means no alignment.\\n\\n"
            prompt_text += f"User interests: {interests}\\n\\n"
            prompt_text += f"Available articles:\\n{articles_detail_text}\\n\\n"
            prompt_text += "Based on the user's interests and the provided titles and summaries, evaluate each article.\\n"
            prompt_text += "Format your response as a list of numbers only, one per article, corresponding to the order above.\\n"
            prompt_text += "Example:\\n"
            prompt_text += "1. [score for Article 1]\\n"
            prompt_text += "2. [score for Article 2]\\n...\\n\\n"
            prompt_text += "Provide only the scores. No other text or explanation is needed. [/INST]"
            
            log_with_color("Sending request to LLM for article scoring...", "1;36")
            
            print_progress_bar(60, "LLM processing articles...")
            response = llm(prompt_text, max_tokens=1024, temperature=0.5) 
            _update_progress(task_id, "LLM processing complete. Parsing scores...", 70, scoring_log_messages)
            print_progress_bar(70, "LLM processing complete. Parsing scores...")
            
            response_text_content = response["choices"][0]["text"].strip()
            scoring_log_messages.append(f"[{current_time()}] LLM response received (first 200 chars): {response_text_content[:200]}...")
            
            score_lines_list = response_text_content.split("\\n")
            print("\n\033[1;36mArticle scores:\033[0m")
            for i, line in enumerate(score_lines_list):
                if i < len(articles_content_list):
                    score_match = re.search(r'(\\d+)', line)
                    if score_match:
                        score = int(score_match.group(1))
                        articles_content_list[i]['match_score'] = score 
                        top_matches_list.append(articles_content_list[i])
                        scoring_log_messages.append(f"[{current_time()}] Article '{articles_content_list[i]['title']}': Assigned score {score} from line '{line}'.")
                        print(f"\033[1;37m{i+1}. '{articles_content_list[i]['title']}': \033[1;33m{score}/100\033[0m")
                    else:
                        articles_content_list[i]['match_score'] = 0 
                        top_matches_list.append(articles_content_list[i])
                        scoring_log_messages.append(f"[{current_time()}] WARNING: Couldn't extract score for '{articles_content_list[i]['title']}' from LLM line: '{line}'. Assigning score 0.")
                        print(f"\033[1;37m{i+1}. '{articles_content_list[i]['title']}': \033[1;31m0/100 (couldn't extract score)\033[0m")
                else:
                    scoring_log_messages.append(f"[{current_time()}] WARNING: More score lines from LLM ({len(score_lines_list)}) than articles ({len(articles_content_list)}). Ignoring extra line: '{line}'")

            _update_progress(task_id, "Scores parsed. Sorting articles...", 85, scoring_log_messages)
            print_progress_bar(85, "Scores parsed. Sorting articles...")
            scoring_log_messages.append(f"[{current_time()}] Scoring complete. Total articles processed with scores: {len(top_matches_list)}.")
            top_matches_list.sort(key=lambda x: x['match_score'], reverse=True)
            scoring_log_messages.append(f"[{current_time()}] Sorted articles by match_score.")
        
        except FileNotFoundError as model_error:
            error_message = f"LLM model file not found: {str(model_error)}"
            print(f"\n\033[1;31m{error_message}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] ERROR: {error_message}")
            logger.error(f"Task {task_id}: {error_message}")
            logger.error(traceback.format_exc())
            _update_progress(task_id, f"Error: {error_message}", 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {
                    "status": "error", "message": "Cannot score articles: LLM model file not found.",
                    "details": str(model_error),
                    "solution": "Please set the LLAMA_MODEL_PATH environment variable or place the model in a known location."}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
        
        except RuntimeError as rt_error:
            error_message = f"LLM runtime error: {str(rt_error)}"
            print(f"\n\033[1;31m{error_message}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] ERROR: {error_message}")
            logger.error(f"Task {task_id}: {error_message}")
            logger.error(traceback.format_exc())
            _update_progress(task_id, f"Error: {error_message}", 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Cannot score articles: LLM runtime error.", "details": str(rt_error)}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
            
        except Exception as e:
            error_message = f"An unexpected error occurred during LLM article scoring: {e}"
            print(f"\n\033[1;31m{error_message}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] ERROR: {error_message}")
            logger.error(f"Task {task_id}: Error using LLM for article matching: {e}")
            logger.error(traceback.format_exc())
            _update_progress(task_id, f"Error: {error_message}", 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "An unexpected error occurred during LLM article scoring.", "details": str(e)}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
        
        best_match_result = top_matches_list[0] if top_matches_list else None
        
        if not best_match_result:
            msg = "No best match found after scoring and sorting."
            print(f"\n\033[1;31m{msg}\033[0m")
            scoring_log_messages.append(f"[{current_time()}] {msg}")
            _update_progress(task_id, msg, 100, scoring_log_messages)
            with PROGRESS_LOCK:
                PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": "Could not find a suitable article match after LLM scoring."}
                PROGRESS_STORE[task_id]['completed'] = True
            _log_scoring_to_new_terminal(scoring_log_messages)
            return
        
        print_progress_bar(100, "Match found. Process complete.")
        scoring_log_messages.append(f"[{current_time()}] Top match selected: '{best_match_result['title']}' (Score: {best_match_result.get('match_score', 'N/A')}).")
        _update_progress(task_id, "Match found. Process complete.", 100, scoring_log_messages)
        
        print(f"\n\n\033[1;32m✓ BEST MATCH: '{best_match_result['title']}' (Score: {best_match_result.get('match_score', 'N/A')})\033[0m\n")
        
        # Get server URL for article HTML link
        try:
            # Get hostname and port from the environment or use default values
            # First try to detect actual server hostname and port
            from flask import request
            try:
                # Try to access the current request context to get actual server URL
                # This may fail if we're running in a background thread
                server_url = request.host_url.rstrip('/')
                logger.info(f"Got server URL from request context: {server_url}")
            except Exception:
                # Fall back to environment variables or defaults
                host = os.environ.get('SERVER_HOST', 'localhost')
                port = os.environ.get('SERVER_PORT', '5001')
                server_url = f"http://{host}:{port}"
                logger.info(f"Using fallback server URL: {server_url}")
            
            # Now create article URLs using the server URL
            article_url = f"{server_url}/article-html/{best_match_result['id']}"
            view_article_url = f"{server_url}/view-article/{task_id}"
            open_article_url = f"{server_url}/open-best-article-html/{task_id}"
            
            scoring_log_messages.append(f"[{current_time()}] Article URLs: article_url={article_url}, view_article_url={view_article_url}")
            log_with_color(f"View article directly at: {article_url}", "1;34")
            log_with_color(f"View article page at: {view_article_url}", "1;34")
            
            # Ensure HTML file exists
            html_filename = f"tech_deep_dive_{best_match_result['id']}.html"
            html_path = os.path.join(HTML_DIR, html_filename)
            logger.info(f"HTML file path: {html_path}, exists: {os.path.exists(html_path)}")
            
            if not os.path.exists(html_path):
                # Generate HTML content and save it
                markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{best_match_result['id']}.md")
                os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                
                # Save markdown content
                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(best_match_result['content'])
                logger.info(f"Saved markdown file: {markdown_path}")
                
                # Convert to HTML
                try:
                    # Try to use python-markdown if available
                    import markdown
                    html_body = markdown.markdown(best_match_result['content'])
                    logger.info("Used python-markdown to convert content")
                except ImportError:
                    # Fall back to simple replacement
                    logger.info("python-markdown not available, using simple replacement")
                    html_body = best_match_result['content'].replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")
                
                # Extract title
                title = best_match_result['title']
                
                # Generate HTML content
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {html_body}
</body>
</html>"""
                
                # Save HTML file
                with open(html_path, 'w', encoding='utf-8') as html_file:
                    html_file.write(html_content)
                logger.info(f"Generated HTML file: {html_path}")
                
                # Also try to access the article-html endpoint to ensure it's cached in the server
                try:
                    from urllib.request import urlopen
                    logger.info(f"Accessing {article_url} to trigger HTML generation")
                    response = urlopen(article_url)
                    html_content = response.read()
                    logger.info(f"Successfully triggered HTML generation, got {len(html_content)} bytes")
                except Exception as html_error:
                    logger.error(f"Error triggering HTML generation: {html_error}")
        except Exception as e:
            logger.error(f"Error generating article URL: {e}")
            article_url = None
            view_article_url = None
            open_article_url = None
        
        final_json_response = {
            "status": "success",
            "message": "Found best matching article",
            "article": {
                "id": best_match_result['id'], 
                "title": best_match_result['title'],
                "match_score": best_match_result.get('match_score', 0),
                "has_html": best_match_result['has_html'] or os.path.exists(html_path), 
                "html_path": best_match_result['html_path'] or html_path,
                "article_url": article_url,
                "view_url": view_article_url,
                "open_url": open_article_url
            }
        }
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id]['final_result'] = final_json_response
            PROGRESS_STORE[task_id]['completed'] = True
        _log_scoring_to_new_terminal(scoring_log_messages)

    except Exception as e:
        # General exception for the whole worker task
        critical_error_msg = f"CRITICAL ERROR in worker task {task_id}: {e}"
        print(f"\n\033[1;31m{critical_error_msg}\033[0m")
        scoring_log_messages.append(f"[{current_time()}] {critical_error_msg}")
        logger.error(critical_error_msg)
        logger.error(traceback.format_exc())
        _update_progress(task_id, f"Critical Error: {e}", 100, scoring_log_messages)
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id]['final_result'] = {"status": "error", "message": f"Critical error in worker: {str(e)}"}
            PROGRESS_STORE[task_id]['completed'] = True
        _log_scoring_to_new_terminal(scoring_log_messages)


@app.route('/get-best-article-match', methods=['POST'])
def get_best_article_match_start(): # Renamed to indicate it starts the task
    """
    Starts the background task for finding the best article match based on user interests.
    Returns a task_id for polling progress.
    """
    try:
        data = request.json
        user_interests = data.get('interests', '')

        if not user_interests:
            return jsonify({"status": "error", "message": "No interests provided"}), 400
        
        task_id = uuid.uuid4().hex
        
        with PROGRESS_LOCK:
            PROGRESS_STORE[task_id] = {
                'status': 'Task initiated. Waiting for worker to start.',
                'percentage': 0,
                'messages': [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id} created for interests: '{user_interests}'"],
                'completed': False,
                'final_result': None,
                'timestamp': time.time()
            }
        
        # For macOS, open a new terminal window to run the article matching process
        if sys.platform == 'darwin':
            # Create a temporary file with the user's interests
            input_file = os.path.join(tempfile.gettempdir(), f'article_match_interests_{task_id}.txt')
            with open(input_file, 'w') as f:
                f.write(f"{user_interests}\n")
                
            # Get the path to the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create a Python script to run the article matching in a terminal
            terminal_script = os.path.join(tempfile.gettempdir(), f'run_article_match_{task_id}.py')
            with open(terminal_script, 'w') as f:
                f.write('''
import sys
import os
import json
import time
import traceback
import glob
import re

# Define task_id at the global scope to avoid reference errors
task_id = ''' + f'"{task_id}"' + '''
input_file = ''' + f'"{input_file}"' + '''

try:
    # Get the script directory
    script_dir = ''' + f'"{script_dir}"' + '''
    sys.path.append(script_dir)
    sys.path.append(os.path.dirname(script_dir))

    # Import necessary modules
    import ansys
    
    def log_with_color(message, color_code="1;37"):
        """Print colored text to the terminal."""
        print(f"\\033[{color_code}m{message}\\033[0m")
    
    # Print header
    log_with_color("===== ARTICLE RATING PROCESS =====", "1;36")
    
    # Load interests
    with open(input_file, "r") as f:
        interests = f.read().strip()
    
    log_with_color(f"Analyzing your interests: {interests}", "1;33")
    
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
        
        # Return an error response
        result_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
        with open(result_file, "w") as f:
            json.dump({"status": "error", "message": "No articles found to rate. Please cache articles first."}, f)
            
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
    
    # Prepare articles with titles for the prompt
    article_details = []
    for i, article in enumerate(final_articles):
        title = article["title"]
        # Clean the title to remove "Deep Dive:" prefixes and be more readable
        clean_title = title
        if "Deep Dive:" in clean_title:
            clean_title = clean_title.replace("Deep Dive:", "").strip()
        
        # Handle the arrow format some titles have
        if "->" in clean_title:
            parts = clean_title.split("->")
            clean_title = parts[-1].strip()
            
        article_details.append({
            "number": i+1,
            "title": clean_title,
            "original_title": title,
            "original_index": i
        })
    
    # Create a clear and simple prompt
    prompt_text = f"""[INST]
I need you to score how well each article matches these interests:

USER INTERESTS: {interests}

For each article below, rate its relevance to the interests on a scale of 0-100.
Higher scores should indicate better matches to the user's interests.

ARTICLES TO SCORE:
"""

    # Add each article title
    for article in article_details:
        prompt_text += f"Article {article['number']}: {article['title']}\\n"
    
    prompt_text += """
Please score each article based on how relevant it would be to someone with the interests above.
Reply with ONLY scores in this exact format:
Article 1: [score]
Article 2: [score]
... and so on.

Use numbers between 10 and 95 for meaningful differentiation. Do not return all zeros.
[/INST]"""

    log_with_color(f"Prompt prepared with approximately {len(prompt_text) // 4} tokens", "1;36")
    log_with_color("Sending request to LLM for article scoring...", "1;36")
    
    # Attempt up to 3 times to get scores
    max_attempts = 3
    success = False
    response_text = ""
    article_scores = {}
    
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log_with_color(f"Retry attempt {attempt}/{max_attempts}...", "1;33")
            # For retries, simplify the prompt further
            prompt_text = f"""[INST]
Rate these articles for a user interested in: {interests}
Score each from 10-95 (not all zeros).

{', '.join([art['title'] for art in article_details])}

Reply ONLY with:
Article 1: [score]
Article 2: [score]
...and so on.
[/INST]"""
            
        try:
            # Score articles
            response = llm(prompt_text, max_tokens=1024, temperature=0.7)
            response_text = response["choices"][0]["text"].strip()
            
            # Log the raw response for debugging
            log_with_color(f"\\nRaw LLM response (attempt {attempt}):", "1;35")
            log_with_color(response_text, "1;37")
            
            # Parse scores using a simple pattern
            lines = response_text.split('\\n')
            article_scores = {}
            
            # Try to extract scores from each line
            for line in lines:
                match = re.search(r'Article (\d+): (\d+)', line)
                if match:
                    try:
                        article_num = int(match.group(1))
                        score = int(match.group(2))
                        if 1 <= article_num <= len(final_articles) and score > 0:
                            article_scores[article_num] = score
                    except:
                        pass
            
            # Check if we got any non-zero scores
            valid_scores = [score for score in article_scores.values() if score > 0]
            if valid_scores:
                success = True
                break
            else:
                log_with_color("All scores are zero or invalid. Retrying with a simplified prompt...", "1;33")
        except Exception as e:
            log_with_color(f"Error during LLM scoring (attempt {attempt}): {e}", "1;31")
    
    log_with_color("\\nParsing scores...", "1;36")
    
    issue_reasons = {}
    # Check for missing articles and note why they're missing
    for i in range(1, len(final_articles) + 1):
        if i not in article_scores:
            issue_reasons[i] = "No valid score found in LLM response"
            # Assign a default score so we have something rather than nothing
            article_scores[i] = 10  # Default low score
    
    # If we still don't have any scores, assign random differentiated scores
    # This is a fallback to avoid returning all zeros
    if not success or all(score == 0 for score in article_scores.values()):
        log_with_color("LLM failed to provide usable scores. Using fallback scoring method.", "1;33")
        # Assign scores based on keyword matching with user interests
        interest_keywords = [kw.strip().lower() for kw in interests.split(',')]
        
        for i, article in enumerate(final_articles):
            title = article["title"].lower()
            content_preview = article["content"][:1000].lower() if article["content"] else ""
            
            # Start with a base score
            score = 30
            
            # Add points for keyword matches in title (more heavily weighted)
            for keyword in interest_keywords:
                if keyword and len(keyword) > 2:  # Skip empty or very short keywords
                    if keyword in title:
                        score += 30
                    if keyword in content_preview:
                        score += 10
            
            # Cap at 95
            score = min(score, 95)
            
            # Ensure some minimum differentiation
            article_scores[i+1] = max(score, 15)
    
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
        log_with_color(f"\\n✓ BEST MATCH: '{best_match['title']}' (Score: {best_match.get('match_score', 'N/A')})", "1;32")
        
        # Calculate server URL details for links
        try:
            host = os.environ.get('SERVER_HOST', 'localhost')
            port = os.environ.get('SERVER_PORT', '5001')
            server_url = f"http://{host}:{port}"
            article_url = f"{server_url}/article-html/{best_match['id']}"
            view_article_url = f"{server_url}/view-article/{task_id}"
            
            log_with_color(f"\\nView article at: {article_url}", "1;34")
            log_with_color(f"View article page at: {view_article_url}", "1;34")
        except Exception as url_error:
            log_with_color(f"Error creating URLs: {url_error}", "1;31")
            article_url = None
            view_article_url = None
        
        # Save the result file
        result_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
        with open(result_file, "w") as f:
            json.dump({
                "status": "success", 
                "message": "Found best matching article", 
                "article": {
                    "id": best_match["id"], 
                    "title": best_match["title"], 
                    "match_score": best_match.get("match_score", 0),
                    "article_url": article_url,
                    "view_url": view_article_url
                }
            }, f)
        log_with_color("\\nResults saved. Generating HTML file...", "1;36")
        
        # Generate HTML file directly
        try:
            # Calculate paths
            project_dir = os.path.dirname(script_dir)
            markdown_dir = os.path.join(project_dir, "final_articles", "markdown")
            html_dir = os.path.join(project_dir, "final_articles", "html")
            
            # Ensure the directories exist
            os.makedirs(markdown_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)
            
            markdown_path = os.path.join(markdown_dir, f"tech_deep_dive_{best_match['id']}.md")
            html_path = os.path.join(html_dir, f"tech_deep_dive_{best_match['id']}.html")
            
            # Write markdown file
            with open(markdown_path, "w", encoding="utf-8") as md_file:
                md_file.write(best_match["content"])
            log_with_color(f"Saved markdown file: {markdown_path}", "1;32")
            
            # Generate HTML content
            title = best_match["title"]
            content = best_match["content"]
            
            # Convert markdown to HTML
            try:
                # Try to use python-markdown if available
                import markdown
                html_body = markdown.markdown(content)
                log_with_color("Used python-markdown to convert content", "1;32")
            except ImportError:
                # Fall back to simple replacement
                log_with_color("python-markdown not available, using simple replacement", "1;33")
                # Improved conversion with paragraph handling
                paragraphs = content.split("\\n\\n")
                html_parts = []
                for p in paragraphs:
                    if p.startswith("# "):
                        html_parts.append(f"<h1>{p[2:]}</h1>")
                    elif p.startswith("## "):
                        html_parts.append(f"<h2>{p[3:]}</h2>")
                    elif p.startswith("### "):
                        html_parts.append(f"<h3>{p[4:]}</h3>")
                    elif p.startswith("#### "):
                        html_parts.append(f"<h4>{p[5:]}</h4>")
                    elif p.startswith("- "):
                        # Convert to unordered list
                        items = p.split("\\n")
                        list_items = "".join([f"<li>{item[2:]}</li>" for item in items if item.startswith("- ")])
                        html_parts.append(f"<ul>{list_items}</ul>")
                    elif p.strip():
                        html_parts.append(f"<p>{p}</p>")
                
                html_body = "\\n".join(html_parts)
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {html_body}
</body>
</html>"""
            
            # Save HTML file
            with open(html_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_content)
            log_with_color(f"Saved HTML file: {html_path}", "1;32")
            
            # Try to open the article in browser
            try:
                import subprocess
                subprocess.run(['open', article_url])
                log_with_color("Opened article in browser", "1;32")
            except Exception as open_error:
                log_with_color(f"Could not open article in browser: {open_error}", "1;33")
                log_with_color(f"Please visit {article_url} to view the article", "1;34")
            
        except Exception as write_error:
            log_with_color(f"Error writing article files: {write_error}", "1;31")
        
        log_with_color("\\nYou can close this window and return to the app.", "1;36")
    else:
        # If we have no valid scores, report the issue
        log_with_color("\\nNO VALID SCORES FOUND. The LLM did not provide any usable article scores.", "1;31")
        log_with_color("This could be due to the complexity of the prompt or limitations of the model.", "1;33")
        log_with_color("Raw LLM response was:", "1;35")
        log_with_color(response_text, "1;37")
        
        # Return an error response
        result_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
        with open(result_file, "w") as f:
            json.dump({"status": "error", "message": "Could not get valid scores from LLM."}, f)
            
    # Keep terminal open for user to read
    log_with_color("\\nPress Enter to close this window...", "1;37")
    input()
    
except Exception as e:
    print(f"\\033[1;31mCritical error: {e}\\033[0m")
    print(traceback.format_exc())
    
    # Try to write failure result
    try:
        result_file = os.path.join(os.path.dirname(input_file), f"article_match_result_{task_id}.json")
        with open(result_file, "w") as f:
            json.dump({"status": "error", "message": f"Critical error: {str(e)}"}, f)
    except:
        pass
        
    print("\\nPress Enter to close this window...")
    input()
''')
            
            # Construct the command to run in Terminal
            terminal_cmd = f"{sys.executable} {terminal_script}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "{terminal_cmd}"
                set position of front window to {{100, 100}}
                set custom title of front window to "Article Rating for Task {task_id}"
            end tell
            '''
            
            # Run the AppleScript
            process = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            if process.returncode != 0:
                logger.error(f"Failed to open Terminal: {process.stderr}")
                
                # Fall back to background thread if terminal fails
                logger.info(f"Falling back to background task for {task_id}")
                thread = threading.Thread(target=_execute_best_article_match_task, args=(task_id, user_interests))
                thread.daemon = True
                thread.start()
            else:
                logger.info(f"Opened new Terminal window for article matching task {task_id}")
                
                # Start a monitoring thread to check for results
                def monitor_terminal_results():
                    result_file = os.path.join(tempfile.gettempdir(), f"article_match_result_{task_id}.json")
                    max_wait_time = 600  # 10 minutes max wait
                    start_time = time.time()
                    check_interval = 2  # Check every 2 seconds
                    
                    logger.info(f"Started monitoring thread for task {task_id} results at {result_file}")
                    
                    while time.time() - start_time < max_wait_time:
                        if os.path.exists(result_file):
                            try:
                                with open(result_file, 'r') as f:
                                    result = json.load(f)
                                
                                logger.info(f"Terminal process for task {task_id} completed with result: {result}")
                                
                                with PROGRESS_LOCK:
                                    PROGRESS_STORE[task_id]['final_result'] = result
                                    PROGRESS_STORE[task_id]['completed'] = True
                                    PROGRESS_STORE[task_id]['status'] = result.get('message', "Match found")
                                    PROGRESS_STORE[task_id]['percentage'] = 100
                                
                                # If result is successful, try to make sure the article's HTML is available
                                if result.get('status') == 'success' and result.get('article'):
                                    article = result.get('article')
                                    article_id = article.get('id')
                                    
                                    if article_id:
                                        html_filename = f"tech_deep_dive_{article_id}.html"
                                        html_path = os.path.join(HTML_DIR, html_filename)
                                        
                                        # If the HTML doesn't exist, try to load it from cache
                                        if not os.path.exists(html_path):
                                            logger.info(f"HTML file doesn't exist at {html_path}, attempting to generate it")
                                            
                                            # Try to access the article-html endpoint to generate it
                                            try:
                                                from urllib.request import urlopen
                                                article_url = f"http://localhost:5001/article-html/{article_id}"
                                                logger.info(f"Accessing {article_url} to trigger HTML generation")
                                                response = urlopen(article_url)
                                                html_content = response.read()
                                                logger.info(f"Successfully triggered HTML generation, got {len(html_content)} bytes")
                                            except Exception as html_error:
                                                logger.error(f"Error triggering HTML generation: {html_error}")
                                
                                # Clean up
                                try:
                                    os.remove(result_file)
                                    os.remove(input_file)
                                    os.remove(terminal_script)
                                except:
                                    pass
                                
                                return
                            except Exception as e:
                                logger.error(f"Error reading terminal results for task {task_id}: {e}")
                        
                        time.sleep(check_interval)
                    
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
                    
                    logger.warning(f"Monitoring thread for task {task_id} timed out after {max_wait_time} seconds")
                
                monitoring_thread = threading.Thread(target=monitor_terminal_results)
                monitoring_thread.daemon = True
                monitoring_thread.start()
        else:
            # For non-macOS platforms, run in the background thread as before
            logger.info(f"Starting background task {task_id} for interests: {user_interests}")
            thread = threading.Thread(target=_execute_best_article_match_task, args=(task_id, user_interests))
            thread.daemon = True
            thread.start()
        
        # Return a 202 Accepted response with a task_id for polling
        # Note: returning a message indicating this is a "process initiated" status rather than an "error"
        return jsonify({
            "task_id": task_id, 
            "message": "Article matching process initiated. Polling is required.",
            "status": "processing"
        }), 202 # 202 Accepted
        
    except Exception as e:
        logger.error(f"Error starting get-best-article-match task: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Exception starting task: {str(e)}"}), 500

@app.route('/get-match-progress/<task_id>', methods=['GET'])
def get_match_progress(task_id):
    """Gets the progress or result of a previously started article matching task."""
    with PROGRESS_LOCK:
        task_info = PROGRESS_STORE.get(task_id)

    if not task_info:
        return jsonify({"status": "error", "message": "Task ID not found or expired."}), 404
    
    # Optionally, clean up very old tasks if they are completed
    # For now, just return the current state.
    return jsonify(task_info)

@app.route('/open-best-article-html/<task_id>', methods=['GET'])
def open_best_article_html(task_id):
    """Open the best matched article in browser directly based on task_id"""
    try:
        with PROGRESS_LOCK:
            task_info = PROGRESS_STORE.get(task_id)
            
        if not task_info:
            return jsonify({"status": "error", "message": "Task ID not found or expired"}), 404
            
        if not task_info.get('completed', False):
            return jsonify({"status": "error", "message": "Task still in progress, please wait"}), 400
            
        final_result = task_info.get('final_result')
        if not final_result or final_result.get('status') != 'success':
            return jsonify({"status": "error", "message": "No successful result found for this task"}), 400
            
        article_data = final_result.get('article')
        if not article_data:
            return jsonify({"status": "error", "message": "No article data found in result"}), 400
            
        article_id = article_data.get('id')
        if not article_id:
            return jsonify({"status": "error", "message": "No article ID found in result"}), 400
            
        # Generate HTML file path
        html_filename = f"tech_deep_dive_{article_id}.html"
        html_path = os.path.join(HTML_DIR, html_filename)
        
        # Check if HTML exists, if not try to generate it
        if not os.path.exists(html_path):
            markdown_path = os.path.join(MARKDOWN_DIR, f"tech_deep_dive_{article_id}.md")
            
            # If no markdown file exists, try to generate it from the cached JSON
            if not os.path.exists(markdown_path):
                logger.info(f"No markdown file found at {markdown_path}, checking cache for content")
                cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
                if not os.path.exists(cache_dir):
                    cache_dir = CACHE_DIR
                
                cache_path = os.path.join(cache_dir, f"final_article_{article_id}.json")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        content = data.get('content', '')
                        
                        # Save markdown file
                        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
                        with open(markdown_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"Generated markdown file from cache: {markdown_path}")
                    except Exception as e:
                        logger.error(f"Error generating markdown file from cache: {e}")
                        return jsonify({"status": "error", "message": f"Error generating markdown file: {str(e)}"}), 500
            
            # If markdown file now exists, generate HTML
            if os.path.exists(markdown_path):
                try:
                    # Read markdown content
                    with open(markdown_path, 'r', encoding='utf-8') as md_file:
                        content = md_file.read()
                    
                    # Extract title
                    title = content.splitlines()[0] if content else 'Unknown Title'
                    if title.startswith('# '):
                        title = title[2:]
                    
                    # Generate HTML content
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/article.css">
</head>
<body>
    {content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>")}
</body>
</html>"""
                    
                    # Save HTML file
                    os.makedirs(os.path.dirname(html_path), exist_ok=True)
                    with open(html_path, 'w', encoding='utf-8') as html_file:
                        html_file.write(html_content)
                    logger.info(f"Generated HTML file for best match: {html_path}")
                except Exception as e:
                    logger.error(f"Error generating HTML file: {e}")
                    return jsonify({"status": "error", "message": f"Error generating HTML file: {str(e)}"}), 500
        
        # Now handle opening in browser (for macOS)
        if sys.platform == 'darwin':
            try:
                import subprocess
                # Use open command to open in default browser
                subprocess.run(['open', html_path])
                logger.info(f"Opened HTML file in browser: {html_path}")
                return jsonify({"status": "success", "message": "Opened best match article in browser", "html_path": html_path})
            except Exception as e:
                logger.error(f"Error opening file in browser: {e}")
                # Fall back to returning the URL
                server_url = request.host_url.rstrip('/')
                article_url = f"{server_url}/article-html/{article_id}"
                return jsonify({
                    "status": "warning", 
                    "message": f"Could not automatically open browser. Please open this URL: {article_url}",
                    "article_url": article_url
                })
        else:
            # For non-macOS, return the URL to access
            server_url = request.host_url.rstrip('/')
            article_url = f"{server_url}/article-html/{article_id}"
            return jsonify({
                "status": "success", 
                "message": "Use this URL to view the best match article",
                "article_url": article_url
            })
    
    except Exception as e:
        logger.error(f"Error opening best article HTML: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"}), 500

@app.route('/view-article/<task_id>', methods=['GET'])
def view_article_page(task_id):
    """Serve a simple HTML page with a button to view the best match article"""
    try:
        with PROGRESS_LOCK:
            task_info = PROGRESS_STORE.get(task_id)
        
        logger.info(f"Serving view-article page for task_id: {task_id}, task_info: {task_info is not None}")
            
        if not task_info:
            return "<html><body><h1>Error</h1><p>Task ID not found or expired</p></body></html>", 404
            
        if not task_info.get('completed', False):
            progress = task_info.get('percentage', 0)
            status = task_info.get('status', 'In progress')
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Article Matching in Progress</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f7f5f2;
                    }}
                    h1 {{
                        color: #34495e;
                        text-align: center;
                    }}
                    .progress-container {{
                        width: 100%;
                        background-color: #e8e8e8;
                        border-radius: 8px;
                        margin: 30px 0;
                        overflow: hidden;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        position: relative;
                    }}
                    .progress-bar {{
                        width: {progress}%;
                        height: 30px;
                        background: linear-gradient(90deg, #3c6382, #4a69bd);
                        text-align: center;
                        line-height: 30px;
                        color: white;
                        transition: width 1s ease;
                        border-radius: 8px;
                        box-shadow: 0 1px 5px rgba(0,0,0,0.1);
                        position: relative;
                        overflow: hidden;
                    }}
                    .progress-bar::after {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: linear-gradient(
                            45deg,
                            rgba(255, 255, 255, 0.2) 25%,
                            transparent 25%,
                            transparent 50%,
                            rgba(255, 255, 255, 0.2) 50%,
                            rgba(255, 255, 255, 0.2) 75%,
                            transparent 75%,
                            transparent
                        );
                        background-size: 50px 50px;
                        animation: progressAnimation 2s linear infinite;
                        border-radius: 8px;
                    }}
                    @keyframes progressAnimation {{
                        0% {{ background-position: 0 0; }}
                        100% {{ background-position: 50px 0; }}
                    }}
                    .status {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f0ece3;
                        border-left: 5px solid #e67e22;
                        border-radius: 0 8px 8px 0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .refresh-btn {{
                        background-color: #3c6382;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        text-align: center;
                        text-decoration: none;
                        display: block;
                        font-size: 16px;
                        margin: 20px auto;
                        cursor: pointer;
                        border-radius: 8px;
                        transition: all 0.3s ease;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .refresh-btn:hover {{
                        background-color: #4a69bd;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    }}
                </style>
                <script>
                    // Auto-refresh every 2 seconds
                    setTimeout(function() {{
                        window.location.reload();
                    }}, 2000);
                </script>
            </head>
            <body>
                <h1>Article Matching in Progress</h1>
                <div class="progress-container">
                    <div class="progress-bar">{progress}%</div>
                </div>
                <div class="status">{status}</div>
                <button class="refresh-btn" onclick="window.location.reload()">Refresh Status</button>
            </body>
            </html>
            """
        
        logger.info(f"Task {task_id} is completed, retrieving final result")    
        final_result = task_info.get('final_result')
        if not final_result or final_result.get('status') != 'success':
            error_msg = "An error occurred during article matching."
            if final_result and final_result.get('message'):
                error_msg = final_result.get('message')
                
            logger.error(f"Task {task_id} has no successful result: {final_result}")
                
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Article Matching Error</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f7f5f2;
                    }}
                    h1 {{
                        color: #e67e22;
                        text-align: center;
                    }}
                    .error-msg {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #fde9e0;
                        border-left: 5px solid #e67e22;
                        text-align: left;
                        border-radius: 0 8px 8px 0;
                    }}
                    pre {{
                        background-color: #f0ece3;
                        padding: 15px;
                        border-radius: 8px;
                        overflow-x: auto;
                        text-align: left;
                        font-size: 14px;
                        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .back-btn {{
                        background-color: #3c6382;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        text-align: center;
                        text-decoration: none;
                        display: block;
                        font-size: 16px;
                        margin: 20px auto;
                        cursor: pointer;
                        border-radius: 8px;
                        transition: background-color 0.3s ease;
                    }}
                    .back-btn:hover {{
                        background-color: #4a69bd;
                    }}
                </style>
            </head>
            <body>
                <h1>Article Matching Error</h1>
                <div class="error-msg">{error_msg}</div>
                <h2>Debugging Information</h2>
                <pre>{json.dumps(final_result, indent=2) if final_result else "No result data available"}</pre>
                <button class="back-btn" onclick="window.history.back()">Go Back</button>
            </body>
            </html>
            """
            
        article_data = final_result.get('article')
        if not article_data:
            logger.error(f"Task {task_id} has no article data in result: {final_result}")
            return "<html><body><h1>Error</h1><p>No article data found in result</p></body></html>", 400
            
        article_id = article_data.get('id')
        logger.info(f"Article ID from result: {article_id}")
        
        article_title = article_data.get('title', 'Unknown Title')
        article_score = article_data.get('match_score', 0)
        
        # Get URLs
        server_url = request.host_url.rstrip('/')
        logger.info(f"Server URL: {server_url}")
        
        article_url = article_data.get('article_url')
        if not article_url:
            article_url = f"{server_url}/article-html/{article_id}"
        logger.info(f"Article URL: {article_url}")
            
        # Check if HTML file exists and attempt to create it if not
        html_filename = f"tech_deep_dive_{article_id}.html"
        html_path = os.path.join(HTML_DIR, html_filename)
        logger.info(f"HTML file path: {html_path}, exists: {os.path.exists(html_path)}")
        
        if not os.path.exists(html_path):
            # Try to create it quickly
            logger.info(f"HTML file doesn't exist, trying to trigger creation via article-html endpoint")
            try:
                import urllib.request
                generation_url = f"{server_url}/article-html/{article_id}"
                logger.info(f"Requesting HTML generation from: {generation_url}")
                with urllib.request.urlopen(generation_url) as response:
                    html_content = response.read()
                    logger.info(f"Generated HTML content, {len(html_content)} bytes")
            except Exception as e:
                logger.error(f"Failed to trigger HTML generation: {e}")
        
        # Generate HTML page with button - add debug info
        debug_info = f"""
        <div style="border: 1px solid #ddd; margin-top: 20px; padding: 10px; text-align: left; background-color: #f0ece3; border-radius: 8px;">
            <h3>Debug Information:</h3>
            <ul>
                <li>Article ID: {article_id}</li>
                <li>HTML File: {html_path}</li>
                <li>HTML File Exists: {os.path.exists(html_path)}</li>
                <li>Generated URL: {article_url}</li>
            </ul>
        </div>
        """
            
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>View Best Match Article</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f7f5f2;
                }}
                h1 {{
                    color: #34495e;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .article-card {{
                    position: sticky;
                    top: 20px;
                    border: 1px solid #ddd;
                    border-radius: 12px;
                    padding: 25px;
                    margin: 0 0 30px 0;
                    background-color: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    text-align: left;
                    z-index: 100;
                    animation: popDown 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                }}
                .article-title {{
                    color: #3c6382;
                    font-size: 24px;
                    margin-bottom: 15px;
                    font-weight: 600;
                }}
                .match-score {{
                    display: inline-block;
                    background-color: {('#e67e22' if article_score >= 70 else '#f39c12' if article_score >= 40 else '#e74c3c')};
                    color: white;
                    padding: 8px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 24px;
                    margin: 10px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 16px;
                    text-align: center;
                    text-decoration: none;
                    border: none;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .primary-btn {{
                    background-color: #3c6382;
                    color: white;
                }}
                .primary-btn:hover {{
                    background-color: #4a69bd;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .secondary-btn {{
                    background-color: #e67e22;
                    color: white;
                }}
                .secondary-btn:hover {{
                    background-color: #f39c12;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .view-btn {{
                    margin-top: 20px;
                    padding: 15px 30px;
                    font-size: 18px;
                    display: block;
                    width: 80%;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .button-group {{
                    display: flex;
                    justify-content: center;
                    margin-top: 30px;
                }}
                @keyframes popDown {{
                    0% {{ transform: translateY(-50px); opacity: 0; }}
                    100% {{ transform: translateY(0); opacity: 1; }}
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
                .content-section {{
                    animation: fadeIn 0.8s ease-in-out;
                }}
            </style>
        </head>
        <body>
            <div class="article-card">
                <div class="article-title">{article_title}</div>
                <div class="match-score">Match Score: {article_score}%</div>
                <p>This article best matches your interests based on our AI analysis.</p>
                <a href="{article_url}" class="btn primary-btn view-btn" target="_blank">View Full Article</a>
            </div>
            
            <div class="content-section">
                <h1>Best Match Article Found</h1>
                
                <div class="button-group">
                    <a href="{server_url}/open-best-article-html/{task_id}" class="btn secondary-btn">Open in Browser</a>
                    <a href="javascript:history.back()" class="btn secondary-btn">Go Back</a>
                </div>
                
                {debug_info}
            </div>
        </body>
        </html>
        """
            
    except Exception as e:
        logger.error(f"Error generating view article page: {e}")
        logger.error(traceback.format_exc())
        return f"""
        <html>
        <body>
            <h1>Error</h1>
            <p>An error occurred: {str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
        </body>
        </html>
        """, 500

@app.route('/direct-article/<task_id>', methods=['GET'])
def direct_article_redirect(task_id):
    """Redirects directly to the best matched article HTML without displaying a page with buttons"""
    try:
        with PROGRESS_LOCK:
            task_info = PROGRESS_STORE.get(task_id)
            
        logger.info(f"Direct article redirect for task: {task_id}, task_info exists: {task_info is not None}")
            
        if not task_info:
            return "<html><body><h1>Error</h1><p>Task ID not found or expired</p></body></html>", 404
            
        if not task_info.get('completed', False):
            progress = task_info.get('percentage', 0)
            status = task_info.get('status', 'In progress')
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Article Matching in Progress</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f7f5f2;
                    }}
                    h1 {{
                        color: #34495e;
                        text-align: center;
                    }}
                    .progress-container {{
                        width: 100%;
                        background-color: #e8e8e8;
                        border-radius: 8px;
                        margin: 30px 0;
                        overflow: hidden;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .progress-bar {{
                        width: {progress}%;
                        height: 30px;
                        background: linear-gradient(90deg, #3c6382, #4a69bd);
                        text-align: center;
                        line-height: 30px;
                        color: white;
                        transition: width 0.5s ease;
                        border-radius: 8px;
                    }}
                    .status {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f0ece3;
                        border-left: 5px solid #e67e22;
                        border-radius: 0 8px 8px 0;
                    }}
                    .refresh-btn {{
                        background-color: #3c6382;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        text-align: center;
                        text-decoration: none;
                        display: block;
                        font-size: 16px;
                        margin: 20px auto;
                        cursor: pointer;
                        border-radius: 8px;
                        transition: background-color 0.3s ease;
                    }}
                    .refresh-btn:hover {{
                        background-color: #4a69bd;
                    }}
                </style>
                <script>
                    // Auto-refresh every 2 seconds
                    setTimeout(function() {{
                        window.location.reload();
                    }}, 2000);
                </script>
            </head>
            <body>
                <h1>Article Matching in Progress</h1>
                <div class="progress-container">
                    <div class="progress-bar">{progress}%</div>
                </div>
                <div class="status">{status}</div>
                <p>You will be automatically redirected to the article when ready.</p>
                <button class="refresh-btn" onclick="window.location.reload()">Refresh Status</button>
            </body>
            </html>
            """
            
        final_result = task_info.get('final_result')
        if not final_result or final_result.get('status') != 'success':
            error_msg = "An error occurred during article matching."
            if final_result and final_result.get('message'):
                error_msg = final_result.get('message')
                
            logger.error(f"Task {task_id} has no successful result: {final_result}")
                
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Article Matching Error</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f7f5f2;
                    }}
                    h1 {{
                        color: #e67e22;
                        text-align: center;
                    }}
                    .error-msg {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #fde9e0;
                        border-left: 5px solid #e67e22;
                        text-align: left;
                        border-radius: 0 8px 8px 0;
                    }}
                    pre {{
                        background-color: #f0ece3;
                        padding: 15px;
                        border-radius: 8px;
                        overflow-x: auto;
                        text-align: left;
                        font-size: 14px;
                        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .back-btn {{
                        background-color: #3c6382;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        text-align: center;
                        text-decoration: none;
                        display: block;
                        font-size: 16px;
                        margin: 20px auto;
                        cursor: pointer;
                        border-radius: 8px;
                        transition: background-color 0.3s ease;
                    }}
                    .back-btn:hover {{
                        background-color: #4a69bd;
                    }}
                </style>
            </head>
            <body>
                <h1>Article Matching Error</h1>
                <div class="error-msg">{error_msg}</div>
                <h2>Debugging Information</h2>
                <pre>{json.dumps(final_result, indent=2) if final_result else "No result data available"}</pre>
                <button class="back-btn" onclick="window.history.back()">Go Back</button>
            </body>
            </html>
            """
            
        article_data = final_result.get('article')
        if not article_data:
            logger.error(f"Task {task_id} has no article data in result: {final_result}")
            return "<html><body><h1>Error</h1><p>No article data found in result</p></body></html>", 400
            
        article_id = article_data.get('id')
        logger.info(f"Article ID from result: {article_id}")
        
        if not article_id:
            logger.error(f"No article ID found in result: {article_data}")
            return "<html><body><h1>Error</h1><p>No article ID found in result</p></body></html>", 400
        
        # Get the article URL
        server_url = request.host_url.rstrip('/')
        logger.info(f"Server URL: {server_url}")
        
        article_url = article_data.get('article_url')
        if not article_url:
            article_url = f"{server_url}/article-html/{article_id}"
        
        logger.info(f"Redirecting to article URL: {article_url}")
        
        # Instead of returning a page with buttons, redirect directly to the article
        # Use a meta refresh tag to do the redirect
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="0;url={article_url}">
            <title>Redirecting to article...</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f7f5f2;
                    text-align: center;
                }}
                h1 {{
                    color: #34495e;
                    text-align: center;
                }}
                p {{
                    margin: 20px 0;
                }}
                a {{
                    color: #3c6382;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                    color: #e67e22;
                }}
            </style>
            <script>
                // Redirect immediately
                window.location.href = "{article_url}";
            </script>
        </head>
        <body>
            <h1>Redirecting...</h1>
            <p>If you are not automatically redirected, <a href="{article_url}">click here</a> to view the article.</p>
        </body>
        </html>
        """
            
    except Exception as e:
        logger.error(f"Error in direct article redirect: {e}")
        logger.error(traceback.format_exc())
        return f"<html><body><h1>Error</h1><p>An error occurred: {str(e)}</p></body></html>", 500

if __name__ == '__main__':
    try:
        # Sync article files on startup
        logger.info("Syncing article files on startup...")
        try:
            copy_articles_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'copy_articles.py')
            if os.path.exists(copy_articles_script):
                subprocess.run([sys.executable, copy_articles_script], check=True)
                logger.info("Article files sync completed successfully")
            else:
                logger.warning(f"Could not find copy_articles.py script at {copy_articles_script}")
        except Exception as e:
            logger.error(f"Error syncing article files: {e}")
            logger.error(traceback.format_exc())
            
        logger.info("Starting server (Flask development server)...")
        app.run(debug=True, port=5001, host='0.0.0.0') # debug=True can cause threads to run twice in some cases, be mindful
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 