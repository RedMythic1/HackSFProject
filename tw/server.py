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
        
        # Count all final_article_*.json files
        final_articles = glob.glob(os.path.join(cache_dir, 'final_article_*.json'))
        final_article_count = len(final_articles)
        
        # Check parent directory if no files found
        if final_article_count == 0 and os.path.exists(CACHE_DIR):
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
            final_article_count = len(final_articles)
        
        # Count unique articles based on title
        unique_titles = set()
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
                except Exception as e:
                    logger.error(f"Error reading article {article_path}: {e}")
        
        unique_article_count = len(unique_titles)
        
        logger.info(f"Found {article_count} cached article summaries and {final_article_count} cached final articles ({unique_article_count} unique)")
        
        return jsonify({
            "status": "success",
            "message": "Cache check successful",
            "cached": article_count > 0,
            "article_count": article_count,
            "final_article_count": unique_article_count  # Return unique count
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
        
        logger.info(f"Found {len(article_data)} cached final articles, {len(unique_articles)} unique")
        
        return jsonify({
            "status": "success",
            "message": "Final articles retrieved successfully",
            "articles": unique_articles
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
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal
            terminal_cmd = f"{sys.executable} {ansys_path} --cache-only"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {script_dir} && echo 'Running: {terminal_cmd}' && {terminal_cmd}"
                set position of front window to {{100, 100}}
                set custom title of front window to "ANSYS Article Caching"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info("Opened new Terminal window to run the command")
            
            return jsonify({
                "status": "success",
                "message": "Started caching articles in a new terminal window. Please check the terminal for progress."
            })
        else:
            # For non-macOS platforms, run in the background as before
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
        
        # Create a temporary file with predefined inputs to feed to ansys.py
        input_file = os.path.join(tempfile.gettempdir(), 'ansys_input.txt')
        with open(input_file, 'w') as f:
            # Add a broad range of interests to ensure most articles are processed
            f.write("technology, programming, science, AI, finance, health, politics, education\n")
        
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal
            terminal_cmd = f"cat {input_file} | {sys.executable} {ansys_path}"
            
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
                "message": "Started full article processing in a new terminal window. Please check the terminal for progress."
            })
        else:
            # For non-macOS platforms, run in the background as before
            # Run ansys.py with full processing in a separate thread
            def run_ansys_full_processing():
                try:
                    logger.info(f"Running ansys.py with predefined interests for full processing")
                    
                    # Create a command that feeds the interests to ansys.py
                    cmd = f'cat {input_file} | {sys.executable} {ansys_path}'
                    logger.info(f"Executing: {cmd}")
                    
                    # Use shell=True to allow piping
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
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
                            for article_file in article_files:
                                # Read the content
                                with open(article_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    
                                # Extract the timestamp from the filename
                                timestamp = article_file.replace('tech_deep_dive_', '').replace('.md', '')
                                
                                # Create a cache path for this final article
                                cache_path = os.path.join(CACHE_DIR, f"final_article_{timestamp}.json")
                                
                                # Cache the content
                                try:
                                    with open(cache_path, 'w', encoding='utf-8') as f:
                                        json.dump({
                                            'content': content,
                                            'timestamp': int(time.time())
                                        }, f)
                                    logger.info(f"Cached final article {article_file} to {cache_path}")
                                except Exception as e:
                                    logger.error(f"Error caching final article {article_file}: {e}")
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
                "message": "Started full article processing with question and answer generation in the background."
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
            
        # Create a temporary file with the user's interests
        input_file = os.path.join(tempfile.gettempdir(), f'ansys_input_{email.replace("@", "_at_")}.txt')
        with open(input_file, 'w') as f:
            f.write(f"{interests}\n")
            
        # On macOS, open a new terminal window to run the command
        if sys.platform == 'darwin':
            # Construct the command to run in Terminal
            terminal_cmd = f"cat {input_file} | {sys.executable} {ansys_path}"
            
            # Create a unique title with the user's email
            terminal_title = f"ANSYS Analysis for {email}"
            
            # Create AppleScript to open new Terminal window
            apple_script = f'''
            tell application "Terminal"
                do script "cd {script_dir} && echo 'Running analysis for: {email}' && echo 'Interests: {interests}' && {terminal_cmd}"
                set position of front window to {{100, 100}}
                set custom title of front window to "{terminal_title}"
            end tell
            '''
            
            # Run the AppleScript
            subprocess.run(['osascript', '-e', apple_script])
            logger.info(f"Opened new Terminal window to run analysis for {email}")
            
            return jsonify({
                "status": "success",
                "message": "Started analysis in a new terminal window. Please check the terminal for progress.",
                "user_email": email
            })
        else:
            # For non-macOS platforms, just acknowledge the request
            # The actual processing would be done separately
            return jsonify({
                "status": "success",
                "message": "Request received and saved. Analysis will be performed in the background.",
                "user_email": email
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

@app.route('/get-best-article-match', methods=['POST'])
def get_best_article_match():
    """Find the best article match based on user interests"""
    try:
        data = request.json
        user_interests = data.get('interests', '')
        
        if not user_interests:
            return jsonify({
                "status": "error",
                "message": "No interests provided"
            }), 400
        
        logger.info(f"Finding best article match for interests: {user_interests}")
        
        # Get all the article files
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(cache_dir):
            cache_dir = CACHE_DIR
            
        # Find all final article files
        final_articles = glob.glob(os.path.join(cache_dir, 'final_article_*.json'))
        
        # Check parent directory if no files found
        if len(final_articles) == 0 and os.path.exists(CACHE_DIR):
            final_articles = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
            
        if not final_articles:
            return jsonify({
                "status": "error", 
                "message": "No articles found in cache."
            }), 404
            
        # Load the content of all articles
        articles_content = []
        
        for article_path in final_articles:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract filename and ID
                filename = os.path.basename(article_path)
                article_id = filename.replace('final_article_', '').replace('.json', '')
                
                # Get the content
                content = data.get('content', '')
                title = content.splitlines()[0] if content else 'Unknown Title'
                if title.startswith('# '):
                    title = title[2:]  # Remove Markdown heading marker
                    
                # Find the corresponding HTML file
                html_filename = f"tech_deep_dive_{article_id}.html"
                html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'final_articles', 'html', html_filename)
                
                has_html = os.path.exists(html_path)
                
                articles_content.append({
                    'id': article_id,
                    'title': title,
                    'content': content,
                    'html_path': html_path if has_html else None,
                    'has_html': has_html
                })
                
            except Exception as e:
                logger.error(f"Error loading article {article_path}: {e}")
                
        if not articles_content:
            return jsonify({
                "status": "error",
                "message": "Could not load any article content."
            }), 500
            
        # Use LLM to determine which article best matches the user's interests
        # This needs to be done in chunks due to potential token limits
        
        # Split articles into manageable chunks if there are many
        chunk_size = 5  # Process 5 articles at a time
        article_chunks = [articles_content[i:i + chunk_size] for i in range(0, len(articles_content), chunk_size)]
        
        top_matches = []
        
        # Get the path to ansys.py to use its Llama model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ansys_path = os.path.join(script_dir, 'ansys.py')
        
        # If not found in the current directory, check the parent
        if not os.path.exists(ansys_path):
            ansys_path = os.path.join(os.path.dirname(script_dir), 'ansys.py')
            
        if not os.path.exists(ansys_path):
            logger.error("ansys.py not found for LLM processing. Using basic keyword matching instead.")
            
            # Fall back to simple keyword matching if LLM not available
            for article in articles_content:
                match_score = 0
                interest_terms = [term.strip().lower() for term in user_interests.split(',')]
                
                content_lower = article['content'].lower()
                title_lower = article['title'].lower()
                
                for term in interest_terms:
                    if term in title_lower:
                        match_score += 10  # Higher weight for title matches
                    if term in content_lower:
                        match_score += 5   # Lower weight for content matches
                        
                article['match_score'] = match_score
                top_matches.append(article)
                
            # Sort by match score, highest first
            top_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
        else:
            # Use the Llama model to analyze matches
            try:
                # Import the LLM functionality from ansys.py
                import sys
                sys.path.append(os.path.dirname(ansys_path))
                import importlib.util
                spec = importlib.util.spec_from_file_location("ansys", ansys_path)
                ansys = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ansys)
                
                # Get the Llama model
                llm = ansys.get_llama_model()
                
                for chunk in article_chunks:
                    chunk_titles = [f"{i+1}. {a['title']}" for i, a in enumerate(chunk)]
                    titles_text = "\n".join(chunk_titles)
                    
                    prompt = f"""[INST] I need to find which article best matches the user's interests.

User interests: {user_interests}

Available articles:
{titles_text}

For each article, provide an alignment score from 0-100 where 100 means perfect alignment with the user's interests.
Format your response as a list of numbers only, one per article:
1. [score]
2. [score]
...

Just provide the scores, no explanation needed.
[/INST]"""

                    with ansys.LLM_LOCK:
                        response = llm(prompt, max_tokens=256, temperature=0.1)
                    
                    response_text = response["choices"][0]["text"].strip()
                    
                    # Parse scores
                    score_lines = response_text.split('\n')
                    for i, line in enumerate(score_lines):
                        if i < len(chunk):  # Ensure we don't go out of bounds
                            # Extract the score using regex
                            import re
                            score_match = re.search(r'(\d+)', line)
                            if score_match:
                                score = int(score_match.group(1))
                                chunk[i]['match_score'] = score
                                top_matches.append(chunk[i])
                            else:
                                # If score not found, give a default score
                                chunk[i]['match_score'] = 0
                                top_matches.append(chunk[i])
                
                # Sort by match score, highest first
                top_matches.sort(key=lambda x: x['match_score'], reverse=True)
                
            except Exception as e:
                logger.error(f"Error using LLM for article matching: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "status": "error",
                    "message": f"Error matching articles with interests: {str(e)}"
                }), 500
        
        # Select the best match
        best_match = top_matches[0] if top_matches else None
        
        if not best_match:
            return jsonify({
                "status": "error",
                "message": "Could not find a suitable article match."
            }), 404
            
        # Return the best match, including HTML path if available
        return jsonify({
            "status": "success",
            "message": "Found best matching article",
            "article": {
                "id": best_match['id'],
                "title": best_match['title'],
                "match_score": best_match.get('match_score', 0),
                "has_html": best_match['has_html'],
                "html_path": best_match['html_path']
            }
        })
        
    except Exception as e:
        logger.error(f"Exception in get_best_article_match: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Exception: {str(e)}"
        }), 500

if __name__ == '__main__':
    try:
        logger.info("Starting server (simplified version - skipping cache initialization)...")
        app.run(debug=True, port=5001, host='0.0.0.0')
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 