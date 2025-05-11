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
import argparse
import math

# Set up logging to file only (avoid console output which causes I/O errors)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
    ]
)
logger = logging.getLogger(__name__)

# Set up frontend logging
frontend_logger = logging.getLogger('frontend')
frontend_logger.setLevel(logging.DEBUG)
frontend_handler = logging.FileHandler('frontend.log')
frontend_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
frontend_logger.addHandler(frontend_handler)
frontend_logger.propagate = False  # Don't propagate to root logger

# Constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')
FINAL_ARTICLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_articles')
HTML_DIR = os.path.join(FINAL_ARTICLES_DIR, 'html')
# Add constant for the processed articles tracking file
PROCESSED_ARTICLES_FILE = os.path.join(CACHE_DIR, 'processed_articles.json')
# Add local cache directory
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_cache')

# Ensure directories exist
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
    logger.info(f"Final articles directory created: {HTML_DIR}")
    logger.info(f"Local cache directory set to: {LOCAL_CACHE_DIR}")
except Exception as e:
    logger.error(f"Error creating directories: {e}")

app = Flask(__name__)
# Enable CORS with support for credentials
CORS(app, supports_credentials=True)  # Add supports_credentials=True
app.secret_key = os.urandom(24)  # Add secret key for cookie signing

# --- Progress Tracking Globals ---
PROGRESS_STORE = {}
PROGRESS_LOCK = threading.Lock()
# --- End Progress Tracking Globals ---

# Add a new logging endpoint for frontend to send logs
@app.route('/log', methods=['POST'])
def log_message():
    """Handle log messages from frontend"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"status": "error", "message": "Missing log message"}), 400
            
        message = data.get('message', '')
        level = data.get('level', 'info').lower()
        source = data.get('source', 'unknown')
        
        # Add source to the message
        full_message = f"[{source}] {message}"
        
        # Log at appropriate level
        if level == 'error':
            frontend_logger.error(full_message)
        elif level == 'debug':
            frontend_logger.debug(full_message)
        else:
            frontend_logger.info(full_message)
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error handling log message: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

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
    logger.info("Received request to check cache")
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
        
        logger.info(f"Found {article_count} summary files and {final_article_count} final article files")
        
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
                            logger.info(f"Creating missing HTML file for article: {title}")
                            # Generate HTML would go here (code not shown for brevity)
                except Exception as e:
                    logger.error(f"Error processing article file {article_path}: {e}")
        
        logger.info(f"Response: article_count={article_count}, final_article_count={final_article_count}, valid_article_count={valid_article_count}")
        return jsonify({
            "status": "success",
            "cached": article_count > 0 or final_article_count > 0,
            "article_count": article_count,
            "final_article_count": final_article_count,
            "valid_article_count": valid_article_count,
            "unique_titles": len(unique_titles)
        })
    except Exception as e:
        logger.error(f"Error checking cache: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error checking cache: {str(e)}"
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

@app.route('/get-final-article/<article_id>', methods=['GET'])
def get_final_article(article_id):
    """Get a specific final article"""
    logger.info(f"Request for article with ID: {article_id}")
    try:
        # Sanitize article_id to ensure it doesn't contain path traversal
        if not re.match(r'^[a-zA-Z0-9_]+$', article_id):
            logger.error(f"Invalid article ID format: {article_id}")
            return jsonify({
                "status": "error",
                "message": "Invalid article ID format"
            }), 400
        
        # Lookup the article in the cache
        article_path = os.path.join(CACHE_DIR, f"final_article_{article_id}.json")
        
        if not os.path.exists(article_path):
            logger.error(f"Article not found: {article_id}")
            return jsonify({
                "status": "error",
                "message": "Article not found"
            }), 404
        
        with open(article_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        
        # Extract title and content
        content = article_data.get('content', '')
        title = 'Unknown Title'
        
        # Extract title from content
        if content:
            lines = content.split('\n')
            if lines and lines[0].startswith('# '):
                title = lines[0][2:].strip()
        
        # Find embedding if available
        embedding = None
        try:
            # Create a hash of the article content to look up potential embedding
            content_hash = hashlib.md5(content.encode()).hexdigest()
            embedding_paths = glob.glob(os.path.join(CACHE_DIR, f"summary_{content_hash}*.json"))
            
            # If no exact hash match found, try all summary files
            if not embedding_paths:
                summary_files = glob.glob(os.path.join(CACHE_DIR, "summary_*.json"))
                logger.info(f"Checking {len(summary_files)} summary files for embedding match")
                
                # Try to find a matching summary by comparing content
                for summary_path in summary_files:
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            summary_data = json.load(f)
                            # If the summary contains an embedding and matches our title
                            summary_title = summary_data.get('title', '')
                            if 'embedding' in summary_data and summary_title.lower() == title.lower():
                                embedding = summary_data.get('embedding')
                                logger.info(f"Found embedding for article by title match: {title}")
                                break
                    except Exception as e:
                        logger.error(f"Error reading summary file {summary_path}: {e}")
                        continue
            else:
                # Use the first matching embedding file
                try:
                    with open(embedding_paths[0], 'r', encoding='utf-8') as f:
                        embedding_data = json.load(f)
                        embedding = embedding_data.get('embedding')
                        logger.info(f"Found direct embedding match for article: {title}")
                except Exception as e:
                    logger.error(f"Error reading embedding file: {e}")
        except Exception as e:
            logger.error(f"Error finding embedding: {e}")
        
        # Return the article data
        response_data = {
            "status": "success",
            "article": {
                "id": article_id,
                "title": title,
                "content": content,
            }
        }
        
        # Add embedding if available
        if embedding:
            response_data["article"]["embedding"] = embedding
            logger.info(f"Returning article with {len(embedding)} dimensional embedding")
        else:
            logger.info("No embedding available for this article")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error retrieving article: {str(e)}"
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

@app.route('/verify-email', methods=['POST'])
def verify_email():
    """Verify email and set cookie"""
    try:
        logger.info("/verify-email endpoint called")
        data = request.get_json()
        logger.info(f"/verify-email received data: {data}")
        email = data.get('email', '').strip() if data else ''
        
        if not email:
            logger.warning("Email verification failed: No email provided")
            return jsonify({
                "status": "error",
                "message": "Email is required"
            }), 400
            
        # Improved email validation: require at least 2 characters for TLD
        email_regex = r'^[^\s@]+@[^\s@]+\.[a-zA-Z0-9]{2,}$'
        if not re.match(email_regex, email):
            logger.warning(f"Email verification failed: Invalid email format: {email}")
            return jsonify({
                "status": "error",
                "message": "Invalid email format. Please use a valid email address (e.g. user@example.com)."
            }), 400
            
        logger.info(f"Setting cookie for email: {email}")
        
        # Create response with success message
        response = make_response(jsonify({
            "status": "success",
            "message": "Email verified successfully"
        }))
        
        # Set cookie that expires in 30 days
        is_local = request.host.startswith('localhost') or request.host.startswith('127.0.0.1')
        
        # Note: Setting secure=False for local development (even on HTTPS) to ensure the cookie works
        response.set_cookie(
            'verified_email',
            email,
            max_age=30*24*60*60,  # 30 days in seconds
            httponly=False,  # Allow JavaScript access for debugging
            secure=False,    # Don't require HTTPS for local development
            samesite='Lax'   # Protect against CSRF
        )
        
        logger.info(f"Cookie set successfully for email: {email}")
        return response
        
    except Exception as e:
        logger.error(f"Error verifying email: {e}")
        logger.error(traceback.format_exc())
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

@app.route('/copy-article-files', methods=['POST'])
def copy_article_files():
    """
    Copy article files from a source .cache directory to the application's .cache directory
    """
    try:
        # Get source directory from request (default to parent .cache if not specified)
        data = request.get_json()
        source_dir = data.get('source_dir', None)
        
        # If no source directory specified, check common locations
        if not source_dir:
            # Check parent directory's .cache
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_source_dirs = [
                os.path.join(parent_dir, '.cache'),
                os.path.expanduser('~/.cache/ansys'),
                os.path.expanduser('~/Code/HackSFProject/.cache')
            ]
            
            for dir_path in possible_source_dirs:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    source_dir = dir_path
                    break
        
        if not source_dir or not os.path.exists(source_dir):
            return jsonify({
                "status": "error",
                "message": "Source directory not found"
            }), 404
            
        # Ensure target directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Find all article files in source directory
        article_files = []
        for pattern in ['final_article_*.json', 'summary_*.json']:
            article_files.extend(glob.glob(os.path.join(source_dir, pattern)))
        
        if not article_files:
            return jsonify({
                "status": "success",
                "message": "No article files found in source directory",
                "copied_count": 0
            })
        
        # Copy files to target directory
        copied_count = 0
        skipped_count = 0
        
        for src_path in article_files:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(CACHE_DIR, filename)
            
            # Only copy if file doesn't exist in target directory or is newer
            if not os.path.exists(dest_path) or os.path.getmtime(src_path) > os.path.getmtime(dest_path):
                try:
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    logger.error(f"Error copying file {src_path} to {dest_path}: {e}")
            else:
                skipped_count += 1
        
        logger.info(f"Copied {copied_count} article files from {source_dir} to {CACHE_DIR}, skipped {skipped_count} existing/older files")
        
        return jsonify({
            "status": "success",
            "message": f"Copied {copied_count} article files, skipped {skipped_count} files",
            "copied_count": copied_count,
            "skipped_count": skipped_count,
            "source_dir": source_dir,
            "target_dir": CACHE_DIR
        })
        
    except Exception as e:
        logger.error(f"Exception in copy_article_files: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Exception: {str(e)}"
        }), 500

@app.route('/analyze-interests', methods=['POST'])
def analyze_interests():
    """
    Analyze articles based on user interests
    Returns JSON with raw article data for frontend vector calculations
    """
    try:
        # Get interests from JSON request
        data = request.get_json()
        interests = data.get('interests', '')
        
        if not interests:
            logger.error("No interests provided in request")
            return jsonify({
                "status": "error",
                "message": "No interests provided"
            }), 400
            
        logger.info(f"Analyzing interests: {interests}")
        
        # Find any final articles in the cache
        article_files = glob.glob(os.path.join(CACHE_DIR, 'final_article_*.json'))
        
        if not article_files:
            logger.error("No articles found for analysis")
            return jsonify({
                "status": "error",
                "message": "No articles found for analysis"
            }), 404
        
        logger.info(f"Found {len(article_files)} article files for analysis")
            
        # Load articles and prepare for frontend analysis
        articles = []
        for article_path in article_files[:10]:  # Limit to 10 articles for performance
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                article_id = os.path.basename(article_path).replace('final_article_', '').replace('.json', '')
                
                # Extract title from content
                title = "Unknown Title"
                content = article_data.get('content', '')
                lines = content.split('\n')
                if lines and lines[0].startswith('# '):
                    title = lines[0][2:].strip()
                
                # Extract a summary from the content
                summary = extract_article_summary(content)
                logger.info(f"Extracted summary for article '{title}': {summary[:100]}...")
                
                # Find article embedding from summary files first
                embedding = None
                try:
                    # First try to find a summary file with matching title
                    summary_files = glob.glob(os.path.join(CACHE_DIR, "summary_*.json"))
                    logger.info(f"Searching {len(summary_files)} summary files for article embedding")
                    
                    # Try to find a matching summary by comparing title
                    for summary_path in summary_files:
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                                # If the summary contains an embedding and matches our title
                                summary_title = summary_data.get('title', '')
                                if 'embedding' in summary_data and summary_title and (
                                   summary_title.lower() == title.lower() or 
                                   summary_title.lower() in title.lower() or 
                                   title.lower() in summary_title.lower()):
                                    embedding = summary_data.get('embedding')
                                    logger.info(f"Found embedding for article: {title} by title match in summary file")
                                    break
                        except Exception as e:
                            logger.error(f"Error reading summary file {summary_path}: {e}")
                            continue
                    
                    # If still no embedding, try hash-based matching
                    if not embedding:
                        # Try matching by content hash
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        embedding_paths = glob.glob(os.path.join(CACHE_DIR, f"summary_{content_hash}*.json"))
                        
                        if embedding_paths:
                            try:
                                with open(embedding_paths[0], 'r', encoding='utf-8') as f:
                                    embedding_data = json.load(f)
                                    embedding = embedding_data.get('embedding')
                                    logger.info(f"Found direct embedding match for article: {title}")
                            except Exception as e:
                                logger.error(f"Error reading embedding file: {e}")
                except Exception as e:
                    logger.error(f"Error finding embedding: {e}")
                
                # Add to articles list
                article_info = {
                    'id': article_id,
                    'title': title,
                    'content': content,
                    'summary': summary  # Include the summary in the response
                }
                
                # Add embedding if available
                if embedding:
                    article_info['embedding'] = embedding
                    logger.info(f"Article '{title}' has embedding with {len(embedding)} dimensions")
                else:
                    logger.info(f"Article '{title}' does not have embedding")
                
                articles.append(article_info)
            except Exception as e:
                logger.error(f"Error loading article {article_path}: {e}")
        
        # If we have no articles, return error
        if not articles:
            logger.error("Failed to load any articles for analysis")
            return jsonify({
                "status": "error",
                "message": "Failed to load any articles for analysis"
            }), 500
        
        logger.info(f"Successfully loaded {len(articles)} articles for analysis")
        
        # Return articles without scores for frontend processing
        return jsonify({
            "status": "success",
            "message": "Retrieved articles for frontend analysis",
            "articles": articles,
            "calculation_steps": [] # Empty array for compatibility
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_interests: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"An error occurred during analysis: {str(e)}"
        }), 500

# Add a simple health check endpoint
@app.route('/status', methods=['GET'])
def status():
    """Simple health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Server is running",
        "time": datetime.datetime.now().isoformat()
    })

# --- Cache Synchronization Functions ---

def sync_cache():
    """
    Synchronize the local cache with the main cache
    Returns statistics about the synchronization
    """
    stats = {
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "totalLocal": 0
    }
    
    try:
        # Ensure cache directories exist
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        
        # Get files from both caches with their last modified times
        main_cache_files = {}
        if os.path.exists(CACHE_DIR):
            for entry in os.scandir(CACHE_DIR):
                if entry.is_file():
                    main_cache_files[entry.name] = entry.stat().st_mtime
        
        local_cache_files = {}
        if os.path.exists(LOCAL_CACHE_DIR):
            for entry in os.scandir(LOCAL_CACHE_DIR):
                if entry.is_file():
                    local_cache_files[entry.name] = entry.stat().st_mtime
        
        stats["totalLocal"] = len(local_cache_files)
        
        # Synchronize from main to local
        for file_name, main_mod_time in main_cache_files.items():
            local_mod_time = local_cache_files.get(file_name)
            
            # If file doesn't exist locally or main version is newer
            if local_mod_time is None or main_mod_time > local_mod_time:
                try:
                    source_file = os.path.join(CACHE_DIR, file_name)
                    dest_file = os.path.join(LOCAL_CACHE_DIR, file_name)
                    
                    shutil.copy2(source_file, dest_file)
                    
                    if local_mod_time is None:
                        stats["added"] += 1
                        logger.info(f"Added: {file_name}")
                    else:
                        stats["updated"] += 1
                        logger.info(f"Updated: {file_name}")
                except Exception as e:
                    logger.error(f"Error copying file {file_name}: {e}")
                    stats["errors"] += 1
            else:
                stats["skipped"] += 1
        
        logger.info(f"Cache sync complete: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} unchanged, {stats['errors']} errors")
        logger.info(f"Local cache now contains {stats['added'] + stats['updated'] + stats['skipped']} files")
        
        return stats
    except Exception as e:
        logger.error(f"Cache sync failed: {e}")
        return stats

def get_cached_file(file_name):
    """
    Get a file from the local cache, falling back to the main cache if needed
    Returns the file content as a JSON object or None if not found
    """
    try:
        # Try local cache first
        local_path = os.path.join(LOCAL_CACHE_DIR, file_name)
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf8') as f:
                content = f.read()
                logger.debug(f"Read from local cache: {file_name}")
                return json.loads(content)
        
        # If not in local cache, try main cache and copy to local
        main_path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(main_path):
            with open(main_path, 'r', encoding='utf8') as f:
                content = f.read()
            
            # Save to local cache
            os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
            with open(local_path, 'w', encoding='utf8') as f:
                f.write(content)
            logger.info(f"Copied from main cache to local: {file_name}")
            
            return json.loads(content)
        
        # Not found in either cache
        return None
    except Exception as e:
        logger.error(f"Error getting cached file {file_name}: {e}")
        return None

def generate_cache_key(input_str):
    """Generate a cache key from a string"""
    return hashlib.md5(input_str.encode()).hexdigest()

# --- Cache Synchronization API Endpoints ---

@app.route('/api/sync-cache', methods=['POST'])
def api_sync_cache():
    """API endpoint to synchronize the cache"""
    try:
        stats = sync_cache()
        return jsonify({
            "status": "success",
            "stats": stats,
            "message": f"Cache synchronized: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} unchanged"
        })
    except Exception as e:
        logger.error(f"Error in sync-cache API: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/get-cached-file', methods=['GET'])
def api_get_cached_file():
    """API endpoint to get a file from cache"""
    try:
        file_name = request.args.get('file')
        if not file_name:
            return jsonify({
                "status": "error",
                "message": "Missing file parameter"
            }), 400
        
        file_data = get_cached_file(file_name)
        if file_data is None:
            return jsonify({
                "status": "error",
                "message": f"File not found: {file_name}"
            }), 404
        
        return jsonify({
            "status": "success",
            "data": file_data,
            "source": "local" if os.path.exists(os.path.join(LOCAL_CACHE_DIR, file_name)) else "main"
        })
    except Exception as e:
        logger.error(f"Error in get-cached-file API: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/get-summary', methods=['GET'])
def api_get_summary():
    """API endpoint to get a summary from cache"""
    try:
        article_id = request.args.get('id')
        if not article_id:
            return jsonify({
                "status": "error",
                "message": "Missing id parameter"
            }), 400
        
        cache_key = generate_cache_key(article_id)
        file_name = f"summary_{cache_key}.json"
        
        summary_data = get_cached_file(file_name)
        if summary_data is None:
            return jsonify({
                "status": "error",
                "message": f"Summary not found for article: {article_id}"
            }), 404
        
        return jsonify({
            "status": "success",
            "data": summary_data,
            "source": "local" if os.path.exists(os.path.join(LOCAL_CACHE_DIR, file_name)) else "main"
        })
    except Exception as e:
        logger.error(f"Error in get-summary API: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/get-article', methods=['GET'])
def api_get_article():
    """API endpoint to get a final article from cache"""
    try:
        article_key = request.args.get('key')
        if not article_key:
            return jsonify({
                "status": "error",
                "message": "Missing key parameter"
            }), 400
        
        # Final articles use a timestamp_subject format
        # We look for files that contain the subject
        safe_key = re.sub(r'[^a-z0-9_]', '_', article_key.lower())
        
        # List all final article files
        matching_files = []
        
        # Check local cache first
        for entry in os.scandir(LOCAL_CACHE_DIR):
            if entry.is_file() and entry.name.startswith('final_article_') and safe_key in entry.name:
                matching_files.append(entry.name)
        
        # If not found locally, check main cache
        if not matching_files:
            for entry in os.scandir(CACHE_DIR):
                if entry.is_file() and entry.name.startswith('final_article_') and safe_key in entry.name:
                    matching_files.append(entry.name)
        
        if not matching_files:
            return jsonify({
                "status": "error",
                "message": f"Article not found for key: {article_key}"
            }), 404
        
        # Use the first match
        file_name = matching_files[0]
        article_data = get_cached_file(file_name)
        
        return jsonify({
            "status": "success",
            "data": article_data,
            "source": "local" if os.path.exists(os.path.join(LOCAL_CACHE_DIR, file_name)) else "main"
        })
    except Exception as e:
        logger.error(f"Error in get-article API: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# --- End Cache Synchronization API Endpoints ---

if __name__ == '__main__':
    # Add command line argument for port
    parser = argparse.ArgumentParser(description='Start the API server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    try:
        # Copy articles to public directory on startup
        logger.info("Syncing article files on startup...")
        result = copy_articles_to_public()
        logger.info(result)
        
        # Check for articles that need summaries
        logger.info("Checking for articles that need summaries...")
        added_summaries = 0
        already_had_summaries = 0
        
        # Check all final article HTML files
        final_articles = glob.glob(os.path.join(HTML_DIR, "*.html"))
        for article_path in final_articles:
            try:
                # Check if the file has a summary section
                with open(article_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "<h3>Summary</h3>" not in content:
                    # Extract markdown content and parse it to get summary
                    article_id = os.path.basename(article_path).replace("tech_deep_dive_", "").replace(".html", "")
                    json_path = os.path.join(CACHE_DIR, f"final_article_{article_id}.json")
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            article_data = json.load(f)
                        
                        markdown_content = article_data.get('content', '')
                        if markdown_content:
                            summary = extract_article_summary(markdown_content)
                            
                            # If we have a summary, update the HTML file
                            if summary:
                                new_content = content.replace("<h2>Introduction</h2>", 
                                                            "<h2>Introduction</h2>\n<h3>Summary</h3>\n<div class='summary'>" + summary + "</div>")
                                
                                with open(article_path, 'w', encoding='utf-8') as f:
                                    f.write(new_content)
                                
                                added_summaries += 1
                    else:
                        already_had_summaries += 1
                else:
                    already_had_summaries += 1
            except Exception as e:
                logger.error(f"Error processing article {article_path}: {e}")
        
        logger.info(f"Added summaries to {added_summaries} articles, {already_had_summaries} already had summaries")
        
        # Start the server
        logger.info("Starting server (Flask development server)...")
        app.run(debug=True, port=args.port, host='0.0.0.0') # Use the port from command line arguments
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())