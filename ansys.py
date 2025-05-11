# Standard library imports
import asyncio
import concurrent.futures
import functools
import json
import os
import re
import hashlib
from time import sleep, time
import threading
import urllib.parse
from collections import defaultdict
import sys
import io
import tempfile
import argparse
import glob

# Third-party imports
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not found. Install with 'pip install transformers'")
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium or WebDriver Manager not found. Install with 'pip install selenium webdriver-manager'")

# Add embedding support
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence Transformers not found. Install with 'pip install sentence-transformers'")

# Add PDF handling libraries
try:
    import PyPDF2
    import pdfplumber
    from PIL import Image
    import pytesseract
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PDF extraction libraries not fully available. Install PyPDF2, pdfplumber, and pytesseract for PDF support.")

# Llama model imports
import llama_cpp._internals as _internals
from llama_cpp import Llama

# Local application imports
from hackernews_summarizer import HackerNewsSummarizer

# Patch Llama sampler
_internals.LlamaSampler.__del__ = lambda self: None

# Global lock for Llama model access
LLM_LOCK = threading.Lock()
# Add print lock for thread safety
PRINT_LOCK = threading.Lock()

# Initialize embedding model (with lazy loading)
EMBEDDING_MODEL = None
EMBEDDING_LOCK = threading.Lock()

def get_embedding_model():
    global EMBEDDING_MODEL
    if not EMBEDDINGS_AVAILABLE:
        return None
        
    if EMBEDDING_MODEL is None:
        with EMBEDDING_LOCK:
            if EMBEDDING_MODEL is None:
                try:
                    safe_print("Initializing embedding model...")
                    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
                    safe_print("Embedding model initialized successfully")
                except Exception as e:
                    safe_print(f"Error initializing embedding model: {e}")
                    return None
    return EMBEDDING_MODEL

# Thread-safe print function
def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        try:
            # Try to print to console first
            print(*args, **kwargs)
        except OSError as e:
            # If console printing fails, try to log instead
            try:
                import logging
                logging.basicConfig(level=logging.INFO)
                logging.info(' '.join(str(arg) for arg in args))
            except Exception:
                # If all else fails, silently ignore
                pass

# Fallback function for when Llama model fails
def fallback_llm_call(prompt, max_tokens=100, temperature=0.5):
    """Simple fallback when the Llama model fails"""
    # Extract key terms from the prompt
    words = prompt.split()
    key_terms = [w for w in words if len(w) > 4 and w.isalnum()]
    
    # Create a simple response based on prompt content
    if "[INST]" in prompt and "[/INST]" in prompt:
        # Extract the instruction
        instruction = prompt.split("[INST]")[1].split("[/INST]")[0].strip()
        
        # Determine what kind of prompt this is
        if "generate two deep" in instruction.lower() or "questions" in instruction.lower():
            return {
                "choices": [{"text": "1. What are the key components of this system?\n2. How does this technology compare to alternatives?"}]
            }
        elif "alignment" in instruction.lower() and "rate" in instruction.lower():
            return {"choices": [{"text": "50"}]}
        elif "subject" in instruction.lower() and "analyze" in instruction.lower():
            # Extract potential subject from the instruction
            for term in key_terms:
                if len(term) > 5:
                    return {"choices": [{"text": term}]}
            return {"choices": [{"text": "Technology"}]}
        else:
            return {"choices": [{"text": "I'm unable to process this request."}]}
    else:
        return {"choices": [{"text": "I'm unable to process this request."}]}

# API Configuration - Create a function to get a shared model instance
def get_llama_model():
    if not hasattr(get_llama_model, "instance") or get_llama_model.instance is None:
        with LLM_LOCK:
            if not hasattr(get_llama_model, "instance") or get_llama_model.instance is None:
                safe_print("Initializing Llama model...")
                
                # Look for model in multiple locations
                possible_model_paths = [
                    "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Original path
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    os.path.expanduser("~/llama-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    os.path.expanduser("~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
                ]
                
                # Check if model path is specified in environment variables
                env_model_path = os.environ.get("LLAMA_MODEL_PATH")
                if env_model_path:
                    possible_model_paths.insert(0, env_model_path)
                
                # Find the first valid model path
                model_path = None
                for path in possible_model_paths:
                    if os.path.exists(path):
                        model_path = path
                        safe_print(f"Found model at: {model_path}")
                        break
                
                if not model_path:
                    error_msg = f"Could not find model file. Checked paths: {possible_model_paths}"
                    safe_print(error_msg)
                    raise FileNotFoundError(error_msg)
                
                try:
                    # First try with minimal GPU usage (only 1 layer on GPU)
                    get_llama_model.instance = Llama(
                        model_path=model_path,
                        n_ctx=24000,  # MODIFIED: Increased context size to 24k
                        n_threads=4, 
                        n_gpu_layers=1, 
                        chat_format="mistral-instruct",
                        verbose=False,
                        stop=None
                    )
                    safe_print("Successfully initialized Llama model with GPU support (n_ctx=24000)")
                except Exception as e:
                    safe_print(f"Error initializing Llama with GPU (n_ctx=24000): {e}, falling back to CPU-only mode")
                    try:
                        # Fall back to CPU-only mode
                        get_llama_model.instance = Llama(
                            model_path=model_path,
                            n_ctx=24000,  # MODIFIED: Increased context size to 24k also for CPU fallback
                            n_threads=4,
                            n_gpu_layers=0,  # CPU only
                            chat_format="mistral-instruct",
                            verbose=False,
                            stop=None
                        )
                        safe_print("Successfully initialized Llama model in CPU-only mode (n_ctx=24000)")
                    except Exception as e2:
                        error_msg = f"Failed to initialize Llama model: {e2}"
                        safe_print(error_msg)
                        raise RuntimeError(error_msg)
    
    return get_llama_model.instance

# Initialize the summarization model for titles
title_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Constants
HACKER_NEWS_URL = "https://news.ycombinator.com/"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Global cache for article summaries
ARTICLE_SUMMARY_CACHE = {}
GLOBAL_SUMMARIZER = None
SUMMARIZER_LOCK = threading.Lock()

# Initialize global summarizer
def get_global_summarizer():
    global GLOBAL_SUMMARIZER
    with SUMMARIZER_LOCK:
        if GLOBAL_SUMMARIZER is None:
            GLOBAL_SUMMARIZER = HackerNewsSummarizer(verbose=False)
    return GLOBAL_SUMMARIZER

# ===== CACHE MANAGEMENT =====

class ArticleCache:
    """Cache manager for storing and retrieving article summaries"""
    
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, article_url, user_interests):
        """Generate a cache file path based on article URL and user interests"""
        # Create a hash based on both URL and interests to ensure cache is user-specific
        cache_key = f"{article_url}:{user_interests}"
        url_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def get_from_cache(self, article_url, user_interests, max_age=3600):
        """Retrieve article info from cache if it exists and is not too old"""
        cache_path = self.get_cache_path(article_url, user_interests)
        
        if os.path.exists(cache_path):
            # Check if cache is fresh enough
            file_age = time() - os.path.getmtime(cache_path)
            if file_age < max_age:
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
        return None
    
    def save_to_cache(self, article_url, user_interests, data):
        """Save article info to cache"""
        cache_path = self.get_cache_path(article_url, user_interests)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            safe_print(f"Error saving to cache: {e}")

# Initialize the cache
article_cache = ArticleCache()

# ===== CONTENT ACQUISITION FUNCTIONS =====

def extract_hn_content(url):
    """
    Extract content specifically from a Hacker News discussion page
    Optimized for Hacker News's specific HTML structure
    """
    try:
        # Get the page content
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract the title
        title_element = soup.select_one(".titleline")
        title = title_element.get_text(strip=True) if title_element else "Unknown Title"
        
        # First check for the main article link
        if title_element and title_element.find('a'):
            original_article_url = title_element.find('a').get('href')
            if original_article_url and not original_article_url.startswith('item?id='):
                safe_print(f"Found original article URL: {original_article_url}")
                # Try to get content from the original article
                try:
                    article_content = extract_web_content(original_article_url)
                    if article_content and len(article_content) > 500:
                        return f"Title: {title}\n\n{article_content}"
                except Exception as e:
                    safe_print(f"Error extracting original article content: {e}, falling back to HN comments")
        
        # Extract the content from the HN post itself
        content = []
        
        # Look for the post text (if any)
        post_text = soup.select_one(".toptext")
        if post_text:
            content.append(post_text.get_text(strip=True))
        
        # Get the top comments
        comments = soup.select(".commtext")
        for i, comment in enumerate(comments[:10]):  # Limit to first 10 comments
            comment_text = comment.get_text(strip=True)
            if len(comment_text) > 50:  # Only include substantial comments
                content.append(f"Comment {i+1}: {comment_text}")
                
            # Limit total content length
            if len('\n\n'.join(content)) > 10000:
                break
                
        # If we still don't have much content, try one more approach
        if len('\n\n'.join(content)) < 500:
            try:
                # Try to find any links to PDFs or GitHub repos or documentation in the comments
                all_links = soup.select("a[href]")
                for link in all_links:
                    href = link.get('href', '')
                    if any(term in href.lower() for term in ['.pdf', 'github.com', 'docs.', 'paper', 'arxiv']):
                        safe_print(f"Found potentially relevant link in comments: {href}")
                        try:
                            link_content = extract_web_content(href) 
                            if link_content and len(link_content) > 500:
                                content.append(f"Content from linked page: {link_content[:5000]}...")
                                break
                        except Exception:
                            continue
            except Exception as e:
                safe_print(f"Error extracting linked content from HN comments: {e}")
        
        if not content:
            return f"Title: {title}\n\nNo substantial content found on the Hacker News page."
            
        return f"Title: {title}\n\n" + '\n\n'.join(content)
    except Exception as e:
        safe_print(f"Error extracting HN content from {url}: {e}")
        return None

def extract_web_content(url):
    """
    Extract content from a general web page (non-PDF)
    Uses Selenium if available to handle dynamic content, falls back to requests.
    """
    if is_pdf_url(url):
        return extract_pdf_content(url)

    content = None
    title = "No title found"

    # Try Selenium first if available
    if SELENIUM_AVAILABLE:
        options = Options()
        options.add_argument("--headless")  # Run headless (no browser window)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        driver = None
        try:
            safe_print(f"Attempting to extract content from {url} using Selenium...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30) # Add timeout
            driver.get(url)
            # Optional: Add a small delay for JavaScript to load
            sleep(3)
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else "No title found"
            content = _parse_soup_content(soup)
            safe_print(f"Successfully extracted content using Selenium for {url}")
        except Exception as e:
            safe_print(f"Selenium extraction failed for {url}: {e}. Falling back to requests.")
            content = None # Ensure content is reset if Selenium fails
        finally:
            if driver:
                driver.quit()

    # Fallback to requests if Selenium failed or is unavailable
    if content is None:
        try:
            safe_print(f"Attempting to extract content from {url} using requests...")
            response = requests.get(url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else "No title found"
            content = _parse_soup_content(soup)
            safe_print(f"Successfully extracted content using requests for {url}")
        except Exception as e:
            safe_print(f"Requests extraction failed for {url}: {e}")
            return None # Failed both ways

    return f"Title: {title}\n\n{content}" if content else None

def _parse_soup_content(soup):
    """Helper function to parse content from BeautifulSoup object"""
    # Remove script, style, and other non-content elements
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        element.decompose()

    # Try different common content containers
    selectors = ['article', 'main', 'div.content', 'div.article', 'div.post', 'div.entry', 'div.blog-post']
    for selector in selectors:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all('p')
            if paragraphs:
                content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])
                if len(content) > 300: # Require a decent amount of text
                    return content

    # If specific containers fail, get all paragraphs from the body
    paragraphs = soup.body.find_all('p') if soup.body else []
    if paragraphs:
        content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])
        if len(content) > 300:
            return content

    # Last resort - get all text, filter lines
    text = soup.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    filtered_lines = [line for line in lines if len(line) > 30] # Only include substantial lines
    if filtered_lines:
        return '\n\n'.join(filtered_lines[:100]) # Limit to 100 lines

    return None

def extract_content_safely(url):
    """
    Extract content from any URL with robust error handling
    Dispatches to the appropriate extraction function based on URL type
    Returns a tuple of (content, error_message)
    """
    content = None
    error = None
    
    try:
        # Check if this is a Hacker News URL
        if 'news.ycombinator.com' in url or 'item?id=' in url:
            content = extract_hn_content(url)
        # Check if this is a PDF
        elif is_pdf_url(url):
            content = extract_pdf_content(url)
        # Otherwise treat as a general web page
        else:
            content = extract_web_content(url)
            
        # Validate content
        if not content:
            error = "Could not extract meaningful content from the URL"
        elif len(content) < 200:
            error = "Extracted content is too short to be useful"
            
        return content, error
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        safe_print(error_msg)
        return None, error_msg

def summarize_content_safely(content, title=None, max_length=1500):
    """
    Safely summarize content with robust error handling
    Returns a tuple of (summary, error_message) or (summary, embedding, error_message) if embeddings are available
    """
    if not content:
        return None, None, "No content provided for summarization" if EMBEDDINGS_AVAILABLE else None, "No content provided for summarization"
        
    try:
        # Create a dedicated summarizer
        summarizer = HackerNewsSummarizer(verbose=False)
        
        try:
            # Clean and normalize the text
            cleaned_content = summarizer.clean_text(content)
            
            # If content is extremely long, truncate before summarizing
            if len(cleaned_content) > 20000:
                safe_print(f"Content is very long ({len(cleaned_content)} chars), truncating before summarization")
                cleaned_content = cleaned_content[:20000]
            
            # Generate the summary
            if len(cleaned_content) > 200:
                summary = summarizer.final_summarize(cleaned_content)
                
                # If we have a title, include it in the summary
                if title and not summary.startswith(title):
                    summary = f"{title}\n\n{summary}"
                
                # Generate embedding if available
                embedding = None
                if EMBEDDINGS_AVAILABLE:
                    try:
                        embedding_model = get_embedding_model()
                        if embedding_model and summary:
                            # Create embedding from the summary (limit to first 1000 chars for efficiency)
                            embedding = embedding_model.encode(summary[:1000]).tolist()  # Convert to list for JSON serialization
                            safe_print(f"Generated embedding vector of length {len(embedding)}")
                    except Exception as e:
                        safe_print(f"Error generating embedding: {e}")
                
                if EMBEDDINGS_AVAILABLE:
                    return summary, embedding, None
                else:
                    return summary, None
            else:
                if EMBEDDINGS_AVAILABLE:
                    return cleaned_content, None, "Content too short for summarization"
                else:
                    return cleaned_content, "Content too short for summarization"
        finally:
            # Always clean up resources
            summarizer.cleanup_resources(silent=True)
    except Exception as e:
        error_msg = f"Error during summarization: {str(e)}"
        safe_print(error_msg)
        if EMBEDDINGS_AVAILABLE:
            return None, None, error_msg
        else:
            return None, error_msg

async def summarize_all_articles_async(articles):
    """
    Asynchronously summarize articles and store results in the global cache
    """
    safe_print(f"Starting background summarization of {len(articles)} articles...")
    
    # Keep track of cache hits and misses
    cache_hits = 0
    disk_cache_hits = 0
    new_summaries = 0
    
    # Define the worker function to process a single article
    def summarize_article(article_data):
        title, link = article_data
        nonlocal cache_hits, disk_cache_hits, new_summaries
        
        # Check if already in memory cache (fastest)
        if link in ARTICLE_SUMMARY_CACHE:
            safe_print(f"✓ Memory cache hit: {title}")
            cache_hits += 1
            return
        
        try:
            # Create a standard cache path for this article
            url_hash = hashlib.md5(link.encode()).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"summary_{url_hash}.json")
            
            # Check disk cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_summary = json.load(f)
                        # Update in-memory cache
                        ARTICLE_SUMMARY_CACHE[link] = cached_summary
                        safe_print(f"✓ Disk cache hit: {title}")
                        disk_cache_hits += 1
                        return
                except Exception as e:
                    safe_print(f"Error reading cache for {title}: {e}")
            
            # Extract content with our specialized functions
            content, error = extract_content_safely(link)
            
            if content:
                # Generate summary and embedding if available
                if EMBEDDINGS_AVAILABLE:
                    summary, embedding, summary_error = summarize_content_safely(content, title)
                else:
                    summary, summary_error = summarize_content_safely(content, title)
                    embedding = None
                
                if summary:
                    # Store in memory cache with embedding if available
                    cache_data = {
                        'title': title,
                        'summary': summary,
                        'timestamp': time()
                    }
                    
                    # Add embedding if available
                    if embedding:
                        cache_data['embedding'] = embedding
                    
                    ARTICLE_SUMMARY_CACHE[link] = cache_data
                    
                    # Save to disk cache
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(ARTICLE_SUMMARY_CACHE[link], f)
                    
                    safe_print(f"✓ Summarized: {title}{' with embedding' if embedding else ''}")
                    new_summaries += 1
                else:
                    safe_print(f"✗ Could not generate summary for {title}: {summary_error}")
            else:
                safe_print(f"✗ Failed to get content: {title} - {error}")
                
        except Exception as e:
            safe_print(f"✗ Error summarizing {title}: {e}")
    
    # Process articles sequentially to avoid resource contention
    # For background processing, sequential is safer and prevents memory issues
    for article in articles:
        summarize_article(article)
    
    safe_print(f"Background summarization complete. Cached {len(ARTICLE_SUMMARY_CACHE)} articles.")
    safe_print(f"Summary: {cache_hits} memory cache hits, {disk_cache_hits} disk cache hits, {new_summaries} new summaries generated")

# ===== HELPER FUNCTIONS =====

def normalize_title(title):
    """Clean and normalize article titles by removing arrow notations and redundant text"""
    # Remove arrow notation (-> text) from titles
    if "->" in title:
        title = title.split("->")[0].strip()
    return title

def is_valid_hn_link(link):
    """Check if link matches the format: item?id= followed by digits"""
    # More flexible pattern that allows for different item IDs
    if not link:
        return False
    
    # Simple check for HN links
    if 'item?id=' in link:
        # Extract the ID
        try:
            id_part = link.split('item?id=')[1]
            # Verify it's a number
            return id_part.isdigit()
        except IndexError:
            return False
    
    return False

def extract_pdf_content(pdf_url):
    """
    Extract text content from a PDF URL
    Returns the extracted text or None if extraction fails
    """
    if not PDF_SUPPORT:
        return "PDF extraction not available. Please install required libraries."
    
    try:
        # Download the PDF
        response = requests.get(pdf_url, stream=True, timeout=30)
        if response.status_code != 200:
            return None
        
        content = ""
        
        # Try PyPDF2 first (faster but less accurate)
        try:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content += page.extract_text() + "\n\n"
                
                # Limit to first 10 pages to avoid memory issues
                if page_num >= 9:
                    content += "[Content truncated - PDF has more pages]"
                    break
        except Exception as e:
            safe_print(f"PyPDF2 extraction failed, trying pdfplumber: {e}")
            
            # If PyPDF2 fails, try pdfplumber (slower but more accurate)
            try:
                pdf_file = io.BytesIO(response.content)
                with pdfplumber.open(pdf_file) as pdf:
                    # Extract text from each page
                    for i, page in enumerate(pdf.pages[:10]):  # Limit to first 10 pages
                        content += page.extract_text() or ""
                        content += "\n\n"
                        
                    if len(pdf.pages) > 10:
                        content += "[Content truncated - PDF has more pages]"
            except Exception as e:
                safe_print(f"PDF extraction failed: {e}")
                return None
                
        return content if content.strip() else None
        
    except Exception as e:
        safe_print(f"Error downloading PDF: {e}")
        return None

def is_pdf_url(url):
    """Check if a URL points to a PDF file"""
    # Check URL extension
    if url.lower().endswith('.pdf'):
        return True
    
    # Check URL path
    if '/pdf/' in url.lower() or 'pdf' in url.lower():
        # Try to check headers
        try:
            head_response = requests.head(url, timeout=5)
            content_type = head_response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                return True
        except Exception:
            # If we can't check headers, make an educated guess based on URL
            return '/pdf/' in url.lower()
    
    return False

def article_grabber(run_background_processing=True):
    """Retrieve articles from Hacker News or fallback sources"""
    all_articles = []
    
    # Try Hacker News first
    try:
        safe_print("Attempting to fetch articles from Hacker News...")
        # Send a GET request to Hacker News with timeout
        response = requests.get(HACKER_NEWS_URL, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all article rows (with class "athing")
        rows = soup.select(".athing")
        safe_print(f"Found {len(rows)} article rows on Hacker News")
        
        for row in rows:
            # Get the article title
            title_element = row.select_one(".titleline > a")
            if not title_element:
                continue
                
            title = title_element.get_text(strip=True)
            
            # Normalize the title to remove arrow notation
            title = normalize_title(title)
            
            # Get the item ID from the row
            item_id = row.get("id")
            if not item_id:
                continue
                
            # Create the comment section URL
            comment_link = f"{HACKER_NEWS_URL}item?id={item_id}"
            
            # Verify it matches our pattern
            if is_valid_hn_link(f"item?id={item_id}"):
                all_articles.append([title, comment_link])
        
        safe_print(f"Successfully retrieved {len(all_articles)} articles from Hacker News")
    except Exception as e:
        safe_print(f"Error retrieving articles from Hacker News: {e}")
    
    # If we couldn't get any articles from Hacker News, try alternative sources
    if not all_articles:
        safe_print("Attempting to fetch articles from alternative sources...")
        
        # Try Reddit r/programming
        try:
            reddit_url = "https://www.reddit.com/r/programming/hot.json"
            headers = {"User-Agent": "Mozilla/5.0 TechDeepDive/1.0"}
            response = requests.get(reddit_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title")
                permalink = post_data.get("permalink")
                
                if title and permalink:
                    # Normalize the title
                    title = normalize_title(title)
                    full_url = f"https://www.reddit.com{permalink}"
                    all_articles.append([title, full_url])
                    
            safe_print(f"Retrieved {len(all_articles)} articles from Reddit")
        except Exception as e:
            safe_print(f"Error retrieving articles from Reddit: {e}")
    
    # If we still don't have any articles, create some dummy ones for testing
    if not all_articles:
        safe_print("Creating fallback dummy articles for testing...")
        dummy_articles = [
            ["Understanding Machine Learning Algorithms", "https://example.com/ml-algorithms"],
            ["The Future of Web Development", "https://example.com/web-dev-future"],
            ["Artificial Intelligence in Healthcare", "https://example.com/ai-healthcare"],
            ["Quantum Computing Explained", "https://example.com/quantum-computing"],
            ["Blockchain Technology and Applications", "https://example.com/blockchain-apps"]
        ]
        all_articles.extend(dummy_articles)
    
    safe_print(f"Total articles collected: {len(all_articles)}")
    
    # Run background processing only if requested
    if run_background_processing:
        safe_print("Starting background processing...")
        # Launch background processing
        if asyncio.get_event_loop().is_running():
            # If we're in an async context already
            asyncio.create_task(summarize_all_articles_async(all_articles))
        else:
            # If we're not in an async context, create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(summarize_all_articles_async(all_articles))
            # Don't close the loop as it might be needed later
    else:
        safe_print("Skipping background processing (disabled)")
    
    return all_articles

async def scrape_cleaned_text(url, min_words_div=5):
    """
    Scrape and clean text from a given URL.
    Extracts all headers and paragraphs, and handles embedded PDFs.
    """
    try:
        # Since HackerNewsSummarizer's methods are synchronous but this function is async,
        # we need to run the synchronous code in a way that doesn't block the event loop
        
        # Define a synchronous function to run
        def extract_with_summarizer():
            # Create an instance with minimal verbosity
            summarizer = HackerNewsSummarizer(verbose=False)
            try:
                # Extract article content using the summarizer's method
                content = summarizer.extract_article_content(url)
                return content
            finally:
                # Clean up resources
                summarizer.cleanup_resources(silent=True)
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, extract_with_summarizer)
        
        if not content:
            raise ValueError(f"Failed to extract content from {url}")
            
        return content
    except Exception as e:
        return f"[!] Failed to load {url}: {e}"

# ===== CONTENT PROCESSING FUNCTIONS =====

def preprocess(articles, user_interests):
    """
    Process and score articles based on user interests in parallel
    """
    # Skip scoring if environment variable ANSYS_NO_SCORE is set
    skip_scoring = os.getenv('ANSYS_NO_SCORE') is not None
    if skip_scoring:
        safe_print(f"*** SCORING DISABLED: ANSYS_NO_SCORE is set - only extracting subjects for {len(articles)} articles ***")
        results = []
        for title, link in articles:
            # Normalize the title to remove arrow notation
            title = normalize_title(title)
            
            # Extract subject using LLM or fallback
            prompt = f"[INST] Extract ONLY the core subject from this title. For example, 'New Advancements in AI Research' should return 'AI Research'. Respond with the subject only, no additional words or explanations. Perform this on: {title}. [/INST]"
            llm = get_llama_model()
            if llm:
                try:
                    with LLM_LOCK:
                        resp = llm(prompt, max_tokens=64, temperature=0.4)
                    subject_line = resp["choices"][0]["text"].strip()
                    safe_print(f"Extracted subject for: {title} -> {subject_line}")
                except Exception:
                    subject_line = fallback_llm_call(prompt)["choices"][0]["text"].strip()
                    safe_print(f"Used fallback for subject: {title} -> {subject_line}")
            else:
                subject_line = fallback_llm_call(prompt)["choices"][0]["text"].strip()
                safe_print(f"Used fallback for subject: {title} -> {subject_line}")
            
            # Set a default score of 50 when not scoring
            results.append({'title': title, 'link': link, 'subject': subject_line, 'score': 50})
        
        # Don't sort by score since we've disabled scoring
        safe_print(f"*** SCORING DISABLED: Processed {len(articles)} articles without scoring ***")
        return results

    # Regular scoring mode - only reached when ANSYS_NO_SCORE is not set
    safe_print(f"*** SCORING ENABLED: Processing {len(articles)} articles with scoring based on user interests ***")

    # Define the worker function to process a single article
    def process_article(article_data):
        title = article_data[0]
        link = article_data[1]
        
        # Normalize the title to remove arrow notation
        title = normalize_title(title)
        
        # Check if we have this article in cache
        cached_data = article_cache.get_from_cache(link, user_interests)
        if cached_data:
            safe_print(f"Using cached scoring for: {title}")
            return cached_data
        
        # Check if the article has been summarized in the global cache
        if link in ARTICLE_SUMMARY_CACHE:
            cached_summary = ARTICLE_SUMMARY_CACHE[link].get('summary', '')
            safe_print(f"Using global summary cache for: {title}")
            link_text = cached_summary
        else:
            # Fetch the discussion page content
            try:
                discussion_response = requests.get(link, timeout=10)
                discussion_soup = BeautifulSoup(discussion_response.text, "html.parser")
                
                # Find the text under class="toptext"
                toptext = discussion_soup.select_one(".toptext")
                link_text = ""
                if toptext:
                    # Get all paragraph text
                    paragraphs = toptext.find_all('p')
                    link_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
            except Exception as e:
                safe_print(f"Error fetching discussion content for {title}: {e}")
                link_text = title  # Fallback to just using the title
        
        try:
            # Keep text short for processing
            if len(link_text) > 500:
                link_text = link_text[:500]
                
            # Get the subject line using Llama
            prompt = f"""[INST] Extract ONLY the core subject from this title. Examples:
'Altair at 50: Remembering the first Personal Computer' → 'Altair; Personal Computing'
'New Study Shows Impact of Machine Learning on Healthcare' → 'Machine Learning; Healthcare'

Respond with just the core subject(s), separated by semicolons if multiple. No phrases like 'This is' or 'The subject is'.
For this title: {title} [/INST]"""
            
            # Use the global Llama model with lock
            llm = get_llama_model()
            
            if llm is not None:
                try:
                    with LLM_LOCK:
                        subject_response = llm(prompt, max_tokens=64, temperature=0.4)
                    subject_line = subject_response["choices"][0]["text"].strip()
                    # Print the subject line and the raw Llama response for debugging
                    safe_print(f"[DEBUG] Article: {title}\n[DEBUG] Llama subject prompt: {prompt}\n[DEBUG] Llama subject response: {subject_response}\n[DEBUG] Extracted subject line: {subject_line}")
                except Exception as e:
                    safe_print(f"Error using Llama for subject extraction: {e}, using fallback")
                    subject_response = fallback_llm_call(prompt)
                    subject_line = subject_response["choices"][0]["text"].strip()
            else:
                # Use fallback if Llama model failed to initialize
                subject_response = fallback_llm_call(prompt)
            subject_line = subject_response["choices"][0]["text"].strip()
            
            # Use AI to determine interest score using article summary from cache
            interest_score = 0
            article_summary = ""
            
            # Get cached summary if available
            if link in ARTICLE_SUMMARY_CACHE:
                article_summary = ARTICLE_SUMMARY_CACHE[link].get('summary', '')
                safe_print(f"Using cached summary for scoring: {article_summary[:100]}...")

            try:
                # Create a prompt for the AI to evaluate article relevance with summary and provide explanation
                score_prompt = f"""[INST] On a scale of 0-100, rate how well this article aligns with these interests: {user_interests}
                
                Article title: {title}
                Article summary: {article_summary[:1000]}  # Truncate for token limits
                
                Consider the summary content, relevance to interests, and potential value.
                First, provide a score between 0-100.
                Then on a new line, provide a brief explanation of your rating (within 30 words).
                Format your response like this:
                85
                This article directly covers machine learning algorithms which aligns with the AI interest.
                [/INST]"""
                safe_print(f"[DEBUG] Score prompt: {score_prompt}")
                with LLM_LOCK:
                    score_response = llm(score_prompt, max_tokens=128, temperature=0.2)
                
                score_text = score_response["choices"][0]["text"].strip()
                safe_print(f"[DEBUG] Raw score response: {score_text}")
                
                # Extract the score and explanation
                lines = score_text.strip().split('\n', 1)
                score_match = re.search(r'(\d+)', lines[0])
                
                explanation = ""
                if len(lines) > 1:
                    explanation = lines[1].strip()
                
                if score_match:
                    interest_score = int(score_match.group(1))
                    # Ensure score is within 0-100 range
                    interest_score = max(0, min(100, interest_score))
                    safe_print(f"[DEBUG] AI-determined score for '{title}': {interest_score}")
                    safe_print(f"[DEBUG] Explanation: {explanation}")
                else:
                    # If we can't extract a score, assign a default of 50
                    interest_score = 50
                    safe_print(f"[DEBUG] Could not extract score for '{title}', setting default of 50")
            except Exception as e:
                safe_print(f"Error using AI for scoring: {e}, setting default score of 50")
                interest_score = 50
                explanation = f"Default score due to error: {str(e)[:50]}"
            
            # Create result data with explanation
            result = {
                'title': title,
                'link': link,
                'subject': subject_line,
                'score': interest_score,
                'explanation': explanation
            }
            
            # Save to cache
            article_cache.save_to_cache(link, user_interests, result)
            
            return result
        except Exception as e:
            safe_print(f"Error processing article {title}: {e}")
            # Return a minimal result with score 0 in case of error
            return {
                'title': title,
                'link': link,
                'subject': "Error Processing Subject",
                'score': 50,
                'explanation': f"Error during processing: {str(e)[:50]}"
            }
    
    safe_print(f"Processing {len(articles)} articles based on user interests...")
    
    # Process articles one at a time to avoid resource issues
    results = []
    for article in articles:
        try:
            result = process_article(article)
            results.append(result)
        except Exception as e:
            safe_print(f"Error processing article: {e}")
    
    # Sort by score in descending order but return all articles
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print the top article's score explanation
    if results and len(results) > 0:
        top_article = results[0]
        safe_print(f"\n===== TOP ARTICLE EXPLANATION =====")
        safe_print(f"Title: {top_article['title']}")
        safe_print(f"Score: {top_article['score']}/100")
        safe_print(f"Why: {top_article.get('explanation', 'No explanation provided')}")
        safe_print(f"=====================================\n")
    
    safe_print(f"Processed {len(articles)} articles, scores: {[r['score'] for r in results[:5]]}")
    
    return results

def generate_deep_dive_questions(article_data):
    """Generate two deep dive questions based on the article summary focusing on fundamental concepts"""
    subject_line = article_data['subject']
    article_link = article_data['link']
    article_title = article_data['title']
    
    # Get the article summary if available in cache
    article_summary = ""
    
    # Check memory cache first (fastest)
    if article_link in ARTICLE_SUMMARY_CACHE:
        article_summary = ARTICLE_SUMMARY_CACHE[article_link].get('summary', '')
        safe_print(f"Using cached article summary from memory for: {subject_line}")
    
    # If not in memory cache, check disk cache
    if not article_summary:
        url_hash = hashlib.md5(article_link.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"summary_{url_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    article_summary = cached_data.get('summary', '')
                    # Update memory cache
                    ARTICLE_SUMMARY_CACHE[article_link] = cached_data
                    safe_print(f"Using cached article summary from disk for: {subject_line}")
            except Exception as e:
                safe_print(f"Error reading cache: {e}")
    
    # If not in cache, try to get it from the link
    if not article_summary:
        try:
            # Create a dedicated summarizer instance
            summarizer = HackerNewsSummarizer(verbose=False)
            try:
                # Extract and summarize content
                content = summarizer.extract_article_content(article_link)
                if content:
                    # Use the full summarization workflow for better results
                    cleaned_text = summarizer.clean_text(content)
                    article_summary = summarizer.final_summarize(cleaned_text)
                    
                    # Save to memory and disk cache
                    cache_data = {
                        'title': article_data['title'],
                        'summary': article_summary,
                        'timestamp': time()
                    }
                    ARTICLE_SUMMARY_CACHE[article_link] = cache_data
                    
                    # Save to disk
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f)
                    
                    safe_print(f"Generated new summary for: {subject_line}")
            finally:
                # Clean up resources
                summarizer.cleanup_resources(silent=True)
        except Exception as e:
            safe_print(f"Error getting article summary: {e}")
            article_summary = subject_line  # Fallback to just using the subject

    # Limit summary to avoid memory issues
    if article_summary and len(article_summary) > 500:
        article_summary = article_summary[:500]
    
    # Default questions in case the model fails
    default_questions = [
        f"What are the fundamental concepts of {subject_line}?",
        f"What are the basic principles related to {subject_line}?"
    ]
    
    # Try to use Llama, but have fallbacks
    llm = get_llama_model()
    if llm is not None:
        try:
            # Use Llama to generate conceptual questions with improved context
            prompt = f"""[INST] Based on this article title: "{article_title}" and subject area: "{subject_line}", 
            generate two focused conceptual questions that would help someone understand the fundamental principles involved.
            
            Each question should:
            1. Be specific yet broadly applicable (not overly technical)
            2. Include just enough context in the question itself to stand alone
            3. Focus on core concepts rather than peripheral details
            
            For example, instead of "What is Y Combinator?" a better question with context would be 
            "What is the concept of amicus curiae brief in legal proceedings?"
            
            Format your response as a numbered list with just the questions:
            1. First question with sufficient context
            2. Second question with sufficient context
            [/INST]"""
            
            # Get the questions using Llama
            with LLM_LOCK:
                questions_response = llm(prompt, max_tokens=256, temperature=0.4)
            
            # Extract questions from the response
            questions_text = questions_response["choices"][0]["text"].strip()
            
            # Parse the numbered list
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):  # Lines starting with a number and period
                    # Extract the question text
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        questions.append(question)
            
            # If we got valid questions, use them
            if questions:
                # Ensure we only return 2 questions
                while len(questions) < 2:
                    questions.append(default_questions[len(questions)])
                return questions[:2]
            
        except Exception as e:
            safe_print(f"Error generating questions with Llama: {e}, using defaults")
    
    # Use default questions if Llama failed
    return default_questions

def generate_additional_questions(subject, article_title):
    """Generate 5 additional questions for further exploration"""
    
    # Default questions in case the model fails
    default_questions = [
        f"What are the historical developments that led to {subject}?",
        f"How is {subject} applied in different industries?",
        f"What are the ethical considerations related to {subject}?",
        f"How might {subject} evolve in the next decade?",
        f"What are the alternative approaches to solving problems that {subject} addresses?"
    ]
    
    # Try to use Llama, but have fallbacks
    llm = get_llama_model()
    if llm is not None:
        try:
            # Use Llama to generate additional questions
            prompt = f"""[INST] Based on this subject: "{subject}" from the article "{article_title}", 
            generate 5 additional questions for readers to explore further.
            
            These should be thought-provoking questions that go beyond the basic concepts
            and encourage deeper research. The questions should cover different aspects like:
            - Historical context
            - Practical applications
            - Ethical considerations
            - Future developments
            - Alternative approaches
            
            Format your response as a numbered list with just the questions:
            1. First question
            2. Second question
            3. Third question
            4. Fourth question
            5. Fifth question
            [/INST]"""
            
            # Get the questions using Llama
            with LLM_LOCK:
                questions_response = llm(prompt, max_tokens=512, temperature=0.4)
            
            # Extract questions from the response
            questions_text = questions_response["choices"][0]["text"].strip()
            
            # Parse the numbered list
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):  # Lines starting with a number and period
                    # Extract the question text
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        questions.append(question)
            
            # If we got valid questions, use them
            if questions and len(questions) >= 5:
                return questions[:5]
            else:
                # If we didn't get enough questions, use defaults to fill in
                combined_questions = questions.copy()
                for i in range(5 - len(combined_questions)):
                    combined_questions.append(default_questions[i])
                return combined_questions
            
        except Exception as e:
            safe_print(f"Error generating additional questions with Llama: {e}, using defaults")
    
    # Use default questions if Llama failed
    return default_questions

def search_for_question(question, start=0, candidate_count=10, result_count=2):
    """Search the web for information related to a specific question"""
    safe_print(f"Searching for: {question} (slice start={start})")
    
    # Create a cache key for this search including the slice start
    cache_key = hashlib.md5(f"{question}:{start}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"search_{cache_key}.json")
    
    # Check if we have cached results for this slice
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                safe_print(f"Using cached search results for: {question}")
                return cached_data
            
        except Exception as e:
            safe_print(f"Error reading search cache: {e}")
    
    search_results = []
    try:
        # Add a significant delay before initiating the search to avoid rate limits
        sleep(5)  # Sleep for 5 seconds before searching
        
        # Use DuckDuckGo for searching
        with DDGS() as ddgs:
            # Add 'filetype:pdf' to the query if it's a technical or academic question
            is_academic = any(term in question.lower() for term in ['research', 'academic', 'paper', 'study', 'theorem', 'theory', 'mathematical'])
            is_technical = any(term in question.lower() for term in ['how', 'technical', 'implementation', 'code', 'programming', 'development'])
            
            # For certain types of questions, include PDFs in search
            pdf_query = f"{question} filetype:pdf" if (is_academic or is_technical) else question
            
            # First try standard search retrieving candidate_count links
            safe_print(f"Performing standard search for: {question}")
            results = list(ddgs.text(question, max_results=candidate_count))
            
            # Add delay between searches to avoid rate limits
            sleep(3)
            
            # For academic/technical questions, also try searching for PDFs
            if is_academic or is_technical:
                safe_print(f"Performing PDF search for: {pdf_query}")
                pdf_results = list(ddgs.text(pdf_query, max_results=candidate_count))
                # Combine and dedupe results to candidate_count
                combined = results + pdf_results
                seen = set()
                deduped = []
                for r in combined:
                    href = r.get('href')
                    if href and href not in seen:
                        seen.add(href)
                        deduped.append(r)
                results = deduped[:candidate_count]
            
            # Extract and validate URLs
            urls = [r['href'] for r in results if 'href' in r and r['href'].startswith('http')]
            
            if not urls:
                safe_print(f"No valid search results for: {question}, using LLM fallback")
                # LLM fallback answer
                llm = get_llama_model()
                if llm:
                    with LLM_LOCK:
                        resp = llm(f"[INST] Based on your knowledge, answer this question: {question} [/INST]", max_tokens=256, temperature=0.5)
                    ans = resp["choices"][0]["text"].strip()
                else:
                    ans = fallback_llm_call(f"[INST] Answer: {question} [/INST]")["choices"][0]["text"].strip()
                return [{ 'url': None, 'summary': ans, 'is_pdf': False }]
            
            # Select the slice of URLs for this question
            selected_urls = urls[start:start+result_count]
            if not selected_urls:
                safe_print(f"No search results in slice for: {question}, using LLM fallback")
                llm = get_llama_model()
                if llm:
                    with LLM_LOCK:
                        resp = llm(f"[INST] Based on your knowledge, answer this question: {question} [/INST]", max_tokens=256, temperature=0.5)
                    ans = resp["choices"][0]["text"].strip()
                else:
                    ans = fallback_llm_call(f"[INST] Answer: {question} [/INST]")["choices"][0]["text"].strip()
                return [{ 'url': None, 'summary': ans, 'is_pdf': False }]
            # Process each selected URL to extract and summarize content
            for url in selected_urls:
                try:
                    safe_print(f"Processing URL: {url}")
                    # Extract content with error handling
                    content, error = extract_content_safely(url)
                    
                    if content:
                        # Generate summary
                        summary, summary_error = summarize_content_safely(content)
                        
                        if summary:
                            # Add to results
                            search_results.append({
                                'url': url,
                                'summary': summary,
                                'is_pdf': is_pdf_url(url)
                            })
                        else:
                            safe_print(f"Could not summarize content from {url}: {summary_error}")
                    else:
                        safe_print(f"Failed to extract content from {url}: {error}")
                    
                    # Add a larger delay to avoid rate limiting
                    sleep(3)  # Increase from 1 to 3 seconds
                except Exception as e:
                    safe_print(f"Error processing URL {url}: {e}")
            
            # If all urls failed to summarize, use LLM fallback
            if not search_results and urls:
                safe_print(f"No summaries generated for: {question}, using LLM fallback")
                llm = get_llama_model()
                if llm:
                    with LLM_LOCK:
                        resp = llm(f"[INST] Based on your knowledge, answer this question: {question} [/INST]", max_tokens=256, temperature=0.5)
                    ans = resp["choices"][0]["text"].strip()
                else:
                    ans = fallback_llm_call(f"[INST] Answer: {question} [/INST]")["choices"][0]["text"].strip()
                search_results.append({ 'url': None, 'summary': ans, 'is_pdf': False })
            
            # Cache the results
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(search_results, f)
            
            return search_results
    except Exception as e:
        safe_print(f"Error searching for question '{question}': {e}, using LLM fallback")
        llm = get_llama_model()
        if llm:
            with LLM_LOCK:
                resp = llm(f"[INST] Based on your knowledge, answer this question: {question} [/INST]", max_tokens=256, temperature=0.5)
            ans = resp["choices"][0]["text"].strip()
        else:
            ans = fallback_llm_call(f"[INST] Answer: {question} [/INST]")["choices"][0]["text"].strip()
        return [{ 'url': None, 'summary': ans, 'is_pdf': False }]

async def process_single_article(article_data):
    """Process a single article to generate questions, search, and summarize results"""
    article_subject = article_data['subject']
    article_link = article_data['link']
    article_title = article_data['title']
    
    safe_print(f"\nProcessing article: {article_subject}")
    
    # Get the cached article summary if available
    article_summary = None
    
    # Check memory cache first (fastest)
    if article_link in ARTICLE_SUMMARY_CACHE:
        article_summary = ARTICLE_SUMMARY_CACHE[article_link].get('summary', '')
        safe_print(f"Using cached article summary for: {article_title}")
    
    # If not in memory cache, check disk cache
    if not article_summary:
        url_hash = hashlib.md5(article_link.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"summary_{url_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    article_summary = cached_data.get('summary', '')
                    # Update memory cache
                    ARTICLE_SUMMARY_CACHE[article_link] = cached_data
                    safe_print(f"Using cached article summary from disk for: {article_title}")
            except Exception as e:
                safe_print(f"Error reading cache: {e}")
        
        # If still not in cache, generate summary
        if article_link not in ARTICLE_SUMMARY_CACHE:
            # Create a dedicated summarizer instance
            summarizer = HackerNewsSummarizer(verbose=False)
            try:
                # Extract and summarize content
                content = summarizer.extract_article_content(article_link)
                if content:
                    # Use the full summarization workflow for better results
                    cleaned_text = summarizer.clean_text(content)
                    article_summary = summarizer.final_summarize(cleaned_text)
                    
                    # Save to memory and disk cache
                    cache_data = {
                        'title': article_title,
                        'summary': article_summary,
                        'timestamp': time()
                    }
                    ARTICLE_SUMMARY_CACHE[article_link] = cache_data
                    
                    # Save to disk
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f)
                    
                    safe_print(f"Generated new summary for: {article_title}")
            finally:
                # Clean up resources
                summarizer.cleanup_resources(silent=True)
    
    # Generate deep dive questions
    questions = generate_deep_dive_questions(article_data)
    safe_print(f"Generated {len(questions)} questions for: {article_subject}")
    
    # Process each question in sequence with slicing (to avoid rate limits)
    all_results = {}
    for idx, question in enumerate(questions):
        # Search for information related to the question using slice offset idx*2
        search_results = search_for_question(question, start=idx*2)
        all_results[question] = search_results
        # Add a delay between searches to avoid rate limits
        safe_print(f"Waiting 5 seconds before next search to avoid rate limits...")
        await asyncio.sleep(5)  # Increase from 2 to 5 seconds
    
    return {
        'article': article_data,
        'questions': all_results
    }

def format_blog_article(all_article_results):
    """Format all the research results into a structured blog article"""
    # Create a text formatter
    summarizer = get_global_summarizer()
    
    # Sort articles by score to get the best one first
    sorted_articles = sorted(all_article_results, key=lambda x: x['article']['score'], reverse=True)
    
    # Create the blog article structure
    blog = []
    
    # Get the top article to use as the main focus
    main_article = sorted_articles[0]['article']
    
    # Normalize the subject to remove arrow notation
    main_subject = normalize_title(main_article['subject'])
    
    # Add the title and introduction
    blog.append(f"# Deep Dive: {main_subject}")
    blog.append("\n## Introduction")
    
    # Get article summaries
    article_summaries = []
    for article_result in sorted_articles:
        article = article_result['article']
        article_link = article['link']
        
        # Try to get the article summary
        if article_link in ARTICLE_SUMMARY_CACHE:
            article_summaries.append(ARTICLE_SUMMARY_CACHE[article_link].get('summary', ''))
    
    # Generate an introduction using Llama based on the article summaries
    intro_prompt = f"""[INST] Based on these article summaries about {main_subject}, write a short introduction paragraph that explains the topic and why it's interesting or important:

{' '.join(article_summaries)}

Keep it under 150 words and make it engaging. [/INST]"""
    
    llm = get_llama_model()
    with LLM_LOCK:
        intro_response = llm(intro_prompt, max_tokens=512, temperature=0.4)
    
    blog.append(intro_response["choices"][0]["text"].strip())
    
    # Process each article
    for article_result in sorted_articles:
        article = article_result['article']
        article_title = article['title']
        # Normalize the subject to remove arrow notation
        article_subject = normalize_title(article['subject'])
        
        # Add article section
        blog.append(f"\n## {article_subject}: {article_title}")
        
        # Add article summary if available
        if article['link'] in ARTICLE_SUMMARY_CACHE:
            summary = ARTICLE_SUMMARY_CACHE[article['link']].get('summary', '')
            blog.append(f"### Summary\n{summary}\n")
        
        # Process questions and search results
        blog.append("### Deep Dive Questions")
        
        for question, results in article_result['questions'].items():
            blog.append(f"#### {question}")
            
            if not results:
                blog.append("*No relevant information found for this question.*\n")
                continue
            
            # Process each search result
            for i, result in enumerate(results):
                # Skip the source display when url is None
                if result.get('url') is None:
                    blog.append(f"{result['summary']}\n")
                    continue
                    
                # Special formatting for PDF sources
                if result.get('is_pdf', False):
                    blog.append(f"**Source {i+1}**: 📄 [{result['url']}]({result['url']}) *(PDF)*")
                else:
                    blog.append(f"**Source {i+1}**: [{result['url']}]({result['url']})")
                
                blog.append(f"{result['summary']}\n")
        
        blog.append("---\n")
    
    # Add a conclusion
    blog.append("## Conclusion")
    
    conclusion_prompt = f"""[INST] Based on the information about {main_subject}, write a brief conclusion paragraph that summarizes the key insights and why this topic matters.

Keep it under 100 words and end with a thought-provoking statement or question. [/INST]"""
    
    with LLM_LOCK:
        conclusion_response = llm(conclusion_prompt, max_tokens=512, temperature=0.4)
    
    blog.append(conclusion_response["choices"][0]["text"].strip())
    
    # Add additional questions for further exploration
    safe_print(f"Generating additional questions for further exploration...")
    additional_questions = generate_additional_questions(main_subject, article_title)
    
    blog.append("\n## Further Exploration")
    blog.append("Want to dive deeper into this topic? Here are some thought-provoking questions to explore:")
    
    for i, question in enumerate(additional_questions):
        blog.append(f"{i+1}. {question}")
    
    blog.append("\nFeel free to research these questions and share your findings!")
    
    # Join all sections into a single document
    blog_article = "\n\n".join(blog)
    
    # Return both the blog content and the main article reference
    return blog_article, main_article

# ===== SCRIPT EXECUTION =====

async def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Analyze articles based on user interests')
        parser.add_argument('--cache-only', action='store_true', help='Only cache articles, no analysis')
        parser.add_argument('--no-background', action='store_true', help='Disable background processing')
        args = parser.parse_args()

        # Check for no-score environment variable early and log it clearly
        skip_scoring = os.getenv('ANSYS_NO_SCORE') is not None
        if skip_scoring:
            safe_print("========================================")
            safe_print("SCORING DISABLED: ANSYS_NO_SCORE is set")
            safe_print("Articles will be cached without scoring")
            safe_print("========================================")

        # Load skip list of processed articles from environment variable
        skip_ids = set()
        skip_file = os.getenv('ANSYS_PROCESSED_IDS_FILE')
        if skip_file and os.path.exists(skip_file):
            try:
                with open(skip_file, 'r', encoding='utf-8') as sf:
                    data = json.load(sf)
                    skip_ids = set(data.get('processed_ids', []))
                safe_print(f"Skipping {len(skip_ids)} already processed articles based on skip file.")
            except Exception as e:
                safe_print(f"Error loading skip file {skip_file}: {e}")

        # Initialize article cache
        article_cache = ArticleCache(CACHE_DIR)
        safe_print(f"Using cache directory: {CACHE_DIR}")

        # If cache-only flag is set, just fetch and cache articles then exit
        if args.cache_only:
            safe_print("Cache-only mode: Fetching and caching articles...")

            # Get articles WITHOUT running background processing
            articles = article_grabber(run_background_processing=False)
            safe_print(f"Found {len(articles)} articles to cache")

            # Filter out already processed articles
            filtered = []
            skipped_count = 0
            for title, link in articles:
                url_hash = hashlib.md5(link.encode()).hexdigest()
                if url_hash in skip_ids:
                    safe_print(f"✓ Skipping already cached article: {title}")
                    skipped_count += 1
                    continue
                filtered.append((title, link))
            articles = filtered
            safe_print(f"{skipped_count} articles skipped; {len(articles)} to process.")

            if not articles:
                safe_print("No new articles to cache.")
                return

            # Manually handle article caching with minimal processing
            safe_print(f"Starting manual caching of {len(articles)} articles...")
            memory_hits = 0
            disk_hits = 0
            new_summaries = 0

            # Count disk cache hits (files in cache dir)
            for article in articles:
                title, link = article
                url_hash = hashlib.md5(link.encode()).hexdigest()
                cache_path = os.path.join(CACHE_DIR, f"summary_{url_hash}.json")

                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached_data = json.load(f)
                        disk_hits += 1
                        safe_print(f"✓ Using existing cache for: {title}")
                    except Exception as e:
                        safe_print(f"✗ Error reading cache for {title}: {e}")
                else:
                    try:
                        # Create a basic cache entry without heavyweight processing
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "title": title,
                                "url": link,
                                "cached_at": time(),
                                "summary": f"Summary for {title}"  # Just a placeholder
                            }, f)
                        new_summaries += 1
                        safe_print(f"✓ Created basic cache entry for: {title}")
                    except Exception as e:
                        safe_print(f"✗ Error creating cache for {title}: {e}")

            # Count the final cache files
            final_disk_hits = len(glob.glob(os.path.join(CACHE_DIR, 'summary_*.json')))

            safe_print("Manual caching complete.")
            safe_print(f"Summary: {memory_hits} memory cache hits, {disk_hits} disk cache hits, {new_summaries} new entries created")
            safe_print(f"Total cached articles: {final_disk_hits}")
            safe_print(f"Cache files in {CACHE_DIR}:")
            for cache_file in glob.glob(os.path.join(CACHE_DIR, 'summary_*.json')):
                try:
                    file_size = os.path.getsize(cache_file)
                    safe_print(f"  - {os.path.basename(cache_file)} ({file_size} bytes)")
                except Exception:
                    safe_print(f"  - {os.path.basename(cache_file)}")

            return

        # Get user interests from stdin if not in no-score mode
        user_interests = "technology, programming, science, AI, machine learning, finance, health, politics, education, data science"
        
        if not skip_scoring:
            # Only prompt for interests if scoring is enabled
            try:
                safe_print("\nWhat are your interests? (comma-separated topics, e.g. AI, finance, health)")
                user_input = input("> ").strip()
                if user_input:
                    user_interests = user_input
            except Exception as e:
                safe_print(f"Error getting user input: {e}, using default interests")
        
        safe_print(f"{'Using' if skip_scoring else 'Scoring based on'} interests: {user_interests}")

        # Get articles and process them
        safe_print("Fetching articles from Hacker News...")
        articles = article_grabber(run_background_processing=not args.no_background)

        # Filter out already processed articles before scoring/Q&A
        filtered = []
        skipped_count = 0
        for title, link in articles:
            url_hash = hashlib.md5(link.encode()).hexdigest()
            if url_hash in skip_ids:
                safe_print(f"✓ Skipping processed article: {title}")
                skipped_count += 1
                continue
            filtered.append((title, link))
        articles = filtered
        safe_print(f"{skipped_count} articles skipped; {len(articles)} to score and process.")

        # Wait a moment for background processing to start
        await asyncio.sleep(1)

        # Process articles based on user interests
        # The preprocess function will handle no-score logic based on environment variable
        ranked_articles = preprocess(articles, user_interests)

        if not ranked_articles:
            safe_print("No relevant articles found.")
            return

        # Process all articles instead of just the top 3
        safe_print(f"\nProcessing all {len(ranked_articles)} articles...")

        # Process each article - limit to 10 for practical reasons if there are too many
        article_limit = min(len(ranked_articles), 10)
        for article_idx in range(article_limit):
            current_article = ranked_articles[article_idx]
            subject = current_article['subject']
            # Sanitize subject for file matching
            safe_subject = re.sub(r'[^A-Za-z0-9]+', '_', subject).strip('_')
            # Skip if final article JSON already exists
            existing_files = glob.glob(os.path.join(CACHE_DIR, f"final_article_*_{safe_subject}.json"))
            if existing_files:
                safe_print(f"✓ Skipping Q&A generation for already processed article: {subject}")
                continue

            safe_print(f"\nProcessing article {article_idx+1}/{article_limit}: {current_article['title']}")
            
            # Create improved deep dive questions with better context
            original_questions = generate_deep_dive_questions(current_article)
            improved_questions = []
            
            # Improve each question with specific context from the article
            llm = get_llama_model()
            subject = current_article['subject']
            article_title = current_article['title']
            
            for q in original_questions:
                # Keep questions focused on fundamental concepts
                # Don't make the questions more specific, just ensure they're well-formatted
                # Check if it's already a well-formed question (ends with ?)
                if q.endswith('?'):
                    improved_questions.append(q)
                else:
                    # If it doesn't end with a question mark, formulate it as a question
                    improved_q = f"What are the core principles of {q}?" if not "what" in q.lower() else q
                    improved_q = f"How do the fundamental concepts of {q} work?" if not "how" in q.lower() and not "what" in q.lower() else improved_q
                    improved_questions.append(improved_q)
            
            safe_print(f"Generated {len(improved_questions)} questions about fundamental concepts for: {subject}")
            
            # Process improved questions
            all_results = {}
            for question in improved_questions:
                # Search for information related to the question
                search_results = search_for_question(question)
                all_results[question] = search_results
                # Add a delay between searches to avoid rate limits
                safe_print(f"Waiting 5 seconds before next search to avoid rate limits...")
                await asyncio.sleep(5)  # Increase from 2 to 5 seconds
            
            # Create a single article result for the current article
            article_result = {
                'article': current_article,
                'questions': all_results
            }
            
            # Format results into a blog article
            safe_print(f"\nGenerating final blog article for {article_title}...")
            
            # Create a blog format focusing on the current article
            blog = []
            
            # Add the title and introduction
            blog.append(f"# Deep Dive: {subject}")
            blog.append("\n## Introduction")
            
            # Get article summary for introduction
            article_summary = ""
            article_link = current_article['link']
            
            # Try to get the article summary
            if article_link in ARTICLE_SUMMARY_CACHE:
                article_summary = ARTICLE_SUMMARY_CACHE[article_link].get('summary', '')
            
            # Generate an introduction using Llama based on the article summary
            intro_prompt = f"""[INST] Based on this article subject: "{subject}" and article title: "{article_title}", 
            write a short introduction paragraph that explains the topic and why it's interesting or important.
            
            If available, use this summary:
            {article_summary}
            
            Keep it under 150 words and make it engaging. [/INST]"""
            
            with LLM_LOCK:
                intro_response = llm(intro_prompt, max_tokens=512, temperature=0.4)
            
            blog.append(intro_response["choices"][0]["text"].strip())
            
            # Add article section
            blog.append(f"\n## {subject}")
            
            # Add article summary if available
            if article_link in ARTICLE_SUMMARY_CACHE:
                summary = ARTICLE_SUMMARY_CACHE[article_link].get('summary', '')
                blog.append(f"### Summary\n{summary}\n")
            
            # Process questions and search results
            blog.append("### Deep Dive Questions")
            
            for question, results in all_results.items():
                blog.append(f"#### {question}")
                
                if not results:
                    blog.append("*No relevant information found for this question.*\n")
                    continue
                
                # Process each search result
                for i, result in enumerate(results):
                    # Skip the source display when url is None
                    if result.get('url') is None:
                        blog.append(f"{result['summary']}\n")
                        continue
                        
                    # Special formatting for PDF sources
                    if result.get('is_pdf', False):
                        blog.append(f"**Source {i+1}**: 📄 [{result['url']}]({result['url']}) *(PDF)*")
                    else:
                        blog.append(f"**Source {i+1}**: [{result['url']}]({result['url']})")
                    
                    blog.append(f"{result['summary']}\n")
            
            # Add a conclusion
            blog.append("## Conclusion")
            
            conclusion_prompt = f"""[INST] Based on the information about {subject} from the article "{article_title}", 
            write a brief conclusion paragraph that summarizes the key insights and why this topic matters.
            
            Keep it under 100 words and end with a thought-provoking statement or question. [/INST]"""
            
            with LLM_LOCK:
                conclusion_response = llm(conclusion_prompt, max_tokens=512, temperature=0.4)
            
            blog.append(conclusion_response["choices"][0]["text"].strip())
            
            # Add additional questions for further exploration
            safe_print(f"Generating additional questions for further exploration...")
            additional_questions = generate_additional_questions(subject, article_title)
            
            blog.append("\n## Further Exploration")
            blog.append("Want to dive deeper into this topic? Here are some thought-provoking questions to explore:")
            
            for i, question in enumerate(additional_questions):
                blog.append(f"{i+1}. {question}")
            
            blog.append("\nFeel free to research these questions and share your findings!")
            
            # Join all sections into a single document
            blog_article = "\n\n".join(blog)
            
            # Save as HTML file only (not saving markdown files anymore)
            timestamp = int(time())
            # Normalize the subject before creating the safe version
            normalized_subject = normalize_title(subject)
            safe_subject = re.sub(r'[^A-Za-z0-9]+', '_', normalized_subject).strip('_')
            
            # Create styled HTML version
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tech Deep Dive: {normalized_subject}</title>
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
        .pdf-icon::before {{
            content: "📄";
            margin-right: 5px;
        }}
        .source {{
            background-color: #e8f4fc;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .summary {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
            margin: 30px 0;
        }}
        .exploration-questions {{
            background-color: #f7f9fa;
            border: 1px solid #e3e6e8;
            border-radius: 8px;
            padding: 15px 20px;
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .exploration-questions ol {{
            padding-left: 25px;
        }}
        .exploration-questions li {{
            margin-bottom: 10px;
            font-weight: 500;
        }}
        .exploration-note {{
            font-style: italic;
            color: #666;
            margin-top: 15px;
        }}
    </style>
</head>
<body>
    {blog_article.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>").replace("**Source", "<div class='source'><strong>Source").replace("**(PDF)*", "</strong><span class='pdf-icon'></span>").replace("**", "</strong>").replace("*No relevant", "<em>No relevant").replace("*\n", "</em></div>").replace("## Further Exploration", "<h2>Further Exploration</h2><div class='exploration-questions'>").replace("Feel free to research these questions and share your findings!", "Feel free to research these questions and share your findings!</div>")}
</body>
</html>
            """
            
            # Save only HTML file - NOT Markdown
            html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'tech_deep_dive_{timestamp}_{safe_subject}.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            safe_print(f"Saved article to: {html_file}")
            
            # Cache the final article
            try:
                # Create a cache path for this final article
                cache_path = os.path.join(CACHE_DIR, f"final_article_{timestamp}_{safe_subject}.json")
                
                # Cache the content - still need to store as markdown format in the cache for compatibility
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'content': blog_article,
                        'timestamp': timestamp
                    }, f)
                safe_print(f"Cached final article to {cache_path}")
            except Exception as e:
                safe_print(f"Error caching final article: {e}")
        
        safe_print("\nAll articles processed successfully!")
        
    except Exception as e:
        safe_print(f"An error occurred in main: {e}")
        import traceback
        safe_print(traceback.format_exc())

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
  