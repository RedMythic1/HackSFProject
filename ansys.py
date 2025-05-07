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

# Third-party imports
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from transformers import pipeline
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium or WebDriver Manager not found. Install with 'pip install selenium webdriver-manager'")

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
        if "generate three deep" in instruction.lower() or "questions" in instruction.lower():
            return {
                "choices": [{"text": "1. What are the key components of this system?\n2. How does this technology compare to alternatives?\n3. What are the future implications of this development?"}]
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
                try:
                    # First try with minimal GPU usage (only 1 layer on GPU)
                    get_llama_model.instance = Llama(
    model_path="/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        n_ctx=4096,  # Reduced context size to save memory
                        n_threads=4,  # Reduced thread count
                        n_gpu_layers=1,  # Minimal GPU usage
    chat_format="mistral-instruct",
    verbose=False,
    stop=None
)
                except Exception as e:
                    safe_print(f"Error initializing Llama with GPU: {e}, falling back to CPU-only mode")
                    try:
                        # Fall back to CPU-only mode
                        get_llama_model.instance = Llama(
                            model_path="/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                            n_ctx=2048,  # Even smaller context
                            n_threads=4,
                            n_gpu_layers=0,  # CPU only
                            chat_format="mistral-instruct",
                            verbose=False,
                            stop=None
                        )
                    except Exception as e2:
                        safe_print(f"Failed to initialize Llama model: {e2}")
                        # Create a dummy model for graceful degradation
                        get_llama_model.instance = None
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
    Returns a tuple of (summary, error_message)
    """
    if not content:
        return None, "No content provided for summarization"
        
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
                
                return summary, None
            else:
                return cleaned_content, "Content too short for summarization"
        finally:
            # Always clean up resources
            summarizer.cleanup_resources(silent=True)
    except Exception as e:
        error_msg = f"Error during summarization: {str(e)}"
        safe_print(error_msg)
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
                # Generate summary
                summary, summary_error = summarize_content_safely(content, title)
                
                if summary:
                    # Store in memory cache
                    ARTICLE_SUMMARY_CACHE[link] = {
                        'title': title,
                        'summary': summary,
                        'timestamp': time()
                    }
                    
                    # Save to disk cache
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(ARTICLE_SUMMARY_CACHE[link], f)
                    
                    safe_print(f"✓ Summarized: {title}")
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

def is_valid_hn_link(link):
    """Check if link matches the format: ends with item?id= followed by 8 digits"""
    pattern = r'item\?id=\d{8}$'
    return bool(re.search(pattern, link))

# ===== CONTENT ACQUISITION FUNCTIONS =====

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

def article_grabber():
    """Retrieve articles from Hacker News"""
    # Send a GET request to Hacker News
    response = requests.get(HACKER_NEWS_URL)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all article rows (with class "athing")
    articles = []
    rows = soup.select(".athing")
    
    for row in rows:
        # Get the article title
        title_element = row.select_one(".titleline > a")
        if not title_element:
            continue
            
        title = title_element.get_text(strip=True)
        
        # Get the item ID from the row
        item_id = row.get("id")
        if not item_id:
            continue
            
        # Create the comment section URL
        comment_link = f"{HACKER_NEWS_URL}item?id={item_id}"
        
        # Verify it matches our pattern (8 digits)
        if is_valid_hn_link(f"item?id={item_id}"):
            articles.append([title, comment_link])
    
    # Launch background processing
    if asyncio.get_event_loop().is_running():
        # If we're in an async context already
        asyncio.create_task(summarize_all_articles_async(articles))
    else:
        # If we're not in an async context, create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(summarize_all_articles_async(articles))
        # Don't close the loop as it might be needed later
    
    return articles

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
    
    Args:
        articles: List of articles from article_grabber
        user_interests: User's interest string
        
    Returns:
        List of processed article data sorted by relevance score
    """
    # Define the worker function to process a single article
    def process_article(article_data):
        title = article_data[0]
        link = article_data[1]
        
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
            prompt = f"""[INST] Analyze this title and tell me what the subject is. Example: Altair at 50: Remembering the first Personal Computer -> Altair, First Personal computer. Perform this on {title}. DO NOT OUTPUT ANYTHING ELSE. NO THANK YOUs or confirmations. I just need the Subject line. [/INST]"""
            
            # Use the global Llama model with lock
            llm = get_llama_model()
            
            if llm is not None:
                try:
                    with LLM_LOCK:
                        subject_response = llm(prompt, max_tokens=64, temperature=0.1)
                    subject_line = subject_response["choices"][0]["text"].strip()
                except Exception as e:
                    safe_print(f"Error using Llama for subject extraction: {e}, using fallback")
                    subject_response = fallback_llm_call(prompt)
                    subject_line = subject_response["choices"][0]["text"].strip()
            else:
                # Use fallback if Llama model failed to initialize
                subject_response = fallback_llm_call(prompt)
            subject_line = subject_response["choices"][0]["text"].strip()
            
            # Simple relevance scoring without additional Llama calls
            # This reduces the load and potential errors
            interest_score = 0
            interest_terms = user_interests.lower().split(',')
            
            # Clean up the terms
            interest_terms = [term.strip() for term in interest_terms]
            
            # Calculate a simple score based on keyword matching
            title_lower = title.lower()
            subject_lower = subject_line.lower()
            
            # Check title and subject against each interest term
            for term in interest_terms:
                if term and len(term) > 1:  # Skip empty or single-char terms
                    if term in title_lower or term in subject_lower:
                        interest_score += 25  # Base score for term match
                    
                    # Check for partial matches
                    for word in term.split():
                        if word and len(word) > 3 and (word in title_lower or word in subject_lower):
                            interest_score += 10
            
            # Ensure score is within 0-100 range
            interest_score = min(100, interest_score)
            
            # Create result data
            result = {
                'title': title,
                'link': link,
                'subject': subject_line,
                'score': interest_score
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
                'subject': "Error processing",
                'score': 0
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
    
    # Sort by score in descending order and keep only top 3
    results.sort(key=lambda x: x['score'], reverse=True)
    top_results = results[:3]
    
    safe_print(f"Processed {len(articles)} articles, top scores: {[r['score'] for r in top_results]}")
    
    return top_results

def generate_deep_dive_questions(article_data):
    """Generate three deep dive questions based on the article summary"""
    subject_line = article_data['subject']
    article_link = article_data['link']
    
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
        f"What are the key aspects of {subject_line}?",
        f"How does {subject_line} compare to alternatives?",
        f"What future developments might we see with {subject_line}?"
    ]
    
    # Try to use Llama, but have fallbacks
    llm = get_llama_model()
    if llm is not None:
        try:
            # Use Llama to generate deep dive questions
            prompt = f"""[INST] Based on this article subject: "{subject_line}" and the following summary snippet: "{article_summary[:200]}...", 
            generate three deep, insightful questions about this topic.
            
            Format your response as a numbered list with just the questions:
            1. First question
            2. Second question
            3. Third question
            [/INST]"""
            
            # Get the questions using Llama
            with LLM_LOCK:
                questions_response = llm(prompt, max_tokens=256, temperature=0.7)
            
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
                # Ensure we only return 3 questions
                while len(questions) < 3:
                    questions.append(default_questions[len(questions)])
                return questions[:3]
            
        except Exception as e:
            safe_print(f"Error generating questions with Llama: {e}, using defaults")
    
    # Use default questions if Llama failed
    return default_questions

def search_for_question(question, max_results=2):
    """Search the web for information related to a specific question"""
    safe_print(f"Searching for: {question}")
    
    # Create a cache key for this search
    cache_key = hashlib.md5(question.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"search_{cache_key}.json")
    
    # Check if we have cached results
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
        # Use DuckDuckGo for searching
        with DDGS() as ddgs:
            # Add 'filetype:pdf' to the query if it's a technical or academic question
            is_academic = any(term in question.lower() for term in ['research', 'academic', 'paper', 'study', 'theorem', 'theory', 'mathematical'])
            is_technical = any(term in question.lower() for term in ['how', 'technical', 'implementation', 'code', 'programming', 'development'])
            
            # For certain types of questions, include PDFs in search
            pdf_query = f"{question} filetype:pdf" if (is_academic or is_technical) else question
            
            # First try standard search
            results = list(ddgs.text(question, max_results=max_results))
            
            # For academic/technical questions, also try searching for PDFs
            if is_academic or is_technical:
                pdf_results = list(ddgs.text(pdf_query, max_results=2))
                # Combine results, but don't exceed max_results total
                all_results = results + pdf_results
                results = all_results[:max_results]
            
            # Extract URLs
            urls = [r['href'] for r in results if 'href' in r]
            
            # Ensure we have valid URLs
            urls = [url for url in urls if url.startswith('http')]
            
            if not urls:
                safe_print(f"No valid search results for: {question}")
                return []
            
            # Process each URL to extract and summarize content
            for url in urls:
                try:
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
                        
                    # Add a small delay to avoid rate limiting
                    sleep(1)
                except Exception as e:
                    safe_print(f"Error processing URL {url}: {e}")
            
            # If all urls failed, add a fallback result
            if not search_results and urls:
                search_results.append({
                    'url': urls[0],
                    'summary': "Could not extract or summarize content from any of the search results. Please visit the source URL directly for more information.",
                    'is_pdf': is_pdf_url(urls[0])
                })
            
            # Cache the results
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(search_results, f)
            
            return search_results
    except Exception as e:
        safe_print(f"Error searching for question '{question}': {e}")
        return []

async def process_single_article(article_data):
    """Process a single article to generate questions, search, and summarize results"""
    article_title = article_data['title']
    article_subject = article_data['subject']
    article_link = article_data['link']
    
    safe_print(f"\nProcessing article: {article_title}")
    
    # Ensure we have a summary for this article before proceeding
    if article_link not in ARTICLE_SUMMARY_CACHE:
        # Check disk cache
        url_hash = hashlib.md5(article_link.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"summary_{url_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Update memory cache
                    ARTICLE_SUMMARY_CACHE[article_link] = cached_data
                    safe_print(f"Loaded cached summary for: {article_title}")
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
    
    # Process each question in sequence (to avoid rate limits)
    all_results = {}
    for question in questions:
        # Search for information related to the question
        search_results = search_for_question(question)
        all_results[question] = search_results
        
        # Add a delay between searches to avoid rate limits
        await asyncio.sleep(2)
                                
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
    
    # Add the title and introduction
    blog.append(f"# Deep Dive: {main_article['subject']}")
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
    intro_prompt = f"""[INST] Based on these article summaries about {main_article['subject']}, write a short introduction paragraph that explains the topic and why it's interesting or important:

{' '.join(article_summaries)}

Keep it under 150 words and make it engaging. [/INST]"""
    
    llm = get_llama_model()
    with LLM_LOCK:
        intro_response = llm(intro_prompt, max_tokens=512, temperature=0.7)
    
    blog.append(intro_response["choices"][0]["text"].strip())
    
    # Process each article
    for article_result in sorted_articles:
        article = article_result['article']
        article_title = article['title']
        article_subject = article['subject']
        
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
                # Special formatting for PDF sources
                if result.get('is_pdf', False):
                    blog.append(f"**Source {i+1}**: 📄 [{result['url']}]({result['url']}) *(PDF)*")
                else:
                    blog.append(f"**Source {i+1}**: [{result['url']}]({result['url']})")
                
                blog.append(f"{result['summary']}\n")
        
        blog.append("---\n")
    
    # Add a conclusion
    blog.append("## Conclusion")
    
    conclusion_prompt = f"""[INST] Based on the information about {main_article['subject']}, write a brief conclusion paragraph that summarizes the key insights and why this topic matters.

Keep it under 100 words and end with a thought-provoking statement or question. [/INST]"""
    
    with LLM_LOCK:
        conclusion_response = llm(conclusion_prompt, max_tokens=512, temperature=0.7)
    
    blog.append(conclusion_response["choices"][0]["text"].strip())
    
    # Join all sections into a single document
    blog_content = "\n\n".join(blog)
    
    # Return both the blog content and the main article reference
    return blog_content, main_article

# ===== SCRIPT EXECUTION =====

async def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Analyze articles based on user interests')
        parser.add_argument('--cache-only', action='store_true', help='Only cache articles, no analysis')
        args = parser.parse_args()
        
        # If cache-only flag is set, just fetch and cache articles then exit
        if args.cache_only:
            safe_print("Cache-only mode: Fetching and caching articles...")
            articles = article_grabber()
            # Wait for background processing to complete (at least partially)
            await asyncio.sleep(5)
            safe_print(f"Cached {len(articles)} articles")
            return

        # Normal flow - Get user interests
        user_interests = input('What are your 4 interests? (separated by commas): \n')
        
        # Get articles and process them
        safe_print("Fetching articles from Hacker News...")
        articles = article_grabber()
        
        # Wait a moment for background processing to start
        await asyncio.sleep(1)
        
        # Process articles based on user interests
        ranked_articles = preprocess(articles, user_interests)
        
        if not ranked_articles:
            safe_print("No relevant articles found. Please try again with different interests.")
            return
        
        # Display top articles but only process the highest scoring one
        safe_print(f"\nTop articles based on your interests:")
        for i, article in enumerate(ranked_articles[:3]):
            safe_print(f"{i+1}. {article['title']} (relevance score: {article['score']})")
        
        # Get the top article
        top_article = ranked_articles[0]
        safe_print(f"\nProcessing the highest scoring article: {top_article['title']}")
        
        # Create improved deep dive questions with better context
        original_questions = generate_deep_dive_questions(top_article)
        improved_questions = []
        
        # Improve each question with specific context from the article
        llm = get_llama_model()
        subject = top_article['subject']
        article_title = top_article['title']
        
        for q in original_questions:
            # Make the question more specific with full context
            improved_q = f"What is the significance of {subject} in the context of {article_title}?" if "What" in q else q
            improved_q = f"How does {subject} work in relation to {article_title}?" if "How" in q else improved_q
            improved_q = f"What future applications might {subject} have in the field of {article_title}?" if "future" in q.lower() else improved_q
            improved_questions.append(improved_q)
        
        safe_print(f"Generated {len(improved_questions)} improved questions for: {subject}")
        
        # Process improved questions
        all_results = {}
        for question in improved_questions:
            # Search for information related to the question
            search_results = search_for_question(question)
            all_results[question] = search_results
            # Add a delay between searches to avoid rate limits
            await asyncio.sleep(2)
        
        # Create a single article result for the top article
        article_result = {
            'article': top_article,
            'questions': all_results
        }
        
        # Format results into a blog article
        safe_print("\nGenerating final blog article...")
        
        # Create a simpler blog format focusing only on the top article
        blog = []
        
        # Add the title and introduction
        blog.append(f"# Deep Dive: {top_article['subject']}")
        blog.append("\n## Introduction")
        
        # Get article summary for introduction
        article_summary = ""
        article_link = top_article['link']
        
        # Try to get the article summary
        if article_link in ARTICLE_SUMMARY_CACHE:
            article_summary = ARTICLE_SUMMARY_CACHE[article_link].get('summary', '')
        
        # Generate an introduction using Llama based on the article summary
        intro_prompt = f"""[INST] Based on this article subject: "{top_article['subject']}" and article title: "{top_article['title']}", 
        write a short introduction paragraph that explains the topic and why it's interesting or important.
        
        If available, use this summary:
        {article_summary}
        
        Keep it under 150 words and make it engaging. [/INST]"""
        
        with LLM_LOCK:
            intro_response = llm(intro_prompt, max_tokens=512, temperature=0.7)
        
        blog.append(intro_response["choices"][0]["text"].strip())
        
        # Add article section
        blog.append(f"\n## {top_article['subject']}")
        
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
                # Special formatting for PDF sources
                if result.get('is_pdf', False):
                    blog.append(f"**Source {i+1}**: 📄 [{result['url']}]({result['url']}) *(PDF)*")
                else:
                    blog.append(f"**Source {i+1}**: [{result['url']}]({result['url']})")
                
                blog.append(f"{result['summary']}\n")
        
        # Add a conclusion
        blog.append("## Conclusion")
        
        conclusion_prompt = f"""[INST] Based on the information about {top_article['subject']} from the article "{top_article['title']}", 
        write a brief conclusion paragraph that summarizes the key insights and why this topic matters.
        
        Keep it under 100 words and end with a thought-provoking statement or question. [/INST]"""
        
        with LLM_LOCK:
            conclusion_response = llm(conclusion_prompt, max_tokens=512, temperature=0.7)
        
        blog.append(conclusion_response["choices"][0]["text"].strip())
        
        # Join all sections into a single document
        blog_article = "\n\n".join(blog)
        
        # Output the final blog
        safe_print("\n" + "="*50)
        safe_print("YOUR PERSONALIZED TECH DEEP DIVE")
        safe_print("="*50 + "\n")
        safe_print(blog_article)
        
        # Save the blog to a file
        timestamp = int(time())
        blog_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'tech_deep_dive_{timestamp}.md')
        with open(blog_file, 'w', encoding='utf-8') as f:
            f.write(blog_article)
        
        # Also save a styled HTML version for better formatting
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tech Deep Dive: {top_article['subject']}</title>
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
    </style>
</head>
<body>
    {blog_article.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>").replace("#### ", "<h4>").replace("\n\n", "<br><br>").replace("**Source", "<div class='source'><strong>Source").replace("**(PDF)*", "</strong><span class='pdf-icon'></span>").replace("**", "</strong>").replace("*No relevant", "<em>No relevant").replace("*\n", "</em></div>")}
</body>
</html>
        """
        
        html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'tech_deep_dive_{timestamp}.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        safe_print(f"\nBlog saved to: {blog_file}")
        safe_print(f"HTML version saved to: {html_file}")
    except Exception as e:
        safe_print(f"An error occurred in main: {e}")
        import traceback
        safe_print(traceback.format_exc())

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
  