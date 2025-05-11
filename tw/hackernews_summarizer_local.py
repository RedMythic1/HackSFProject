"""
HACKERNEWS SUMMARIZER - A tool to extract, summarize and analyze Hacker News posts with their comments

This tool takes a Hacker News URL, extracts both the linked article and the comments,
and provides a synthesized summary of both the article content and the community discussion.
"""

# ===============================================================================
# CLEANUP PATCHES - Prevent segmentation faults and errors during garbage collection
# ===============================================================================

# Patch problematic destructors in llama-cpp-python to prevent errors during garbage collection
import llama_cpp
import types

# First disable sampler __del__ method
if hasattr(llama_cpp, '_internals') and hasattr(llama_cpp._internals, 'LlamaSampler'):
    llama_cpp._internals.LlamaSampler.__del__ = lambda self: None

# Replace Llama's __del__ with a safer version
def _safe_llama_del(self):
    try:
        # Handle Llama cleanup safely
        if hasattr(self, 'ctx') and self.ctx is not None:
            # Clear the reference that might cause errors
            self.ctx = None
    except:
        pass

# Find and patch the Llama class
if hasattr(llama_cpp, 'llama') and hasattr(llama_cpp.llama, 'Llama'):
    # Save original for reference
    if hasattr(llama_cpp.llama.Llama, '__del__'):
        original_del = llama_cpp.llama.Llama.__del__
        # Apply the safer version
        llama_cpp.llama.Llama.__del__ = _safe_llama_del

# ===============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# ===============================================================================

# Core libraries for AI models
from llama_cpp import Llama  # AI model for text processing
try:
    from transformers import pipeline, AutoTokenizer, AutoModel  # For summarization and embedding
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not found. Install with 'pip install transformers'")

# Standard libraries
import sys
import os
import re  # For text pattern matching
from time import time  # For timing and pauses
import requests  # For fetching web content
from bs4 import BeautifulSoup  # For parsing HTML
import hashlib  # For creating unique file names
import threading  # For thread management
try:
    import torch  # For AI model operations
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Install with 'pip install torch'")

from typing import List, Dict, Optional, Union, Tuple, Any  # For type hints
from functools import lru_cache  # For function result caching
import concurrent.futures  # For parallel processing
import logging  # For better logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('hackernews_summarizer')

# ===============================================================================
# MAIN SUMMARIZER CLASS
# ===============================================================================

class HackerNewsSummarizer:
    """Main class for summarizing Hacker News posts and their linked articles"""
    
    def __init__(self, 
                 model_path: str = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the summarizer with models and configuration
        
        Args:
            model_path: Path to the LLM model file
            cache_dir: Directory to cache web content and model results
            verbose: Whether to show detailed progress output
        """
        # Set verbosity
        self.verbose = verbose
        
        # Initialize resource locks
        self._model_cache_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        
        # Initialize caches
        self._model_cache = {}
        self._vector_model_cache = {}
        self._timings = {}
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # If model_path is None, we'll search for the model based on common locations
        if model_path is None:
            # Look for model in multiple locations
            possible_model_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                os.path.expanduser("~/llama-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                os.path.expanduser("~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            ]
            
            # Check if model path is specified in environment variables
            env_model_path = os.environ.get("LLAMA_MODEL_PATH")
            if env_model_path:
                possible_model_paths.insert(0, env_model_path)
            
            # Find the first valid model path
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    if verbose:
                        print(f"Found model at: {model_path}")
                    break
            
            # If we still don't have a model path after searching, use the default
            if model_path is None:
                model_path = "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
                
        # Store model path
        self.model_path = model_path
        
        # Initialize models (lazy loading)
        self._llm = None
        self._summarizer = None
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        try:
            self.cleanup_resources(silent=True)
        except:
            # Prevent any exceptions during garbage collection
            pass
    
    def log(self, level: str, message: str) -> None:
        """Log a message if verbose mode is enabled"""
        if not self.verbose:
            return
        
        try:
            if level.lower() == 'info':
                logger.info(message)
            elif level.lower() == 'warning':
                logger.warning(message)
            elif level.lower() == 'error':
                logger.error(message)
            elif level.lower() == 'debug':
                logger.debug(message)
        except:
            # Fallback if logger is not available (during shutdown)
            if level.lower() != 'debug':  # Skip debug messages in fallback
                print(f"[{level.upper()}] {message}")
    
    def time_function(self, func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            execution_time = end_time - start_time
            
            if func_name in self._timings:
                self._timings[func_name].append(execution_time)
            else:
                self._timings[func_name] = [execution_time]
                
            self.log('debug', f"Function {func_name} took {execution_time:.3f}s")
            return result
        return wrapper

    def get_cache_path(self, url: str) -> str:
        """Generate a cache file path for a given URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.html")
    
    def fetch_url_with_cache(self, url: str, cache_timeout: int = 3600) -> str:
        """Fetch URL content with caching to reduce network requests"""
        cache_path = self.get_cache_path(url)
        
        # Check if we have a valid cache file
        if os.path.exists(cache_path):
            file_age = time() - os.path.getmtime(cache_path)
            if file_age < cache_timeout:
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        self.log('info', f"Using cached version of {url}")
                        return f.read()
                except Exception as e:
                    self.log('warning', f"Cache read error: {e}")
        
        # Fetch fresh content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            self.log('info', f"Fetching {url}...")
            response = requests.get(url, headers=headers, timeout=30)
            
            # Special handling for 403 Forbidden errors
            if response.status_code == 403:
                error_msg = f"Access forbidden (HTTP 403) - Unable to access {url}. The site may be blocking web scrapers."
                self.log('error', error_msg)
                raise PermissionError(error_msg)
                
            response.raise_for_status()
            content = response.text
            
            # Save to cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                self.log('warning', f"Cache write error: {e}")
                
            return content
        except requests.exceptions.HTTPError as e:
            # Handle various HTTP errors
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                error_msg = f"HTTP Error {status_code} for {url}"
                
                if status_code == 403:
                    error_msg = f"Access forbidden (HTTP 403) - The site {url} is blocking our access. Try manual browsing instead."
                elif status_code == 404:
                    error_msg = f"Page not found (HTTP 404) - The URL {url} does not exist."
                elif status_code == 429:
                    error_msg = f"Too many requests (HTTP 429) - The site {url} is rate limiting our access. Try again later."
                elif status_code >= 500:
                    error_msg = f"Server error (HTTP {status_code}) - The site {url} is experiencing issues. Try again later."
                
                self.log('error', error_msg)
                raise RuntimeError(error_msg)
            self.log('error', f"Failed to fetch {url}: {e}")
            raise
        except Exception as e:
            self.log('error', f"Failed to fetch {url}: {e}")
            # Try to use expired cache if available
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        self.log('info', f"Using expired cache for {url}")
                        return f.read()
                except Exception:
                    pass
            raise
            
    @property
    def llm(self):
        """Lazy-loaded LLM model"""
        if self._llm is None:
            with self._model_cache_lock:
                if self._llm is None:
                    self.log('info', "Initializing LLM model...")
                    self._llm = Llama(
                        model_path=self.model_path,
                        n_ctx=16384,  # Reduced context for less memory usage
                        n_threads=4,
                        n_gpu_layers=1,  # Minimal GPU usage
                        chat_format="mistral-instruct",
                        verbose=False,
                        stop=None
                    )
        return self._llm
        
    @property
    def summarizer(self):
        """Lazy-loaded summarization model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        if self._summarizer is None:
            with self._model_cache_lock:
                if self._summarizer is None:
                    self.log('info', "Initializing summarization model...")
                    try:
                        self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                        self.log('info', "Summarization model initialized successfully")
                    except Exception as e:
                        self.log('error', f"Failed to initialize summarization model: {e}")
                        self._summarizer = None
        return self._summarizer
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common JavaScript artifacts
        text = re.sub(r'undefined|null|NaN|Infinity', '', text)
        
        # Remove URLs to reduce noise
        text = re.sub(r'https?://\S+', '[URL]', text)
        
        return text.strip()
    
    def offline_summarize(self, text: str) -> str:
        """Summarize text using the offline LLM model"""
        if not text:
            return ""
            
        # Clean the text
        clean_text = self.clean_text(text)
        
        # If text is too short, just return it
        if len(clean_text.split()) < 100:
            return clean_text
            
        try:
            # Sampling the text to reduce length for faster processing
            sampled_text = self.sample_text(clean_text)
            
            # Prompt for summarization
            prompt = f"""[INST]
Summarize the following text in a clear, concise way while preserving the key information and important details.
Write the summary in third person and stay true to the original information without adding your own opinions.
Aim for approximately 250-300 words.

TEXT TO SUMMARIZE:
{sampled_text}
[/INST]"""

            # Call the LLM with proper locking
            with self._llm_lock:
                response = self.llm(prompt, max_tokens=512, temperature=0.3)
                
            # Extract and return the summary
            return response["choices"][0]["text"].strip()
        except Exception as e:
            self.log('error', f"Error in offline summarization: {e}")
            # Return a truncated version of the text as fallback
            return clean_text[:1000] + "..."
    
    def sample_text(self, text: str, target_words: int = 1000) -> str:
        """Sample a large text to reduce size while maintaining representativeness"""
        words = text.split()
        
        # If text is already short enough, just return it
        if len(words) <= target_words:
            return text
            
        # For very long texts, take beginning, middle and end sections
        if len(words) > target_words * 3:
            # Calculate section sizes
            section_size = target_words // 3
            
            # Extract beginning
            beginning = ' '.join(words[:section_size])
            
            # Extract middle
            middle_start = len(words) // 2 - section_size // 2
            middle = ' '.join(words[middle_start:middle_start+section_size])
            
            # Extract end
            end = ' '.join(words[-section_size:])
            
            return f"{beginning}\n\n[...content omitted...]\n\n{middle}\n\n[...content omitted...]\n\n{end}"
        
        # For moderately long texts, sample sentences
        sentence_end = re.compile(r'[.!?]\s+')
        sentences = sentence_end.split(text)
        
        # Calculate rough sampling rate
        sample_rate = max(1, len(sentences) // (target_words // 20))  # Approximate 20 words per sentence
        
        # Always include the first few and last few sentences
        num_bookend_sentences = min(3, len(sentences) // 10)
        
        # Sample the middle sentences
        middle_sentences = sentences[num_bookend_sentences:-num_bookend_sentences]
        sampled_middle = [s for i, s in enumerate(middle_sentences) if i % sample_rate == 0]
        
        # Combine beginning, sampled middle, and end
        sampled_sentences = sentences[:num_bookend_sentences] + sampled_middle + sentences[-num_bookend_sentences:]
        
        return '. '.join(sampled_sentences) + '.'
    
    def final_summarize(self, text: str) -> str:
        """Generate a final summary of text, using transformers if available or falling back to LLM"""
        if not text or len(text.strip()) == 0:
            return "No content available to summarize."
            
        # Clean the text
        clean_text = self.clean_text(text)
        
        # If text is short, just return it
        if len(clean_text.split()) < 50:
            return clean_text
            
        # Try to use transformers pipeline if available
        if TRANSFORMERS_AVAILABLE and self.summarizer:
            try:
                self.log('info', "Using transformers for summarization")
                
                # For very long text, we need to chunk it
                if len(clean_text) > 1000:
                    chunks = self.split_into_chunks(clean_text)
                    chunk_summaries = []
                    
                    # Process each chunk
                    for chunk in chunks:
                        result = self.summarizer(chunk["text"], max_length=100, min_length=30, do_sample=False)
                        chunk_summaries.append(result[0]["summary_text"])
                    
                    # If we have multiple chunks, recursively summarize the combined summaries
                    if len(chunk_summaries) > 1:
                        combined_summaries = " ".join(chunk_summaries)
                        return self.final_summarize(combined_summaries)
                    else:
                        return chunk_summaries[0]
                else:
                    # For shorter text, summarize directly
                    result = self.summarizer(clean_text, max_length=150, min_length=50, do_sample=False)
                    return result[0]["summary_text"]
            except Exception as e:
                self.log('warning', f"Error using transformers for summarization: {e}, falling back to LLM")
                
        # Fall back to offline LLM summarization
        return self.offline_summarize(clean_text)
    
    def split_into_chunks(self, input_string: str, chunk_size: int = 500) -> List[Dict]:
        """Split text into chunks for processing with maximum token length constraints"""
        # First split by paragraph
        paragraphs = input_string.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph itself is longer than chunk_size, need to split it
            if len(paragraph.split()) > chunk_size:
                # Process the current chunk if not empty
                if current_chunk:
                    chunks.append({"text": current_chunk.strip()})
                    current_chunk = ""
                
                # Split long paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_sentence_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would exceed chunk size
                    if len((current_sentence_chunk + " " + sentence).split()) > chunk_size:
                        if current_sentence_chunk:
                            chunks.append({"text": current_sentence_chunk.strip()})
                        current_sentence_chunk = sentence
                    else:
                        if current_sentence_chunk:
                            current_sentence_chunk += " " + sentence
                        else:
                            current_sentence_chunk = sentence
                
                # Add any remaining sentence chunk
                if current_sentence_chunk:
                    chunks.append({"text": current_sentence_chunk.strip()})
            
            # Check if adding this paragraph would exceed chunk size
            elif len((current_chunk + "\n\n" + paragraph).split()) > chunk_size:
                chunks.append({"text": current_chunk.strip()})
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({"text": current_chunk.strip()})
        
        return chunks
    
    def extract_article_content(self, url: str) -> str:
        """Extract article content from a URL"""
        try:
            # Fetch the HTML content
            html_content = self.fetch_url_with_cache(url)
            
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract title
            title = soup.title.text.strip() if soup.title else "Untitled Article"
            
            # Try to find the main content
            main_content = ""
            
            # Check for main content elements
            content_elements = soup.select('article, main, .content, .post, .entry-content')
            
            if content_elements:
                # Use the first content element found
                content = content_elements[0]
                
                # Get all paragraphs
                paragraphs = content.find_all('p')
                
                if paragraphs:
                    main_content = "\n\n".join(p.get_text().strip() for p in paragraphs)
            
            # If no content found, try to get all paragraphs
            if not main_content:
                paragraphs = soup.find_all('p')
                main_content = "\n\n".join(p.get_text().strip() for p in paragraphs)
            
            # Combine title and content
            result = f"Title: {title}\n\n{main_content}"
            
            # Clean the result
            return self.clean_text(result)
        
        except Exception as e:
            self.log('error', f"Error extracting content from {url}: {e}")
            return f"Failed to extract content from {url}: {str(e)}"
    
    def cleanup_resources(self, silent=False):
        """Clean up all resources used by the summarizer"""
        if not silent:
            self.log('info', "Cleaning up resources...")
        
        # Clean up LLM
        if hasattr(self, '_llm') and self._llm is not None:
            try:
                self._llm = None
            except Exception as e:
                if not silent:
                    self.log('warning', f"Error cleaning up LLM: {e}")
        
        # Clean up summarizer
        if hasattr(self, '_summarizer') and self._summarizer is not None:
            try:
                self._summarizer = None
            except Exception as e:
                if not silent:
                    self.log('warning', f"Error cleaning up summarizer: {e}")
        
        # Clean up any other resources
        self._model_cache = {}
        self._vector_model_cache = {}
        
        if not silent:
            self.log('info', "Resources cleaned up")


# Simple standalone function for direct use
def summarize_hn(url, verbose=False):
    """Standalone function to summarize a Hacker News URL"""
    summarizer = HackerNewsSummarizer(verbose=verbose)
    try:
        content = summarizer.extract_article_content(url)
        summary = summarizer.final_summarize(content)
        return summary
    finally:
        summarizer.cleanup_resources(silent=True) 