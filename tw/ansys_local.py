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

# Local application imports - use the local version in tw directory
from hackernews_summarizer_local import HackerNewsSummarizer

# Patch Llama sampler
_internals.LlamaSampler.__del__ = lambda self: None

# Global lock for Llama model access
LLM_LOCK = threading.Lock()
# Add print lock for thread safety
PRINT_LOCK = threading.Lock()

# Initialize embedding model (with lazy loading)
EMBEDDING_MODEL = None
EMBEDDING_LOCK = threading.Lock()

# Constants - set up local cache directory for tw
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

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
                
                # Look for model in multiple locations with emphasis on local paths first
                possible_model_paths = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    os.path.expanduser("~/llama-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    os.path.expanduser("~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                    "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Last resort path
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