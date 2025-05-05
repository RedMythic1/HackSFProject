"""
PAGESUMZ - A tool to extract, summarize and analyze webpage content

==== BEGINNER'S GUIDE ====

What is this tool?
-----------------
PageSumz is a program that takes a web page URL, extracts the important content, 
and creates an AI-powered summary of that content. It's like having someone read 
a webpage for you and tell you the most important points.

How to use it:
-----------------
1. Run this program from the command line: python pagesumz.py
2. When prompted, enter any web URL you'd like to summarize
3. Wait for the AI to process the content (this may take a minute or two)
4. Read the summary that's generated!

Technical Overview (for more advanced users):
-----------------
This program works in several steps:
1. It extracts webpage content using multiple methods for best results
2. It processes the text using AI models to understand the content
3. It generates summaries at different levels of detail
4. It handles special cases like Hacker News comment threads

Requirements:
-----------------
- Python 3.7+
- Various Python libraries (installed via pip)
- A Llama model file (for AI processing)

=========================
"""

# Prevent segmentation faults by disabling sampler __del__ method
import llama_cpp._internals as _internals
_internals.LlamaSampler.__del__ = lambda self: None

# ===============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# ===============================================================================

# Core libraries for AI models
from llama_cpp import Llama  # AI model for text processing
from transformers import pipeline  # Used for summarization tasks

# Standard libraries
import re  # For text pattern matching
from time import sleep, time  # For timing and pauses
import requests  # For fetching web content
from bs4 import BeautifulSoup  # For parsing HTML
import argparse  # For command line arguments
import sys  # For system operations
import html2text  # For converting HTML to text
import trafilatura  # For extracting content from webpages
from readability import Document  # For isolating main content
from urllib.parse import urlparse  # For URL handling
from fpdf import FPDF  # For creating PDF files
import os  # For file operations
from datetime import datetime  # For timestamps
import concurrent.futures  # For parallel processing
import hashlib  # For creating unique file names
from functools import lru_cache  # For caching results
import threading  # For thread management
import numpy as np  # For numerical operations
from typing import List  # For type hints
from numpy.typing import NDArray  # For array type hints
import torch  # For AI model operations
from transformers import AutoTokenizer, AutoModel  # For AI models

# Storage for our models so we don't reload them repeatedly
_MODEL_CACHE = {}  # Stores loaded models
_MODEL_CACHE_LOCK = threading.Lock()  # Prevents multiple threads from accessing models at once
_LLM_LOCK = threading.Lock()  # Ensures only one thread uses the AI model at a time
_VECTOR_MODEL_CACHE = {}  # Stores vector models for similarity comparisons

# Where to save cached web content (saves time when reloading)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)  # Create cache directory if it doesn't exist

# For tracking how long different operations take
TIMINGS = {}

# These phrases usually indicate UI elements rather than actual content
USELESS_PHRASES = [
    "open menu", "log in", "get app", "expand user menu", "create your account",
    "members online", "weekly newsquiz", "reddit home", "settings menu", "close button"
]

# ===============================================================================
# VECTOR MODEL FUNCTIONS
# ===============================================================================

def get_vector_model():
    """Get or initialize the vector model"""
    with _MODEL_CACHE_LOCK:
        if 'vector_model' not in _VECTOR_MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            model.eval()
            _VECTOR_MODEL_CACHE['vector_model'] = (tokenizer, model)
        return _VECTOR_MODEL_CACHE['vector_model']

def string_to_vector(text: str) -> torch.Tensor:
    """
    Convert a string into a vector using a proper transformer model.
    """
    tokenizer, model = get_vector_model()
    
    # Tokenize the text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :]
    
    # Normalize the embedding to unit length
    embedding = embedding.squeeze(0)
    norm = torch.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def compute_angle(vector1: torch.Tensor, vector2: torch.Tensor) -> float:
    """
    Compute the angle between two vectors in degrees.
    """
    cos_angle = torch.dot(vector1, vector2)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle_radians = torch.acos(cos_angle)
    angle_degrees = torch.rad2deg(angle_radians)
    return angle_degrees.item()

def filter_similar_comments(comments: List[str], angle_threshold: float = 30.0) -> List[str]:
    """
    Filter out comments that are too similar to each other based on vector angle.
    
    Args:
        comments: List of comment strings
        angle_threshold: Minimum angle (in degrees) required to keep a comment
        
    Returns:
        List of filtered comments
    """
    if not comments:
        return []
    
    # Convert all comments to vectors
    print("\n[INFO] Converting comments to vectors...")
    vectors = []
    for i, comment in enumerate(comments):
        print(f"\n[INFO] Processing comment {i+1}/{len(comments)}:")
        print("-" * 80)
        print(comment)
        print("-" * 80)
        vectors.append(string_to_vector(comment))
    
    # Keep track of which comments to keep
    keep_indices = [0]  # Always keep the first comment
    
    # Compare each comment with all previously kept comments
    print("\n[INFO] Comparing comments for similarity...")
    for i in range(1, len(comments)):
        keep = True
        for j in keep_indices:
            angle = compute_angle(vectors[i], vectors[j])
            print(f"\nComparing comment {i+1} with comment {j+1}:")
            print(f"Angle: {angle:.2f}°")
            if angle < angle_threshold:
                print(f"[INFO] Comment {i+1} is too similar to comment {j+1} (angle: {angle:.2f}°)")
                print("Comment 1:")
                print("-" * 80)
                print(comments[j])
                print("-" * 80)
                print("Comment 2:")
                print("-" * 80)
                print(comments[i])
                print("-" * 80)
                keep = False
                break
        
        if keep:
            keep_indices.append(i)
            print(f"[INFO] Keeping comment {i+1}")
    
    # Return only the kept comments
    return [comments[i] for i in keep_indices]

# ===============================================================================
# PERFORMANCE TRACKING AND CACHING FUNCTIONS
# ===============================================================================

def time_function(func):
    """Decorator to time function execution and log performance metrics"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        if func_name in TIMINGS:
            TIMINGS[func_name].append(execution_time)
        else:
            TIMINGS[func_name] = [execution_time]
        return result
    return wrapper

def get_cache_path(url):
    """Generate a cache file path for a given URL using MD5 hash"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.html")

@time_function
def fetch_url_with_cache(url, cache_timeout=3600):
    """
    Fetch URL content with caching to reduce network requests
    
    Args:
        url: The URL to fetch
        cache_timeout: Cache validity in seconds (default: 1 hour)
        
    Returns:
        str: The HTML content of the webpage
    """
    cache_path = get_cache_path(url)
    
    # Check if we have a valid cache file
    if os.path.exists(cache_path):
        file_age = time() - os.path.getmtime(cache_path)
        if file_age < cache_timeout:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    print(f"[INFO] Using cached version of {url}")
                    return f.read()
            except Exception as e:
                print(f"[WARNING] Cache read error: {e}")
    
    # Fetch fresh content
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Save to cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"[WARNING] Cache write error: {e}")
            
        return content
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        # Try to use expired cache if available
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    print(f"[INFO] Using expired cache for {url}")
                    return f.read()
            except Exception:
                pass
        raise

# ===============================================================================
# MODEL INITIALIZATION AND BASIC SUMMARIZATION
# ===============================================================================

@time_function
def initialize_models():
    """
    Initialize the LLM and summarization models with caching
    Uses thread-safe locking to prevent race conditions
    
    Returns:
        tuple: (llm, summarizer) - The initialized models
    """
    print("[INFO] Initializing models...")
    
    with _MODEL_CACHE_LOCK:
        # Check if models are already initialized
        if 'llm' in _MODEL_CACHE and 'summarizer' in _MODEL_CACHE:
            print("[INFO] Using cached models")
            return _MODEL_CACHE['llm'], _MODEL_CACHE['summarizer']
        
        # Initialize models
        llm = Llama(
            model_path="/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=32768,
            n_threads=12,  # Increased threads for better CPU utilization
            n_gpu_layers=35,
            chat_format="mistral-instruct",
            verbose=False,
            stop=None
        )
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Cache the models
        _MODEL_CACHE['llm'] = llm
        _MODEL_CACHE['summarizer'] = summarizer
        
        return llm, summarizer

@time_function
def offline_summarize(text, summarizer):
    """
    Quick summarization using Hugging Face BART model
    Used for intermediate summarization steps
    
    Args:
        text: Text to summarize
        summarizer: The BART summarization pipeline
        
    Returns:
        str: Summarized text
    """
    max_input_length = 4096
    if len(text.split()) > max_input_length:
        text = " ".join(text.split()[:max_input_length])
    
    summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# ===============================================================================
# CONTENT EXTRACTION AND CLEANING FUNCTIONS
# ===============================================================================

def is_useful_block(text):
    """Filter out UI elements and short content blocks"""
    text_lower = text.lower()
    if len(text_lower.split()) < 10:
        return False
    for phrase in USELESS_PHRASES:
        if phrase in text_lower:
            return False
    return True

@time_function
def clean_text(text):
    """
    Clean the text by removing extra whitespace and normalizing punctuation
    Uses optimized regex patterns for better performance
    """
    # Remove extra spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove duplicate spaces around punctuation
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    # Fix spacing after periods that don't end sentences
    text = re.sub(r'(\w\.\w\.)\s+', r'\1', text)
    return text.strip()

@time_function
def process_hacker_news_comments(url, soup):
    """
    Special function to extract and process Hacker News comments properly
    
    Args:
        url: The URL being processed
        soup: BeautifulSoup object of the page
        
    Returns:
        list: List of processed comment blocks
    """
    print("[INFO] Processing Hacker News comments...")
    content_blocks = []
    
    # Extract the article title
    article_title = ""
    if soup.title:
        article_title = soup.title.get_text().strip()
        content_blocks.append(f"# {article_title}\n")
    
    # Extract article link (if present)
    article_text = soup.find('td', class_='title')
    if article_text and article_text.find('a'):
        article_link = article_text.find('a').get('href', '')
        content_blocks.append(f"Source: {article_link}\n")
    
    # Find all comments
    comment_elements = soup.find_all('tr', class_='athing comtr')
    
    print(f"[INFO] Found {len(comment_elements)} comments on Hacker News page")
    
    comments = []
    for comment in comment_elements:
        # Get indentation level
        indent = comment.find('td', class_='ind')
        indent_level = int(indent.get('indent', '0')) if indent else 0
        
        # Get comment metadata
        comhead = comment.find('span', class_='comhead')
        username = "Anonymous"
        age = ""
        
        if comhead:
            username_elem = comhead.find('a', class_='hnuser')
            if username_elem:
                username = username_elem.text
            
            age_elem = comhead.find('span', class_='age')
            if age_elem:
                age = age_elem.text
        
        # Get comment content
        commtext = comment.find('div', class_='commtext c00')
        if commtext:
            # Get pure text without links or formatting
            text = commtext.get_text(separator=' ', strip=True)
            
            # Format the comment
            if text and len(text.split()) > 3:  # Keep comments with at least 4 words
                # Format with username, timestamp, and proper indentation
                prefix = "  " * indent_level
                formatted_comment = f"{prefix}[{username} {age}]\n{prefix}{text}\n"
                comments.append(formatted_comment)
    
    # Only keep a subset of comments to avoid too much content
    # Focus on high-level (less indented) comments first
    comments.sort(key=lambda x: len(x.split('\n')[0]) - len(x.split('\n')[0].lstrip()))
    
    # Keep at most 20 comments, prioritizing top-level comments
    selected_comments = comments[:min(20, len(comments))]
    
    if selected_comments:
        # Add a header for comments section
        content_blocks.append("\n=== COMMENTS ===\n")
        content_blocks.extend(selected_comments)
        # Add a separator after comments
        content_blocks.append("\n" + "="*50 + "\n")
    
    return content_blocks

@time_function
def extract_webpage_content(url, save_pdf=False):
    """
    Primary content extraction function - tries multiple methods in sequence
    """
    print(f"[INFO] Fetching content from {url}...")
    
    try:
        # Use cached/fetched content
        content = fetch_url_with_cache(url)
        
        # Special handling for known sites
        soup = BeautifulSoup(content, 'html.parser')
        
        if "news.ycombinator.com" in url:
            content_blocks = process_hacker_news_comments(url, soup)
            extracted_content = "\n".join(content_blocks)
            
            # Generate PDF if requested
            if save_pdf:
                save_as_pdf(extracted_content, url)
                
            return extracted_content
            
        # Method 1: Use trafilatura for extraction (generally high quality)
        try:
            trafilatura_text = trafilatura.extract(content, include_comments=False, 
                                    include_tables=True, output_format="text")
            if trafilatura_text and len(trafilatura_text.split()) > 100:
                print(f"[INFO] Successfully extracted content using trafilatura: {len(trafilatura_text.split())} words")
                extracted_content = clean_text(trafilatura_text)
                
                # Generate PDF if requested
                if save_pdf:
                    save_as_pdf(extracted_content, url)
                    
                return extracted_content
        except Exception as e:
            print(f"[WARNING] Trafilatura extraction failed: {e}")
        
        # Method 2: Try with readability
        doc = Document(content)
        readability_text = html2text.html2text(doc.summary())
        if readability_text and len(readability_text.split()) > 100:
            print(f"[INFO] Successfully extracted content using readability: {len(readability_text.split())} words")
            extracted_content = clean_text(readability_text)
            
            # Generate PDF if requested
            if save_pdf:
                save_as_pdf(extracted_content, url)
                
            return extracted_content
        
        # Method 3: Fallback to BeautifulSoup extraction
        # Remove non-content elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript', 'iframe', 'form']):
            tag.decompose()
            
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text() + "\n\n"
            
        # Extract meaningful content blocks prioritizing main content areas
        content_blocks = []
        
        # Add main content areas with priority
        for main_tag in soup.find_all(['main', 'article', 'section', '[role="main"]']):
            main_content = main_tag.get_text(separator='\n', strip=True)
            if main_content and len(main_content.split()) > 30:
                content_blocks.append(main_content)
                break  # If we find a main content area, no need to continue
        
        # If no main content was found, extract from all paragraph and heading tags
        if not content_blocks:
            content_tags = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])
            for tag in content_tags:
                text = tag.get_text(strip=True)
                if text and len(text.split()) > 5:
                    content_blocks.append(text)
        
        if content_blocks:
            # Join blocks with double newlines to ensure proper separation
            bs_text = title + "\n\n".join(content_blocks)
            print(f"[INFO] Successfully extracted content using BeautifulSoup: {len(bs_text.split())} words")
            extracted_content = clean_text(bs_text)
            
            # Generate PDF if requested
            if save_pdf:
                save_as_pdf(extracted_content, url)
                
            return extracted_content
            
        # If all methods fail, return the entire visible text
        all_text = soup.get_text(separator='\n', strip=True)
        print(f"[INFO] Falling back to raw text extraction: {len(all_text.split())} words")
        extracted_content = clean_text(all_text)
        
        # Generate PDF if requested
        if save_pdf:
            save_as_pdf(extracted_content, url)
            
        return extracted_content
            
    except Exception as e:
        print(f"[ERROR] Failed to extract content from {url}: {str(e)}")
        return f"Failed to extract content from {url}: {str(e)}"

# ===============================================================================
# TEXT PROCESSING AND CHUNKING FUNCTIONS
# ===============================================================================

@time_function
def sample_text(text):
    """
    Sample the text to get approximately 2000 words
    Takes sentences in a pattern to preserve coherence while reducing volume
    Returns original text if it's already 2000 words or less
    
    Args:
        text: Original text to sample
        
    Returns:
        str: Sampled text or original text if short enough
    """
    word_count = len(text.split())
    
    # Return original text if it's already 2000 words or less
    if word_count <= 2000:
        print(f"[INFO] Text already short enough ({word_count} words), skipping sampling")
        return text
    
    # Calculate target sample percentage based on text length
    # Aim for around 2000 words in the final sample
    sample_percentage = round(2000/word_count, 2)
    print(f"[INFO] Sampling text to {100*sample_percentage}% of original ({word_count} words)")
    
    # Split into sentences to maintain sentence integrity - using efficient regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    total_sentences = len(sentences)
    
    if total_sentences <= 20:  # If very few sentences, return the original
        return text
    
    # Calculate dynamic skip count based on sample percentage
    # For example, if we want 15% sampling (0.15), we need to take 1 sentence and skip (1/0.15 - 1) sentences
    # This ensures we get approximately the desired percentage
    take_count = 1
    skip_count = max(1, int(1/sample_percentage - 1))  # Ensure skip_count is at least 1
    
    sampled_sentences = []
    
    # Efficient algorithm for sampling
    i = 0
    while i < total_sentences:
        # Take sentences
        end_idx = min(i + take_count, total_sentences)
        for j in range(i, end_idx):
            sampled_sentences.append(sentences[j])
        
        # Skip sentences
        i = end_idx + skip_count
    
    result = " ".join(sampled_sentences)
    print(f"[INFO] Sampled text has {len(result.split())} words ({(len(result.split()) / word_count * 100):.1f}% of original)")
    return result

@time_function
def extract_subjects(text, summarizer):
    """
    Extract important subjects from text to maintain context between chunks
    
    Args:
        text: Text to analyze
        summarizer: The summarization pipeline
        
    Returns:
        str: Extracted key subjects
    """
    try:
        summary = summarizer(text, max_length=80, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[ERROR] Error extracting subjects: {e}")
        return ""

@time_function
def split_into_chunks(input_string, chunk_size=500):
    """
    Split input text into chunks for processing
    Preserves sentence boundaries when splitting
    
    Args:
        input_string: The text to split
        chunk_size: Target size (in words) for each chunk
        
    Returns:
        list: List of chunk dictionaries with indices and text
    """
    # Efficient sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', input_string)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append({'chunk': len(chunks), 'text': ' '.join(current_chunk)})
                current_chunk = []
                current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append({'chunk': len(chunks), 'text': ' '.join(current_chunk)})

    print(f"[DEBUG] split_into_chunks: {len(chunks)} chunks created.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"[DEBUG] Chunk {i}: {len(chunk['text'].split())} words, preview: {chunk['text'][:100]}")
    return chunks

# ===============================================================================
# CHUNK PROCESSING AND SUMMARIZATION FUNCTIONS
# ===============================================================================

@time_function
def process_chunk(chunk_data):
    """
    Process a single chunk of text in the summarization pipeline
    Maintains context between chunks using previous summaries
    Implements thread safety with locks for LLM access
    
    Args:
        chunk_data: Tuple containing the chunk and context information
        
    Returns:
        tuple: (chunk_idx, explanation, summary) for the processed chunk
    """
    chunk, previous_summary, previous_explanation, llm, summarizer, explanation_prompt = chunk_data
    
    print(f"[INFO] Processing chunk {chunk['chunk']} with {len(chunk['text'].split())} words.")
    important_subjects = ""
    if previous_explanation and previous_explanation != "No explanation returned." and previous_explanation != "Error during explanation.":
        important_subjects = extract_subjects(previous_explanation, summarizer)
    
    if chunk['chunk'] == 0:
        explanation_prompt_text = f"[INST] {explanation_prompt}: {chunk['text']} [/INST]"
    else:
        explanation_prompt_text = f"""[INST]

Key topics and themes from previous context: {important_subjects}

Building upon this context, analyze the following new content.

New content to analyze: {explanation_prompt}: {previous_summary}\n{chunk['text']} [/INST]"""
    
    prompt_length = len(explanation_prompt_text.split())
    print(f"[INFO] Prompt length for chunk {chunk['chunk']}: {prompt_length} words")
    
    if prompt_length > 1000:
        print(f"[WARNING] Prompt too long for chunk {chunk['chunk']}, truncating...")
        explanation_prompt_text = " ".join(explanation_prompt_text.split()[:1000])
    
    try:
        # Use lock to ensure thread safety when using the LLM
        with _LLM_LOCK:
            output = llm(explanation_prompt_text, max_tokens=1024, temperature=0.1)
            explanation = output["choices"][0]["text"].strip()
        
        print(f"[INFO] Explanation generated: {len(explanation.split())} words")
        if not explanation:
            explanation = "No explanation returned."
        if explanation and explanation != "No explanation returned.":
            new_summary = offline_summarize(explanation, summarizer)
        else:
            new_summary = "Previous explanation was empty or errored."
    except Exception as e:
        print(f"[ERROR] Error processing chunk {chunk['chunk']}: {e}")
        explanation = "Error during explanation."
        new_summary = "Error occurred in previous chunk processing."
    return chunk['chunk'], explanation, new_summary

@time_function
def final_summarize(text, llm):
    """
    Create a concise executive summary from comprehensive text using Llama
    Generates a coherent paragraph without repetition
    
    Args:
        text: The text to summarize (usually the comprehensive summary)
        llm: The Llama model instance
        
    Returns:
        str: The final, polished executive summary
    """
    max_input_length = 2048
    
    if len(text.split()) > max_input_length:
        # Only use the beginning portion if too long
        text = " ".join(text.split()[:max_input_length])
    
    # Create a prompt that encourages a coherent, non-repetitive summary
    prompt = f"""[INST]
Create a concise, coherent paragraph that summarizes the following text.
The summary should be well-structured and avoid repeating any information.
Focus on the key points and maintain a natural flow.

Text to summarize:
{text}

Please provide a single, well-written paragraph that captures the essence of the text without repetition.
[/INST]"""
    
    try:
        # Use lock to ensure thread safety when using the LLM
        with _LLM_LOCK:
            output = llm(prompt, max_tokens=512, temperature=0.1)
            summary = output["choices"][0]["text"].strip()
        
        # Basic post-processing to fix common formatting issues
        processed_summary = summary
        
        # Fix periods followed by words without spaces
        processed_summary = re.sub(r'\.([A-Z])', r'. \1', processed_summary)
        
        # Fix spaces before punctuation
        processed_summary = re.sub(r'\s+([.,;:!?])', r'\1', processed_summary)
        
        # Fix double spaces
        processed_summary = re.sub(r'\s{2,}', ' ', processed_summary)
        
        return processed_summary.strip()
        
    except Exception as e:
        print(f"[ERROR] Error in final_summarize: {e}")
        return "Error generating final summary."

# ===============================================================================
# PDF GENERATION FUNCTIONS
# ===============================================================================

@time_function
def save_as_pdf(text, url, filename=None):
    """
    Save the scraped webpage content as a plaintext PDF file
    Handles Unicode characters by safely replacing them with ASCII equivalents
    
    Args:
        text: The text content to save
        url: The URL of the webpage (for metadata)
        filename: Custom filename for the PDF
    
    Returns:
        str: Path to the saved PDF file or None if failed
    """
    if not filename:
        # Create a filename based on the URL domain and timestamp
        domain = urlparse(url).netloc.replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webpage_{domain}_{timestamp}.pdf"
    
    # Ensure filename ends with .pdf
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    print(f"[INFO] Generating PDF: {filename}")
    
    try:
        # Create ASCII-only version of the text for PDF compatibility
        ascii_text = ""
        for char in text:
            if ord(char) < 128:
                ascii_text += char
            else:
                # Replace non-ASCII characters with appropriate replacements
                if char in "''""":
                    ascii_text += "'"
                elif char == '—':
                    ascii_text += "--"
                elif char == '–':
                    ascii_text += "-"
                elif char == '…':
                    ascii_text += "..."
                else:
                    ascii_text += " "
        
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set metadata
        pdf.set_title(f"Webpage Content: {url}")
        pdf.set_author("PAGESUMZ Web Scraper")
        pdf.set_creator("PAGESUMZ using FPDF")
        
        # Use built-in font
        pdf.set_font("Courier", size=11)
        
        # Add URL and timestamp
        pdf.cell(200, 10, txt=f"Source: {url}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
        pdf.ln(5)
        
        # Add a title
        pdf.set_font("Courier", 'B', size=14)
        pdf.cell(200, 10, txt="WEBPAGE CONTENT", ln=True, align='C')
        pdf.ln(5)
        
        # Add content with line wrapping - process in larger chunks for better performance
        pdf.set_font("Courier", size=10)
        
        # Split text into chunks and process in batches
        max_chunk_size = 5000  # Process 5000 chars at a time
        for i in range(0, len(ascii_text), max_chunk_size):
            chunk = ascii_text[i:i+max_chunk_size]
            lines = chunk.split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    pdf.multi_cell(0, 5, txt=line)
        
        # Save the PDF
        pdf.output(filename)
        
        print(f"[INFO] PDF successfully created: {os.path.abspath(filename)}")
        return os.path.abspath(filename)
    
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        return None

# ===============================================================================
# MAIN PROCESSING PIPELINE
# ===============================================================================

@time_function
def process_text(input_text, llm, summarizer, explanation_prompt):
    """
    Main processing pipeline that coordinates the entire summarization workflow
    
    Steps:
    1. Sample text to reduce volume
    2. Optionally pre-summarize very long text
    3. Split into chunks for processing
    4. Process chunks sequentially with context preservation
    5. Generate executive summary
    6. Report performance metrics
    
    Args:
        input_text: Raw text to process
        llm: The large language model for explanations
        summarizer: The summarization pipeline
        explanation_prompt: The prompt to guide LLM explanations
        
    Returns:
        str: The executive summary
    """
    # Check if content is from Hacker News by looking for comment formatting
    is_hacker_news = False
    if "\n=== COMMENTS ===\n" in input_text:
        is_hacker_news = True
        print("[INFO] Detected Hacker News format - preserving comment structure")
        
        # Split content into article and comments
        parts = input_text.split("\n=== COMMENTS ===\n", 1)
        article_content = parts[0]
        comments_section = parts[1] if len(parts) > 1 else ""
        
        # Process comments differently
        if comments_section:
            # Extract individual comments
            comments = []
            current_comment = ""
            comment_lines = comments_section.split('\n')
            
            for line in comment_lines:
                if line.strip() and line.lstrip().startswith('[') and ']' in line:
                    # This is a new comment header
                    if current_comment:
                        comments.append(current_comment.strip())
                    current_comment = line + "\n"
                elif current_comment:
                    current_comment += line + "\n"
                    
            # Add the last comment if there is one
            if current_comment:
                comments.append(current_comment.strip())
            
            print(f"\n[INFO] Extracted {len(comments)} individual comments")
            
            # Filter out similar comments
            print(f"\n[INFO] Filtering {len(comments)} comments for semantic similarity...")
            filtered_comments = filter_similar_comments(comments)
            print(f"[INFO] Kept {len(filtered_comments)} unique comments after filtering")
            
            print("\n=== FILTERED COMMENTS ===")
            for i, comment in enumerate(filtered_comments):
                print(f"\nKept Comment {i+1}:")
                print("-" * 80)
                print(comment)
                print("-" * 80)
            
            # Combine article and filtered comments
            summarized_text = article_content + "\n=== COMMENTS ===\n" + "\n\n".join(filtered_comments)
        else:
            summarized_text = article_content
    else:
        # Sample the text first if not Hacker News format
        sampled_text = sample_text(input_text)
        
        # Split into comments (using single newlines)
        comments = [c.strip() for c in sampled_text.split('\n') if c.strip()]
        
        print("\n=== COMMENTS FOUND ===")
        for i, comment in enumerate(comments):
            print(f"\nComment {i+1}:")
            print("-" * 80)
            print(comment)
            print("-" * 80)
        
        # Filter out similar comments
        print(f"\n[INFO] Filtering {len(comments)} comments for semantic similarity...")
        filtered_comments = filter_similar_comments(comments)
        print(f"[INFO] Kept {len(filtered_comments)} unique comments after filtering")
        
        print("\n=== FILTERED COMMENTS ===")
        for i, comment in enumerate(filtered_comments):
            print(f"\nKept Comment {i+1}:")
            print("-" * 80)
            print(comment)
            print("-" * 80)
        
        # Rejoin the filtered comments
        summarized_text = '\n'.join(filtered_comments)
    
    # Split the text into manageable chunks
    chunks = split_into_chunks(summarized_text)
    
    # Prepare for chunk processing
    all_explanations = [""] * len(chunks)
    
    print(f"[INFO] Processing {len(chunks)} chunks sequentially...")
    
    # First chunk must be processed first to establish context
    first_chunk_idx, first_explanation, first_summary = process_chunk(
        (chunks[0], "", "", llm, summarizer, explanation_prompt)
    )
    all_explanations[first_chunk_idx] = first_explanation
    previous_summary = first_summary
    previous_explanation = first_explanation
    
    # Process remaining chunks sequentially to avoid model conflicts
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        chunk_idx, explanation, new_summary = process_chunk(
            (chunk, previous_summary, previous_explanation, llm, summarizer, explanation_prompt)
        )
        all_explanations[chunk_idx] = explanation
        previous_summary = new_summary
        previous_explanation = explanation
    
    # Generate the final comprehensive summary
    comprehensive_summary = ""
    for explanation in all_explanations:
        if explanation and not explanation.isspace():
            if comprehensive_summary:
                comprehensive_summary += "\n\n" + explanation
            else:
                comprehensive_summary = explanation
    
    # Apply final summarization to get a concise executive summary
    print("[INFO] Creating executive summary...")
    executive_summary = final_summarize(comprehensive_summary, llm)
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY:")
    print("="*80)
    print(executive_summary)
    print("="*80)
    
    # Print performance metrics
    print("\nPERFORMANCE METRICS:")
    for func_name, times in TIMINGS.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{func_name}: avg {avg_time:.3f}s, total {sum(times):.3f}s, calls {len(times)}")
    
    return executive_summary

@time_function
def summarize_webpage(url, explanation_prompt="Explain what this means in simple terms", save_pdf=False):
    """
    Main entry point for webpage summarization
    Coordinates the entire extraction and summarization pipeline
    
    Args:
        url: The webpage URL to summarize
        explanation_prompt: Prompt for the LLM to guide explanations
        save_pdf: Whether to save extracted content as PDF
        
    Returns:
        str: The executive summary
    """
    # Extract content from the webpage
    webpage_text = extract_webpage_content(url, save_pdf=save_pdf)
    
    if webpage_text.startswith("Failed to extract"):
        print(f"[ERROR] {webpage_text}")
        return "Failed to extract content from the webpage."
    
    # Initialize models
    llm, summarizer = initialize_models()
    
    try:
        # Process the extracted text
        print("[INFO] Processing webpage content...")
        summary = process_text(webpage_text, llm, summarizer, explanation_prompt)
        return summary
    finally:
        # Don't close the models - they're cached
        pass

# ===============================================================================
# COMMAND-LINE INTERFACE
# ===============================================================================

def main():
    """
    Main function to parse command line arguments and run the summarization pipeline
    Handles command-line options and provides defaults for testing
    """
    start_time = time()
    
    parser = argparse.ArgumentParser(description="Summarize web page content using Llama.")
    parser.add_argument("url", nargs="?", help="URL of the webpage to summarize")
    parser.add_argument("--prompt", "-p", default="Explain the main points of this content in simple terms",
                        help="Custom prompt for the summarization")
    parser.add_argument("--pdf", "-pdf", action="store_true",
                        help="Save the scraped webpage content as a plaintext PDF")
    parser.add_argument("--no-cache", "-nc", action="store_true",
                        help="Bypass the cache and fetch fresh content")
    parser.add_argument("--sequential", "-seq", action="store_true",
                        help="Force sequential processing (safer but slower)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Interactive mode if no URL is provided
    if not args.url:
        # Super simple input prompt
        url = input("link here mf: ")
        
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            print(f"Added https:// prefix: {url}")
        
        prompt = args.prompt
        save_pdf = args.pdf
    else:
        # Normal mode with command line arguments
        url = args.url
        prompt = args.prompt
        save_pdf = args.pdf
        if args.no_cache:
            # Clear the URL cache if requested
            cache_path = get_cache_path(url)
            if os.path.exists(cache_path):
                os.remove(cache_path)
    
    print(f"[INFO] Summarizing: {url}")
    print(f"[INFO] Using prompt: '{prompt}'")
    if save_pdf:
        print("[INFO] Will save webpage content as PDF")
    
    try:
        result = summarize_webpage(url, prompt, save_pdf=save_pdf)
        
        # Calculate and display total execution time
        total_time = time() - start_time
        print(f"\n[INFO] Total execution time: {total_time:.2f} seconds")
        
        return result
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        # Clean up in case of error
        with _MODEL_CACHE_LOCK:
            if 'llm' in _MODEL_CACHE:
                try:
                    _MODEL_CACHE['llm'].close()
                except Exception:
                    pass
                _MODEL_CACHE.pop('llm', None)
        return f"Error: {str(e)}"

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure proper cleanup on exit
        with _MODEL_CACHE_LOCK:
            if 'llm' in _MODEL_CACHE:
                try:
                    _MODEL_CACHE['llm'].close()
                except Exception:
                    pass
