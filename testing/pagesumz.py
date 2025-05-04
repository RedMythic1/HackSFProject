"""
PAGESUMZ - A tool to extract, summarize and analyze webpage content
Extracts content using multiple methods, processes it with LLMs, and provides multi-level summaries
"""

# Prevent segmentation faults by disabling sampler __del__ method
import llama_cpp._internals as _internals
_internals.LlamaSampler.__del__ = lambda self: None

# ===============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# ===============================================================================

from llama_cpp import Llama
from transformers import pipeline
import re
from time import sleep, time
import requests
from bs4 import BeautifulSoup
import argparse
import sys
import html2text
import trafilatura
from readability import Document
from urllib.parse import urlparse
from fpdf import FPDF
import os
from datetime import datetime
import concurrent.futures
import hashlib
from functools import lru_cache
import threading

# Global cache for models to avoid reloading
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LLM_LOCK = threading.Lock()  # Thread safety lock for LLM inference

# Web content cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Performance timing dictionary
TIMINGS = {}

# Phrases that indicate non-useful content
USELESS_PHRASES = [
    "open menu", "log in", "get app", "expand user menu", "create your account",
    "members online", "weekly newsquiz", "reddit home", "settings menu", "close button"
]

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
@lru_cache(maxsize=10)  # Cache recent extractions
def extract_webpage_content(url, save_pdf=False):
    """
    Primary content extraction function - tries multiple methods in sequence
    Uses a cascading approach with fallbacks to ensure best possible extraction:
    1. Trafilatura (best quality for most sites)
    2. Readability (Mozilla's algorithm)
    3. BeautifulSoup custom extraction
    4. Raw text extraction (last resort)
    
    Args:
        url: The URL to scrape
        save_pdf: Whether to save the extracted content as a PDF
        
    Returns:
        str: The extracted content as clean text
    """
    print(f"[INFO] Fetching content from {url}...")
    
    try:
        # Use cached/fetched content
        content = fetch_url_with_cache(url)
        
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
        soup = BeautifulSoup(content, 'html.parser')
        
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
            main_content = main_tag.get_text(separator=' ', strip=True)
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
            bs_text = title + " ".join(content_blocks)
            print(f"[INFO] Successfully extracted content using BeautifulSoup: {len(bs_text.split())} words")
            extracted_content = clean_text(bs_text)
            
            # Generate PDF if requested
            if save_pdf:
                save_as_pdf(extracted_content, url)
                
            return extracted_content
            
        # If all methods fail, return the entire visible text
        all_text = soup.get_text(separator=' ', strip=True)
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
def sample_text(text, sample_percentage=15):
    """
    Sample the text to get approximately 15% of the original content
    Takes sentences in a pattern to preserve coherence while reducing volume
    
    For 15% sampling: take 1 sentence, skip 6 sentences (≈ 14.3% sampling rate)
    
    Args:
        text: Original text to sample
        sample_percentage: Target percentage of text to keep
        
    Returns:
        str: Sampled text
    """
    print(f"[INFO] Sampling text to {sample_percentage}% of original ({len(text.split())} words)")
    
    # Split into sentences to maintain sentence integrity - using efficient regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    total_sentences = len(sentences)
    
    if total_sentences <= 20:  # If very few sentences, return the original
        return text
    
    # For 15% sampling: take 1 sentence, skip 6 sentences
    take_count = 1
    skip_count = 6
    
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
    print(f"[INFO] Sampled text has {len(result.split())} words ({(len(result.split()) / len(text.split()) * 100):.1f}% of original)")
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

@time_function
def pre_summarize_text(text, summarizer, max_input_length=4096, max_length=200, min_length=50):
    """
    Pre-summarize very long text before chunking to reduce overall processing time
    Used for extremely long articles/pages to improve processing efficiency
    
    Args:
        text: Long text to pre-summarize
        summarizer: The summarization pipeline
        max_input_length: Maximum length the summarizer can handle
        max_length/min_length: Parameters for summary generation
        
    Returns:
        str: Pre-summarized text
    """
    try:
        print(f"[INFO] Pre-summarizing {len(text.split())} words of text...")
        
        # If text is already short enough, return as is
        if len(text.split()) < max_input_length * 2:
            return text
            
        # Split text into chunks for the summarizer
        words = text.split()
        chunks = []
        
        # Process in larger chunks for better performance
        for i in range(0, len(words), max_input_length):
            chunks.append(' '.join(words[i:i+max_input_length]))
        
        print(f"[INFO] Summarizing {len(chunks)} pre-chunks...")
        
        # Use ThreadPoolExecutor for parallel summarization
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(
                    lambda c: summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'],
                    chunk
                ))
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as e:
                    print(f"[ERROR] Error summarizing pre-chunk: {e}")
                    # Add a placeholder if summarization fails
                    summaries.append("Error in summarization.")
        
        result = " ".join(summaries)
        print(f"[INFO] Pre-summarization complete: reduced to {len(result.split())} words")
        return result
    except Exception as e:
        print(f"[ERROR] Error in pre-summarization: {e}")
        return text

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
        explanation_prompt_text = f"[INST] Previous context summary: {previous_summary}\n\nImportant subjects from previous context: {important_subjects}\n\nConsidering the previous context and these subjects, fulfill the following prompt: {explanation_prompt}: {chunk['text']} [/INST]"
    
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
def final_summarize(text, summarizer):
    """
    Create a concise executive summary from comprehensive text
    Includes post-processing to fix common formatting issues
    
    Args:
        text: The text to summarize (usually the comprehensive summary)
        summarizer: The summarization pipeline
        
    Returns:
        str: The final, polished executive summary
    """
    max_input_length = 2048
    
    if len(text.split()) > max_input_length:
        # Only use the beginning portion if too long
        text = " ".join(text.split()[:max_input_length])
    
    # Parameters optimized for final summarization
    summary = summarizer(text, max_length=200, min_length=100, do_sample=False)
    raw_summary = summary[0]['summary_text']
    
    # Post-process the summary to fix common issues
    processed_summary = raw_summary
    
    # Fix periods followed by words without spaces
    processed_summary = re.sub(r'\.([A-Z])', r'. \1', processed_summary)
    
    # Fix issue with "views are:" followed by a period
    processed_summary = processed_summary.replace("views are:.", "views are:")
    
    # Fix spaces before punctuation
    processed_summary = re.sub(r'\s+([.,;:!?])', r'\1', processed_summary)
    
    # Fix double spaces
    processed_summary = re.sub(r'\s{2,}', ' ', processed_summary)
    
    # Fix "the" + "AI" without space
    processed_summary = processed_summary.replace("theAI", "the AI")
    
    return processed_summary

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
    # Sample the text first to get approximately 15% of it
    sampled_text = sample_text(input_text)
    
    if len(sampled_text.split()) > 8000:
        print("[INFO] Long text detected. Pre-summarizing text to reduce length...")
        summarized_text = pre_summarize_text(sampled_text, summarizer)
    else:
        summarized_text = sampled_text
    
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
    executive_summary = final_summarize(comprehensive_summary, summarizer)
    
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
    parser.add_argument("url", help="URL of the webpage to summarize")
    parser.add_argument("--prompt", "-p", default="Explain the main points of this content in simple terms",
                        help="Custom prompt for the summarization")
    parser.add_argument("--pdf", "-pdf", action="store_true",
                        help="Save the scraped webpage content as a plaintext PDF")
    parser.add_argument("--no-cache", "-nc", action="store_true",
                        help="Bypass the cache and fetch fresh content")
    parser.add_argument("--sequential", "-seq", action="store_true",
                        help="Force sequential processing (safer but slower)")
    
    # Parse arguments or use defaults if called directly
    if len(sys.argv) > 1:
        args = parser.parse_args()
        url = args.url
        prompt = args.prompt
        save_pdf = args.pdf
        if args.no_cache:
            # Clear the URL cache if requested
            cache_path = get_cache_path(url)
            if os.path.exists(cache_path):
                os.remove(cache_path)
    else:
        # Example default for testing
        url = "https://news.ycombinator.com/item?id=43878850"
        prompt = "What are the main points of this content (generate 5 questions for a power user to search up and dive deeper into the content)?"
        save_pdf = False
        print(f"[INFO] No arguments provided. Using default URL: {url}")
        print(f"[INFO] Default prompt: '{prompt}'")
    
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
