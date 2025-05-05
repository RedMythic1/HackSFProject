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
from transformers import pipeline, AutoTokenizer, AutoModel  # For summarization and embedding

# Standard libraries
import sys
import os
import re  # For text pattern matching
from time import time  # For timing and pauses
import requests  # For fetching web content
from bs4 import BeautifulSoup  # For parsing HTML
import hashlib  # For creating unique file names
import threading  # For thread management
import torch  # For AI model operations
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
                 model_path: str = "/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
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
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
                        n_ctx=32768,
                        n_threads=12,
                        n_gpu_layers=35,
                        chat_format="mistral-instruct",
                        verbose=False,
                        stop=None
                    )
        return self._llm
    
    @property
    def summarizer(self):
        """Lazy-loaded summarization model"""
        if self._summarizer is None:
            with self._model_cache_lock:
                if self._summarizer is None:
                    self.log('info', "Initializing summarization model...")
                    self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return self._summarizer
    
    def get_vector_model(self):
        """Get or initialize the vector model for similarity comparisons"""
        with self._model_cache_lock:
            if 'vector_model' not in self._vector_model_cache:
                self.log('info', "Initializing vector model...")
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                model.eval()
                self._vector_model_cache['vector_model'] = (tokenizer, model)
            return self._vector_model_cache['vector_model']
    
    def string_to_vector(self, text: str) -> torch.Tensor:
        """Convert a string into a vector using a transformer model"""
        tokenizer, model = self.get_vector_model()
        
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
    
    def compute_angle(self, vector1: torch.Tensor, vector2: torch.Tensor) -> float:
        """Compute the angle between two vectors in degrees"""
        cos_angle = torch.dot(vector1, vector2)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle_radians = torch.acos(cos_angle)
        angle_degrees = torch.rad2deg(angle_radians)
        return angle_degrees.item()
    
    def filter_similar_comments(self, comments: List[str], angle_threshold: float = 30.0) -> List[str]:
        """Filter out comments that are too similar to each other based on vector angle"""
        if not comments:
            return []
        
        # Convert all comments to vectors
        self.log('info', "Converting comments to vectors...")
        vectors = []
        for i, comment in enumerate(comments):
            self.log('debug', f"Processing comment {i+1}/{len(comments)}")
            vectors.append(self.string_to_vector(comment))
        
        # Keep track of which comments to keep
        keep_indices = [0]  # Always keep the first comment
        
        # Compare each comment with all previously kept comments
        self.log('info', "Comparing comments for similarity...")
        for i in range(1, len(comments)):
            keep = True
            for j in keep_indices:
                angle = self.compute_angle(vectors[i], vectors[j])
                self.log('debug', f"Comparing comment {i+1} with comment {j+1}: Angle: {angle:.2f}°")
                if angle < angle_threshold:
                    self.log('info', f"Comment {i+1} is too similar to comment {j+1} (angle: {angle:.2f}°)")
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
                self.log('info', f"Keeping comment {i+1}")
        
        # Return only the kept comments
        return [comments[i] for i in keep_indices]
    
    def offline_summarize(self, text: str) -> str:
        """Quick summarization using Hugging Face BART model"""
        max_input_length = 4096
        if len(text.split()) > max_input_length:
            text = " ".join(text.split()[:max_input_length])
        
        summary = self.summarizer(text, max_length=60, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean the text by removing extra whitespace and normalizing punctuation"""
        # Remove extra spaces, newlines and tabs
        text = re.sub(r'\s+', ' ', text)
        # Remove duplicate spaces around punctuation
        text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
        # Fix spacing after periods that don't end sentences
        text = re.sub(r'(\w\.\w\.)\s+', r'\1', text)
        return text.strip()
    
    def sample_text(self, text: str, target_words: int = 2000) -> str:
        """Sample the text to reduce volume while preserving coherence"""
        word_count = len(text.split())
        
        # Return original text if it's already short enough
        if word_count <= target_words:
            self.log('info', f"Text already short enough ({word_count} words), skipping sampling")
            return text
        
        # Calculate target sample percentage
        sample_percentage = round(target_words/word_count, 2)
        self.log('info', f"Sampling text to {100*sample_percentage}% of original ({word_count} words)")
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        total_sentences = len(sentences)
        
        if total_sentences <= 20:  # If very few sentences, return the original
            return text
        
        # Calculate dynamic skip count
        take_count = 1
        skip_count = max(1, int(1/sample_percentage - 1)) 
        
        # Efficient sampling algorithm
        sampled_sentences = []
        i = 0
        while i < total_sentences:
            # Take sentences
            end_idx = min(i + take_count, total_sentences)
            for j in range(i, end_idx):
                sampled_sentences.append(sentences[j])
            
            # Skip sentences
            i = end_idx + skip_count
        
        result = " ".join(sampled_sentences)
        self.log('info', f"Sampled text has {len(result.split())} words ({(len(result.split()) / word_count * 100):.1f}% of original)")
        return result
    
    def extract_subjects(self, text: str) -> str:
        """Extract important subjects from text to maintain context between chunks"""
        try:
            summary = self.summarizer(text, max_length=80, min_length=20, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            self.log('error', f"Error extracting subjects: {e}")
            return ""
    
    def split_into_chunks(self, input_string: str, chunk_size: int = 500) -> List[Dict]:
        """Split input text into chunks for processing"""
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

        self.log('info', f"Split into {len(chunks)} chunks")
        return chunks
    
    def process_chunk(self, chunk_data: Tuple) -> Tuple:
        """Process a single chunk of text in the summarization pipeline"""
        chunk, previous_summary, previous_explanation, explanation_prompt = chunk_data
        
        self.log('info', f"Processing chunk {chunk['chunk']} with {len(chunk['text'].split())} words")
        important_subjects = ""
        if previous_explanation and previous_explanation != "No explanation returned." and previous_explanation != "Error during explanation.":
            important_subjects = self.extract_subjects(previous_explanation)
        
        if chunk['chunk'] == 0:
            explanation_prompt_text = f"[INST] {explanation_prompt}: {chunk['text']} [/INST]"
        else:
            explanation_prompt_text = f"""[INST]

Key topics and themes from previous context: {important_subjects}

Building upon this context, analyze the following new content.

New content to analyze: {explanation_prompt}: {previous_summary}\n{chunk['text']} [/INST]"""
        
        prompt_length = len(explanation_prompt_text.split())
        if prompt_length > 1000:
            self.log('warning', f"Prompt too long for chunk {chunk['chunk']}, truncating...")
            explanation_prompt_text = " ".join(explanation_prompt_text.split()[:1000])
        
        try:
            # Use lock to ensure thread safety when using the LLM
            with self._llm_lock:
                output = self.llm(explanation_prompt_text, max_tokens=1024, temperature=0.1)
                explanation = output["choices"][0]["text"].strip()
            
            self.log('info', f"Explanation generated: {len(explanation.split())} words")
            if not explanation:
                explanation = "No explanation returned."
            if explanation and explanation != "No explanation returned.":
                new_summary = self.offline_summarize(explanation)
            else:
                new_summary = "Previous explanation was empty or errored."
        except Exception as e:
            self.log('error', f"Error processing chunk {chunk['chunk']}: {e}")
            explanation = "Error during explanation."
            new_summary = "Error occurred in previous chunk processing."
        return chunk['chunk'], explanation, new_summary
    
    def final_summarize(self, text: str) -> str:
        """Create a concise executive summary from comprehensive text"""
        max_input_length = 2048
        
        if len(text.split()) > max_input_length:
            # Only use the beginning portion if too long
            text = " ".join(text.split()[:max_input_length])
        
        prompt = f"""[INST]
Create a concise, coherent paragraph that summarizes the following text.
The summary should be well-structured and avoid repeating any information.
Focus on the key points and maintain a natural flow.

Text to summarize:
{text}

Please provide a single, well-written paragraph that captures the essence of the text without repetition.
[/INST]"""
        
        try:
            with self._llm_lock:
                output = self.llm(prompt, max_tokens=512, temperature=0.1)
                summary = output["choices"][0]["text"].strip()
            
            # Basic post-processing
            processed_summary = summary
            processed_summary = re.sub(r'\.([A-Z])', r'. \1', processed_summary)
            processed_summary = re.sub(r'\s+([.,;:!?])', r'\1', processed_summary)
            processed_summary = re.sub(r'\s{2,}', ' ', processed_summary)
            
            return processed_summary.strip()
            
        except Exception as e:
            self.log('error', f"Error in final_summarize: {e}")
            return "Error generating final summary."
    
    def process_text(self, input_text: str, explanation_prompt: str) -> str:
        """Main processing pipeline for summarization"""
        # Sample the text to reduce volume
        sampled_text = self.sample_text(input_text)
        
        # Split the text into manageable chunks
        chunks = self.split_into_chunks(sampled_text)
        
        # Prepare for chunk processing
        all_explanations = [""] * len(chunks)
        
        # First chunk must be processed first to establish context
        first_chunk_idx, first_explanation, first_summary = self.process_chunk(
            (chunks[0], "", "", explanation_prompt)
        )
        all_explanations[first_chunk_idx] = first_explanation
        previous_summary = first_summary
        previous_explanation = first_explanation
        
        # Process remaining chunks sequentially
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            chunk_idx, explanation, new_summary = self.process_chunk(
                (chunk, previous_summary, previous_explanation, explanation_prompt)
            )
            all_explanations[chunk_idx] = explanation
            previous_summary = new_summary
            previous_explanation = explanation
        
        # Generate the final comprehensive summary
        comprehensive_summary = "\n\n".join([exp for exp in all_explanations if exp and not exp.isspace()])
        
        # Apply final summarization
        self.log('info', "Creating executive summary...")
        executive_summary = self.final_summarize(comprehensive_summary)
        
        if self.verbose:
            print("\n" + "="*80)
            print("EXECUTIVE SUMMARY:")
            print("="*80)
            print(executive_summary)
            print("="*80)
        
        return executive_summary
    
    def extract_article_content(self, url: str) -> str:
        """Extract the main article content from a webpage"""
        try:
            article_html = self.fetch_url_with_cache(url)
            
            # Parse and extract visible text
            article_soup = BeautifulSoup(article_html, 'html.parser')
            for tag in article_soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript', 'iframe', 'form']):
                tag.decompose()
            text = article_soup.get_text(separator='\n', strip=True)
            # Replace all single newlines with a space, keep double newlines
            cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
            return cleaned_text
        except Exception as e:
            self.log('error', f"Error fetching the article page: {e}")
            return ""
    
    def extract_hn_comments(self, hn_html: str) -> List[str]:
        """Extract and process comments from Hacker News"""
        soup = BeautifulSoup(hn_html, 'html.parser')
        comment_elements = soup.find_all('tr', class_='athing comtr')
        
        self.log('info', f"Found {len(comment_elements)} comments on Hacker News page")
        
        comments = []
        for comment in comment_elements:
            # Get comment content
            commtext = comment.find('div', class_='commtext c00')
            if commtext:
                # Get pure text without links or formatting
                text = commtext.get_text(separator=' ', strip=True)
                
                # Format the comment
                if text and len(text.split()) > 3:  # Keep comments with at least 4 words
                    comments.append(text)
        
        return comments
    
    def generate_final_summary(self, article_summary: str, comments_summary: str) -> str:
        """Generate a final summary paragraph combining article insights and comment reactions"""
        prompt = f"""[INST]
Create a single coherent paragraph that combines these two elements:
1. A summary of an article: {article_summary}
2. A summary of community reactions: {comments_summary}

The paragraph should give equal weight to both the article content AND the community's response to it.
Make sure to highlight at least 2-3 specific points from the community discussion.
Include areas of consensus and disagreement in the comments if they exist.
Focus on creating a synthesis that's informative and could naturally lead to discussion questions.
Make it concise but complete - it should be easy to generate questions from this summary.

Remember: The community's reaction and discussion is just as important as the article content itself.
[/INST]"""
        
        try:
            with self._llm_lock:
                output = self.llm(prompt, max_tokens=512, temperature=0.1)
                final_paragraph = output["choices"][0]["text"].strip()
                
                # Check if the final paragraph actually includes comment discussion
                if "communit" not in final_paragraph.lower() and "discussion" not in final_paragraph.lower() and "react" not in final_paragraph.lower():
                    # If not, try again with an even more explicit prompt
                    retry_prompt = f"""[INST]
The article can be summarized as: {article_summary}

The community discussion of this article raised these key points: {comments_summary}

Write ONE coherent paragraph that MUST include BOTH what the article is about AND what the community thought about it.
Dedicate at least 40% of your response to the community's reaction.
Highlight specific opinions, disagreements, or insights from the comments section.

Your response MUST cover both the article content and community reaction in a balanced way.
[/INST]"""
                    
                    with self._llm_lock:
                        output = self.llm(retry_prompt, max_tokens=512, temperature=0.1)
                        final_paragraph = output["choices"][0]["text"].strip()
                
            return final_paragraph
        except Exception as e:
            self.log('error', f"Error generating final summary: {e}")
            return f"Error generating final summary: {e}"
    
    def cleanup_resources(self, silent=False):
        """
        Safely clean up model resources
        
        Args:
            silent: If True, suppresses all logging during cleanup
        """
        # Store original verbosity and temporarily disable if silent=True
        original_verbose = self.verbose
        if silent:
            self.verbose = False
            
        try:
            if not silent:
                self.log('info', "Cleaning up model resources...")
            
            with self._model_cache_lock:
                if self._llm is not None:
                    try:
                        # Safely close the model in a specific sequence to avoid errors
                        if hasattr(self._llm, 'ctx') and self._llm.ctx is not None:
                            # First set the context to None
                            ctx = self._llm.ctx
                            self._llm.ctx = None
                            # Then clear other references
                            if hasattr(ctx, '_llama_free_model'):
                                ctx._llama_free_model = None
                            
                        self._llm = None
                        if not silent:
                            self.log('info', "Successfully cleaned up LLM")
                    except Exception as e:
                        if not silent:
                            self.log('warning', f"Error during LLM cleanup: {e}")
                
                # Clear model caches
                self._vector_model_cache.clear()
                self._model_cache.clear()
        finally:
            # Restore original verbosity setting
            self.verbose = original_verbose
    
    def summarize(self, hn_url: str) -> str:
        """
        Main function to summarize both the article and comments from a Hacker News post
        
        Args:
            hn_url: URL of the Hacker News post
            
        Returns:
            str: Final summary paragraph
        """
        try:
            self.log('info', f"Processing Hacker News URL: {hn_url}")
            
            # Ensure URL has protocol
            if not hn_url.startswith(('http://', 'https://')):
                hn_url = 'https://' + hn_url
                self.log('info', f"Added https:// prefix: {hn_url}")
            
            # Fetch the Hacker News page
            try:
                hn_html = self.fetch_url_with_cache(hn_url)
            except Exception as e:
                self.log('error', f"Error fetching Hacker News page: {e}")
                return f"Error: {e}"
            
            soup = BeautifulSoup(hn_html, 'html.parser')
            
            # Find the main article link
            main_link = None
            for td in soup.find_all('td', class_='title'):
                a = td.find('a', href=True)
                if a and a.text.lower() != 'more':
                    main_link = a['href']
                    break
            
            if not main_link:
                self.log('error', "Could not find the main article link.")
                return "Error: Could not find the main article link."
            
            # Make sure the link is absolute
            if not main_link.startswith(('http://', 'https://')):
                # Check if it's a relative link or just missing the protocol
                if main_link.startswith('//'):
                    main_link = 'https:' + main_link
                elif not main_link.startswith('/'):
                    main_link = 'https://' + main_link
                else:
                    # It's a relative link to HN itself, get the domain
                    hn_domain = re.match(r'https?://[^/]+', hn_url).group(0)
                    main_link = hn_domain + main_link
            
            # Extract and summarize article content
            self.log('info', f"Extracting content from article: {main_link}")
            article_content = self.extract_article_content(main_link)
            
            article_explanation_prompt = "Summarize the key points of this article"
            article_summary = self.process_text(article_content, article_explanation_prompt)
            
            if self.verbose:
                print("\n--- Article Summary ---\n")
                print(article_summary)
            
            # Extract and summarize comments
            self.log('info', "Extracting and analyzing comments...")
            comments = self.extract_hn_comments(hn_html)
            
            # Filter similar comments
            filtered_comments = self.filter_similar_comments(comments)
            self.log('info', f"Kept {len(filtered_comments)} unique comments after filtering")
            
            comments_text = "\n\n".join(filtered_comments)
            comments_explanation_prompt = "Summarize how people are reacting to the article, analyzing key themes in the discussion"
            comments_summary = self.process_text(comments_text, comments_explanation_prompt)
            
            if self.verbose:
                print("\n--- Comments Analysis ---\n")
                print(comments_summary)
            
            # Generate final combined summary
            self.log('info', "Generating final synthesis...")
            final_summary = self.generate_final_summary(article_summary, comments_summary)
            
            if self.verbose:
                print("\n--- Final Summary ---\n")
                print(final_summary)
            
            # Print performance metrics
            if self.verbose:
                print("\nPERFORMANCE METRICS:")
                for func_name, times in self._timings.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        print(f"{func_name}: avg {avg_time:.3f}s, total {sum(times):.3f}s, calls {len(times)}")
            
            return final_summary
        finally:
            # Always clean up resources, even if an error occurs
            self.cleanup_resources()

# ===============================================================================
# SIMPLE API FUNCTIONS
# ===============================================================================

def summarize_hn(url, verbose=False):
    """
    Simple API function to summarize a Hacker News post from another file.
    
    Args:
        url: Hacker News URL to summarize
        verbose: Whether to print progress information (default: False)
    
    Returns:
        str: Final summary paragraph combining article and comments
    """
    # Create a summarizer with the specified verbosity
    summarizer = HackerNewsSummarizer(verbose=verbose)
    
    try:
        # Run the summarization
        result = summarizer.summarize(url)
        return result
        
    except Exception as e:
        return f"Error summarizing the Hacker News post: {str(e)}"

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            hn_url = sys.argv[1]
        else:
            hn_url = input("Enter a Hacker News link: ").strip()
        
        # Create summarizer and run
        summarizer = HackerNewsSummarizer(verbose=True)
        result = summarizer.summarize(hn_url)
        
        print("\n=== FINAL RESULT ===\n")
        print(result)
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

# Example usage from another file:
'''
from hackernews_summarizer import summarize_hn

# Simple usage
summary = summarize_hn("news.ycombinator.com/item?id=12345678")
print(summary)

# With verbose output 
summary = summarize_hn("news.ycombinator.com/item?id=12345678", verbose=True)
print(summary)
''' 