import llama_cpp._internals as _internals
_internals.LlamaSampler.__del__ = lambda self: None

from llama_cpp import Llama
from transformers import pipeline
import re
from time import sleep
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from io import BytesIO
import urllib.parse

def clean_text(text):
    """Clean the text by removing extra spaces and unwanted characters."""
    text = ' '.join(text.split())
    text = re.sub(r'\s*\|\s*', ' ', text)
    return text

def scrape_cleaned_text(url, min_words_div=5):
    """Scrape and clean text from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
    except Exception as e:
        return f"[!] Failed to load {url}: {e}"

    soup = BeautifulSoup(content, 'html.parser')

    for tag in soup(['script', 'style']):
        tag.decompose()

    body = soup.body
    text_chunks = []
    
    if body:
        for tag in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            text = tag.get_text(separator=' ', strip=True)
            if text and "reply" not in text.lower():
                cleaned_text = clean_text(text)
                if cleaned_text:
                    text_chunks.append(cleaned_text)

        for div in body.find_all('div'):
            text = div.get_text(separator=' ', strip=True)
            if len(text.split()) > min_words_div and "reply" not in text.lower():
                cleaned_text = clean_text(text)
                if cleaned_text:
                    text_chunks.append(cleaned_text)

    pdf_urls = set()
    if body:
        for tag in body.find_all(['iframe', 'embed', 'object']):
            src = tag.get('data') or tag.get('src')
            if src and src.lower().endswith('.pdf'):
                pdf_urls.add(urllib.parse.urljoin(url, src))

    for pdf_url in pdf_urls:
        try:
            resp = requests.get(pdf_url)
            resp.raise_for_status()
            pdf_text = extract_text(BytesIO(resp.content))
            if pdf_text:
                cleaned_pdf_text = clean_text(pdf_text)
                text_chunks.append(f"\n\n--- PDF content from {pdf_url} ---\n\n" + cleaned_pdf_text)
        except Exception as e:
            text_chunks.append(f"\n\n[!] Failed to extract PDF at {pdf_url}: {e}\n")
        
    return "\n\n".join(text_chunks)

def initialize_models():
    """Initialize the LLM and summarization models"""
    llm = Llama(
        model_path="/Users/avneh/llama-models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=32768,
        n_threads=6,
        n_gpu_layers=35,
        chat_format="mistral-instruct",
        verbose=False,
        stop=None
    )
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return llm, summarizer

def offline_summarize(text, summarizer):
    """Summarize the provided text using Hugging Face BART model."""
    max_input_length = 1024
    if len(text.split()) > max_input_length:
        text = text[:max_input_length]
    
    summary = summarizer(text, max_length=50, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def split_into_chunks(input_string, chunk_size=467):
    """Split input text into chunks of approximately equal size"""
    sentences = re.split(r'(?<=[.!?]) +', input_string)
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

    return chunks

def process_chunk(chunk, previous_summary, llm, summarizer, explanation_prompt="Explain what this means in simple terms"):
    """Process a single chunk of text and return its explanation and summary"""
    if chunk['chunk'] == 0:
        explanation_prompt_text = f"[INST] {explanation_prompt}: {chunk['text']} [/INST]"
    else:
        explanation_prompt_text = f"[INST] Previous context summary: {explanation_prompt}: {previous_summary}, considering the previous context, fulfill the following prompt: {explanation_prompt}: {chunk['text']} [/INST]"
    
    prompt_length = len(explanation_prompt_text.split())
    print(f"Prompt length for chunk {chunk['chunk']}: {prompt_length} words")
    
    if prompt_length > 600:
        print(f"Prompt too long for chunk {chunk['chunk']}, skipping...")
        return None, previous_summary
    
    try:
        output = llm(explanation_prompt_text, max_tokens=1024, temperature=0.1)
        explanation = output["choices"][0]["text"].strip()
        
        if not explanation:
            explanation = "No explanation returned."
        
        print(f"Chunk {chunk['chunk']} Explanation: {explanation}\n")

        if explanation and explanation != "No explanation returned.":
            new_summary = offline_summarize(explanation, summarizer)
        else:
            new_summary = "Previous explanation was empty or errored."
            
    except Exception as e:
        print(f"Error processing chunk {chunk['chunk']}: {e}")
        explanation = "Error during explanation."
        new_summary = "Error occurred in previous chunk processing."

    return explanation, new_summary

def process_text(input_text, llm, summarizer, explanation_prompt="Explain what this means in simple terms"):
    """Process the entire input text through chunks"""
    chunks = split_into_chunks(input_text)
    previous_summary = ""
    
    for chunk in chunks:
        explanation, previous_summary = process_chunk(chunk, previous_summary, llm, summarizer, explanation_prompt)
        sleep(0.7)  # Delay to avoid rate limiting

def summarize_webpage(url, explanation_prompt="Explain what this means in simple terms"):
    """Scrape a webpage and summarize its content using the chunked explanation approach"""
    # Scrape and clean the webpage content
    print(f"Scraping content from {url}...")
    webpage_text = scrape_cleaned_text(url)
    
    if webpage_text.startswith("[!]"):
        print(f"Error scraping webpage: {webpage_text}")
        return
    
    # Initialize models
    llm, summarizer = initialize_models()
    
    try:
        # Process the scraped text
        print("\nProcessing webpage content...")
        process_text(webpage_text, llm, summarizer, explanation_prompt)
    finally:
        # Clean up
        llm.close()

def main():
    """Main function to run the text processing pipeline"""
    # Example usage
    url = "https://www.nba.com/warriors/news/warriors-vs-celtics-game-6-nba-finals-2022"
    explanation_prompt = "How would this affect the United States gold exchange rate?"
    
    summarize_webpage(url, explanation_prompt)

if __name__ == "__main__":
    main()
