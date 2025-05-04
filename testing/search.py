import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from io import BytesIO
import urllib.parse
import re

def clean_text(text):
    """
    Clean the text by removing extra spaces and unwanted characters.
    """
    # Remove extra spaces
    text = ' '.join(text.split())
    # Remove unwanted characters (e.g., '|')
    text = re.sub(r'\s*\|\s*', ' ', text)  # Replace ' | ' with a single space
    return text

def scrape_cleaned_text(url, min_words_div=5):
    """
    Scrape and clean text from a given URL.
    Extracts all headers, paragraphs, and <div> texts with more than min_words_div words,
    and handles embedded PDFs.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        content = response.text
    except Exception as e:
        return f"[!] Failed to load {url}: {e}"

    soup = BeautifulSoup(content, 'html.parser')

    # Remove scripts and styles
    for tag in soup(['script', 'style']):
        tag.decompose()

    body = soup.body
    text_chunks = []
    
    if body:
        # Collect text from all header and paragraph tags
        for tag in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            text = tag.get_text(separator=' ', strip=True)
            if text and "reply" not in text.lower():  # Exclude texts containing "reply"
                cleaned_text = clean_text(text)
                if cleaned_text:  # Ensure cleaned text is not empty
                    text_chunks.append(cleaned_text)

        # Collect text from <div> tags with more than min_words_div words
        for div in body.find_all('div'):
            text = div.get_text(separator=' ', strip=True)
            if len(text.split()) > min_words_div and "reply" not in text.lower():  # Exclude texts containing "reply"
                cleaned_text = clean_text(text)
                if cleaned_text:  # Ensure cleaned text is not empty
                    text_chunks.append(cleaned_text)

    # Look for embedded PDFs and extract text
    pdf_urls = set()
    if body:
        for tag in body.find_all(['iframe', 'embed', 'object']):
            src = tag.get('data') or tag.get('src')
            if src and src.lower().endswith('.pdf'):
                pdf_urls.add(urllib.parse.urljoin(url, src))

    # Extract text from each PDF
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

# Example usage
if __name__ == "__main__":
    url = 'https://www.reddit.com/r/BambuLab/'
    extracted_text = scrape_cleaned_text(url)
    print(extracted_text)