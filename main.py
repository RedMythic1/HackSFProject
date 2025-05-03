import requests
from time import sleep
import re
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import asyncio
from playwright.async_api import async_playwright
from pdfminer.high_level import extract_text
from transformers import pipeline
from io import BytesIO
import urllib.parse

# API Configuration
client = OpenAI(api_key='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')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Constants
HACKER_NEWS_URL = "https://news.ycombinator.com/"

# ===== HELPER FUNCTIONS =====

def offline_summarize(text):
    """
    Summarize the provided text using Hugging Face BART model.
    Returns the summarized version of the input text.
    """
    # The model limits the length of input to around 1024 tokens, so we split long text
    max_input_length = 1024
    if len(text.split()) > max_input_length:
        text = text[:max_input_length]
    
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def is_valid_hn_link(link):
    """Check if link matches the format: ends with item?id= followed by 8 digits"""
    pattern = r'item\?id=\d{8}$'
    return bool(re.search(pattern, link))


# ===== CONTENT ACQUISITION FUNCTIONS =====

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
    
    return articles

async def scrape_cleaned_text(url, min_words_div=5):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            content = await page.content()
            await browser.close()
    except Exception as e:
        return f"[!] Failed to load {url}: {e}"

    soup = BeautifulSoup(content, 'html.parser')

    for tag in soup(['script', 'style']):
        tag.decompose()

    body = soup.body
    text_chunks = []
    if body:
        for tag in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = tag.get_text(separator=' ', strip=True)
            if text:
                text_chunks.append(text)

        for div in body.find_all('div'):
            if div.find(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                continue
            text = div.get_text(separator=' ', strip=True)
            if len(text.split()) >= min_words_div:
                text_chunks.append(text)

    html_text = "\n\n".join(text_chunks).strip()

    # PDF content (iframe/embed/object)
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
                html_text += (
                    f"\n\n--- PDF content from {pdf_url} ---\n\n"
                    + "\n".join(
                        [line for line in pdf_text.splitlines() if len(line.split()) >= min_words_div]
                    )
                )
        except Exception as e:
            html_text += f"\n\n[!] Failed to extract PDF at {pdf_url}: {e}\n"

    return html_text

# ===== CONTENT PROCESSING FUNCTIONS =====

def preprocess(articles):
    """Process and score articles based on user interests"""
    finalized_articles = []
    for article_data in articles:
        title = article_data[0]
        link = article_data[1]
        
        # Fetch the discussion page content
        discussion_response = requests.get(link)
        discussion_soup = BeautifulSoup(discussion_response.text, "html.parser")
        
        # Find the text under class="toptext"
        toptext = discussion_soup.select_one(".toptext")
        link_text = ""
        if toptext:
            # Get all paragraph text
            paragraphs = toptext.find_all('p')
            link_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        
        # Get the subject line
        subject_response = client.responses.create(
            model="gpt-4.1-nano",
            input=f"Analyze this title and passage, and tell me what the subject is. Example: Altair at 50: Remembering the first Personal Computer -> Altair, First Personal computer. Perform this on {title} and this passage: {link_text}. DO NOT OUTPUT ANYTHING ELSE. NO THANK YOUs or confirmations. I just need the Subject line. Do not say anything else. ONLY GIVE 1 SUBJECT, Summarize. Return in the form: [subject_line]"
        )
        subject_line = subject_response.output_text
        
        # Check if subject line aligns with user interests
        interest_response = client.responses.create(
            model="gpt-4.1-nano",
            input=f"Does the subject '{subject_line}' align with the user's interests: '{user_settings}'? Rate this alignment from 0 to 100, where 0 means no alignment and 100 means perfect alignment. ONLY return a number between 0 and 100. Do not add any explanation, just the number."
        )
        
        # Extract integer from response
        try:
            interest_score = int(''.join(filter(str.isdigit, interest_response.output_text)))
            # Ensure the score is between 0 and 100
            interest_score = max(0, min(100, interest_score))
            
            # Add article with score to finalized list
            finalized_articles.append({
                'title': title,
                'link': link,
                'subject': subject_line,
                'score': interest_score
            })
        except ValueError:
            # If no integer found, assign a default score
            finalized_articles.append({
                'title': title,
                'link': link,
                'subject': subject_line,
                'score': 0
            })
    
    # Sort by score in descending order and keep only top 3
    finalized_articles.sort(key=lambda x: x['score'], reverse=True)
    finalized_articles = finalized_articles[:3]
    
    return finalized_articles


def research(article_data):
    """Generate research questions, search links, and scrape content"""
    subject_line = article_data['subject']
    question_research_data = {}

    r_questions = client.responses.create(
        model="gpt-4.1-nano",
        input=f"I do not know what {subject_line} is. I wish to learn about it. Please give me ten search query prompts that someone decently technically savvy would search for deep and meaningful research. Answer this question to the fullest of your extent. answer in the form [insert question here]~[insert question here]~[insert question here]~[insert question here] etc. etc. Only return the questions, and nothing else."
    )
    
    questions_list = [q.strip() for q in r_questions.output_text.split('~') if q.strip()]

    async def process_question(question):
        summarized_results = []
        with DDGS() as ddgs:
            links = [r['href'] for r in ddgs.text(question, max_results=3)]
            for url in links:
                try:
                    raw_text = await scrape_cleaned_text(url)
                    if not raw_text or len(raw_text.split()) < 30:
                        continue  # Skip if not enough content to summarize

                    # Use offline summarization here
                    summary = offline_summarize(raw_text)

                    summarized_results.append({
                        'url': url,
                        'summary': summary
                    })

                except Exception as e:
                    summarized_results.append({
                        'url': url,
                        'summary': f"[!] Failed to summarize content: {e}"
                    })

        return question, summarized_results

    async def process_all():
        tasks = [process_question(q) for q in questions_list]
        return dict(await asyncio.gather(*tasks))

    return asyncio.run(process_all())

# ===== SCRIPT EXECUTION =====

# Get user interests
user_settings = input('What are your interests? (please enter 4): \n')
 
# Convert interests to vectors
  