import requests
from time import sleep
import re
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from openai import OpenAI
client = OpenAI(api_key='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')

# URL of the first page of Hacker News
url = "https://news.ycombinator.com/"

# Send a GET request to the page
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Find all the story links
headlines = soup.select(".athing .titleline a")

def is_valid_hn_link(link):
    # Check if link matches the exact format: ends with item?id= followed by 8 digits
    pattern = r'item\?id=\d{8}$'
    return bool(re.search(pattern, link))
        
article_data = ['Show HN: I built a synthesizer based on 3D physics (anukari.com)', 'https://news.ycombinator.com/item?id=43873074']

def research(article_data):
    question_research_links = {}
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
    
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=f"Analyze this title and passage, and tell me what the subject is. Example: Altair at 50: Remembering the first Personal Computer -> Altair, First Personal computer. Perform this on {title} and this passage: {link_text}. DO NOT OUTPUT ANYTHING ELSE. NO THANK YOUs or confirmations. I just need the Subject line. Do not say anything else. ONLY GIVE 1 SUBJECT, Summarize. Return in the form: [subject_line]"
    )
    
    subject_line = response.output_text
    r_questions = client.responses.create(
        model="gpt-4.1-nano",
        input=f"I do not know what {subject_line} is. I wish to learn about it. Please give me ten search query prompts that someone decently technically savvy would search for deep and meaningful research. Answer this question to the fullest of your extent. answer in the form [insert question here]~[insert question here]~[insert question here]~[insert question here] etc. etc. Only return the questions, and nothing else."
    )
    # Split the questions into a list using ~ as delimiter
    questions_list = r_questions.output_text.split('~')
    
    # Print each question on a new line
    for question in questions_list:
        links = []
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=3))
            for r in results:
                links.append(r['href'])
        question_research_links[question] = links
        print(question_research_links)
        sleep(2)
        
    print(question_research_links)
    
research(article_data)    