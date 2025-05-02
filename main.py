import requests
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
headlines = soup.select(".athing .titleline")

# Print the headlines
for idx, headline in enumerate(headlines, start=1):
    title = headline.get_text(strip=True)
    response = client.responses.create(
    model="gpt-4.1-nano",
    input=f"I am a business investor interested in computer chips. After reading this articles title, Tell me with a simply Yes or No if I would be interested in this. If Yes, return 1, if No, return 0. Only return 1 or 0. This is the article title: {title}")
    print(f"{title}: {response.output_text}")
    
    
def research(title):
    
    