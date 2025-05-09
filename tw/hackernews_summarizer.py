import re
import logging
from typing import Optional

class HackerNewsSummarizer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def extract_article_content(self, url: str) -> Optional[str]:
        """Extract content from a URL."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error extracting content from {url}: {e}")
            return None

    def final_summarize(self, text: str) -> str:
        """Generate a summary of the text."""
        if not text:
            return ""
        
        # For now, just return the first 500 characters as a simple summary
        return text[:500] + "..." if len(text) > 500 else text

    def cleanup_resources(self, silent: bool = False) -> None:
        """Clean up any resources used by the summarizer."""
        if not silent and self.verbose:
            self.logger.info("Cleaning up summarizer resources") 