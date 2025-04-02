# Example usage:
from abc import ABC, abstractmethod
import requests
import cloudscraper
import random
import re
from bs4 import BeautifulSoup
from omegaconf import DictConfig

class MediaSource(ABC):
    @abstractmethod
    def fetch_content(self, identifier):
        pass
    # @abstractmethod
    # def get_transcript(self, identifier):
    #     pass

class WebContent():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.headers = self._get_random_headers()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.api_token = cfg.api_keys['diffbot']
        self.base_url = "https://api.diffbot.com/v3/article"
        print('#'*7+'WebContent init'+'#'*7)

    def extract_text(self, url: str):
        """
        Extracts the main text and essential information from a given URL.
        
        :param url: str - The URL of the page to analyze.
        :return: dict - Contains 'title', 'author', 'date', and 'text' if extraction is successful.
        """
        params = {
            'token': self.api_token,
            'url': url,
            'discussion': 'false'  # Optional parameter to disable extraction of comments.
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check if article data is available in the response
            if 'objects' in data and len(data['objects']) > 0:
                article_data = data['objects'][0]
                extracted_content = {
                    'title': article_data.get('title', ''),
                    'author': article_data.get('author', ''),
                    'date': article_data.get('date', ''),
                    'text': article_data.get('text', '')
                }
                return extracted_content
            else:
                print("No article content found in the response.")
                return {}
        
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return {}
    

    def clean_text(self, raw_text: str) -> str:
        """
        Clean the extracted text by removing HTML tags and unnecessary whitespace.
        
        :param raw_text: str - Raw text containing HTML tags and unwanted characters.
        :return: str - Cleaned text with only the main content.
        """
        if raw_text is None:
            return ""
            
        if isinstance(raw_text, list):
            raw_text = " ".join(str(item) for item in raw_text)
            
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
            
        if not raw_text:
            return ""
            
        # 1. Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text(separator=" ")

        # 2. Remove unwanted characters or patterns (e.g., extra spaces, newlines)
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _get_random_headers(self):
        """랜덤 User-Agent와 함께 요청 헤더 생성"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1',
            'Cache-Control': 'max-age=0'
        }

    
    def _is_valid_content(self, content):
        """컨텐츠가 유효한지 확인"""
        if not content or len(content) < 100:
            return False
        
        # 봇 감지/보안 체크 키워드
        security_keywords = [
            'cf-browser-verification',
            'security check',
            'captcha',
            'robot verification',
            'access denied',
            'please verify',
            '차단된 페이지',
            '접근이 거부되었습니다'
        ]
        
        content_lower = content.lower()
        return not any(keyword in content_lower for keyword in security_keywords)