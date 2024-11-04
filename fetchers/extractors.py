from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from readability import Document
from newspaper import Article
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import trafilatura
import json
from ..processors.text import ContentProcessor

class BaseExtractor(ABC):
    """기본 추출기 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.processor = ContentProcessor()
    
    @abstractmethod
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        pass
    
    def _get_response(self, url: str) -> Optional[requests.Response]:
        """URL로부터 응답 받기"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"URL 요청 실패: {e}")
            return None
    
    def _is_valid_content(self, content: Dict[str, Any]) -> bool:
        """컨텐츠 유효성 검사"""
        if not content or not content.get('text'):
            return False
        return len(content['text']) >= 500

class TrafilaturaExtractor(BaseExtractor):
    """Trafilatura 기반 추출기"""
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._get_response(url)
            if not response:
                return None
            
            downloaded = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                output_format='json'
            )
            
            if not downloaded:
                return None
            
            content = json.loads(downloaded)
            result = {
                'title': content.get('title', ''),
                'text': self.processor.process(content.get('text', '')),
                'description': content.get('description', ''),
                'author': content.get('author', ''),
                'date': content.get('date', ''),
                'url': url,
                'source': 'trafilatura'
            }
            
            return result if self._is_valid_content(result) else None
            
        except Exception as e:
            print(f"Trafilatura 추출 실패: {e}")
            return None

class ReadabilityExtractor(BaseExtractor):
    """Readability 기반 추출기"""
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._get_response(url)
            if not response:
                return None
            
            doc = Document(response.text)
            
            result = {
                'title': doc.title(),
                'text': self.processor.process(doc.summary()),
                'description': doc.meta_description or '',
                'url': url,
                'source': 'readability'
            }
            
            return result if self._is_valid_content(result) else None
            
        except Exception as e:
            print(f"Readability 추출 실패: {e}")
            return None

class NewspaperExtractor(BaseExtractor):
    """Newspaper3k 기반 추출기"""
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            result = {
                'title': article.title,
                'text': self.processor.process(article.text),
                'description': article.meta_description or '',
                'author': article.authors[0] if article.authors else '',
                'date': str(article.publish_date) if article.publish_date else '',
                'url': url,
                'source': 'newspaper'
            }
            
            return result if self._is_valid_content(result) else None
            
        except Exception as e:
            print(f"Newspaper 추출 실패: {e}")
            return None

class BeautifulSoupExtractor(BaseExtractor):
    """BeautifulSoup 기반 추출기"""
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._get_response(url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 불필요한 요소 제거
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            # 메타데이터 추출
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            
            # 본문 추출
            main_content = self._extract_main_content(soup)
            if not main_content:
                return None
            
            result = {
                'title': title,
                'text': self.processor.process(main_content),
                'description': description,
                'url': url,
                'source': 'beautifulsoup'
            }
            
            return result if self._is_valid_content(result) else None
            
        except Exception as e:
            print(f"BeautifulSoup 추출 실패: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """제목 추출"""
        title = soup.find('meta', property='og:title')
        if title:
            return title.get('content', '')
        
        title = soup.find('title')
        if title:
            return title.text.strip()
        
        return ''
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """설명 추출"""
        desc = soup.find('meta', property='og:description')
        if desc:
            return desc.get('content', '')
        
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            return desc.get('content', '')
        
        return ''
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """주요 컨텐츠 영역 추출"""
        # 우선순위 순서대로 탐색
        selectors = [
            'article', 'main', '.content', '#content',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content.get_text(separator=' ').strip()
        
        # 대안: body 전체 텍스트
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ').strip()
        
        return None