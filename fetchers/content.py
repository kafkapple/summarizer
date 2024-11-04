from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

from ..core.base import ContentFetcher
from ..processors.text import ContentProcessor
from .extractors import TrafilaturaExtractor, ReadabilityExtractor, NewspaperExtractor
from .youtube import YouTubeFetcher

class WebContentFetcher(ContentFetcher):
    """웹 컨텐츠 수집기"""
    
    def __init__(self, config):
        self.config = config
        self.processor = ContentProcessor()
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.extractors = [
            TrafilaturaExtractor(config),
            ReadabilityExtractor(config),
            NewspaperExtractor(config)
        ]
        self.youtube = YouTubeFetcher(config)
    
    def validate(self, url: str) -> bool:
        """URL 유효성 검사"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            
            if result.scheme not in ['http', 'https']:
                return False
                
            # YouTube URL 별도 처리
            if self._detect_content_type(url) == 'youtube':
                video_id, is_playlist = self.youtube.parse_youtube_url(url)
                return bool(video_id)
                
            # 일반 웹 URL 접근성 검사
            response = requests.head(url, headers=self.headers, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            print(f"URL 검증 실패: {e}")
            return False
    
    def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """웹 컨텐츠 수집"""
        try:
            content_type = self._detect_content_type(url)
            
            if content_type == 'youtube':
                return self._fetch_youtube_content(url)
            return self._fetch_web(url)
            
        except Exception as e:
            print(f"컨텐츠 수집 실패: {e}")
            return None
    
    def _fetch_youtube_content(self, url: str) -> Optional[Dict[str, Any]]:
        """YouTube 컨텐츠 수집"""
        try:
            video_id, is_playlist = self.youtube.parse_youtube_url(url)
            if not video_id:
                return None
            
            if is_playlist:
                return self.youtube.fetch_playlist(video_id)
            return self.youtube.fetch_video(video_id)
            
        except Exception as e:
            print(f"YouTube 컨텐츠 수집 실패: {e}")
            return None
    
    def _fetch_web(self, url: str) -> Optional[Dict[str, Any]]:
        """일반 웹 컨텐츠 수집"""
        # 먼저 추출기들을 시도
        for extractor in self.extractors:
            try:
                content = extractor.extract(url)
                if content and self._is_valid_content(content):
                    return content
            except Exception as e:
                print(f"추출기 {extractor.__class__.__name__} 실패: {e}")
                continue
        
        # 추출기들이 실패하면 기본 방식 시도
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            content = self._extract_content(soup)
            
            if not content:
                return None
            
            result = {
                'title': title,
                'description': description,
                'text': content,
                'url': url,
                'type': 'web'
            }
            
            return result if self._is_valid_content(result) else None
            
        except Exception as e:
            print(f"웹 컨텐츠 수집 실패: {e}")
            return None
    
    @staticmethod
    def _detect_content_type(url: str) -> str:
        """컨텐츠 타입 감지"""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        return 'web'
    
    def _is_valid_content(self, content: Dict[str, Any]) -> bool:
        """컨텐츠 유효성 검사"""
        if not content or not content.get('text'):
            return False
        return len(content['text']) >= 500
    
    # ... (나머지 헬퍼 메서드들은 동일)