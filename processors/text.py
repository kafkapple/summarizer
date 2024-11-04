import re
from typing import Union, List, Dict, Optional
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter
from ..core.base import TextProcessor

class ContentProcessor(TextProcessor):
    """컨텐츠 텍스트 처리 클래스"""
    
    def __init__(self, max_tokens: int = 4096, chunk_overlap: int = 100):
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=chunk_overlap
        )
    
    def process(self, text: Union[str, List[Dict[str, str]], List[str]], 
                clean_options: Optional[Dict] = None) -> str:
        """
        텍스트 전처리 통합 메서드
        
        Args:
            text: 처리할 텍스트
            clean_options: 전처리 옵션
                - remove_special_chars: 특수문자 제거
                - to_lowercase: 소문자 변환
                - remove_numbers: 숫자 제거
                - clean_tags: [음악], (박수) 등의 태그 제거
        """
        if clean_options is None:
            clean_options = {
                'remove_special_chars': False,
                'to_lowercase': False,
                'remove_numbers': False,
                'clean_tags': True
            }
            
        try:
            processed_text = self._preprocess_text(text, **clean_options)
            return self._clean_text(processed_text)
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {e}")
            return ''
    
    def chunk(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
        return self.text_splitter.split_text(text)
    
    def _preprocess_text(self, text: Union[str, List], **options) -> str:
        """텍스트 전처리"""
        if not text:
            return ''
            
        if isinstance(text, list):
            return self._process_list(text, **options)
        
        if not isinstance(text, str):
            raise ValueError("입력은 문자열 또는 리스트 형식이어야 합니다")
            
        return text
    
    def _process_list(self, text_list: List, **options) -> str:
        """리스트 형식 텍스트 처리"""
        if all(isinstance(item, dict) and 'text' in item for item in text_list):
            return self._process_transcript(text_list, **options)
        
        return ' '.join(str(item).strip() for item in text_list if str(item).strip())
    
    def _process_transcript(self, transcript: List[Dict], clean_tags: bool = True) -> str:
        """자막 형식 텍스트 처리"""
        text_parts = []
        for entry in transcript:
            cleaned_text = entry['text'].strip()
            if clean_tags:
                cleaned_text = self._remove_tags(cleaned_text)
            
            if cleaned_text:
                if not cleaned_text[-1] in '.!?':
                    cleaned_text += '.'
                text_parts.append(cleaned_text)
        
        return ' '.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제
        
        Args:
            text: 정제할 텍스트
        
        Returns:
            정제된 텍스트
        """
        if not text:
            return ""
            
        # HTML 제거
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        
        # 여러 줄 바꿈 정리
        text = re.sub(r'\n+', '\n', text)
        
        # 여러 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    @staticmethod
    def _remove_tags(text: str) -> str:
        """태그 제거"""
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text
    
    @staticmethod
    def _apply_options(text: str, 
                      remove_special_chars: bool = False,
                      to_lowercase: bool = False,
                      remove_numbers: bool = False) -> str:
        """옵션 적용"""
        if remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        if to_lowercase:
            text = text.lower()
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        return text 