from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path

class ContentFetcher(ABC):
    """컨텐츠 수집 기본 클래스"""
    
    @abstractmethod
    def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """컨텐츠 수집"""
        pass
    
    @abstractmethod
    def validate(self, content: Dict[str, Any]) -> bool:
        """컨텐츠 유효성 검사"""
        pass

class TextProcessor(ABC):
    """텍스트 처리 기본 클래스"""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """텍스트 처리"""
        pass
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """텍스트 청크 분할"""
        pass

class StorageHandler(ABC):
    """저장소 처리 기본 클래스"""
    
    @abstractmethod
    def save(self, data: Dict[str, Any], path: Optional[Path] = None) -> bool:
        """데이터 저장"""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> Dict[str, Any]:
        """데이터 로드"""
        pass 