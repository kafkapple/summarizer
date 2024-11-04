from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional

class Settings:
    """애플리케이션 설정 관리"""
    
    def __init__(self, env_path: Optional[str] = None):
        # 기본 .env 파일 위치 설정
        if env_path is None:
            env_path = Path(__file__).parent / 'data' / '.env'
            print(str(env_path))
        
        load_dotenv(env_path)
        self._load_settings()
    
    def _load_settings(self):
        """모든 설정 로드"""
        # API 키 설정
        self.api_keys = {
            'openai': self._get_required_env('OPENAI_API_KEY'),
            'youtube': self._get_env('YOUTUBE_API_KEY'),
            'notion': self._get_env('NOTION_TOKEN'),
            'diffbot': self._get_env('DIFFBOT_API_TOKEN'),
        }
        
        # Notion 설정
        self.notion = {
            'database_ids': {
                'youtube': self._get_env('NOTION_DB_YOUTUBE_ID'),
                'pocket': self._get_env('NOTION_DB_POCKET_ID'),
                'raindrop': self._get_env('NOTION_DB_RAINDROP_ID'),
            }
        }
        
        # 모델 설정
        self.model = {
            'name': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 4096,
            'buffer_tokens': 1000,
            'max_response_tokens': 600,
            'min_response_tokens': 150,
        }
        
        # 일반 설정
        self.general = {
            'output_language': 'ko',
            'include_keywords': True,
            'include_full_text': False,
            'enable_chapters': True,
            'max_chunks_per_chapter': 5,
        }
        
        # 경로 설정
        self.paths = {
            'data': Path(__file__).parent.parent.parent / 'data',
            'cache': Path(__file__).parent.parent.parent / 'data' / 'cache',
            'logs': Path(__file__).parent.parent.parent / 'data' / 'logs',
            'results': Path(__file__).parent.parent.parent / 'data' / 'results',
        }
        
        # 경로 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _get_env(self, key: str, default: str = '') -> str:
        """환경 변수 가져오기"""
        return os.getenv(key, default)
    
    def _get_required_env(self, key: str) -> str:
        """필수 환경 변수 가져오기"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value 