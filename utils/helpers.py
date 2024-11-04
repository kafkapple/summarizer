from typing import Dict, Any, Optional
import re
from urllib.parse import urlparse, parse_qs

def extract_youtube_id(url: str) -> Optional[str]:
    """YouTube URL에서 비디오 ID 추출"""
    try:
        if 'youtube.com' in url:
            parsed = urlparse(url)
            return parse_qs(parsed.query).get('v', [None])[0]
        elif 'youtu.be' in url:
            return url.split('/')[-1]
        return None
    except Exception:
        return None

def clean_filename(text: str) -> str:
    """파일명으로 사용 가능하도록 텍스트 정제"""
    # 특수문자 제거 및 공백 처리
    text = re.sub(r'[\\/*?:"<>|]', '', text)
    text = text.strip().replace(' ', '_')
    # 길이 제한
    return text[:50]

def format_duration(seconds: int) -> str:
    """초 단위 시간을 보기 좋은 형식으로 변환"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"

def remove_empty_lines(text: str) -> str:
    """빈 줄 제거"""
    return '\n'.join(line for line in text.splitlines() if line.strip())

def truncate_text(text: str, max_length: int = 100) -> str:
    """텍스트 길이 제한"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..." 