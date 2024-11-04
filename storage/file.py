from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle
from datetime import datetime
from ..core.base import StorageHandler

class FileStorage(StorageHandler):
    """파일 저장소 핸들러"""
    
    def __init__(self, config):
        self.config = config
    
    def save(self, data: Dict[str, Any], path: Optional[Path] = None) -> bool:
        """데이터를 파일로 저장"""
        try:
            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = self.config.paths['results'] / f'content_{timestamp}.json'
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False
    
    def load(self, path: Path) -> Optional[Dict[str, Any]]:
        """파일에서 데이터 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"파일 로드 실패: {e}")
            return None
    
    def save_backup(self, data: Dict[str, Any], prefix: str = 'backup') -> Path:
        """백업 파일 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.config.paths['cache'] / f'{prefix}_{timestamp}.pkl'
        
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, 'wb') as f:
                pickle.dump(data, f)
            return backup_path
            
        except Exception as e:
            print(f"백업 저장 실패: {e}")
            raise 