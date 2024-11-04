from typing import Dict, List, Any, Optional
from notion_client import Client
from datetime import datetime
from pathlib import Path
from ..core.base import StorageHandler

class NotionBlock:
    """Notion 블록 생성 유틸리티"""
    
    @staticmethod
    def text(content: str, block_type: str = "paragraph") -> Dict:
        """기본 텍스트 블록 생성"""
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": content}
                }]
            }
        }
    
    @staticmethod
    def bullet(content: str) -> Dict:
        """글머리 기호 블록"""
        return NotionBlock.text(content, "bulleted_list_item")
    
    @staticmethod
    def heading(content: str, level: int = 2) -> Dict:
        """제목 블록"""
        return NotionBlock.text(content, f"heading_{level}")
    
    @staticmethod
    def divider() -> Dict:
        """구분선 블록"""
        return {"object": "block", "type": "divider", "divider": {}}

class NotionClient(StorageHandler):
    """Notion 저장소 핸들러"""
    
    def __init__(self, config):
        self.config = config
        self.client = Client(auth=config.api_keys['notion'])
        self.block_builder = NotionBlock()
    
    def save(self, data: Dict[str, Any], database_id: str) -> bool:
        """
        데이터를 Notion 데이터베이스에 저장
        
        Args:
            data: 저장할 데이터
            database_id: Notion 데이터베이스 ID
        """
        try:
            # 페이지 생성
            page = self.client.pages.create(
                parent={"database_id": database_id},
                properties=self._create_properties(data),
                children=self._create_blocks(data)
            )
            
            print(f"Notion에 저장됨: {page['url']}")
            return True
            
        except Exception as e:
            print(f"Notion 저장 실패: {e}")
            self._backup_failed_data(data)
            return False
    
    def load(self, page_id: str) -> Dict[str, Any]:
        """
        Notion 페이지 데이터 로드
        
        Args:
            page_id: Notion 페이지 ID
        """
        try:
            page = self.client.pages.retrieve(page_id)
            blocks = self.client.blocks.children.list(page_id)
            
            return {
                'properties': page['properties'],
                'blocks': blocks['results']
            }
            
        except Exception as e:
            print(f"Notion 페이지 로드 실패: {e}")
            return {}
    
    def _create_properties(self, data: Dict[str, Any]) -> Dict:
        """페이지 속성 생성"""
        properties = {
            "Title": {
                "title": [{
                    "text": {"content": data.get('title', 'Untitled')}
                }]
            },
            "URL": {"url": data.get('url', '')},
            "Type": {"select": {"name": data.get('type', 'web')}},
            "Created": {"date": {"start": datetime.now().isoformat()}}
        }
        
        # YouTube 전용 속성 추가
        if data.get('type') == 'youtube' and 'metadata' in data:
            metadata = data['metadata']
            properties.update({
                "Channel": {"select": {"name": metadata.get('channel', '')}},
                "Duration": {"number": metadata.get('duration', 0)},
                "Views": {"number": metadata.get('views', 0)},
                "Likes": {"number": metadata.get('likes', 0)}
            })
        
        return properties
    
    def _create_blocks(self, data: Dict[str, Any]) -> List[Dict]:
        """페이지 블록 생성"""
        blocks = []
        summary = data.get('summary', {})
        
        # 한 줄 요약
        if one_sentence := summary.get('one_sentence_summary'):
            blocks.extend([
                self.block_builder.heading("한 줄 요약"),
                self.block_builder.text(one_sentence),
                self.block_builder.divider()
            ])
        
        # 전체 요약
        if full_summary := summary.get('full_summary'):
            blocks.extend([
                self.block_builder.heading("전체 요약"),
                *[self.block_builder.bullet(text) for text in full_summary],
                self.block_builder.divider()
            ])
        
        # 섹션별 요약
        if sections := summary.get('sections'):
            blocks.append(self.block_builder.heading("섹션별 요약"))
            for section in sections:
                blocks.extend([
                    self.block_builder.heading(section['title'], 3),
                    *[self.block_builder.bullet(text) for text in section['summary']]
                ])
        
        return blocks
    
    def _backup_failed_data(self, data: Dict[str, Any]):
        """실패한 데이터 백업"""
        try:
            backup_dir = self.config.paths['cache'] / 'notion_backup'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f'backup_{timestamp}.json'
            
            import json
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"백업 파일 저장됨: {backup_path}")
            
        except Exception as e:
            print(f"백업 저장 실패: {e}") 