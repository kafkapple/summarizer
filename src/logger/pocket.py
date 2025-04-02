from src.logger.base import NotionBase
from omegaconf import DictConfig

class Pocket2Notion(NotionBase):
    def __init__(self, cfg: DictConfig, verbose=False, quiet=False):
        super().__init__(cfg, verbose, quiet)
        self.database_id = cfg.notion.pocket_id
        if not self.database_id:
            raise ValueError("Notion database ID for Pocket is not configured (notion.pocket_id)")
        self.pocket_client = None
    
    def initialize(self, pocket_client):
        """Pocket 클라이언트 초기화"""
        self.pocket_client = pocket_client
        
    def save_to_notion_pocket(self, data: dict):
        """Pocket 데이터를 Notion에 저장"""
        try:
            # 1. Notion Properties 생성 (매핑 기반)
            properties = self._create_notion_properties(data)

            # 2. Notion Children 생성
            children = self.organize_summary(data)

            # 3. Notion 저장 (save_to_notion 호출 시 인자 수정)
            self.save_to_notion(properties=properties, children=children)
            
            if not self.quiet:
                print(f"Saving to Notion: {data.get('title', 'Untitled')}")
                
        except Exception as e:
            if not self.quiet:
                print(f"Error saving to Notion: {str(e)}")
            raise