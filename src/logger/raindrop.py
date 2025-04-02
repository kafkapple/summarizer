from src.logger.base import NotionBase
from omegaconf import DictConfig

class Raindrop2Notion(NotionBase):
    def __init__(self, cfg: DictConfig, verbose=False, quiet=False):
        super().__init__(cfg, verbose, quiet)
        self.database_id = cfg.notion.raindrop_id
        if not self.database_id:
            raise ValueError("Notion database ID for Raindrop is not configured (notion.raindrop_id)")
        
    def save_to_notion_raindrop(self, data: dict):
        """Raindrop 데이터를 Notion에 저장"""
        try:
            # 1. Notion Properties 생성 (매핑 기반)
            properties = self._create_notion_properties(data)

            # 2. Notion Children 생성
            children = self.organize_summary(data)

            # 3. Notion 저장 (save_to_notion 호출 시 인자 수정)
            self.save_to_notion(properties=properties, children=children)

        except Exception as e:
            print(f"Error processing Raindrop data for Notion: {data.get('title', 'N/A')} - {e}")
