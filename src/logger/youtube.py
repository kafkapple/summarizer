from src.logger.base import NotionBase
from omegaconf import DictConfig
from datetime import datetime
from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)

class YouTube2Notion(NotionBase):
    def __init__(self, cfg: DictConfig, verbose=False, quiet=False):
        super().__init__(cfg, verbose, quiet)
        # Use the main database ID from base class or override if needed
        # self.database_id = cfg.notion.youtube_ch_id # youtube_ch_id is likely for channels, not videos
        if not self.database_id: # self.database_id is set in NotionBase init
            logger.error("Notion database ID (notion.database_id) is not configured.")
            raise ValueError("Notion database ID is not configured.")

    def save_to_notion(self, properties: Dict[str, Any], children: List[Dict[str, Any]] = None) -> bool:
        # This method now correctly uses self.client due to previous base.py fix
        # No changes needed here unless specific error handling is desired
        return super().save_to_notion(properties, children)

    def save_to_notion_youtube(self, data: dict):
        """YouTube 데이터를 Notion에 저장 (매핑 기반으로 수정됨)"""
        try:
            # 1. Prepare data for the generic property creation method
            # Merge video_info (which is 'data' here) and summary_result
            combined_data = data.copy() # Start with video_info
            summary_result = combined_data.pop('summary_result', {}) # Extract and remove summary_result

            # Add summary details under a 'summary' key as expected by base._create_notion_properties
            if isinstance(summary_result, dict):
                 combined_data['summary'] = summary_result
                 # Base method _create_notion_properties handles extracting details like 
                 # full_summary, keywords etc. from this 'summary' dict based on mapping
            else:
                 combined_data['summary'] = {} # Ensure summary key exists

            # Ensure base keys expected by the mapping exist directly in combined_data
            # The input 'data' dict (originally video_info) usually has these keys.
            # Add playlist explicitly if mapping requires it
            if 'Playlist' in self.mapping: # Check if Playlist is in the mapping
                 combined_data['playlist'] = data.get('playlist', '')
            
            # Add LLM info (already handled by base method, but good practice)
            # combined_data['gpt_model'] = self.cfg.llm.get('model')
            # combined_data['output_language'] = self.cfg.output.get('language')

            # 2. Notion Properties 생성 (매핑 기반 - Base 클래스 메서드 호출)
            properties = self._create_notion_properties(combined_data) # Use base class method

            if not properties:
                 logger.error(f"Failed to create Notion properties for {data.get('title', 'N/A')}. Check mapping and data.")
                 return # Stop if no properties could be created

            # 3. Notion Children 생성 (organize_summary 호출)
            children = self.organize_summary(combined_data) 

            # 4. Notion 저장 (API 호출 - Base 클래스 메서드 사용)
            success = self.save_to_notion(properties=properties, children=children)
            if not success:
                 logger.warning(f"Failed to save YouTube data to Notion for: {data.get('title', 'N/A')}")

        except Exception as e:
            logger.error(f"Error processing YouTube data for Notion: {data.get('title', 'N/A')} - {e}", exc_info=True) # Add traceback

    # save_to_notion_youtube_ch 함수는 채널 정보 저장이므로 별도 매핑/처리 필요
    # def save_to_notion_youtube_ch(self, data):
    #     # Needs its own mapping or specific property creation logic for channel data
    #     pass

