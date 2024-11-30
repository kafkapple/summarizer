from notion_client import Client
import requests
from datetime import datetime
import re
from typing import List, Dict, Optional
import time
import random   

class NotionBase:
    def __init__(self, config, verbose=False, quiet=False):
        self.config = config
        self.client = Client(auth=self.config.NOTION_TOKEN)
        self.database_id = self.config.NOTION_DATABASE_ID
        self.keywords = []
        self.verbose = verbose
        self.quiet = quiet

    def change_id(self, id):
        self.database_id = id

    def save_to_notion(self, data, properties, children=None):
        try:
            self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children or []
            )
            print(f"Summary for '{data['title']}' has been saved to Notion.\n")
        except Exception as e:
            print(f"Error saving to Notion: {e}")
    # organize_summary 메소드 수정
    def create_text_block(self, content: str, block_type: str = "paragraph", keywords: List[str] = None) -> Dict:
        """키워드 강조가 포함된 텍스트 블록을 생성합니다."""
        if keywords:
            rich_text = self.highlight_keywords(content, keywords)
        else:
            rich_text = [{"type": "text", "text": {"content": content}}]
        
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": rich_text
            }
        }

    def create_bulleted_list_item(self, content: str, keywords: List[str] = None) -> Dict:
        """키워드 강조가 포함된 글머리 기호 항목을 생성합니다."""
        if keywords:
            rich_text = self.highlight_keywords(content, keywords)
        else:
            rich_text = [{"type": "text", "text": {"content": content}}]
        
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": rich_text
            }
        }

    def organize_summary(self, data, heading='Section summary', contents='summary'):
        children = [
            {"object": "block", "type": "table_of_contents", "table_of_contents": {}},
        ]
        
        try:
            if 'thumbnail' in data:
                children.append({
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "external",
                        "external": {"url": data['thumbnail']}
                    }
                })

            summary = data.get('summary', {})
            keywords = []
            if self.config.INCLUDE_KEYWORDS and self.config.HIGHLIGHT_KEYWORDS:
                keywords = summary.get('keywords', [])  # keywords_original 사용

            if isinstance(summary, dict):
                chapters = summary.get('chapters', []) if self.config.ENABLE_CHAPTERS else []
                
                if 'sections' in summary:
                    children.append(self.create_text_block("Detailed Section Summaries", "heading_1"))
                    
                    # 챕터 정보가 있는 경우
                    if chapters:
                        current_chapter_idx = 0
                        current_chapter = chapters[current_chapter_idx]
                        
                        for i, segment in enumerate(summary['sections']):
                            # 새로운 챕터의 시작인지 확인
                            while (current_chapter_idx < len(chapters) and 
                                   i >= current_chapter['section_indices']['end']):
                                current_chapter_idx += 1
                                if current_chapter_idx < len(chapters):
                                    current_chapter = chapters[current_chapter_idx]
                            
                            # 현재 섹션이 새로운 챕터의 시작이면 챕터 제목 추가
                            if current_chapter_idx < len(chapters) and \
                               i == current_chapter['section_indices']['start']:
                                children.append(self.create_text_block(
                                    f"{current_chapter_idx + 1}. {current_chapter['chapter_title']}",
                                    "heading_1"
                                ))
                            
                            # 섹션 추가
                            if isinstance(segment, dict):
                                # 섹션 제목
                                children.append(self.create_text_block(
                                    f'{i+1}. {segment.get("title", "")}',
                                    "heading_2",
                                    keywords if keywords else None
                                ))
                                
                                # 섹션 내용
                                summary_content = segment.get('summary', [])
                                if isinstance(summary_content, list):
                                    for item in summary_content:
                                        children.append(self.create_bulleted_list_item(
                                            item,
                                            keywords if keywords else None
                                        ))
                                
                                children.append(self.create_text_block(""))  # 빈 줄
                    else:
                        # 챕터가 없는 경우의 처리
                        for i, segment in enumerate(summary['sections']):
                            if isinstance(segment, dict):
                                children.append(self.create_text_block(
                                    f'{i+1}. {segment.get("title", "")}',
                                    "heading_2",
                                    keywords if keywords else None
                                ))
                                
                                summary_content = segment.get('summary', [])
                                if isinstance(summary_content, list):
                                    for item in summary_content:
                                        children.append(self.create_bulleted_list_item(
                                            item,
                                            keywords if keywords else None
                                        ))
                                
                                children.append(self.create_text_block(""))  # 빈 줄

        except Exception as e:
            print(f'Error organizing summary: {e}')
        
        return children

    def common_properties(self, data):
        playlist = data.get('playlist', '')
        print(playlist)
        properties = {
            "Title": {"title": [{"text": {"content": data.get('title', '')}}]},
            "URL": {"url": data.get('url', '')},
            "GPT Model": {"select": {"name": data.get('model', '')}},
            "Keywords": {"multi_select": [{"name": keyword} for keyword in data.get('keywords', [])]},
           # "Channel": {"rich_text": [{"text": {"content": data.get('channel', '')}}]},
        }
        # Channel, Like Count, Comment Count, One Sentence Summary 
        # Published Date 처리
        #published_date = data.get('published_date')
        
        summary = data.get('summary', {})
        if isinstance(summary, dict):
            full_summary = summary.get('full_summary', '')
            if isinstance(full_summary, list):
                full_summary = ' '.join(full_summary)
            elif not isinstance(full_summary, str):
                full_summary = str(full_summary)
            
            properties["Summary"] = {"rich_text": [{"text": {"content": full_summary[:2000]}}]}
            
            one_sentence_summary = summary.get('one_sentence_summary', '')
            
            if not isinstance(one_sentence_summary, str):
                one_sentence_summary = str(one_sentence_summary)
            properties["One Sentence Summary"] = {"rich_text": [{"text": {"content": one_sentence_summary[:2000]}}]}
        
        return properties

    def highlight_keywords(self, text: str, keywords: List[str]) -> List[Dict]:
        """텍스트에서 키워드를 찾아 강조 표시를 추가합니다."""
        if not text or not keywords:
            return [{"type": "text", "text": {"content": text or ""}}]

        result = []
        current_pos = 0
        text_lower = text.lower()
        
        # 키워드를 길이 순으로 정렬 (긴 키워드부터 처리)
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        while current_pos < len(text):
            found_keyword = False
            
            for keyword in sorted_keywords:
                keyword_lower = keyword.lower()
                pos = text_lower.find(keyword_lower, current_pos)
                
                if pos == current_pos:
                    # 키워드 이전 텍스트 추가
                    if pos > 0:
                        result.append({
                            "type": "text",
                            "text": {"content": text[current_pos:pos]}
                        })
                    
                    # 키워드 추가 (강조 표시)
                    result.append({
                        "type": "text",
                        "text": {
                            "content": text[pos:pos + len(keyword)],
                        },
                        "annotations": {
                            "bold": True,
                            "color": "yellow_background"
                        }
                    })
                    
                    current_pos = pos + len(keyword)
                    found_keyword = True
                    break
            
            if not found_keyword:
                # 키워드를 찾지 못한 경우, 다음 문자를 일반 텍스트로 추가
                next_pos = len(text)
                for keyword in sorted_keywords:
                    keyword_pos = text_lower.find(keyword_lower, current_pos + 1)
                    if keyword_pos != -1 and keyword_pos < next_pos:
                        next_pos = keyword_pos
                
                result.append({
                    "type": "text",
                    "text": {"content": text[current_pos:next_pos]}
                })
                current_pos = next_pos
        
        return result

    
class Pocket2Notion(NotionBase):
    def __init__(self, config, verbose=False, quiet=False):
        """
        Pocket2Notion 초기화
        Args:
            config: 설정 객체 (NOTION_TOKEN, NOTION_DATABASE_ID 등 포함)
            verbose: 상세 로깅 여부
            quiet: 로깅 최소화 여부
        """
        super().__init__(config, verbose, quiet)
        self.pocket_client = None
    
        
    def initialize(self, pocket_client):
        """
        Pocket 클라이언트와 Summarizer 설정
        Args:
            pocket_client: PocketClient 인스턴스
            summarizer: Summarizer 인스턴스
        """
        self.pocket_client = pocket_client
        
    def save_to_notion_pocket(self, data):
        """
        Pocket 아이템을 Notion에 저장
        Args:
            data: 처리된 Pocket 아이템 데이터
        """
        try:
            children = self.organize_summary(data)
            properties = self.common_properties(data)
            
            # Pocket 특화 properties 추가
            properties.update({
                "Excerpt": {"rich_text": [{"text": {"content": data.get('excerpt', '')[:2000]}}]},
                "Word count": {"number": int(data.get('word_count', 0))},
                "Date": {"date": {"start": data.get('time_added', '')}},
                "Language": {"select": {"name": data.get('lang', 'unknown')}},
                "Favorite": {"select": {"name": str(data.get('favorite', False))}},
                "Status": {"select": {"name": data.get('status', 'unread')}},
                "Video": {"select": {"name": str(data.get('has_video', False))}}
            })
            
            if not self.quiet:
                print(f"Saving to Notion: {data.get('title', 'Untitled')}")
                
            self.save_to_notion(data, properties, children)
            
        except Exception as e:
            if not self.quiet:
                print(f"Error saving to Notion: {str(e)}")
            raise
            

class Raindrop2Notion(NotionBase):
    def save_to_notion_raindrop(self, data):
        properties = self.common_properties(data)
        properties.update({
            "Collection": {"select": {"name": data['collection']}}
        })
        children = self.organize_summary(data)
        self.save_to_notion(data, properties, children)

class YouTube2Notion(NotionBase):
    def __init__(self, config, verbose=False, quiet=False):
        super().__init__(config, verbose, quiet)
        
    def save_to_notion_youtube_ch(self, data):
        properties = self.common_properties(data)
        properties.update({
            "Subscribers": {"number": int(data['Subscribers'])},
            "View Count": {"number": int(data['View Count'])},
            "Video Count": {"number": int(data['Video Count'])},
            "Country": {"select": {"name": data['Country']}},
            "Category": {"multi_select": [{"name": topic} for topic in data['Category']]},
            "Publish Date": {"date": {"start": data['Published At']}},
            "Description": {"rich_text": [{"text": {"content": data['Description']}}]},
            

        })
        children = self.organize_summary(data)
        self.save_to_notion(data, properties, children)

    def save_to_notion_youtube(self, data):
        children = self.organize_summary(data)
        properties = self.common_properties(data)
        summary = data.get('summary', {})
        try:
            keywords = [topic for topic in summary.get('keywords_original', '')]
        except:
            keywords = []
        properties.update({
            "Channel": {"select": {"name": data.get('channel_title', '')}},  # "Channel Name"을 "Channel"로 변경
            "Category": {"select": {"name": data.get('category', '')}},
            "Tags": {"multi_select": [{"name": topic} for topic in data.get('tags', [])]},
            "Output Language": {"select": {"name": data.get('output_language', '')}},
            "Published Date": {"date": {"start": data.get('publish_date', '')}},  # "Publish Date"를 "Published Date"로 변경
            "Duration": {"rich_text": [{"text": {"content": str(data.get('duration', ''))}}]}, #"Duration": {"number": data.get('duration', 0)},
            "View Count": {"number": int(data.get('view_count', 0))},
            "Like Count": {"number": int(data.get('like_count', 0))},
            "Comment Count": {"number": int(data.get('comment_count', 0))},
          
            "Playlist": {"select": {"name": data.get('playlist', '')}}
        })
        
        self.save_to_notion(data, properties, children)



