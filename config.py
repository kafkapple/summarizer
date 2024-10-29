import os
from dotenv import load_dotenv
import copy
import argparse
import logging
# httpx 로거의 레벨을 WARNING으로 설정하여 INFO 레벨 로그를 숨깁니다.
logging.getLogger("httpx").setLevel(logging.WARNING)
# 또는 전체 로깅 레벨을 설정할 수 있습니다.
#logging.basicConfig(level=logging.WARNING)

load_dotenv()

class Config:
    def __init__(self):
        # 기본 설정값
        self.INCLUDE_KEYWORDS = True
        self.INCLUDE_FULL_TEXT = False
        self.ENABLE_CHAPTERS = True
        
        # 환경 변수 및 기타 설정...
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(self.base_path, 'src')
        self.save_path = os.path.join(self.base_path, 'save')
        self.result_path = os.path.join(self.base_path, 'result')

        os.makedirs(self.src_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        
        load_dotenv(os.path.join(self.src_path, '.env'))
        
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.NOTION_TOKEN = os.getenv("NOTION_TOKEN")

        self.DIFFBOT_API_TOKEN = os.getenv("DIFFBOT_API_TOKEN")
        
        self.NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
        self.NOTION_DB_YOUTUBE_CH_ID = os.getenv("NOTION_DB_YOUTUBE_CH_ID")
        self.NOTION_DB_RAINDROP_ID = os.getenv("NOTION_DB_RAINDROP_ID")
        self.NOTION_DB_POCKET_ID = os.getenv("NOTION_DB_POCKET_ID")

        self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
        self.RAINDROP_TOKEN = os.getenv("RAINDROP_TOKEN")

        self.POCKET_CONSUMER_KEY = os.getenv("POCKET_CONSUMER_KEY")
        self.POCKET_ACCESS_TOKEN = os.getenv("POCKET_ACCESS_TOKEN")

        self.OUTPUT_LANGUAGE = 'ko'

        self.GPT_MODEL = 'gpt-3.5-turbo'
        self.MAX_TOKEN = 4096
        self.max_token_response = 600
        self.min_token_response = 150
        self.TEMPERATURE = 0.1
        self.system_content = """You are a logical summary assistant. Follow these rules:
        1. Please respond in JSON format only. Do not include comments or any additional text.
        2. Cover different aspects without overlap.
        3. Use concise language.
        4. Maintain consistent formatting.
        """
   
        # 설정 로그 출력
        print("\n=== Configuration ===")
        print(f"Keywords Enabled: {self.INCLUDE_KEYWORDS}")
        print(f"Full Text Enabled: {self.INCLUDE_FULL_TEXT}")
        print(f"Chapters Enabled: {self.ENABLE_CHAPTERS}")

        # 스키마 생성 및 할당
        self.json_function_section, self.json_function_final, self.json_function_full = self.create_schema()
    
    def update_runtime_settings(self, keywords=None, full_text=None, chapters=None):
        """실행 시 설정 업데이트"""
        if keywords is not None:
            self.INCLUDE_KEYWORDS = keywords
        if full_text is not None:
            self.INCLUDE_FULL_TEXT = full_text
        if chapters is not None:
            self.ENABLE_CHAPTERS = chapters
            
        # 설정 로그 출력
        print("\n=== Configuration ===")
        print(f"Keywords Enabled: {self.INCLUDE_KEYWORDS}")
        print(f"Full Text Enabled: {self.INCLUDE_FULL_TEXT}")
        print(f"Chapters Enabled: {self.ENABLE_CHAPTERS}")
        print("==================\n")

    # 공통 스키마 정의
    def create_schema(self):
        # Detailed descriptions for better control
        description_section = """Divide content into meaningful sections. Provide detailed section summaries following these rules:
        1. Each section should have 2-3 detailed bullet points
        2. Each bullet point should be 20-30 words long
        3. Focus on specific details and examples
        4. Avoid generic statements
        5. Include relevant numbers and facts when available"""

        description_full = """Create a concise overall summary following these rules:
        1. Maximum 3 bullet points for the entire text
        2. Each bullet point should be 15-20 words
        3. Focus on high-level key points only
        4. Avoid detailed examples
        5. Maintain broad perspective"""

        description_bullet = """Create detailed bullet points that:
        1. Are 20-30 words each
        2. Include specific examples or data
        3. Focus on distinct aspects"""


        # Schema for sections with length controls
        section_schema = {
            "sections": {
                "type": "array",
                "description": description_section,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Concise title (3-5 words)",
                            "maxLength": 30
                        },
                        "summary": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 50,  # Enforce minimum length for detail
                                "maxLength": 100  # Enforce maximum length for clarity
                            },
                            "description": description_bullet,
                            "minItems": 2,
                            "maxItems": 3
                        }
                    },
                    "required": ["title", "summary"]
                },
                "minItems": 1
            }
        }

        # Schema for full summary with stricter length controls
        full_summary_schema = {
            "full_summary": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 30,  # Shorter minimum for conciseness
                    "maxLength": 60   # Shorter maximum for brevity
                },
                "description": description_full,
                "minItems": 2,
                "maxItems": 3
            }
        }

        one_sentence_summary_schema = {
            "one_sentence_summary": {
                "type": "string",
                "description": "Single sentence capturing the main idea (15-20 words)",
                "minLength": 30,
                "maxLength": 50
            }
        }

        keyword_schema = {
            "keywords": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Key concept or term",
                            "maxLength": 20
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of occurrences"
                        }
                    },
                    "required": ["term", "count"]
                },
                "maxItems": 3
            }
        }

        def create_function(properties_dict, required_fields):
            return [{
                "name": "create_summary",
                "description": "Generate structured summary with specified detail levels",
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": required_fields
                }
            }]

        # Create function schemas with updated properties
        section_properties = {**section_schema, **keyword_schema}
        json_function_section = create_function(section_properties, ["sections"])

        final_properties = {**full_summary_schema, **one_sentence_summary_schema, **keyword_schema}
        json_function_final = create_function(final_properties, ["full_summary", "one_sentence_summary"])

        full_properties = {**section_schema, **full_summary_schema, **one_sentence_summary_schema, **keyword_schema}
        json_function_full = create_function(full_properties, ["sections", "full_summary", "one_sentence_summary"])

        return [json_function_section, json_function_final, json_function_full]
    