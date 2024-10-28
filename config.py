import os
from dotenv import load_dotenv
import copy
import argparse
load_dotenv()

class Config:
    def __init__(self):
        # 기존 설정들...
        # CLI 인자 파싱
        parser = argparse.ArgumentParser(description='Text Summarization Configuration')
        
        # 키워드 옵션
        parser.add_argument('--keywords', action='store_true', default=True,
                          help='Include keywords in the summary')
        parser.add_argument('--no-keywords', action='store_false', dest='keywords',
                          help='Exclude keywords from the summary')
        
        # 전체 텍스트 옵션
        parser.add_argument('--full-text', action='store_true', default=False,
                          help='Include full text in the summary')
        parser.add_argument('--no-full-text', action='store_false', dest='full_text',
                          help='Exclude full text from the summary')
        
        # 챕터 옵션
        parser.add_argument('--chapters', action='store_true', default=True,
                          help='Enable chapter-based summarization')
        parser.add_argument('--no-chapters', action='store_false', dest='chapters',
                          help='Disable chapter-based summarization')
        
        # 키워드 하이라이팅 옵션
        parser.add_argument('--highlight-keywords', action='store_true', default=True,
                          help='Highlight keywords in the text')
        parser.add_argument('--no-highlight-keywords', action='store_false', 
                          dest='highlight_keywords',
                          help='Disable keyword highlighting')
        # CLI 인자 파싱 및 설정 적용
        args = parser.parse_args()
        
        # 기능 설정 초기화
        self.INCLUDE_KEYWORDS = args.keywords
        self.INCLUDE_FULL_TEXT = args.full_text
        self.ENABLE_CHAPTERS = args.chapters
        self.HIGHLIGHT_KEYWORDS = args.highlight_keywords

        self.base_path =os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(self.base_path, 'src')
        self.save_path = os.path.join(self.base_path, 'save')
        self.result_path = os.path.join(self.base_path, 'result')

        os.makedirs(self.src_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        
        load_dotenv(os.path.join(self.src_path,'.env' ))
        
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

        self.GPT_MODEL =  'gpt-3.5-turbo'#'gpt-4o-mini'#'gpt-4o'#' #'o1-preview'#'gpt-4o' #'gpt-4o-mini'#,, 'gpt-3.5-turbo' 
        self.MAX_TOKEN = 4096
        self.max_token_response = 500
        self.min_token_response = 100
        self.TEMPERATURE = 0.2
        self.system_content = """You are a helpful assistant that creates summaries in JSON format. Follow these rules strictly: 
            Use clear language.
            Avoid redundancy while keeping key details.
            Ensure each section summary is self-contained and informative.
            """    
        # 설정 로그 출력
        print("\n=== Configuration ===")
        print(f"Keywords Enabled: {self.INCLUDE_KEYWORDS}")
        print(f"Full Text Enabled: {self.INCLUDE_FULL_TEXT}")
        print(f"Chapters Enabled: {self.ENABLE_CHAPTERS}")
        print(f"Keyword Highlighting: {self.HIGHLIGHT_KEYWORDS}")
        print("==================\n")
    
    # 공통 스키마 정의
    def create_schema():
        description_summary = "Provide summaries focusing on key details."
        description_section = "Divide the text into meaning-based section, considering the context. Each section should capture the essence of the contents." 
        description_bullet = "Approximately 3 bullet points summarizing the text."
        
        description_full_summary = "A concise and comprehensive summary of the entire text."
        description_one_sentence = "Response in a single sentence, capturing the essence of the main idea."

        title_section = "A descriptive title reflecting its main idea."
        # 기본 스키마 정의
        base_array_schema = {
            "type": "array",
            "items": {"type": "string"}
        }

        # 섹션 스키마
        section_schema = {
            "sections": {
                "type": "array",
                "description": description_section,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string", 
                            "description": title_section,
                            "maxLength":10
                          },  # 제목 최대 길이 제한},
                        "summary": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "maxLength": 30  # 각 요약 문장의 최대 길이 제한
                            },
                            "description": description_bullet,
                            "maxItems": 3  # 각 섹션당 최대 bullet point 수 제한
                        }
                    },
                    "required": ["title", "summary"]
                },
                "minItems": 2,
                "maxItems": 3  # 최대 섹션 수 제한
            }
        }

        # 키워드 스키마
        keyword_schema = {
            "keywords": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Key concepts extracted from the text"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of occurrences in text"
                        }
                    },
                    "required": ["term", "count"]
                },
                "maxItems": 5
            }
        }

        # 요약 스키마
        full_summary_schema = {
            "full_summary": {
                "type": "array",
                "items": {"type": "string"},
                "description": description_full_summary + description_bullet
            }
        }

        one_sentence_summary_schema = {
            "one_sentence_summary": {
                "type": "string",
                "description": description_one_sentence,
                "maxLength": 30
            }
        }

        def create_function(properties_dict, required_fields):
            return [{
                "name": "create_summary",
                "description": description_summary,
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": required_fields
                }
            }]

        # 각 함수 스키마 생성
        
        section_properties = {}
        section_properties.update(section_schema)
        section_properties.update(keyword_schema)
        json_function_section = create_function(section_properties, ["sections", "keywords"])

        # final 함수 스키마 생성
        final_properties = {}
        final_properties.update(full_summary_schema)
        final_properties.update(one_sentence_summary_schema)
        
        json_function_final = create_function(final_properties, 
                                                ["full_summary", "one_sentence_summary"])

        # full 함수 스키마 생성
        full_properties = {}
        full_properties.update(section_schema)
        full_properties.update(full_summary_schema)
        full_properties.update(one_sentence_summary_schema)
        #full_properties.update(keyword_schema)
        json_function_full = create_function(full_properties,
                                               ["sections", "full_summary", "one_sentence_summary"])#, "keywords"])

        return json_function_section, json_function_final, json_function_full

    # 스키마 생성 및 할당
    json_function_section, json_function_final, json_function_full = create_schema()
    