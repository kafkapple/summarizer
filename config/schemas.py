from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Section(BaseModel):
    """섹션 정보"""
    title: str = Field(description="섹션 제목", max_length=30)
    summary: List[str] = Field(
        description="섹션 상세 요약",
        min_items=1,
        max_items=4
    )

class Keyword(BaseModel):
    """키워드 정보"""
    term: str = Field(description="키워드 용어", max_length=20)
    count: Optional[int] = Field(description="출현 빈도", default=1)

class Summary(BaseModel):
    """요약 정보"""
    sections: List[Section] = Field(description="주요 섹션들")
    full_summary: List[str] = Field(
        description="전체 요약",
        min_items=2,
        max_items=3
    )
    one_sentence_summary: str = Field(
        description="한 문장 요약",
        min_length=30,
        max_length=50
    )
    keywords: List[Keyword] = Field(
        description="주요 키워드",
        max_items=5
    )
    # 응답은 반드시 다음 JSON 형식을 따라야 합니다:
    # {
    #     "full_summary": ["전체 요약 문장 1", "전체 요약 문장 2"],
    #     "one_sentence_summary": "한 문장으로 된 전체 요약"
    # }
class PromptTemplates:
    """Prompt templates"""
    
    MAP_TEMPLATE = """
    Analyze the following text and create section-by-section summaries.

    Text:
    {text}

    Format instructions:
    {format_instructions}
    """

    COMBINE_TEMPLATE = """
    Combine these summaries into a concise overall summary.

    Summaries:
    {text}

    Format instructions:
    {format_instructions}
    """

class SummarySchemas:
    """요약 스키마 관리"""
    
    def __init__(self):
        self.prompts = PromptTemplates()
        self._setup_schemas()
    
    def _setup_schemas(self):
        """스키마 설정"""
        self.descriptions = {
            'section': self._get_section_description(),
            'full': self._get_full_description(),
            'bullet': self._get_bullet_description()
        }
        
        schemas = self._create_schemas()
        
        # 섹션 요약용 스키마
        section_properties = {**schemas['section']}
        self.json_function_section = self._create_function(section_properties, ["sections"])
        
        # 최종 요약용 스키마
        final_properties = {
            **{"full_summary": schemas['final']['full_summary']},
            **{"one_sentence_summary": schemas['final']['one_sentence_summary']},
            **{"keywords": schemas['final']['keywords']}
        }
        self.json_function_final = self._create_function(final_properties, ["full_summary", "one_sentence_summary"])
        
        # 전체 요약용 스키마
        full_properties = {**schemas['section'], **schemas['final']}
        self.json_function_full = self._create_function(full_properties, ["sections", "full_summary", "one_sentence_summary"])
    
    @staticmethod
    def _get_section_description() -> str:
        return """Divide content into meaningful sections. Provide detailed section summaries following these rules:
        1. Each section should have 2-4 detailed bullet points
        2. Each bullet point should be 20-30 words long
        3. Focus on specific details and examples
        4. Avoid generic statements
        5. Include relevant numbers and facts when available"""
    
    @staticmethod
    def _get_full_description() -> str:
        return """Create a concise overall summary following these rules:
        1. Maximum 3 bullet points for the entire text
        2. Each bullet point should be 15-20 words
        3. Focus on high-level key points only
        4. Avoid detailed examples
        5. Maintain broad perspective"""
    
    @staticmethod
    def _get_bullet_description() -> str:
        return """Create detailed bullet points that:
        1. Are 20-30 words each
        2. Include specific examples or data
        3. Focus on distinct aspects"""
    
    def _create_schemas(self) -> Dict:
        """스키마 생성"""
        return {
            'section': {
                "sections": {
                    "type": "array",
                    "description": self.descriptions['section'],
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
                                    "description": self.descriptions['bullet'],
                                    "minLength": 50,
                                    "maxLength": 100
                                },
                                "minItems": 1,
                                "maxItems": 4
                            }
                        },
                        "required": ["title", "summary"]
                    },
                    "minItems": 1
                }
            },
            'final': {
                "full_summary": {
                    "type": "array",
                    "description": self.descriptions['full'],
                    "items": {
                        "type": "string",
                        "minLength": 30,
                        "maxLength": 60
                    },
                    "minItems": 2,
                    "maxItems": 3
                },
                "one_sentence_summary": {
                    "type": "string",
                    "description": "Single sentence capturing the main idea (maximum 50 characters)",
                    "maxLength": 50
                },
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
        }
    
    @staticmethod
    def _create_function(properties_dict: Dict, required_fields: List[str]) -> List[Dict]:
        """함수 스키마 생성"""
        return [{
            "name": "create_summary",
            "description": "Generate structured summary with specified detail levels",
            "parameters": {
                "type": "object",
                "properties": properties_dict,
                "required": required_fields
            }
        }]