from typing import Dict, List, Any

class Schemas:
    """스키마 정의 클래스"""
    
    def __init__(self):
        self.prompts = self.Prompts()
        self.descriptions = self._load_descriptions()
        self.json_function_section = self._load_json_functions()
    
    class Prompts:
        """프롬프트 템플릿"""
        MAP_TEMPLATE = """다음 텍스트를 분석하여 주요 섹션별로 요약해주세요:

{text}

다음 형식으로 출력해주세요:
{format_instructions}"""

        REDUCE_TEMPLATE = """다음 섹션들을 하나의 통합된 요약으로 만들어주세요:

{sections}

다음 형식으로 출력해주세요:
{format_instructions}"""
    
    def _load_descriptions(self) -> Dict[str, str]:
        """출력 형식 설명"""
        return {
            'section': """각 섹션은 다음 형식을 따라야 합니다:
- title: 섹션의 제목
- content: 섹션의 주요 내용 요약
- key_points: 핵심 포인트들 (리스트)""",
            
            'summary': """최종 요약은 다음 형식을 따라야 합니다:
- title: 전체 내용의 제목
- overview: 전체 내용의 개요
- key_points: 주요 핵심 포인트들 (리스트)
- sections: 각 섹션별 요약 (리스트)"""
        }
    
    def _load_json_functions(self) -> List[Dict[str, Any]]:
        """JSON 함수 스키마 정의"""
        return [{
            "name": "create_summary",
            "description": "텍스트를 섹션별로 요약",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "섹션의 제목"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "섹션의 주요 내용 요약"
                                },
                                "key_points": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "핵심 포인트들"
                                }
                            },
                            "required": ["title", "content", "key_points"]
                        }
                    }
                },
                "required": ["sections"]
            }
        }] 