from typing import Dict, List, Any, Optional
import json
from openai import OpenAI
from summarizer.config.schemas import SummarySchemas, Summary
from summarizer.processors.text import ContentProcessor

class Summarizer:
    """텍스트 요약 처리기"""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.api_keys['openai'])
        self.schemas = SummarySchemas()
        self.processor = ContentProcessor()
    
    def summarize(self, text: str, title: str = "") -> Optional[Dict[str, Any]]:
        """
        텍스트 요약 생성
        
        Args:
            text: 요약할 텍스트
            title: 문서 제목 (선택)
        """
        try:
            # 텍스트 청크 분할
            chunks = self.processor.chunk(text) # 깨짐
            if not chunks:
                return None
            
            # 섹션별 요약 생성
            sections = self._generate_section_summaries(chunks)
            if not sections:
                return None
            
            # 전체 요약 생성
            full_summary = self._generate_full_summary(sections, title)
            if not full_summary:
                return None
            
            # 요약 결과 검증 및 반환
            summary = Summary(
                sections=sections['sections'],
                full_summary=full_summary['full_summary'],
                one_sentence_summary=full_summary['one_sentence_summary'],
                keywords=full_summary['keywords']
            )
            
            return summary.model_dump()
            
        except Exception as e:
            print(f"요약 생성 실패: {e}")
            return None
    
    def _generate_section_summaries(self, chunks: List[str]) -> Optional[Dict]:
        """섹션별 요약 생성"""
        try:
            sections = []
            
            for chunk in chunks:
                response = self.client.chat.completions.create(
                    model=self.config.model['name'],
                    messages=[{
                        "role": "system",
                        "content": "You are a precise summarizer that creates detailed section summaries."
                    }, {
                        "role": "user",
                        "content": self.schemas.prompts.MAP_TEMPLATE.format(
                            text=chunk,
                            format_instructions=self.schemas.descriptions['section']
                        )
                    }],
                    functions=self.schemas.json_function_section,
                    function_call={"name": "create_summary"},
                    temperature=self.config.model['temperature']
                )
           
                result = json.loads(response.choices[0].message.function_call.arguments)
                
                # 섹션 제목 길이 검증 및 수정
                for section in result['sections']:
                    if len(section['title']) > 30:
                        section['title'] = section['title'][:27] + "..."
                
                sections.extend(result['sections'])
            
            return {'sections': sections}
            
        except Exception as e:
            print(f"섹션 요약 생성 실패: {e}")
            return None
    
    def _generate_full_summary(self, sections: Dict, title: str = "") -> Optional[Dict]:
        """전체 요약 생성"""
        try:
            # 섹션 요약 텍스트 준비
            section_texts = []
            if title:
                section_texts.append(f"Title: {title}")
            
            for section in sections['sections']:
                section_texts.append(f"Section: {section['title']}")
                section_texts.extend(f"- {point}" for point in section['summary'])
            
            summary_text = "\n".join(section_texts)
            print('section summary', summary_text)
            
            # 전체 요약 생성
            response = self.client.chat.completions.create(
                model=self.config.model['name'],
                messages=[{
                    "role": "system",
                    "content": "You are a precise summarizer that creates concise overall summaries."
                }, {
                    "role": "user",
                    "content": self.schemas.prompts.COMBINE_TEMPLATE.format(
                        text=summary_text,
                        format_instructions=self.schemas.descriptions['full']
                    )
                }],
                functions=self.schemas.json_function_final,
                function_call={"name": "create_summary"},
                temperature=self.config.model['temperature']
            )
            
            result = json.loads(response.choices[0].message.function_call.arguments)
            print('summary', result)
            
            # one_sentence_summary 길이 검증 및 수정
            if len(result.get('one_sentence_summary', '')) > 50:
                result['one_sentence_summary'] = result['one_sentence_summary'][:47] + "..."
            
            return result
            
        except Exception as e:
            print(f"전체 요약 생성 실패: {e}")
            return None
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[Dict[str, Any]]:
        """키워드 추출"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model['name'],
                messages=[{
                    "role": "system",
                    "content": "Extract the most important keywords from the text."
                }, {
                    "role": "user",
                    "content": f"Extract up to {max_keywords} important keywords from:\n\n{text}"
                }],
                temperature=0.3
            )
            
            keywords = response.choices[0].message.content.split(',')
            return [{'term': k.strip(), 'count': 1} for k in keywords[:max_keywords]]
            
        except Exception as e:
            print(f"키워드 추출 실패: {e}")
            return [] 