import  json
import openai
from openai import OpenAI
from langdetect import detect
from deep_translator import GoogleTranslator
from typing import List, Dict, Optional
import tiktoken
import re
from utils import Utils

class BaseSummarizer:
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        openai.api_key = config.OPENAI_API_KEY
        self.gpt_model = config.GPT_MODEL
        self.output_language = config.OUTPUT_LANGUAGE
        if self.output_language == 'ko':
            self.output_language_full = 'Korean'
        else:
            self.output_language_full = 'English'
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.json_function_full = config.json_function_full
        self.json_function_section = config.json_function_section
        self.json_function_final = config.json_function_final
        
        self.max_token = config.MAX_TOKEN
        self.system_content = config.system_content
        self.response_token = 700
        self.buffer_token = 100
        
        self.max_translate_length = 4500
        self.max_response_token = 600

        self.MAX_CHUNKS_PER_CHAPTER =  6  # 한 챕터당 최대 청크 수

        print( '\n'+'#'*7 +' Initialization of Summarizer ' +'#'*7+ f"\nGPT 모델 = {self.gpt_model}\n초기화: 대상 언어 = {self.output_language}")
        self.system_token = Utils.num_tokens_from_string(self.system_content, self.gpt_model)
        self.json_token = Utils.num_tokens_from_string(json.dumps(self.json_function_full), self.gpt_model)
        self.fixed_token = self.system_token + self.json_token

        self.prompt_token = self.max_token - self.system_token - self.json_token - self.response_token  
        if self.verbose:
            print(f'\nPutative Max/System/Json/Response:{self.max_token}/{self.system_token}/{self.json_token}/{self.response_token}\nPrompt:{self.prompt_token}')
    
    def summarize(self, text: str, title: str) -> Optional[Dict]:
        try:
            processed_text = Utils.preprocess_text(text)
            self.source_lang = detect(processed_text)
            print(f'Prompt Token per chunk: {self.prompt_token}')
            chunks = Utils.split_text_into_chunks(text=processed_text, max_length=self.prompt_token, by_token=True, gpt_model=self.gpt_model)
            n_chunks = len(chunks)
            print(f'청크 수: {n_chunks}')
            
            if n_chunks == 1:
                text = chunks[0]
                prompt = f'Title: {title}/ {text}'
                summary = self.get_chunk_summary(prompt, self.json_function_full)
                
            elif n_chunks <= self.MAX_CHUNKS_PER_CHAPTER:
                # 각 청크별 요약 생성 및 검증
                summary_chunks = []
                for chunk in chunks:
                    if chunk:
                        chunk_summary = self.get_chunk_summary(chunk, self.json_function_section)
                        if isinstance(chunk_summary, dict) and 'sections' in chunk_summary:
                            summary_chunks.append(chunk_summary)
                
                # 요약 병합
                concat, merged = self.merge_summaries(summary_chunks, chunks)
                
                # 최종 요약 생성
                prompt = f'Title: {title}. {concat}'
                final_summary = self.get_chunk_summary(prompt, self.json_function_final)
                
                # 병합된 섹션과 키워드 추가
                summary = {
                    'sections': merged['sections'],
                    'keywords': merged['keywords'],
                    'full_summary': final_summary.get('full_summary', []),
                    'one_sentence_summary': final_summary.get('one_sentence_summary', '')
                }
                
            elif n_chunks > self.MAX_CHUNKS_PER_CHAPTER: # chapter 단위로 요약
                return self.process_large_text(chunks, title, self.MAX_CHUNKS_PER_CHAPTER)
            
            if self.source_lang != self.output_language or self.source_lang == 'unknown':
                summary = self.translate_summary(summary)

            #summary['keywords_original'] = [item['term'] for item in summary.get('keywords', [])]
            final_summary = self.format_summary(summary, processed_text)
            if self.verbose:
                print(f'One sentence summary: {final_summary['one_sentence_summary']}\nFinal Summary: \n{final_summary['full_summary']}')

            return final_summary
            
        except Exception as e:
            print(f"요약 처리 중 오류 발생: {e}")
            return None

    def divide_chunks_into_chapters(self, chunks: List[str], max_chunks_per_chapter: int) -> List[List[str]]:
        """청크들을 가능한 균등한 크기의 챕터로 나눕니다."""
        n_chunks = len(chunks)
        
        if n_chunks <= max_chunks_per_chapter:
            return [chunks]
        
        # 적절한 챕터 수 계산
        n_chapters = (n_chunks + max_chunks_per_chapter - 1) // max_chunks_per_chapter
        # 더 균등한 분배를 위해 챕터 수 조정
        if n_chunks / n_chapters < max_chunks_per_chapter * 0.5:  # 챕터당 청크가 너무 적은 경우
            n_chapters = max(2, (n_chunks + 1) // 2)  # 최소 2개의 챕터
        
        # 챕터당 기본 청크 수 계산
        base_chunks_per_chapter = n_chunks // n_chapters
        remaining_chunks = n_chunks % n_chapters
        
        chapters = []
        start_idx = 0
        
        for i in range(n_chapters):
            # 남은 청크를 균등하게 분배
            chapter_size = base_chunks_per_chapter + (1 if i < remaining_chunks else 0)
            end_idx = start_idx + chapter_size
            chapters.append(chunks[start_idx:end_idx])
            start_idx = end_idx
        
        if self.verbose:
            print(f'총 {n_chunks}개 청크를 {n_chapters}개 챕터로 나눔')
            print(f'챕터별 청크 수: {[len(chapter) for chapter in chapters]}')
        
        return chapters

    def process_large_text(self, chunks: List[str], title: str, max_chunks_per_chapter: int) -> Optional[Dict]:
        try:
            chapters = self.divide_chunks_into_chapters(chunks, max_chunks_per_chapter)
            print(f'챕터 수: {len(chapters)}')
            
            chapter_info = []
            chapter_summaries = []
            total_section_count = 0  # 전체 섹션 수 추적
            
            for i, chapter_chunks in enumerate(chapters):
                chapter_num = i + 1
                print(f'챕터 {chapter_num} 처리 중... (청크 수: {len(chapter_chunks)})')
                
                # 각 챕터의 청크들을 요약
                chunk_summaries = [self.get_chunk_summary(chunk, 
                                 self.json_function_section) 
                                 for chunk in chapter_chunks if chunk]
                
                # 챕터 내용 병합
                chapter_concat, chapter_merged = self.merge_summaries(
                    chunk_summaries, chapter_chunks)
                chapter_sections = chapter_merged["sections"]
                chapter_keywords = chapter_merged["keywords"]
                chapter_title = f"Chapter {chapter_num}"
                 
                # 챕터 전체 요약 생성
                chapter_summary = self.get_chunk_summary(chapter_concat, 
                                                       self.json_function_section)
                chapter_summary['sections'] = chapter_sections
                chapter_summary['keywords'] = chapter_keywords
                
                # 섹션 인덱스 계산
                section_start = total_section_count
                total_section_count += len(chapter_sections)
                
                # 챕터 정보 저장
                chapter_info.append({
                    'chapter_number': chapter_num,
                    'chapter_title': chapter_title,
                    'section_indices': {
                        'start': section_start,
                        'end': total_section_count
                    },
                    'sections': chapter_sections,
                    'keywords': chapter_keywords,
                    #'summary': chapter_summary.get('full_summary', []),
                    #'one_sentence_summary': chapter_summary.get('one_sentence_summary', '')
                })
                
                chapter_summaries.append(chapter_summary)
                if self.verbose:
                    print(f'One sentence summary: {chapter_summary.get("one_sentence_summary", "")}')
                    print(f'Full summary: {chapter_summary.get("full_summary", "")}')
                    
            # 최종 요약 생성
            final_concat, final_merged = self.merge_summaries(
                chapter_summaries, [sum((list(chapter_chunks) for chapter_chunks in chapters), [])])
            final_sections = final_merged["sections"]
            final_keywords = final_merged["keywords"]

            prompt = f'Title: {title}/ {final_concat}'
            final_summary = self.get_chunk_summary(prompt,  self.json_function_final)
            final_summary['sections'] = final_sections
            final_summary['keywords'] = final_keywords
            
            if self.source_lang != self.output_language or self.source_lang == 'unknown':
                final_summary = self.translate_summary(final_summary)
                chapter_info = self.translate_chapter_info(chapter_info)
                
            final_summary['keywords_original'] = [item['term'] for item in final_keywords]
            final_summary['chapters'] = chapter_info
            
            return self.format_summary(final_summary, ' '.join(chunks))
            
        except Exception as e:
            print(f"대용량 텍스트 처리 중 오류 발생: {e}")
            return None

    def translate_chapter_info(self, chapter_info: List[Dict]) -> List[Dict]:
        """챕터 정보 번역"""
        try:
            translator = GoogleTranslator(source=self.source_lang, target=self.output_language)
            
            for chapter in chapter_info:
                # 챕터 제목 번역
                chapter['chapter_title'] = translator.translate(chapter['chapter_title'])
                
                # 섹션 정보 번역
                for section in chapter['sections']:
                    section['title'] = translator.translate(section['title'])
                    section['summary'] = [translator.translate(s) for s in section['summary']]
                
                # 챕터 요약 번역
                chapter['summary'] = [translator.translate(s) for s in chapter['summary']]
                chapter['one_sentence_summary'] = translator.translate(chapter['one_sentence_summary'])
                
            return chapter_info
        except Exception as e:
            print(f"챕터 정보 번역 중 오류 발생: {e}")
            return chapter_info

    def get_chunk_summary(self, chunk: str, json_function: List[Dict] = None) -> Optional[Dict]:
        try:
            system_content = self.system_content + f'Respond in {self.output_language_full}'
            system_token = Utils.num_tokens_from_string(system_content, self.gpt_model)
            json_token = Utils.num_tokens_from_string(json.dumps(json_function), self.gpt_model)
            prompt_token = Utils.num_tokens_from_string(chunk, self.gpt_model)
            
            # 여유있게 response token 설정
            response_token = self.max_token - system_token - json_token - prompt_token - self.buffer_token
            
            # response_token = min(
            #     response_token,
            #     self.config.max_token_response  # 최대값 설정
            # )
            # response_token = max(response_token, self.config.min_token_response)  # 최소값 보장
            
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": chunk}
                ],
                functions=json_function,
                function_call={"name": "create_summary"},
                max_tokens=response_token,
                temperature=self.config.TEMPERATURE
            )
            
            result_json = response.choices[0].message.function_call.arguments
            if self.verbose:
                print(f'System Token: {system_token + json_token} / Prompt Token: {self.prompt_token}\nResponse Token: {response_token}')
                print(f'Actual Response Token: {Utils.num_tokens_from_string(result_json, self.gpt_model)}')
            
            try:
                result = json.loads(result_json)
                result = self.sanitize_summary_structure(result)
                # 스키마 검증
                
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 실패: {e}")
                if self.verbose:
                    print(result_json)
                result = self.extract_valid_json_parts(result_json)
                result = self.sanitize_summary_structure(result)
            
            if not result:
                print("유효한 요약 구조 생성 실패")
                return None
            
            # 섹션 수 검증
            # if len(result['sections']) < 2:
            #     # 섹션이 부족한 경우 다시 시도
            #     print("Warning: 섹션 수가 부족합니다. 다시 시도합니다.")
            #     system_content += "\nIMPORTANT: You must create at least 2 distinct sections."
            #     response = self.client.chat.completions.create(
            #         model=self.gpt_model,
            #         messages=[
            #             {"role": "system", "content": system_content},
            #             {"role": "user", "content": chunk}
            #         ],
            #         functions=json_function,
            #         function_call={"name": "create_summary"},
            #         max_tokens=response_token,
            #         temperature=self.config.TEMPERATURE
            #     )
            #     result = json.loads(response.choices[0].message.function_call.arguments)
            #     result = self.sanitize_summary_structure(result)
            
            return result
            
        except Exception as e:
            print(f"요약 오류: {e}")
            return None

    def sanitize_summary_structure(self, summary: Dict) -> Dict:
        """요약 구조를 정리하고 검증"""
        try:
            sanitized = {
                'sections': [],
                'full_summary': [],
                'one_sentence_summary': '',
                'keywords': []
            }
            
            # sections 정리
            if 'sections' in summary and isinstance(summary['sections'], list):
                for item in summary['sections']:
                    if isinstance(item, dict) and 'title' in item and 'summary' in item:
                        sanitized['sections'].append({
                            'title': item['title'],
                            'summary': item['summary'] if isinstance(item['summary'], list) else []
                        })
            
            # full_summary 정리
            if 'full_summary' in summary:
                if isinstance(summary['full_summary'], str):
                    sanitized['full_summary'] = [summary['full_summary']]
                elif isinstance(summary['full_summary'], list):
                    sanitized['full_summary'] = summary['full_summary']
            
            # one_sentence_summary 정리
            if 'one_sentence_summary' in summary:
                sanitized['one_sentence_summary'] = str(summary['one_sentence_summary'])
            
            # keywords 정리
            if 'keywords' in summary:
                if isinstance(summary['keywords'], list):
                    sanitized['keywords'] = [
                        {'term': k['term']} if isinstance(k, dict) and 'term' in k
                        else {'term': str(k)} for k in summary['keywords']
                        if k is not None
                    ]
            
            return sanitized
            
        except Exception as e:
            print(f"요약 구조 정리 중 오류 발생: {e}")
            return {
                'sections': [],
                'full_summary': [],
                'one_sentence_summary': '',
                'keywords': []
            }

    def fix_truncated_json(self, json_str: str) -> str:
        """잘린 JSON 문자열을 복구 시도"""
        # 열린 괄호 수 계산
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 필요한 만큼 닫는 괄호 추가
        json_str = json_str.rstrip('.')
        json_str += '}' * (open_braces - close_braces)
        json_str += ']' * (open_brackets - close_brackets)
        
        return json_str

    def extract_valid_json_parts(self, json_str: str) -> Dict:
        """부분적으로 유효한 JSON 구조 추출"""
        result = {}
        
        try:
            # sections 추출
            sections_match = re.search(r'"sections":\s*(\[.*?\])', json_str, re.DOTALL)
            if sections_match:
                sections_str = sections_match.group(1)
                try:
                    result['sections'] = json.loads(sections_str)
                except:
                    pass
            
            # keywords 추출
            keywords_match = re.search(r'"keywords":\s*(\[.*?\])', json_str, re.DOTALL)
            if keywords_match:
                keywords_str = keywords_match.group(1)
                try:
                    result['keywords'] = json.loads(keywords_str)
                except:
                    pass
            
            # full_summary 추출
            summary_match = re.search(r'"full_summary":\s*(\[.*?\])', json_str, re.DOTALL)
            if summary_match:
                summary_str = summary_match.group(1)
                try:
                    result['full_summary'] = json.loads(summary_str)
                except:
                    pass
            
            # one_sentence_summary 추출
            sentence_match = re.search(r'"one_sentence_summary":\s*"([^"]*)"', json_str)
            if sentence_match:
                result['one_sentence_summary'] = sentence_match.group(1)
        
        except Exception as e:
            print(f"JSON 부분 추출 중 오류: {e}")
        
        return result

    def merge_summaries(self, summaries: List[Dict], chunks: List[str]) -> tuple[str, list, list]:
        merged = {
            "sections": [],
            "full_text": [],  # 각 섹션의 원본 텍스트를 저장할 리스트
            "keywords": []
        }
        concat = ''
        
        # 각 chunk의 섹션들을 순차적으로 리
        for i, summary in enumerate(summaries):
            if not isinstance(summary, dict):
                print(f"Warning: Invalid summary type: {type(summary)}")
                continue
            
            sections = summary.get('sections', [])
            
            # 현재 chunk의 섹션들을 처리
            for i, section in enumerate(sections): 
                # 나머지 처리는 동일
                # section['start_index'] = chunks[i].lower().find(section['start_sentence'].lower())
                # section['end_index'] = chunks[i].rfind(section['end_sentence'])+1
                
                #  # 원본 텍스트에서 해당 섹션의 전체 텍스트 추출
                # original_text = chunks[i][section['start_index']:section['end_index']]
                # merged["full_text"].append(original_text)
                
                section_text = str(i+1) + '. ' + section.get('title', '') + ': '
                section_text += ' '.join(section.get('summary', [])) + ' '
                concat += section_text
            
                merged["sections"].append(section)
            
            merged["keywords"].extend(summary.get('keywords', []))
        
        # 중복 키워드 제거
        merged["keywords"] = list({keyword['term']: keyword for keyword in merged["keywords"]}.values())
        
        # 검증: sections와 full_text 개수가 일치하는지 확인
        # if len(merged["sections"]) != len(merged["full_text"]):
        #     print(f"Warning: Number of sections ({len(merged['sections'])}) does not match number of full texts ({len(merged['full_text'])})")
        
        return concat, merged#['sections']#, merged['keywords']#, merged["full_text"]
    def clean_text(self, text: str) -> str:
        """Remove unnecessary spaces from the text."""
        return re.sub(r'\s+', ' ', text).strip()
    def translate_summary(self, summary: Dict):
        def translate_text(text: str, translator) -> str:
            """Translate text in chunks of 4500 characters."""
            if not text:  # 빈 텍스트 체크
                return ""
            if not isinstance(text, str):  # 문자열이 아닌 경우 변환
                text = str(text)
                
            max_length = self.max_translate_length
            translated_text = ""
            for i in range(0, len(text), max_length):
                chunk = text[i:i + max_length]
                translated_text += translator.translate(chunk)
            return self.clean_text(translated_text)

        try:
            if not summary:  # summary가 None인 경우 처리
                return {}
                
            translator = GoogleTranslator(source=self.source_lang, target=self.output_language)
            
            # 섹션 번역
            if 'sections' in summary:
                summary['sections'] = [
                    {
                        'title': translate_text(section.get('title', ''), translator),
                        'summary': [translate_text(s, translator) for s in section.get('summary', [])]
                    }
                    for section in summary.get('sections', [])
                ]
            
            # 전체 요약 번역
            if 'full_summary' in summary:
                if isinstance(summary['full_summary'], str):
                    summary['full_summary'] = [summary['full_summary']]
                summary['full_summary'] = [translate_text(s, translator) for s in summary.get('full_summary', [])]
            
            # 키워드 번역
            if 'keywords' in summary:
                summary['keywords'] = [
                    {'term': translate_text(keyword['term'] if isinstance(keyword, dict) else keyword, translator)}
                    for keyword in summary.get('keywords', [])
                ]

            # 한 문장 요약 번역
            if 'one_sentence_summary' in summary:
                one_sentence = summary['one_sentence_summary']
                if isinstance(one_sentence, list):
                    one_sentence = ' '.join(one_sentence)
                summary['one_sentence_summary'] = translate_text(one_sentence, translator)
            
            return summary
            
        except Exception as e:
            print(f"번역 중 오류 발생: {e}")
            print(f"Summary 구조: {summary}")  # 디버깅을 위한 출력
            return summary  # 오류 발생시 원본 반환
    # def translate_summary(self, summary: Dict):
    #     def translate_text(text: str, translator) -> str:
    #         """Translate text in chunks of 4500 characters."""
    #         max_length = self.max_translate_length
    #         translated_text = ""
    #         for i in range(0, len(text), max_length):
    #             chunk = text[i:i + max_length]
    #             translated_text += translator.translate(chunk)
    #         return self.clean_text(translated_text)

    #     try:
    #         translator = GoogleTranslator(source=self.source_lang, target=self.output_language)
            
    #         # 섹션 번역
    #         summary['sections'] = [
    #             {
    #                 'title': translate_text(section.get('title', ''), translator),
    #                 'summary': [translate_text(s, translator) for s in section.get('summary', [])]
    #             }
    #             for section in summary.get('sections', [])
    #         ]
            
    #         # 전체 요약 번역
    #         summary['full_summary'] = [translate_text(s, translator) for s in summary.get('full_summary', [])]
    #         summary['keywords'] = [{
    #             'term': translate_text(section.get('term', ''), translator),
    #             #'context': translate_text(section.get('context', ''), translator),
    #             }
    #             for section in summary.get('keywords', [])
    #         ]

    #         # 한 문장 요약 번역
    #         one_sentence = summary.get('one_sentence_summary', '')
            
    #         if isinstance(one_sentence, str):
    #             print(f"Warning: Unexpected one_sentence_summary type (list): {type(one_sentence)}")
    #         elif isinstance(one_sentence, list):
    #             one_sentence = ''.join(one_sentence)
    #         else:
    #             print(f"Warning: Unexpected one_sentence_summary type: {type(one_sentence)}")
                
    #         summary['one_sentence_summary'] = translate_text(one_sentence, translator)
            
    #         return summary
    #     except Exception as e:
    #         print(f"번역 중 오류 발생: {e}")
    def format_summary(self, merged_summary: Dict, processed_text: str) -> Dict:
        try:
            one_sentence = merged_summary.get('one_sentence_summary', '')
            if isinstance(one_sentence, list):
                one_sentence = ''.join(one_sentence)
            
            formatted_summary = {
                'sections': merged_summary.get('sections', []),
                'full_summary': "-"+"\n-".join(merged_summary.get('full_summary', '')),
                'one_sentence_summary': one_sentence,
                'text': processed_text,
            }
            
            # 옵션에 따른 추가 정보
            if self.config.INCLUDE_KEYWORDS:
                keywords = merged_summary.get('keywords', [])
                sorted_keywords = sorted(keywords, 
                                       key=lambda x: x.get('count', 0), 
                                       reverse=True)[:5]
                formatted_summary['keywords'] = sorted_keywords
                #formatted_summary['keywords_original'] = [item['term'] for item in sorted_keywords]
            
            if self.config.ENABLE_CHAPTERS:
                formatted_summary['chapters'] = merged_summary.get('chapters', [])
            
            if self.config.INCLUDE_FULL_TEXT:
                formatted_summary['full_text'] = merged_summary.get('full_text', '')
            
            return formatted_summary
            
        except Exception as e:
            print(f"포맷팅 중 오류 발생: {e}")
            return {
                'sections': [],
                'full_summary': '',
                'one_sentence_summary': '',
                'text': processed_text
            }





