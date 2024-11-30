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
        self.response_token = 600
        self.buffer_token = 0
        self.max_translate_length = 4500
        self.max_response_token = 600

        self.MAX_CHUNKS_PER_CHAPTER =  6  # 한 챕터당 최대 청크 수

        print( '\n'+'#'*7 +' Initialization of Summarizer ' +'#'*7+ f"\nGPT 모델 = {self.gpt_model}\n초기화: 대상 언어 = {self.output_language}")
        self.system_token = Utils.num_tokens_from_string(self.system_content, self.gpt_model)
        self.json_token = Utils.num_tokens_from_string(json.dumps(self.json_function_full), self.gpt_model)
        self.prompt_token = self.max_token - self.system_token - self.json_token - self.response_token  -self.buffer_token
        if self.verbose:
            print(f'\nPutative Max/System/Json/Response:{self.max_token}/{self.system_token}/{self.json_token}/{self.response_token}\nPrompt:{self.prompt_token}')
        
    def summarize(self, text: str, title: str) -> Optional[Dict]:
        try:
            processed_text = Utils.preprocess_text(text)
            self.source_lang = detect(processed_text)
            chunks = Utils.split_text_into_chunks(text=processed_text, max_length=self.prompt_token, by_token=True, gpt_model=self.gpt_model)
            n_chunks = len(chunks)
            if self.verbose:
                print('Max token per chunk:', self.prompt_token)
                print(f'청크 수: {n_chunks}')
            
            # 청크가 너무 많은 경우 (예: 10개 이상) 챕터 단위로 처리
            MAX_CHUNKS_PER_CHAPTER =  self.MAX_CHUNKS_PER_CHAPTER  # 한 챕터당 최대 청크 수
            
            # 기존 로직
            if n_chunks == 1: # 
                text = chunks[0]
                prompt = f'Title: {title}/ {text}'
                summary = self.get_chunk_summary(prompt,  self.json_function_full)
                #summary['full_text'] = [text[section.get('start_index', 0):section.get('end_index', 0)] 
                #                      for section in summary.get('sections', [])]
            elif n_chunks <= MAX_CHUNKS_PER_CHAPTER:  # section 단위로 요약
                summary_chunks = [self.get_chunk_summary(chunk, self.json_function_section) 
                                for chunk in chunks if chunk]
                concat, merged = self.merge_summaries(summary_chunks, chunks)
                prompt = f'Title: {title}/ {concat}'
                summary = self.get_chunk_summary(prompt,  self.json_function_final)
                summary['sections'] = merged["sections"]
                summary["keywords"] = merged["keywords"]
                
            elif n_chunks > MAX_CHUNKS_PER_CHAPTER: # chapter 단위로 요약
                return self.process_large_text(chunks, title, MAX_CHUNKS_PER_CHAPTER)
            
            summary["keywords"] = list({keyword['term']: keyword for keyword in summary["keywords"]}.values())

            if self.source_lang != self.output_language or self.source_lang == 'unknown':
                summary = self.translate_summary(summary)

            summary['keywords_original'] = [item['term'] for item in summary["keywords"]]
            final_summary = self.format_summary(summary, processed_text)
            if self.verbose:
                print(final_summary['one_sentence_summary'], final_summary['full_summary'])

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
            system_content = self.system_content + f'Respond in {self.output_language_full}, maintain consistency in formatting throughout the response.'#
             # When encountering proper nouns, English abbreviations, or technical terminology from the original text, preserve them in their original English form without translation.'
            #f'Always respond in {self.output_language_full} language, and maintain consistency in language and formatting throughout the response. Keep proper nouns, English abbreviations, and technical terms in their original English form.'
            system_token = Utils.num_tokens_from_string(system_content, self.gpt_model)
            json_token = Utils.num_tokens_from_string(json.dumps(json_function), self.gpt_model)
            
            prompt = chunk
            prompt_token = Utils.num_tokens_from_string(prompt, self.gpt_model)
            response_token = self.max_token - system_token - json_token - prompt_token
            #response_token = max(response_token, self.max_response_token)
            if self.verbose:
                print(f'Response Token: {response_token}')
            #print(f'\nActual Max/System/Json/Response/Prompt:{self.max_token}/{self.system_token}/{self.json_token}/{self.response_token}/{prompt_token}: buffer = {self.RESPONSE_BUFFER}')
        
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt  }
                ],
                functions = json_function,
                function_call={"name": "create_summary"},
                max_tokens=response_token,
                temperature=self.config.TEMPERATURE
            )
            result_json = response.choices[0].message.function_call.arguments
            
            keys = json_function[0]['parameters']['properties'].keys()
            default_structure = {key: [] for key in keys}
            result = self.process_json_response(result_json, default_structure)
            if self.verbose:
                print(f'OutputTokens: {Utils.num_tokens_from_string(result_json, self.gpt_model)}')
            return result
        except Exception as e:
            print(f"요약 오류: {e}")
            return None

    def process_json_response(self, response_text: str, default_structure: Dict[str, List] = None) -> Optional[Dict]:
        try:
            # 불필요한 공백, 줄바꿈, 탭 제거
            response_text = re.sub(r'\s+', ' ', response_text.strip())

            # 기본 구조 설정
            if default_structure is None:
                default_structure = {
                    "keywords": [],
                    "sections": [],
                    "full_summary": [],
                    "one_sentence_summary": []
                }
            
            # JSON 파싱 시도
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}\n{response_text}")
                return default_structure
            
            # 결과 타입 확인
            if not isinstance(result, dict):
                print(f"예상치 못한 결과 타입: {type(result)}")
                return default_structure
            
            # 기본 구조에 따라 결과 보정
            for key in default_structure.keys():
                if key not in result:
                    result[key] = default_structure[key]
                elif not isinstance(result[key], list):
                    result[key] = [result[key]] if result[key] else []
            
            return result
        
        except Exception as e:
            print(f"JSON 처리 중 예외 발생: {e}")
            return None

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
            max_length = self.max_translate_length
            translated_text = ""
            for i in range(0, len(text), max_length):
                chunk = text[i:i + max_length]
                translated_text += translator.translate(chunk)
            return self.clean_text(translated_text)

        try:
            translator = GoogleTranslator(source=self.source_lang, target=self.output_language)
            
            # 섹션 번역
            summary['sections'] = [
                {
                    'title': translate_text(section.get('title', ''), translator),
                    'summary': [translate_text(s, translator) for s in section.get('summary', [])]
                }
                for section in summary.get('sections', [])
            ]
            
            # 전체 요약 번역
            summary['full_summary'] = [translate_text(s, translator) for s in summary.get('full_summary', [])]
            summary['keywords'] = [{
                'term': translate_text(section.get('term', ''), translator),
                #'context': translate_text(section.get('context', ''), translator),
                }
                for section in summary.get('keywords', [])
            ]

            # 한 문장 요약 번역
            one_sentence = summary.get('one_sentence_summary', '')
            
            if isinstance(one_sentence, str):
                print(f"Warning: Unexpected one_sentence_summary type (list): {type(one_sentence)}")
            elif isinstance(one_sentence, list):
                one_sentence = ''.join(one_sentence)
            else:
                print(f"Warning: Unexpected one_sentence_summary type: {type(one_sentence)}")
                
            summary['one_sentence_summary'] = translate_text(one_sentence, translator)
            
            return summary
        except Exception as e:
            print(f"번역 중 오류 발생: {e}")
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
                                       key=lambda x: x.get('frequency', 0), 
                                       reverse=True)[:10]
                formatted_summary['keywords'] = sorted_keywords
                formatted_summary['keywords_original'] = [item['term'] for item in sorted_keywords]
            
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





