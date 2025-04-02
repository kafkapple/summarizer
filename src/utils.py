import time
import os
import re
from datetime import datetime
import pandas as pd
import tiktoken
from typing import List, Union, Dict
import logging
import json

class Utils:
    def __init__(self):
        self.script_dir = self.get_script_directory()
    @staticmethod
    def num_tokens_from_string(string: str, gpt_model: str) -> int: #encoding_name: str = "cl100k_base"
        """Returns the number of tokens in a text string."""
        #encoding = openai.Encoding.get_encoding(self.gpt_model)
        #encoding = tiktoken.get_encoding(encoding_name)
        if not isinstance(string, str):
            print(f"Warning: Expected string, got {type(string)}. Converting to string.")
            string = str(string)
        encoding = tiktoken.encoding_for_model(gpt_model)
        return len(encoding.encode(string))
    @staticmethod
    def split_text_into_chunks(text: str, max_length: int = 2000, by_token: bool = False, gpt_model: str = None) -> List[str]:
        """
        텍스트를 청크로 분할하는 공통 함수
        
        Args:
            text (str): 분할할 텍스트
            max_length (int): 청크당 최대 길이 (토큰 또는 문자)
            by_token (bool): 토큰 기준 분할 여부
        
        Returns:
            List[str]: 분할된 청크 리스트
        """
        if not text:
            return []
        
        # 먼저 기본적인 문장 종결 부호로 시도
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        # 문장 종결 부호로 분리가 안 된 경우에만 다른 패턴 시도
        if len(sentences) <= 1:
            sentence_patterns = [
                r'[\n]{2,}',             # 빈 줄
                r'(?<=[;:])[\s\n]+',     # 세미콜론/콜론 뒤의 공백이나 줄바꿈
                r'[\s]{2,}',             # 연속된 공백
                r'(?<=[\,])[\s]+'        # 쉼표 뒤의 공백
            ]
            
            current_text = text.strip()
            for pattern in sentence_patterns:
                sentences = [s.strip() for s in re.split(pattern, current_text) if s.strip()]
                if len(sentences) > 1:
                    break
        
        # 여전히 분리가 안 된 경우 단순 길이 기준 분할
        if len(sentences) <= 1:
            sentences = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        # 청크 생성
        chunks = []
        current_chunk = ""
        
        for i in range(len(sentences)):
            sentence = sentences[i].strip()
            if not sentence:
                continue
                
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            # 길이 체크 (토큰 또는 문자 기준)
            if by_token:
                current_length = Utils.num_tokens_from_string(potential_chunk, gpt_model)  # 토큰 계산 함수 필요
                length_exceeded = current_length >= max_length
            else:
                length_exceeded = len(potential_chunk) >= max_length
            
            if length_exceeded and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    # @staticmethod
    # def prep_text():
    #     #Remove special characters (optional, uncomment if needed)
    #     text = re.sub(r'[^\w\s]', '', text)
    #     #Convert to lowercase (optional, uncomment if needed)
    #     text = text.lower()
    #     #Remove numbers (optional, uncomment if needed)
    #     text = re.sub(r'\d+', '', text)
    
    def get_latest_file_with_string(directory, search_string):
        """특정 문자열을 포함하는 파일 중 가장 최신 파일의 경로를 반환합니다."""
        latest_file = None
        latest_time = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if search_string in file:
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path

        return latest_file

    def save_file(self, data, filepath):
        """파일 확장자를 확인한 후 데이터를 저장합니다."""
        filename = self.add_timestamp_to_filename(filename)
        ext = os.path.splitext(filepath)[1]
        full_path = self.create_path_relative_to_script(filepath)
        self.ensure_directory_exists(os.path.dirname(full_path))
        print('Saved to ', full_path)
        if ext == '.txt':
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        elif ext == '.json':
            import json
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif ext == '.csv':
            if isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(filepath,encoding='utf-8-sig' )
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {ext}")
    def load_file(self, filepath):
        """파일 확장자를 확인한 후 파일을 불러옵니다."""
        ext = os.path.splitext(filepath)[1]
        full_path = self.create_path_relative_to_script(filepath)
        if ext == '.txt':
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.json':
            import json
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext =='.csv':
            return pd.read_csv(filepath, encoding='utf-8-sig') 
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {ext}")
    @staticmethod
    def print_data_info(data):
        """데이터의 타입, 크기, 내용 출력합니다."""
        print(f"타입: {type(data)}")
        if hasattr(data, '__len__'):
            print(f"크기: {len(data)}")
        print(f"내용: {data}")

    @staticmethod
    def timeit(func):
        """함수의 실행 시간을 측정하는 데코레이터입니다."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"'{func.__name__}' 함수 실행 시간: {end_time - start_time:.6f}초")
            return result
        return wrapper
    
    @staticmethod
    def find_elements_with_substring(lst, substring):
        indices = [index for index, element in enumerate(lst) if substring in element]
        elements = [lst[index] for index in indices]
        return indices, elements

    @staticmethod
    def list_csv_files(directory):
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        return csv_files

    @staticmethod
    def iterate_and_print(iterable):
        """Iterable한 데이터의 내용을 하나씩 출력합니다."""
        for index, item in enumerate(iterable):
            print(f"인덱스 {index}: {item}")

    @staticmethod
    def extract_strings(pattern, text):
        """주어진 패턴에 매칭되는 문자열을 추출합니다."""
        return re.findall(pattern, text)

    @staticmethod
    def add_timestamp_to_filename(filename):
        """현재 시각의 타임스탬프를 파일 이름에 추가합니다."""
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base}_{timestamp}{ext}"

    @staticmethod
    def get_script_directory():
        """현재 스크립트 파일의 디렉토리 경로를 반환합니다."""
        return os.path.dirname(os.path.abspath(__file__))

    def create_path_relative_to_script(self, relative_path):
        """현재 스크립트 파일 위치를 기준으로 한 상대 경로를 생성합니다."""
        path=os.path.join(self.script_dir, relative_path)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def ensure_directory_exists(directory_path):
        """디렉토리가 존재하지 않으면 생성합니다."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def extract_keys_from_dict(dictionary, key_list):
        """딕셔너리에서 특정 키들만 추출하여 새로운 딕셔너리를 반환합니다."""
        return {key: dictionary[key] for key in key_list if key in dictionary}

    @staticmethod
    def flatten_list(nested_list):
        """중첩된 리스트를 평탄화합니다."""
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def merge_dicts(*dicts):
        """여러 딕셔너리를 하나로 병합합니다."""
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    @staticmethod
    def chunks(lst, n):
        """리스트를 n개씩 분할합니다."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def save_summary_to_files(summary: Dict, output_dir: str = "outputs/summaries"):
        """요약 결과를 JSON, MD, TXT 파일로 저장합니다."""
        try:
            # outputs 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성 (타임스탬프와 제목 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = summary.get('metadata', {})
            title = metadata.get('title', 'untitled')
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:50]  # 파일명 안전 문자 처리
            
            base_filename = f"{safe_title}_{timestamp}"
            
            # --- TXT 파일 저장 (원본 텍스트 + 기본 메타데이터) ---
            txt_filepath = os.path.join(output_dir, f"{base_filename}.txt")
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"제목: {title}\n")
                    f.write(f"URL: {metadata.get('url', 'N/A')}\n")
                    f.write(f"채널: {metadata.get('channel', 'N/A')}\n")
                    f.write(f"발행일: {metadata.get('published_date', 'N/A')}\n")
                    # 추가 메타데이터 필요시 여기에 추가
                    f.write("\n" + "="*50 + "\n\n")
                    # full_text 키가 있으면 사용, 없으면 빈 문자열
                    f.write(summary.get('full_text', '')) 
            except Exception as txt_e:
                 logging.error(f"Error saving TXT file {txt_filepath}: {txt_e}")

            # --- JSON 파일 저장 (전체 요약 정보, 원본 텍스트 제외) ---
            json_filepath = os.path.join(output_dir, f"{base_filename}.json")
            try:
                # 원본 텍스트 제외한 복사본 생성
                summary_for_json = summary.copy()
                summary_for_json.pop('full_text', None) # full_text 키 제거
                
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary_for_json, f, ensure_ascii=False, indent=2)
            except Exception as json_e:
                 logging.error(f"Error saving JSON file {json_filepath}: {json_e}")

            # --- MD 파일 저장 (가독성 좋은 요약) ---
            md_filepath = os.path.join(output_dir, f"{base_filename}.md")
            try:
                with open(md_filepath, 'w', encoding='utf-8') as f:
                    f.write("# 요약 결과\n\n")
                    
                    # 메타데이터
                    f.write("## 메타데이터\n\n")
                    f.write(f"- 제목: {title}\n")
                    f.write(f"- URL: {metadata.get('url', 'N/A')}\n") # 기본값 N/A 추가
                    f.write(f"- 채널: {metadata.get('channel', 'N/A')}\n")
                    f.write(f"- 발행일: {metadata.get('published_date', 'N/A')}\n")
                    f.write(f"- 조회수: {metadata.get('view_count', 'N/A')}\n")
                    f.write(f"- 좋아요: {metadata.get('like_count', 'N/A')}\n")
                    f.write(f"- 댓글: {metadata.get('comment_count', 'N/A')}\n")
                    category_val = metadata.get('category', 'N/A')
                    if isinstance(category_val, list):
                        f.write(f"- 카테고리: {', '.join(category_val) if category_val else 'N/A'}\n")
                    else:
                        f.write(f"- 카테고리: {category_val}\n")
                    tags_list = metadata.get('tags', [])
                    f.write(f"- 태그: {', '.join(tags_list) if tags_list else 'N/A'}\n\n")
                    
                    # 한 문장 요약
                    f.write("## 한 문장 요약\n\n")
                    f.write(f"{summary.get('one_sentence_summary', '요약 정보 없음')}\n\n") # 기본값 추가
                    
                    # 전체 요약
                    f.write("## 전체 요약\n\n")
                    full_summary_val = summary.get('full_summary', '요약 정보 없음') # 기본값 추가
                    if isinstance(full_summary_val, list):
                        f.write("\n".join([f"- {str(item)}" for item in full_summary_val]) if full_summary_val else "요약 정보 없음")
                    elif isinstance(full_summary_val, str):
                         f.write(full_summary_val if full_summary_val.strip() else "요약 정보 없음")
                    else: # 다른 타입일 경우 문자열 변환
                         f.write(str(full_summary_val) if full_summary_val else "요약 정보 없음")
                    f.write("\n\n")

                    # 핵심 주제
                    f.write("## 핵심 주제\n\n")
                    core_summary_val = summary.get('core_summary', '요약 정보 없음') # 기본값 추가
                    if isinstance(core_summary_val, list):
                         f.write("\n".join([f"- {str(item)}" for item in core_summary_val]) if core_summary_val else "요약 정보 없음")
                    elif isinstance(core_summary_val, str):
                        points = core_summary_val.split('\n')
                        f.write("\n".join([f"- {p.strip()}" for p in points if p.strip()]) if core_summary_val.strip() else "요약 정보 없음")
                    else:
                        f.write(str(core_summary_val) if core_summary_val else "요약 정보 없음")
                    f.write("\n\n")
                    
                    # 섹션별 요약
                    f.write("## 섹션별 요약\n\n")
                    sections_summary_val = summary.get('sections_summary', '요약 정보 없음') # 기본값 추가
                    # sections_summary는 보통 문자열로 생성됨 (_format_sections_summary 참고)
                    if isinstance(sections_summary_val, str):
                        f.write(sections_summary_val if sections_summary_val.strip() else "요약 정보 없음")
                    elif isinstance(sections_summary_val, list): # 혹시 리스트 형태일 경우 처리
                         for section in sections_summary_val:
                             f.write(f"### {section.get('title', 'Untitled Section')}\n")
                             sec_summary = section.get('summary', '')
                             if isinstance(sec_summary, list):
                                 f.write("\n".join([f"- {str(s)}" for s in sec_summary]))
                             else:
                                 f.write(str(sec_summary))
                             f.write("\n\n")
                    else:
                        f.write(str(sections_summary_val) if sections_summary_val else "요약 정보 없음")
                    f.write("\n\n")
                    
                    # 키워드
                    keywords = summary.get('keywords', [])
                    if keywords:
                        f.write("## 키워드\n\n")
                        for keyword in keywords:
                            if isinstance(keyword, dict):
                                f.write(f"- {keyword.get('term', 'N/A')} (빈도: {keyword.get('frequency', 'N/A')})\n")
                            else:
                                f.write(f"- {str(keyword)}\n")
                        f.write("\n")
                    else:
                         f.write("키워드 정보 없음\n\n")

                    # 챕터 정보 (존재하는 경우)
                    chapters = summary.get('chapters', [])
                    if chapters:
                        f.write("## 챕터 정보\n\n")
                        for chapter in chapters:
                            f.write(f"### {chapter.get('chapter_title', 'Untitled Chapter')}\n\n")
                            f.write(f"한 문장 요약: {chapter.get('one_sentence_summary', 'N/A')}\n\n")
                            f.write("주요 내용:\n")
                            chapter_summary_points = chapter.get('summary', [])
                            if isinstance(chapter_summary_points, list):
                                for point in chapter_summary_points:
                                    f.write(f"- {point}\n")
                            else:
                                f.write(f"- {str(chapter_summary_points)}\n") # 리스트가 아닌 경우 처리
                            f.write("\n")
                        f.write("\n")

                logging.info(f"요약 결과가 저장되었습니다: {json_filepath}, {md_filepath}, {txt_filepath}") # txt_filepath 로깅 복원
            except Exception as md_e:
                 logging.error(f"Error saving Markdown file {md_filepath}: {md_e}")
            
        except Exception as e:
            logging.error(f"요약 결과 저장 중 오류 발생: {e}")

    @staticmethod
    def preprocess_text(text, clean_tags=True, remove_special_chars=False, to_lowercase=False, remove_numbers=False):
        """
        Unified text preprocessing function with enhanced type checking
        
        Args:
            text: Text to process (string, list of strings, or list of dicts with 'text' key)
            clean_tags: Whether to remove [music], (applause) tags
            remove_special_chars: Whether to remove special characters
            to_lowercase: Whether to convert to lowercase
            remove_numbers: Whether to remove numbers
        """
        try:
            if text is None:
                return ''
                
            # Type handling
            if isinstance(text, list):
                # Handle list of dictionaries (transcript format)
                if all(isinstance(item, dict) and 'text' in item for item in text):
                    text_parts = []
                    for entry in text:
                        cleaned_text = entry['text'].strip()
                        if clean_tags:
                            cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
                            cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)
                        
                        if cleaned_text:
                            # Add period if missing sentence-ending punctuation
                            if not cleaned_text[-1] in '.!?':
                                cleaned_text += '.'
                            text_parts.append(cleaned_text)
                    text = ' '.join(text_parts)
                else:
                    # Handle list of strings or other items
                    text = ' '.join(str(item).strip() for item in text if str(item).strip())
            
            # Ensure string type at this point
            if not isinstance(text, str):
                # Log type for debugging
                print(f"Warning: Expected string after processing, got {type(text)}. Converting to string.")
                text = str(text)
            
            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            # Remove tags if requested
            if clean_tags:
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(r'\(.*?\)', '', text)
            
            # Optional processing
            if remove_special_chars:
                text = re.sub(r'[^\w\s]', '', text)
            if to_lowercase:
                text = text.lower()
            if remove_numbers:
                text = re.sub(r'\d+', '', text)
                
            return text
            
        except Exception as e:
            print(f"Text preprocessing error: {e}, type: {type(text)}")
            if isinstance(text, str):
                return text.strip()  # Return original text as fallback
            return ''  # Empty string as last resort

# 로깅 설정
def setup_logging():
    # 기본 로거 설정
    logger = logging.getLogger() # Get root logger
    logger.setLevel(logging.DEBUG) # Set root logger level
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    debug_log_file = os.path.join(log_dir, f"llm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 포맷터 설정
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s') # Include logger name

    # 파일 핸들러 (모든 로그)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Capture all levels in the main log file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 (INFO 이상 -> DEBUG 이상으로 변경)
    console_handler = logging.StreamHandler() # Defaults to sys.stderr
    console_handler.setLevel(logging.DEBUG) # INFO -> DEBUG 변경
    console_handler.setFormatter(formatter)
    # Add encoding for console if needed, though usually handled by terminal/env
    # console_handler.stream.reconfigure(encoding='utf-8') # Example for Python 3.7+
    # Set encoding explicitly for Windows environments prone to cp949 issues
    try:
        import sys
        if sys.platform == "win32":
             console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        # For other OS or if the above fails, rely on default stream handling
    except Exception as e:
        logger.warning(f"Could not set console handler encoding to utf-8: {e}")
        
    logger.addHandler(console_handler)

    # 디버그 로거 (llm_debug) 설정
    debug_logger = logging.getLogger('llm_debug')
    debug_logger.setLevel(logging.DEBUG)
    # debug_logger.propagate = False # 루트 로거로 전파 방지 -> 주석 처리 또는 True로 변경
    debug_logger.propagate = True # 명시적으로 True 설정

    # 디버그 로거의 기존 핸들러 제거 (중복 방지)
    for handler in debug_logger.handlers[:]:
        debug_logger.removeHandler(handler)

    # 디버그 로그 파일 핸들러
    debug_file_handler = logging.FileHandler(debug_log_file, encoding='utf-8')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(formatter)
    debug_logger.addHandler(debug_file_handler)

    # 기본 로거 ('__main__' 또는 다른 모듈 이름) 와 디버그 로거 반환
    main_logger = logging.getLogger(__name__) # Get logger for the current module (utils)
    return main_logger, debug_logger