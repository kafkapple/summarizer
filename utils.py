import time
import os
import re
from datetime import datetime
import pandas as pd
import tiktoken
from typing import List, Union, Dict
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
    def preprocess_text(text: Union[str, List[Dict[str, str]], List[str]], 
                       remove_special_chars: bool = False, 
                       to_lowercase: bool = False, 
                       remove_numbers: bool = False,
                       clean_tags: bool = True) -> str:
        """
        통합된 텍스트 전처리 함수
        
        Args:
            text: 처리할 텍스트
            remove_special_chars: 특수문자 제거 여부
            to_lowercase: 소문자 변환 여부
            remove_numbers: 숫자 제거 여부
            clean_tags: [음악], (박수) 등의 태그 제거 여부
        """
        try:
            if not text:
                return ''
                
            # 리스트 처리
            if isinstance(text, list):
                if all(isinstance(item, dict) and 'text' in item for item in text):
                    # 자막 딕셔너리 리스트 처리
                    text_parts = []
                    for entry in text:
                        cleaned_text = entry['text'].strip()
                        if clean_tags:
                            cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
                            cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)
                        
                        if cleaned_text:
                            if not cleaned_text[-1] in '.!?':
                                cleaned_text += '.'
                            text_parts.append(cleaned_text)
                    text = ' '.join(text_parts)
                else:
                    text = ' '.join(str(item).strip() for item in text if str(item).strip())
            
            elif not isinstance(text, str):
                raise ValueError("입력은 문자열 또는 리스트 형식이어야 합니다")

            # 기본 전처리
            text = text.strip()
            text = ' '.join(text.split())
            
            # 선택적 전처리
            if remove_special_chars:
                text = re.sub(r'[^\w\s]', '', text)
            if to_lowercase:
                text = text.lower()
            if remove_numbers:
                text = re.sub(r'\d+', '', text)
                
            return text
            
        except Exception as e:
            print(f"텍스트 전처리 중 오류 발생: {e}")
            return ''
