import time
import os
import re
from datetime import datetime
import pandas as pd

from typing import List, Union, Dict
class Utils:
    def __init__(self):
        self.script_dir = self.get_script_directory()
    
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

    
