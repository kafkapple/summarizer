from src.fetcher.base import MediaSource, WebContent
from omegaconf import DictConfig
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
import time
import random
import requests
import os
from dotenv import load_dotenv

class PocketClient(WebContent):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.consumer_key = cfg.api_keys['pocket_consumer']
        self.access_token = cfg.api_keys['pocket_access']
        self.base_url = "https://getpocket.com/v3"
        self.all_items = []
        print('#'*7+'PocketClient init'+'#'*7)

    def fetch_content(self, batch_size=500, state='all', detail_type='complete', sort='newest', offset=0, tags=None):
        """
        1. 전체 아이템 목록 먼저 수집
        2. 각 아이템별 으로 콘텐츠 수집 및 요약
        """
        # sort: newest, oldest, title, site
        # 1. 전체 아이템 목록 수집
        items = self._get_items_with_params(batch_size, state, detail_type, sort, offset, tags)
        processed_items = self.process_items(items)
        
        print(f"총 {len(processed_items)}개의 아이템을 처리합니다.")
        return processed_items
        

    def _fetch_single_content(self, item, max_retries=1):
        """단일 아이템의 웹 콘텐츠 수집"""
        for attempt in range(max_retries):
            try:
                # 요청  딜
                time.sleep(random.uniform(2, 6))
                
                content = self.fetch_web_content(item['url'])
                if self._is_valid_content(content):
                    print(f"성공: {item['title']}")
                    return content
                    
            except Exception as e:
                print(f"시도 {attempt + 1}/{max_retries} 실패 ({item['url']}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # 재시도마다 대기 시간 증가
                continue
                
        print(f"최종 실패: {item['url']}")
        return None

    def _get_items_with_params(self, batch_size, state, detail_type, sort, offset, tags):
        """Pocket API 파라미터 설정 및 아이템 수집"""
        params = {
            "consumer_key": self.consumer_key,
            "access_token": self.access_token,
            "count": batch_size,
            "offset": offset,
            "state": state,
            "detailType": detail_type,
            'sort': sort,
        }
        
        if tags:
            if isinstance(tags, list):
                params['tag'] = ','.join(tags)
            else:
                params['tag'] = tags
        
        return self.get_all_items(params)

    def get_contents(self, processed_items, max_retries=3):
        """processed_items의 각 URL에서 실제 컨텐츠 추출"""
        for item in tqdm(processed_items, desc="Fetching contents"):
            try:
                content = self.fetch_web_content(item['url'])
                if self._is_valid_content(content):
                    item['content'] = content
                else:
                    item['content'] = None
                    print(f'Invalid content: {item["url"]}')
            except Exception as e:
                print(f"Error fetching content for {item['url']}: {e}")
                item['content'] = None
                
            # 요청 간 짧은 딜레이 추가
            time.sleep(random.uniform(0.5, 1.5))
                
        return processed_items

    def get_items(self, params, max_retries: int = 3, retry_delay: int = 1) -> List[Dict]:
        headers = {
            'Content-Type': 'application/json',
            'X-Accept': 'application/json'
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/get", json=params, headers=headers)
                response.raise_for_status()
                return response.json().get("list", {})
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)

    def get_all_items(self, params, max_items = None,batch_size = 500, favorite = None, tag=None, content_type=None ):
        total_items = 0
        
        if favorite is not None:
            params['favorite'] = favorite
        if tag is not None:
            params['tag'] = tag
        if content_type is not None:
            params['contentType'] = content_type

        while True:
            items_batch = self.get_items(params)
            if not items_batch:
                break
            self.all_items.extend(items_batch.values())
            total_items += len(items_batch)
            
            print(f"Retrieved {len(items_batch)} items. Total: {total_items}")
            
            if max_items and total_items >= max_items:
                self.all_items = self.all_items[:max_items]
                break
            
            params['offset'] += batch_size  # 다음 배치를 위해 offset 증가
           # self.print_pocket_items(items_batch)
            time.sleep(1)
            if len(self.all_items) % 500 == 0:
                print(f'n: {len(self.all_items)}\n')
        return self.all_items
    
    def process_items(self, items: List[Dict]) -> List[Dict]:
        """Pocket 아이템을 Notion 데이터베이스 속성에 맞게 처리"""
        processed_items = []
        for item in items:
            processed_item = {
                'Title': {'title': [{'text': {'content': item.get('title', '')}}]},
                'URL': {'url': item.get('url', '')},
                'Description': {'rich_text': [{'text': {'content': item.get('excerpt', '')}}]},
                'Category': {'multi_select': [{'name': tag} for tag in item.get('tags', [])]},
                'Favorite': {'checkbox': item.get('favorite', False)},
                'Status': {'select': {'name': item.get('status', 'unread')}},
                'Word Count': {'number': item.get('word_count', 0)},
                'Tags': {'multi_select': [{'name': tag} for tag in item.get('tags', [])]},
                'Date': {'date': {'start': item.get('date_added', '')}},
                'Author': {'rich_text': [{'text': {'content': item.get('author', '')}}]},
                'Source': {'select': {'name': 'pocket'}}
            }
            processed_items.append(processed_item)
        return processed_items

    def get_transcript(self, identifier):
        return []

    def get_pocket_auth_token(self):
        """
        초기 인증 및 액세스 토큰 발급을 위한 함수
        처음 한 번만 실행하면 됨
        """
        load_dotenv()
        consumer_key = os.getenv('POCKET_CONSUMER_KEY')
        if not consumer_key:
            raise ValueError("POCKET_CONSUMER_KEY를 환경 변수에 설정해주세요.")
        
        # Step 1: 요청 토큰 얻기
        request_token_url = "https://getpocket.com/v3/oauth/request"
        redirect_uri = "http://localhost:8080/"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Accept': 'application/json'
        }
        
        data = {
            'consumer_key': consumer_key,
            'redirect_uri': redirect_uri
        }
        
        response = requests.post(request_token_url, json=data, headers=headers)
        if response.status_code != 200:
            raise Exception(f"요청 토큰 획득 실패: {response.text}")
            
        request_token = response.json()['code']
        
        # Step 2: 사용자 인증 URL 생성
        auth_url = f"https://getpocket.com/auth/authorize?request_token={request_token}&redirect_uri={redirect_uri}"
        
        print(f"\n다음 URL을 브라우저에서 열어 인증을 완료해주세요:\n{auth_url}\n")
        input("인증을 완료한 후 Enter를 눌러주세요...")
        
        # Step 3: 액세스 토큰 얻기
        access_token_url = "https://getpocket.com/v3/oauth/authorize"
        data = {
            'consumer_key': consumer_key,
            'code': request_token
        }
        
        response = requests.post(access_token_url, json=data, headers=headers)
        if response.status_code != 200:
            raise Exception(f"액세스 토큰 획득 실패: {response.text}")
        
        result = response.json()
        access_token = result['access_token']
        username = result['username']
        
        print(f"\n인증 성공!")
        print(f"Username: {username}")
        print(f"\n.env 파일에 다음 내용을 추가해주세요:")
        print(f"POCKET_ACCESS_TOKEN={access_token}")
            
if __name__ == "__main__":
    pocket = PocketClient(DictConfig())
    pocket.get_pocket_auth_token()
