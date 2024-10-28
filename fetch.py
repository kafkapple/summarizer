from abc import ABC, abstractmethod
import requests
import isodate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import Utils
import time
from datetime import datetime
from typing import Optional, Dict, List
import os
import random
from abc import ABC, abstractmethod
import cloudscraper
import re
import requests


# Example usage:
class MediaSource(ABC):
    @abstractmethod
    def fetch_content(self, identifier):
        pass
    # @abstractmethod
    # def get_transcript(self, identifier):
    #     pass

class WebContent():
    def __init__(self, config):
        self.config = config
        self.headers = self._get_random_headers()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.api_token = config.DIFFBOT_API_TOKEN
        self.base_url = "https://api.diffbot.com/v3/article"
        print('#'*7+'WebContent init'+'#'*7)

        
    def extract_text(self, url):
        """
        Extracts the main text and essential information from a given URL.
        
        :param url: str - The URL of the page to analyze.
        :return: dict - Contains 'title', 'author', 'date', and 'text' if extraction is successful.
        """
        params = {
            'token': self.api_token,
            'url': url,
            'discussion': 'false'  # Optional parameter to disable extraction of comments.
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check if article data is available in the response
            if 'objects' in data and len(data['objects']) > 0:
                article_data = data['objects'][0]
                extracted_content = {
                    'title': article_data.get('title', ''),
                    'author': article_data.get('author', ''),
                    'date': article_data.get('date', ''),
                    'text': article_data.get('text', '')
                }
                return extracted_content
            else:
                print("No article content found in the response.")
                return {}
        
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return {}
    

    def clean_text(self, raw_text):
        """
        Clean the extracted text by removing HTML tags and unnecessary whitespace.
        
        :param raw_text: str - Raw text containing HTML tags and unwanted characters.
        :return: str - Cleaned text with only the main content.
        """
        # 1. Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text(separator=" ")

        # 2. Remove unwanted characters or patterns (e.g., extra spaces, newlines)
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _get_random_headers(self):
        """랜덤 User-Agent와 함께 요청 헤더 생성"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1',
            'Cache-Control': 'max-age=0'
        }

    
    def _is_valid_content(self, content):
        """컨텐츠가 유효한지 확인"""
        if not content or len(content) < 100:
            return False
        
        # 봇 감지/보안 체크 키워드
        security_keywords = [
            'cf-browser-verification',
            'security check',
            'captcha',
            'robot verification',
            'access denied',
            'please verify',
            '차단된 페이지',
            '접근이 거부되었습니다'
        ]
        
        content_lower = content.lower()
        return not any(keyword in content_lower for keyword in security_keywords)


class YouTube(MediaSource):
    def __init__(self, config):
        self._init_youtube_client(config)
    
    def _init_youtube_client(self, config):
        """YouTube API 클라이언트 초기화"""
        self.api_key = config.YOUTUBE_API_KEY
        self.config = config
        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
        creds = self._get_or_refresh_credentials(SCOPES)
        self.youtube = build("youtube", "v3", credentials=creds)

    def _get_or_refresh_credentials(self, SCOPES):
        """인증 보 가져오기 또는 갱신"""
        token_file = os.path.join(self.config.src_path, 'token.json')
        client_secret_file = os.path.join(self.config.src_path, 'client_secret.json')
        
        creds = None
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    @staticmethod
    def parse_youtube_url(url):
        if "&si=" in url:
            url = url.split("&si=")[0] 

        if "list=" in url:
            id = url.split("list=")[-1]
            return id, True
        elif "v=" in url:
            id = url.split("v=")[-1]
            return id, False
        return 
    @staticmethod
    def _get_best_thumbnail(thumbnails):
        """최적의 썸네일 URL 반환"""
        for res in ["high", "medium", "default"]:
            if res in thumbnails:
                return thumbnails[res].get("url")
        return None

    @staticmethod
    def _safe_get_count(stats, key, default=0):
        """통계 정보 안전하게 가져오기"""
        try:
            return int(stats.get(key, default))
        except (ValueError, TypeError):
            return default
    def get_playlist_name(self, playlist_id):
        # 플레이리스트 정보 요청
        request = self.youtube.playlists().list(
            part='snippet',
            id=playlist_id
        )
        response = request.execute()

        # 플레이리스트 이름 추출
        if 'items' in response and len(response['items']) > 0:
            playlist_name = response['items'][0]['snippet']['title']
            return playlist_name
        else:
            return ''

    def fetch_content(self, video_id):
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={self.api_key}"
        response = requests.get(url)
        data = response.json()
        
        if not data.get("items"):
            return None
            
        item = data["items"][0]
        snippet = item["snippet"]
        content_details = item["contentDetails"]
        stats = item["statistics"]
        
        return {
            'title': snippet["title"],
            'channel_title': snippet["channelTitle"],
            'duration': str(isodate.parse_duration(content_details["duration"])),
            'publish_date': snippet["publishedAt"],
            'view_count': self._safe_get_count(stats, "viewCount"),
            'like_count': self._safe_get_count(stats, "likeCount"),
            'comment_count': self._safe_get_count(stats, "commentCount"),
            'description': snippet['description'],
            'thumbnail': self._get_best_thumbnail(snippet.get("thumbnails", {})),
            'chapters': [],
            'category': self.fetch_category_name(snippet.get("categoryId", "Unknown")),
            'tags': snippet.get("tags", []),
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'model': self.config.GPT_MODEL,
            'output_language': self.config.OUTPUT_LANGUAGE,
        }

    def get_subscribed_channels_save(self, export_result):
        export_result.change_id(self.config.NOTION_DB_YOUTUBE_CH_ID)
        channels_info = []
        next_page_token = None
        
        while True:
            channels_batch = self._fetch_subscription_batch(next_page_token)
            if not channels_batch:
                break
                
            for channel in channels_batch:
                channel_info = self._process_channel_info(channel)
                channels_info.append(channel_info)
                export_result.save_to_notion_ch(channel_info)
            
            next_page_token = channels_batch.get('nextPageToken')
            if not next_page_token:
                break
                
        Utils.save_file(channels_info, 'youtube_channels.csv')
        return channels_info

    def _fetch_subscription_batch(self, page_token=None):
        """구독 채널 배치 가져오기"""
        request = self.youtube.subscriptions().list(
            part='snippet',
            mine=True,
            maxResults=50,
            pageToken=page_token
        )
        return request.execute()

    def _process_channel_info(self, channel):
        """채널 정보 처리"""
        ch_snip = channel['snippet']
        ch_stat = channel.get('statistics', {})
        
        return {
            'Title': ch_snip['title'],
            'Subscribers': ch_stat.get('subscriberCount', 'N/A'),
            'View Count': ch_stat.get('viewCount', 'N/A'),
            'Video Count': ch_stat.get('videoCount', 'N/A'),
            'Published At': ch_snip['publishedAt'],
            'Description': ch_snip['description'],
            'URL': ch_snip.get('customUrl', 'No custom URL available'),
            'Thumbnail': self._get_best_thumbnail(ch_snip.get("thumbnails", {})),
            'Country': ch_snip.get('country', 'N/A'),
            'Category': self._get_channel_categories(channel.get('topicDetails', {}))
        }

    @staticmethod
    def _get_channel_categories(topic_details):
        """채널 카테고리 추출"""
        try:
            category_urls = topic_details.get('topicCategories', [])
            return [url.split("/")[-1].replace("_", " ") for url in category_urls]
        except:
            return []

    def get_transcript(self, video_id, preferred_languages=['ko', 'en', 'ja']):
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # 1. 수동 자막 시도
                for lang in preferred_languages:
                    try:
                        transcript = transcript_list.find_transcript([lang])
                        if transcript:
                            # Utils의 preprocess_text 사용
                            return Utils.preprocess_text(transcript.fetch(), clean_tags=True)
                    except NoTranscriptFound:
                        continue
                
                # 2. 자동 생성 자막 시도 (모든 언어)
                try:
                    generated_transcripts = transcript_list.find_generated_transcript(['ko', 'en', 'ja', 'auto'])
                    if generated_transcripts:
                        return Utils.preprocess_text(generated_transcripts.fetch())
                except NoTranscriptFound:
                    pass
                
                # 3. 사용 가능한 모든 자막 확인
                for transcript in transcript_list:
                    try:
                        return Utils.preprocess_text(transcript.fetch())
                    except NoTranscriptFound:
                        continue
            
            except TranscriptsDisabled:
                print(f"자막이 비활성화됨: {video_id}")
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"자막 추출 재시도 중... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                print(f"자막 추출 최종 실패: {e}")
                return None
        
        return None

    def fetch_category_name(self, category_id):
        url = f"https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&id={category_id}&key={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if "items" in data and data["items"]:
            return data["items"][0]["snippet"]["title"]
        return "Unknown"

    def fetch_playlist_videos(self, playlist_id):
        videos = []
        nextToken = None
        while True:
            request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=nextToken
            )
            response = request.execute()
            for item in response['items']:
                videos.append({'video_id': item['snippet']['resourceId']['videoId'], 'title': item['snippet']['title']})
            nextToken = response.get('nextPageToken')
            if not nextToken:
                break

        return videos

class PocketClient(WebContent):
    def __init__(self, config):
        super().__init__(config)
        self.consumer_key = config.POCKET_CONSUMER_KEY
        self.access_token = config.POCKET_ACCESS_TOKEN
        self.base_url = "https://getpocket.com/v3"
        self.all_items = []
        print('#'*7+'PocketClient init'+'#'*7)

    def fetch_content(self, batch_size=500, state='all', detail_type='complete', sort='newest', offset=0, tags=None):
        """
        1. 전체 아이템 목록 먼저 수집
        2. 각 아이템별로 순차적으로 콘텐츠 수집 및 요약
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
                # 요청 간 딜레이
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
    
    def process_items(self, items):
        processed = []
        for item in items:
            processed_item = {
                'title': item.get('resolved_title') or item.get('given_title'),
                'url': item.get('resolved_url') or item.get('given_url'),
                'excerpt': item.get('excerpt'),
                'cover': item.get('top_image_url'),
                'tags': list(item.get('tags', {}).keys()),
                'time_added': datetime.fromtimestamp(int(item.get('time_added', 0))).strftime('%Y-%m-%d %H:%M:%S'),
                'favorite': item.get('favorite') == '1',
                'status': 'archived' if item.get('status') == '1' else 'unread',
                'lang': item.get('lang'),
                'word_count': item.get('word_count'),
                'has_video': item.get('has_video')
            }
            processed.append(processed_item)
            
        return processed

    def get_transcript(self, identifier):
        return []

class RaindropClient(WebContent):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.RAINDROP_TOKEN
        self.base_url = "https://api.raindrop.io/rest/v1/"
        self.filename_collection = 'raindrop_collections.csv'
        self.filename_item = 'raindrop_items.csv'

    def fetch_content(self, identifier):
        """Raindrop API를 통해 아이템 가져오기"""
        items = self._fetch_raindrop_items(identifier)
        
        # 각 아이템의 웹 컨텐츠 가져오기
        for item in items:
            try:
                item['content'] = self.fetch_web_content(item['link'])
            except Exception as e:
                print(f"Error fetching content for {item['link']}: {e}")
                item['content'] = None
                
        return items

    def _fetch_raindrop_items(self, identifier):
        """Raindrop API에서 아이템 가져오기"""
        url = f"{self.base_url}raindrops/{identifier}"
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        response.raise_for_status()
        return response.json().get('items', [])

    def scrape_save(self):
        ## Export result
        bookmarks = self.get_bookmarks()
        collections = self.get_collections()
        items = self.get_item_from_collection(collections)
        try:
            Utils.save_file(collections, self.filename_collection)
            Utils.save_file(items, self.filename_item)
        except:
            print('save error')
 
    def get_collections(self):
        url = self.base_url +"collections"
        collections = self.get_response(url)

        print('Collections: ', len(collections))
        results = []
        for collect in collections:
            dict_collect = {
                'title': collect['title'],
                'id': collect['_id'],
                'count': collect['count'],
                'expanded': collect['expanded'],
                #'access': collect['access'],
                'parent':''
                }

            print(dict_collect)
            results.append(dict_collect)

        collections_child = self.get_child_collections()

        results+=collections_child
        return results

    def get_child_collections(self):
        url = self.base_url +"collections/childrens"
        collections = self.get_response(url)
        print('Collections: ', len(collections))
        results = []
        for collect in collections:
            dict_collect = {
                'title': collect['title'],
                'id': collect['_id'],
                'count': collect['count'],
                'expanded': collect['expanded'],
               # 'access': collect['access'],
                'parent':collect['parent']
                }
            try: 
                dict_collect['parent'] =collect['parent']['$ref']+ '_'+str(collect['parent']['$id'])
            except:
                print('parent parsing error.')
            print(dict_collect)
            results.append(dict_collect)
        return results
        
    def get_item_from_collection(self, collections):
        items = []
        for collect in tqdm(collections):
            id = collect['id']
            url = self.base_url+f'raindrops/{id}'
            collect_items = self.get_response(url)
            #collect_items = self.get_item_from_collection(collect['id'])
            print('Items in the collection: ', len(collect_items))
            for item in tqdm(collect_items):
                id_item = item['_id']
                url = self.base_url +"raindrop/{id_item}"
                item_info = self.get_response(url)
                dict_item ={
                    'collection': collect['title'],
                    'title': item['title'],
                    'type': item['type'],
                    'excerpt': item['excerpt'],
                    'note': item['note'],
                    'link': item['link'],
                    'id': id_item,
                    'cover': item['cover']
                }
                items.append(dict_item)
        return items
    


#python -c "from pocket_sync import get_pocket_auth_token; get_pocket_auth_token()"











