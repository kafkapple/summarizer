import requests
import isodate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from src.utils import Utils
import time
from typing import Optional, Dict, List, Tuple
import os
import logging

# 로거 설정
logger = logging.getLogger(__name__)

import requests
from omegaconf import DictConfig
from src.fetcher.base import MediaSource
from pathlib import Path
class YouTube(MediaSource):
    def __init__(self, cfg: DictConfig):
        self._init_youtube_client(cfg)
    
    def _init_youtube_client(self, cfg: DictConfig):
        """YouTube API 클라이언트 초기화"""
        self.api_key = cfg.api_keys['youtube']
        self.cfg = cfg
        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
        print("===========")
        creds = self._get_or_refresh_credentials(SCOPES)
        self.youtube = build("youtube", "v3", credentials=creds)

    def _get_or_refresh_credentials(self, scopes):
        """인증 정보를 가져오거나 갱신합니다."""
        
        try:
            token_file = 'token.json'
            creds = None
            print(self.cfg.paths.base)
            
            token_file = Path(self.cfg.paths.base) / Path(token_file)
            secrete_file = Path(self.cfg.paths.base) / Path('client_secret.json')
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, scopes)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        secrete_file, scopes)
                    creds = flow.run_local_server(port=0)
                
                # 인증 정보 저장
                os.makedirs(os.path.dirname(token_file), exist_ok=True)
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            return creds
            
        except Exception as e:
            print(f"인증 정보 처리 중 오류 발생: {e}")
            return None
    @staticmethod
    def parse_youtube_url(url: str) -> Tuple[str, bool]:
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
    def get_playlist_name(self, playlist_id: str) -> str:
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

    def fetch_content(self, video_id: str) -> Optional[Dict]:
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={self.api_key}"
        response = requests.get(url)
        data = response.json()
        
        if not data.get("items"):
            return None
            
        item = data["items"][0]
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        stats = item.get("statistics", {})
        
        # 내부 키를 사용하여 데이터 구성
        processed_data = {
            'title': snippet.get("title"),
            'channel': snippet.get("channelTitle"),
            'category': [self.fetch_category_name(snippet.get("categoryId", "Unknown"))], # 리스트 형태 유지
            'tags': snippet.get("tags", []), # 리스트 형태 유지
            'published_date': snippet.get("publishedAt"),
            'duration': str(isodate.parse_duration(content_details["duration"])) if content_details.get("duration") else None,
            'like_count': self._safe_get_count(stats, "likeCount"),
            'comment_count': self._safe_get_count(stats, "commentCount"),
            'view_count': self._safe_get_count(stats, "viewCount"), # View Count 추가
            'description': snippet.get('description'),
            'thumbnail': self._get_best_thumbnail(snippet.get("thumbnails", {})),
            'url': f"https://www.youtube.com/watch?v={video_id}",
            # 'llm_model', 'output_language' 등은 config에서 직접 접근 가능하므로 여기서 제외 가능
            # 'keywords', 'summary' 등은 Summarizer 단계에서 추가됨
        }
        
        # None 값 제거 (선택 사항, logger에서 처리 가능)
        # processed_data = {k: v for k, v in processed_data.items() if v is not None}
        
        return processed_data

    def get_subscribed_channels_save(self, export_result):
        export_result.change_id(self.cfg.notion['youtube_ch_id'])
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

    def get_transcript(self, video_id: str) -> str:
        """YouTube 동영상의 스크립트 가져오기"""
        try:
            # 스크립트 가져오기 시도
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en']) # 한국어, 영어 순서로 시도
            
            # 반환된 결과 처리 (딕셔너리 리스트 예상)
            if isinstance(transcript_list, list):
                full_transcript = " ".join([item['text'] for item in transcript_list])
                return full_transcript
            else:
                # 예상치 못한 타입 처리 (예: FetchedTranscript 객체)
                logger.warning(f"Unexpected transcript type received for video {video_id}: {type(transcript_list)}. Attempting string conversion.")
                # 강제 문자열 변환 대신, 객체 구조 확인 후 텍스트 추출 시도 (어려움)
                # 우선 빈 문자열 반환 또는 오류 발생시키는 것이 나을 수 있음
                return "" # 또는 raise TypeError("Unexpected transcript format")
                
        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video: {video_id}")
            return ""
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video: {video_id} in specified languages ('ko', 'en')")
            # 자동 생성 스크립트 시도 (선택적)
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id).find_generated_transcript(['ko', 'en']).fetch()
                if isinstance(transcript_list, list):
                    return " ".join([item['text'] for item in transcript_list])
            except Exception as gen_e:
                logger.warning(f"Could not fetch generated transcript for {video_id}: {gen_e}")
            return ""
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            return ""

    def fetch_category_name(self, category_id):
        url = f"https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&id={category_id}&key={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if "items" in data and data["items"]:
            return data["items"][0]["snippet"]["title"]
        return "Unknown"

    def fetch_playlist_videos(self, playlist_id: str) -> List[Dict]:
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
