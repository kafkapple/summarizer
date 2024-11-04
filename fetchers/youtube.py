from typing import Dict, Optional, List, Any, Tuple
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import requests
import isodate
from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeFetcher:
    def __init__(self, config):
        self._init_youtube_client(config)
    
    def _init_youtube_client(self, config):
        """YouTube API 클라이언트 초기화"""
        self.api_key = config.api_keys['youtube']
        self.config = config
        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
        
        creds = self._get_or_refresh_credentials(SCOPES)
        self.youtube = build("youtube", "v3", credentials=creds)

    def _get_or_refresh_credentials(self, SCOPES):
        """인증 정보 가져오기 또는 갱신"""
        # 프로젝트 루트의 data 디렉토리 경로
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        token_file = os.path.join(data_dir, 'token.json')
        client_secret_file = os.path.join(data_dir, 'client_secret.json')
        
        creds = None
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"토큰 갱신 실패: {e}")
                    creds = self._run_auth_flow(client_secret_file, SCOPES)
            else:
                creds = self._run_auth_flow(client_secret_file, SCOPES)
            
            # data 디렉토리가 없으면 생성
            os.makedirs(data_dir, exist_ok=True)
            
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds

    def _run_auth_flow(self, client_secret_file, SCOPES):
        """인증 흐름 실행"""
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
        creds = flow.run_local_server(port=0)
        return creds

    def fetch_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """비디오 정보 및 자막 수집"""
        try:
            # 비디오 메타데이터 수집
            metadata = self.fetch_content(video_id)
            if not metadata:
                return None
            
            # 자막 수집
            transcript = self._fetch_transcript(video_id)
            if not transcript:
                return None
            
            return {
                **metadata,
                'text': transcript,
                'type': 'youtube'
            }
            
        except Exception as e:
            print(f"비디오 수집 실패: {e}")
            return None

    def fetch_playlist(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """플레이리스트 정보 및 모든 비디오 수집"""
        try:
            # 플레이리스트 메타데이터
            playlist_name = self.get_playlist_name(playlist_id)
            if not playlist_name:
                return None
            
            # 플레이리스트의 모든 비디오 수집
            videos = []
            for video in self._get_playlist_videos(playlist_id):
                try:
                    video_content = self.fetch_video(video['id'])
                    if video_content:
                        videos.append(video_content)
                except Exception as e:
                    print(f"비디오 수집 실패 ({video['id']}): {e}")
                    continue
            
            # 모든 비디오의 텍스트 결합
            combined_text = "\n\n".join(
                f"[{v['title']}]\n{v['text']}" 
                for v in videos
            )
            
            return {
                'title': playlist_name,
                'text': combined_text,
                'type': 'youtube_playlist',
                'videos': videos,
                'url': f"https://www.youtube.com/playlist?list={playlist_id}"
            }
            
        except Exception as e:
            print(f"플레이리스트 수집 실패: {e}")
            return None

    def fetch_content(self, video_id: str) -> Optional[Dict[str, Any]]:
        """비디오 메타데이터 수집"""
        try:
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
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
                'category': snippet.get("categoryId", "Unknown"),
                'tags': snippet.get("tags", []),
                'url': f"https://www.youtube.com/watch?v={video_id}"
            }
        except Exception as e:
            print(f"API 요청 실패: {e}")
            return None

    def _fetch_transcript(self, video_id: str) -> Optional[str]:
        """자막 수집 및 처리"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['ko', 'en']
            )
            
            text_parts = []
            for entry in transcript:
                text = entry['text'].strip()
                if text:
                    if not text[-1] in '.!?':
                        text += '.'
                    text_parts.append(text)
            print('Transcript fetched.')
            return ' '.join(text_parts)
            
        except Exception as e:
            print(f"자막 수집 실패: {e}")
            return None

    def _get_playlist_videos(self, playlist_id: str) -> List[Dict[str, str]]:
        """플레이리스트의 비디오 목록 수집"""
        videos = []
        try:
            request = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=playlist_id,
                maxResults=50
            )
            
            while request:
                response = request.execute()
                
                for item in response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    videos.append({
                        'id': video_id,
                        'title': item['snippet']['title']
                    })
                
                request = self.youtube.playlistItems().list_next(request, response)
                
        except Exception as e:
            print(f"플레이리스트 비디오 목록 수집 실패: {e}")
        
        return videos

    @staticmethod
    def parse_youtube_url(url: str) -> Tuple[Optional[str], bool]:
        """YouTube URL 파싱"""
        if "&si=" in url:
            url = url.split("&si=")[0] 

        if "list=" in url:
            id = url.split("list=")[-1]
            return id, True
        elif "v=" in url:
            id = url.split("v=")[-1]
            return id, False
        return None, False

    @staticmethod
    def _get_best_thumbnail(thumbnails: Dict) -> Optional[str]:
        """최적의 썸네일 URL 반환"""
        for res in ["high", "medium", "default"]:
            if res in thumbnails:
                return thumbnails[res].get("url")
        return None

    @staticmethod
    def _safe_get_count(stats: Dict, key: str, default: int = 0) -> int:
        """통계 정보 안전하게 가져오기"""
        try:
            return int(stats.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_playlist_name(self, playlist_id: str) -> str:
        """플레이리스트 이름 조회"""
        try:
            request = self.youtube.playlists().list(
                part='snippet',
                id=playlist_id
            )
            response = request.execute()

            if 'items' in response and len(response['items']) > 0:
                return response['items'][0]['snippet']['title']
            return ''
            
        except Exception as e:
            print(f"플레이리스트 이름 조회 실패: {e}")
            return ''