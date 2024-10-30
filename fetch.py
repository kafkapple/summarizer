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
from newspaper import Article, Config
from readability import Document
import nltk
nltk.download('punkt_tab')
# Example usage:
# class MediaSource(ABC):
#     @abstractmethod
#     def fetch_content(self, identifier):
#         pass
#     # @abstractmethod
#     # def get_transcript(self, identifier):
#     #     pass

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
        print('#'*7+' WebContent init '+ '#'*7)
    
    @staticmethod
    def is_meaningful_content(text):
        # 1. 특정 키워드가 과도하게 반복될 경우 무의미한 텍스트로 간주합니다.
        exclusion_keywords = ["공지", "회원가입", "게시판", "공지사항", "로그인", "홈", "이용약관", "개인정보"]
        keyword_threshold = 5  # 각 키워드가 5번 이상 반복되면 무의미한 텍스트로 간주
        for keyword in exclusion_keywords:
            if text.count(keyword) >= keyword_threshold:
                print(f'Exclusion keyword found: {keyword}')
                return False

        # 2. 특수 기호가 연속적으로 등장하는 경우 무의미한 텍스트로 간주합니다.
        special_pattern = re.compile(r"[\[\]>|]{2,}")
        if len(special_pattern.findall(text)) >= 10:
            print('Special pattern found.')
            return False

        # 3. 정보 밀도를 판단하여 대부분이 고유명사나 제목일 경우 무의미한 텍스트로 간주합니다.
        # 여기서는 ">"와 "|" 기호를 기준으로 제목/링크가 과도한 경우를 필터링
        title_like_patterns = re.findall(r"\b(?:\S+>|\|\S+)\b", text)
        if len(title_like_patterns) > 15:  # 제목 패턴이 15개 이상이면 무의미한 텍스트로 간주
            
            return False

        # # 4. 정보 밀도가 낮은 경우 필터링합니다.
        # word_count = len(re.findall(r"\b\w+\b", text))
        # meaningful_words = len(re.findall(r"\b(뉴스|영상|분석|토론|기술|개발|경제|정책|사회)\b", text))
        # if word_count > 200 and meaningful_words < 10:
        #     print('Low information density.')
        #     return False

        return True

    def extract_text_readability(self, url):
        try:
            headers = {'User-Agent': self.headers['User-Agent']}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
                
            # 원본 HTML 파싱
            original_soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Readability 시도
            doc = Document(response.text)
            content = doc.summary()
            
            # 2. Readability 실패시 대체 방법
            if "ruthless removal did not work" in content or len(content) < 100:
                print("Readability fallback: using alternative extraction method")
                
                # 불필요한 태그 제거
                for tag in original_soup.find_all(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside']):
                    tag.decompose()
                
                # 주요 콘텐츠 영역 찾기
                content_candidates = [
                    original_soup.find('article'),
                    original_soup.find('main'),
                    original_soup.find(class_='content'),
                    original_soup.find(class_='post-content'),
                    original_soup.find(class_='entry-content'),
                    original_soup.find(id='content'),
                    original_soup.find(class_='article-content'),
                    original_soup.find(role='main')
                ]
                
                content = None
                for candidate in content_candidates:
                    if candidate:
                        content = candidate
                        break
                
                if not content:
                    # 가장 많은 <p> 태그를 포함한 div 찾기
                    max_p_count = 0
                    max_div = None
                    for div in original_soup.find_all('div'):
                        p_count = len(div.find_all('p'))
                        if p_count > max_p_count:
                            max_p_count = p_count
                            max_div = div
                    
                    if max_div:
                        content = max_div
                    else:
                        content = original_soup
            
            # 최종 텍스트 추출
            if isinstance(content, str):
                soup = BeautifulSoup(content, 'html.parser')
            else:
                soup = content
                
            # 남은 불필요 요소 제거
            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
                tag.decompose()
                
            # 텍스트 추출 및 정제
            paragraphs = []
            for p in soup.find_all(['p', 'article', 'section']):
                text = p.get_text().strip()
                if len(text) > 20:  # 짧은 텍스트 필터링
                    paragraphs.append(text)
            
            text = '\n\n'.join(paragraphs)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 사이트 정보 추출
            site_info = self._extract_site_info(url, soup=original_soup)
            
            article_data = {
                'title': doc.title() or '',
                'text': text,
                'source': {
                    'method': 'readability',
                    **site_info
                }
            }
            
            if len(text) < 100:
                print("Warning: Extracted text is too short")
                return None
                
            return article_data
            
        except Exception as e:
            print(f"Readability 추출 오류: {str(e)}")
            return None
    def extract_text_soup(self, url):
        try:
            header = {'User-Agent': self.headers['User-Agent']}
            response = requests.get(url, header)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            # Get text
            text = ' '.join(soup.stripped_strings)
            text = {'text': text}
            return text
        except Exception as e:
            print(f'Error parsing article at {url}: {e}')
    def extract_text_newspaper(self, url):
        try:
            config = Config()
            config.browser_user_agent = self.headers['User-Agent']
            config.request_timeout = 10
            
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            try:
                article.nlp()
            except:
                print("NLP 처리 실패")
            
            # 사이트 정보 추출
            site_info = self._extract_site_info(
                url, 
                response_text=article.html
            )
            
            article_data = {
                'title': article.title,
                'text': article.text,
                'source': {
                    'method': 'newspaper',
                    **site_info,
                    'canonical_link': article.canonical_link
                }
            }
            
            return article_data
            
        except Exception as e:
            print(f"Newspaper 추출 오류: {str(e)}")
            return None

    def extract_text_diffbot(self, url):
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
                    'text': article_data.get('text', ''),
                    'source': {
                        'method': 'diffbot',
                        'site_name': article_data.get('siteName', ''),
                        'domain': article_data.get('resolvedDomain', ''),
                        'url': url,
                        'publisher_region': article_data.get('publisherRegion', ''),
                        'publisher_country': article_data.get('publisherCountry', '')
                    }
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

    def _extract_site_info(self, url, soup=None, response_text=None):
        """공통 사이트 정보 추출 메서드"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            if soup is None and response_text:
                soup = BeautifulSoup(response_text, 'html.parser')
            
            site_name = None
            if soup:
                # 다양한 메타 태그에서 사이트 이름 찾기
                meta_tags = [
                    ('property', 'og:site_name'),
                    ('name', 'application-name'),
                    ('name', 'publisher'),
                    ('name', 'author'),
                    ('property', 'twitter:site')
                ]
                
                for attr, value in meta_tags:
                    meta = soup.find('meta', {attr: value})
                    if meta and meta.get('content'):
                        site_name = meta.get('content')
                        break
                
                # 대체 방법: title 태그에서 추출
                if not site_name and soup.title:
                    site_parts = soup.title.string.split('|')
                    if len(site_parts) > 1:
                        site_name = site_parts[-1].strip()
            
                # 도메인에서 사이트 이름 추출 (다른 방법 실패 시)
                if not site_name:
                    site_name = domain.split('.')[0].capitalize()
                
                return {
                    'site_name': site_name,
                    'domain': domain,
                    'url': url
                }
                
        except Exception as e:
            print(f"사이트 정보 추출 오류: {e}")
            return {
                'site_name': '',
                'domain': urlparse(url).netloc,
                'url': url
            }

    def save_extracted_text(self, article_data: dict, save_path: str = None) -> str:
        """
        추출된 텍스트를 파일로 저장
        
        Args:
            article_data: 추출된 기사 데이터
            save_path: 저장할 경로 (기본값: None, 이 경우 기본 경로 사용)
        
        Returns:
            str: 저장된 파일의 경로
        """
        try:
            if not article_data or 'text' not in article_data:
                print("저장할 텍스트가 없습니다.")
                return None
                
            # 기본 저장 경로 설정
            if not save_path:
                save_path = os.path.join(self.config.save_path, 'extracted_texts')
            os.makedirs(save_path, exist_ok=True)
            
            # 파일명 구성 요소 준비
            source_info = article_data.get('source', {})
            domain = source_info.get('domain', '').replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extractor = source_info.get('method', 'unknown')  # 추출 방법
            
            # 제목 정제
            title = article_data.get('title', '').strip()
            title = re.sub(r'[\\/*?:"<>|]', '', title)
            title = title[:50] if title else 'untitled'  # 제목 길이 제한
            
            # 파일명 형식: domain_title_extractor_timestamp.txt
            filename = f"{domain}_{title}_{extractor}_{timestamp}.txt"
            filepath = os.path.join(save_path, filename)
            
            info ={'source_info': source_info, 'extractor': extractor, 'title': title, 'timestamp': timestamp, 'filepath': filepath}
            # 텍스트 파일 작성
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article_data.get('title', '')}\n")
                f.write(f"Source: {source_info.get('site_name', '')}\n")
                f.write(f"URL: {source_info.get('url', '')}\n")
                f.write(f"Extracted Date: {timestamp}\n")
                f.write(f"Method: {extractor}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(article_data['text'])
            
            print(f"\n텍스트 파일이 저장되었습니다: {filepath}\n")
            return info
            
        except Exception as e:
            print(f"텍스트 저장 중 오류 발생: {e}")
            return None

    def fetch_web_content(self, url: str) -> Optional[Dict]:
        """
        웹 컨텐츠 추출 및 저장
        """
        try:
            # 기존 추출 로직
            for extractor in [
                self.extract_text_diffbot,
                self.extract_text_readability,
                self.extract_text_newspaper,
                self.extract_text_soup
            ]:
                try:
                    content = extractor(url)
                    text = content.get('text' , '')
                    text = self.clean_text(text)
                    meaningful = self.is_meaningful_content(text)
                    char_count = len(text)
                    if content and char_count > 500 and meaningful:
                        # 텍스트 파일로 저장
                        filepath = self.save_extracted_text(content)
                        saved_path = filepath
                        if saved_path:
                            content['saved_file'] = saved_path
                        content['char_count'] = char_count
                     
                        return content
                    else:
                        print(f'Not meaningful content.')
                except Exception as e:
                    print(f"{extractor.__name__} 실패: {e}")
                    continue
                    
            return None
            
        except Exception as e:
            print(f"컨텐츠 추출 실패: {e}")
            return None

class YouTube():#MediaSource):
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
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"토큰 갱신 실패: {e}")
                    creds = self._run_auth_flow(client_secret_file, SCOPES)
            else:
                creds = self._run_auth_flow(client_secret_file, SCOPES)
            
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds

    def _run_auth_flow(self, client_secret_file, SCOPES):
        """인증 흐름 실행"""
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
        creds = flow.run_local_server(port=0)
        return creds

    def fetch_content(self, video_id):
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
                'chapters': [],
                'category': self.fetch_category_name(snippet.get("categoryId", "Unknown")),
                'tags': snippet.get("tags", []),
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'model': self.config.GPT_MODEL,
                'output_language': self.config.OUTPUT_LANGUAGE,
            }
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
            return None

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
#pocket
# fetch_content ->_get_items_with_params -> get_all_items -> get_items
class PocketClient(WebContent):
    def __init__(self, config):
        super().__init__(config)
        self.consumer_key = config.POCKET_CONSUMER_KEY
        self.access_token = config.POCKET_ACCESS_TOKEN
        self.base_url = "https://getpocket.com/v3"
        self.all_items = []
        print('#'*7+' PocketClient init '+ '#'*7)

    def fetch_items(self, batch_size=500, state='all', detail_type='complete', sort='newest', offset=0, tags=None):
        """
        1. 전체 아이템 목록 먼저 수
        2. 각 아이템별로 순차적으로 콘텐츠 수집 및 요약
        """
        # sort: newest, oldest, title, site
        # 1. 전체 아이템 목록 수집
        items = self._get_items_with_params(batch_size, state, detail_type, sort, offset, tags)
        processed_items = self._process_items(items)
        
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
        
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/get", json=params, headers=self.headers)
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
            
            print(f"\n- Retrieved {len(items_batch)} items. Total: {total_items}")
            
            if max_items and total_items >= max_items:
                self.all_items = self.all_items[:max_items]
                break
            
            params['offset'] += batch_size  # 다음 배치를 위해 offset 증가
           # self.print_pocket_items(items_batch)
            time.sleep(1)
            if len(self.all_items) % 500 == 0:
                print(f'n: {len(self.all_items)}\n')
        return self.all_items
    
    def _process_items(self, items):
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
                #'has_video': item.get('has_video')
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
        self.all_items = []
        print('#'*7+' RaindropClient init '+ '#'*7)

    def fetch_items(self, collection_name=None, tags=None, favorite=None):
        """Raindrop API를 통해 아이템 목록화 및 주요 정보 획득"""
        collection_id = self._get_collection_id_by_name(collection_name) if collection_name else None
        params = {}
        if tags:
            params['tag'] = tags
        if favorite is not None:
            params['favorite'] = favorite

        items = self._get_items(collection_id, params)
        processed_items = self._process_items(items)
        print(f"총 {len(processed_items)}개의 아이템을 처리합니다.")
        return processed_items

    def _get_collection_id_by_name(self, collection_name):
        """컬렉션 이름으로 컬렉션 ID를 가져옵니다."""
        url = f"{self.base_url}collections"
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        response.raise_for_status()
        collections = response.json().get('items', [])
        
        for collection in collections:
            if collection.get('title') == collection_name:
                return collection.get('_id')
        
        print(f"Collection '{collection_name}' not found.")
        return None

    def _get_items(self, collection_id, params):
        """Raindrop API에서 아이템 가져오기"""
        if collection_id:
            url = f"{self.base_url}raindrops/{collection_id}"
        else:
            url = f"{self.base_url}raindrops"

        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"}, params=params)
        response.raise_for_status()
        return response.json().get('items', [])

    def _process_items(self, items):
        """아이템의 주요 정보 처리"""
        processed = []
        
        
        for item in items:
            created_date_str = item.get('created', 'No Date')
            created_date = datetime.fromisoformat(created_date_str.replace('Z', '+00:00'))
            
            processed_item = {
                'title': item.get('title', 'No title'),
                'url': item.get('link'),
                'excerpt': item.get('excerpt'),
                'time_added': created_date.isoformat(),
                'tags': item.get('tags', []),
                'favorite': item.get('important', False),
                'status': item.get('status', 'Unknown'),
                'lang': item.get('language', 'Unknown'),
                'collection': item.get('collection', {}).get('title', 'Unknown'),
                'source': item.get('domain', 'Unknown')
            }
            processed.append(processed_item)
        return processed

#python -c "from pocket_sync import get_pocket_auth_token; get_pocket_auth_token()"











