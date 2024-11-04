from typing import Optional
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import requests

def get_pocket_auth_token():
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
    
class OAuthHandler(BaseHTTPRequestHandler):
    """OAuth 콜백 핸들러"""
    
    def do_GET(self):
        """GET 요청 처리"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # 인증 코드 추출
        if 'code=' in self.path:
            auth_code = self.path.split('code=')[1].split('&')[0]
            self.server.auth_code = auth_code
            
            response = """
            <html><body>
            <h1>인증이 완료되었습니다.</h1>
            <p>이 창을 닫으셔도 됩니다.</p>
            </body></html>
            """
        else:
            response = """
            <html><body>
            <h1>인증에 실패했습니다.</h1>
            <p>다시 시도해주세요.</p>
            </body></html>
            """
        
        self.wfile.write(response.encode())

class AuthManager:
    """인증 관리자"""
    
    def __init__(self, config):
        self.config = config
        self.credentials_path = Path(config.paths['data']) / 'credentials.json'
    
    def authenticate(self, service: str) -> Optional[str]:
        """서비스 인증 처리"""
        # 저장된 인증 정보 확인
        if self.credentials_path.exists():
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
                if service in credentials:
                    return credentials[service]
        
        # OAuth 인증 프로세스
        auth_code = self._run_oauth_flow(service)
        if auth_code:
            # 인증 정보 저장
            credentials = {}
            if self.credentials_path.exists():
                with open(self.credentials_path, 'r') as f:
                    credentials = json.load(f)
            
            credentials[service] = auth_code
            
            with open(self.credentials_path, 'w') as f:
                json.dump(credentials, f)
            
            return auth_code
        
        return None
    
    def _run_oauth_flow(self, service: str) -> Optional[str]:
        """OAuth 인증 플로우 실행"""
        # 임시 서버 시작
        server = HTTPServer(('localhost', 8080), OAuthHandler)
        server.auth_code = None
        
        # 브라우저로 인증 페이지 열기
        auth_url = self._get_auth_url(service)
        webbrowser.open(auth_url)
        
        # 콜백 대기
        while server.auth_code is None:
            server.handle_request()
        
        return server.auth_code
    
    def _get_auth_url(self, service: str) -> str:
        """서비스별 인증 URL 반환"""
        if service == 'notion':
            return f"https://api.notion.com/v1/oauth/authorize?client_id={self.config.api_keys['notion_client_id']}&response_type=code&owner=user"
        # 다른 서비스 추가 가능
        raise ValueError(f"지원하지 않는 서비스: {service}") 