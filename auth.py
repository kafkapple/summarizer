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