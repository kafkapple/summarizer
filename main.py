import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
# 프로젝트 루트 경로를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import argparse
import logging
from datetime import datetime

from summarizer.config.settings import Settings
from summarizer.core.summarizer import Summarizer
from summarizer.fetchers.content import WebContentFetcher
from summarizer.storage.notion import NotionClient

class SummarizerApp:
    """요약 애플리케이션 메인 클래스"""
    
    def __init__(self):
        self._setup_logging()
        self.settings = Settings()
        self.fetcher = WebContentFetcher(self.settings)
        self.summarizer = Summarizer(self.settings)
        self.notion = NotionClient(self.settings)
        self.logger = logging.getLogger(__name__)
    
    def run(self, url: str = None, save_to_notion: bool = True):
        """
        요약 프로세스 실행
        
        Args:
            url: 처리할 컨텐츠 URL (없으면 사용자 입력 요청)
            save_to_notion: Notion 저장 여부
        """
        if url is None:
            url = 'https://www.youtube.com/watch?v=ZUwWpNEu8-k'#self._get_url_from_user()
        
        try:
            # 1. 컨텐츠 수집
            self.logger.info(f"컨텐츠 수집 시작: {url}")
            content = self.fetcher.fetch(url)
            if not content:
                self.logger.error("컨텐츠 수집 실패")
                return None
            
            # 2. 요약 생성
            self.logger.info("요약 생성 시작")
            summary = self.summarizer.summarize(
                text=content['text'],
                title=content['title']
            )
            if not summary:
                self.logger.error("요약 생성 실패")
                return None
            
            # 3. 결과 데이터 준비
            result = {
                'title': content['title'],
                'url': url,
                'type': content.get('type', 'web'),
                'metadata': content.get('metadata', {}),
                'summary': summary
            }
            
            # 4. 결과 저장
            self._save_result(result, save_to_notion)
            
            return result
            
        except Exception as e:
            self.logger.error(f"처리 중 오류 발생: {e}")
            return None
    
    def _get_url_from_user(self) -> str:
        """사용자로부터 URL 입력 받기"""
        while True:
            url = input("\n요약할 컨텐츠의 URL을 입력하세요: ").strip()
            if url:
                return url
            print("URL을 입력해주세요!")

    def _save_result(self, result: Dict[str, Any], save_to_notion: bool):
        """결과 저장"""
        # 로컬 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"summary_{timestamp}.json"
        
        save_path = self.settings.paths['results'] / filename
        self._save_json(result, save_path)
        
        # Notion 저장
        if save_to_notion:
            content_type = result.get('type', 'web')
            database_id = self.settings.notion['database_ids'].get(content_type)
            
            if database_id:
                self.logger.info("Notion 저장 시작")
                if self.notion.save(result, database_id):
                    self.logger.info("Notion 저장 완료")
                else:
                    self.logger.error("Notion 저장 실패")
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path(__file__).parent.parent / 'data' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"summarizer_{datetime.now():%Y%m%d}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def _save_json(data: Dict, path: Path):
        """JSON 파일 저장"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='컨텐츠 요약 도구')
    parser.add_argument('--url', help='처리할 컨텐츠 URL')
    parser.add_argument('--no-notion', action='store_true', help='Notion 저장 비활성화')
    
    args = parser.parse_args()
    
    app = SummarizerApp()
    result = app.run(
        url=args.url,  # URL이 없으면 None이 전달되어 사용자 입력 요청
        save_to_notion=not args.no_notion
    )
    
    if result:
        print("\n요약 완료!")
        print(f"결과 파일: {app.settings.paths['results']}")
    else:
        print("\n요약 실패!")

if __name__ == "__main__":
    main()
