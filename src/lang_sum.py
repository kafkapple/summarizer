from src.utils import setup_logging
from omegaconf import DictConfig
import re
from typing import List, Dict, Optional, Any, Union
import asyncio
import logging
import httpx

# 로거 초기화
logger, debug_logger = setup_logging()

# 필요한 모듈 임포트
from src.llm_interface import LLMInterface
from src.summarizer.base import BaseSummarizer
from src.summarizer.factory import create_summarizer
from src.schema import get_default_summary_schema # 기본 스키마 임포트

class LangChainSummarizerSync:
    """Asynchronous Summarizer를 위한 동기 래퍼"""

    def __init__(self, cfg: DictConfig):
        """동기 래퍼 초기화. LLM 인터페이스와 팩토리를 통해 Summarizer 생성"""
        self.cfg = cfg
        self.logger = logger
        self.debug_llm = cfg.get('debug_llm', False)
        self.summarizer: Optional[BaseSummarizer] = None # 초기화 실패 대비

        try:
            # 1. LLMInterface 초기화
            # LLMInterface 초기화 실패 시 Summarizer 생성 불가
            self.llm_interface = LLMInterface(cfg)

            # 2. Summarizer Factory를 통해 실제 Summarizer 인스턴스 생성
            self.summarizer = create_summarizer(cfg, self.llm_interface)

            self.logger.info(f"Summarizer Sync Wrapper initialized with strategy: {cfg.summary.get('strategy', 'default')}")

        except Exception as e:
            self.logger.error(f"Summarizer Sync Wrapper initialization error: {str(e)}", exc_info=True)
            # 초기화 실패 시 summarize 호출에서 오류 발생하도록 None 상태 유지
            # 또는 raise RuntimeError("Failed to initialize Summarizer Sync Wrapper") from e

    def _get_empty_summary(self) -> Dict:
        """실패 시 반환할 빈 요약 딕셔너리를 생성합니다."""
        summary = get_default_summary_schema().copy()
        summary['summary_strategy_used'] = 'error' # 또는 'failed_initialization'
        # 모델 정보는 llm_interface가 초기화되었을 경우에만 추가 시도
        if hasattr(self, 'llm_interface') and self.llm_interface:
             summary['model'] = {
                  'provider': self.llm_interface.provider,
                  'model': self.llm_interface.model_name,
                  'output_language': self.cfg.output.get('language', 'ko')
             }
        return summary

    def summarize(self, text: str, title: Optional[str] = None) -> Optional[Dict]:
        """
        요약 메소드의 동기 버전.
        팩토리에서 생성된 Summarizer의 비동기 summarize 메소드를 호출합니다.
        """
        try:
            # Summarizer 초기화 성공 여부 확인
            if not self.summarizer:
                 self.logger.error("Summarizer was not initialized successfully. Cannot summarize.")
                 return self._get_empty_summary()

            # 입력 유효성 검사
            if not isinstance(text, str):
                self.logger.warning(f"Input type {type(text)} is not str. Attempting conversion.")
                text = str(text)
            if not text.strip():
                self.logger.error("Input text is empty or whitespace.")
                return self._get_empty_summary()

            if self.debug_llm:
                print(f"\n[DEBUG] Starting sync summarization wrapper for: {title}")
                # debug_logger 사용 시:
                # debug_logger.debug(f"===== Starting Summarization via Sync Wrapper for: {title} =====")

            # 실제 Summarizer의 비동기 summarize 메소드 호출
            result = asyncio.run(self.summarizer.summarize(text, title))

            if self.debug_llm:
                print(f"[DEBUG] Finished sync summarization wrapper for: {title}")
                # debug_logger.debug(f"===== Finished Summarization via Sync Wrapper for: {title} =====")

            # 결과가 None인 경우 (요약 실패) 처리
            if result is None:
                 self.logger.error(f"Summarization returned None for title '{title}'.")
                 return self._get_empty_summary()

            return result

        except Exception as e:
            self.logger.error(f"Synchronous summarization error for title '{title}': {str(e)}", exc_info=True)
            # debug_llm 로깅은 summarizer 내부에서 처리될 수 있음
            return self._get_empty_summary() # 일반 실패 시 빈 요약 반환