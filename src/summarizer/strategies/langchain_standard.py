# src/summarizer/strategies/langchain_standard.py
import logging
from typing import Dict, Optional, List
import asyncio

from omegaconf import DictConfig
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain

from src.summarizer.base import BaseSummarizer
from src.llm_interface import LLMInterface
from src.schema import get_default_summary_schema, get_minimal_summary_schema

logger = logging.getLogger(__name__)

class StandardLangChainSummarizer(BaseSummarizer):
    """표준 LangChain 요약 체인(map_reduce, refine)을 사용하는 전략"""

    def __init__(self, cfg: DictConfig, llm_interface: LLMInterface, strategy: str):
        super().__init__(cfg, llm_interface)
        if strategy not in ['map_reduce', 'refine']:
            raise ValueError(f"Unsupported standard LangChain strategy: {strategy}. Use 'map_reduce' or 'refine'.")
        self.strategy = strategy
        logger.info(f"Initialized StandardLangChainSummarizer with strategy: {self.strategy}")

    async def summarize(self, text: str, title: Optional[str] = None) -> Dict:
        """설정된 표준 LangChain 전략(map_reduce 또는 refine)을 사용하여 요약합니다."""
        logger.info(f"Executing {self.strategy} strategy for: {title or 'Untitled'}")
        default_summary = self._get_default_summary(self.strategy)

        try:
            # 1. 텍스트 정리 및 언어 감지
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.error("Cleaned text is empty.")
                return default_summary
            source_lang = self.detect_language(cleaned_text)
            logger.info(f"Detected source language: {source_lang}")

            # 2. LLM 가져오기
            llm = self.llm_interface.get_llm()
            if not llm:
                logger.error(f"LLM not available for {self.strategy} strategy.")
                return default_summary

            # 3. 텍스트 청킹 (Base 클래스 분할기 사용)
            docs = self.text_splitter.create_documents([cleaned_text])
            n_chunks = len(docs)
            logger.info(f"Split text into {n_chunks} chunks for {self.strategy}.")
            if n_chunks == 0:
                 logger.error("No chunks generated from text.")
                 return default_summary

            # 4. 표준 LangChain 요약 체인 실행
            chain = load_summarize_chain(llm, chain_type=self.strategy)
            # 참고: chain.run은 동기 메소드이므로 비동기 환경에서 실행 시 주의 필요
            # asyncio.to_thread 사용하여 비동기 이벤트 루프 블로킹 방지
            chain_result = await asyncio.to_thread(chain.run, docs)
            logger.info(f"Successfully executed {self.strategy} chain.")

            # 5. 결과 매핑 (Minimal Schema 기준)
            final_summary = get_minimal_summary_schema().copy() # 표준 체인은 주로 full_summary만 반환
            final_summary['full_summary'] = chain_result.strip() if chain_result else ""
            final_summary['summary_strategy_used'] = self.strategy
            # 모델 정보 추가
            final_summary['model'] = default_summary['model'] # 기본 스키마에서 가져옴

            # 6. 번역 처리
            translated_summary = await self._translate_summary(final_summary)

            # 7. 최종 결과 정리
            final_result = self._get_default_summary(self.strategy) # 전체 스키마 로드
            final_result.update(translated_summary) # 번역된 내용 업데이트
            if self.include_full_text:
                final_result['full_text'] = cleaned_text
            final_result['summary_strategy_used'] = self.strategy # 재확인

            logger.info(f"Successfully completed {self.strategy} summarization.")
            return final_result

        except Exception as e:
            logger.error(f"Error during {self.strategy} summarization: {str(e)}", exc_info=True)
            return default_summary 