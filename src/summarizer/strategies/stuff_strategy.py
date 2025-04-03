# src/summarizer/strategies/stuff_strategy.py
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

class StuffSummarizer(BaseSummarizer):
    """LangChain의 'stuff' 요약 체인을 사용하는 전략 (짧은 텍스트용)"""

    def __init__(self, cfg: DictConfig, llm_interface: LLMInterface):
        super().__init__(cfg, llm_interface)
        self.strategy = 'stuff'
        logger.info(f"Initialized StuffSummarizer.")

    async def summarize(self, text: str, title: Optional[str] = None) -> Dict:
        """'stuff' 전략을 사용하여 요약합니다. 텍스트가 길면 실패할 수 있습니다."""
        strategy_name = self.strategy
        logger.info(f"Executing {strategy_name} strategy for: {title or 'Untitled'}")
        default_summary = self._get_default_summary(strategy_name)

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
                logger.error(f"LLM not available for {strategy_name} strategy.")
                return default_summary

            # 3. Stuff 전략용 Document 생성 (청킹 없이 전체 텍스트 사용)
            # 주의: 텍스트가 LLM 컨텍스트 제한을 초과하면 여기서 오류 발생 가능
            doc = [Document(page_content=cleaned_text)]
            # 토큰 수 검사 (선택적이지만 권장)
            total_tokens = self._num_tokens_from_string(cleaned_text)
            if total_tokens > self.prompt_token: # 청크 크기 계산에 사용된 prompt_token 활용
                 logger.warning(f"Input text ({total_tokens} tokens) likely exceeds context window for stuff strategy ({self.prompt_token} tokens). May fail.")
                 # 여기서 에러를 반환하거나, 실행을 시도할 수 있음
                 # return default_summary # 안전하게 실패 처리

            # 4. Stuff 요약 체인 실행
            chain = load_summarize_chain(llm, chain_type=self.strategy)
            chain_result = await asyncio.to_thread(chain.run, doc)
            logger.info(f"Successfully executed {strategy_name} chain.")

            # 5. 결과 매핑 (Minimal Schema 기준)
            final_summary = get_minimal_summary_schema().copy()
            final_summary['full_summary'] = chain_result.strip() if chain_result else ""
            final_summary['summary_strategy_used'] = strategy_name
            final_summary['model'] = default_summary['model']

            # 6. 번역 처리
            translated_summary = await self._translate_summary(final_summary)

            # 7. 최종 결과 정리
            final_result = self._get_default_summary(strategy_name)
            final_result.update(translated_summary)
            if self.include_full_text:
                final_result['full_text'] = cleaned_text
            final_result['summary_strategy_used'] = strategy_name # 재확인

            logger.info(f"Successfully completed {strategy_name} summarization.")
            return final_result

        except Exception as e:
            # Stuff 전략은 토큰 제한 초과 시 에러 발생 가능성이 높음
            logger.error(f"Error during {strategy_name} summarization (possibly due to context length): {str(e)}", exc_info=True)
            return default_summary 