import logging
import time # 재시도를 위한 time 모듈 추가
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any # Any 타입 추가
from omegaconf import DictConfig

from src.llm_interface import LLMInterface
from src.schema import get_default_summary_schema
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

logger = logging.getLogger(__name__)

class BaseSummarizer(ABC):
    """모든 요약 전략 클래스의 기반이 되는 추상 클래스"""

    def __init__(self, cfg: DictConfig, llm_interface: LLMInterface):
        """
        기본 초기화 로직. 설정값, LLM 인터페이스, 텍스트 분할기 등을 설정합니다.

        Args:
            cfg: 전체 설정 객체
            llm_interface: 초기화된 LLMInterface 인스턴스
        """
        self.cfg = cfg
        self.llm_interface = llm_interface
        self.verbose = cfg.get('verbose', False)
        self.debug_llm = cfg.get('debug_llm', False)
        self.output_language = cfg.output.get('language', 'ko')
        self.include_full_text = cfg.get('include_full_text', False)
        self.max_token = int(cfg.llm.get('max_tokens', 4000))
        self.response_token = int(cfg.llm.get('response_tokens', 600))
        self.buffer_token = int(cfg.llm.get('buffer_tokens', 200))

        # 텍스트 분할기 및 청크 크기 계산 (공통 로직)
        self.system_content = cfg.prompt.get('system_content', '') # 토큰 계산용
        self.system_token = self._num_tokens_from_string(self.system_content)
        self.prompt_token = self.max_token - self.system_token - self.response_token - self.buffer_token

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.prompt_token,
            chunk_overlap=int(cfg.llm.get('chunk_overlap', 200)), # 설정값 사용
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len # 토큰 기반 분할 대신 길이 기반 사용 (LangChain 기본)
            # 참고: 토큰 기반 분할이 필요하면 tiktoken과 통합 필요
        )

        if self.verbose:
            logger.info(f"{self.__class__.__name__} Initialized.")
            logger.info(f'Max/System/Response/Prompt Tokens: {self.max_token}/{self.system_token}/{self.response_token}/{self.prompt_token}')
            logger.info(f'Chunk overlap: {self.text_splitter._chunk_overlap}')


    def _num_tokens_from_string(self, string: str, encoding_name: Optional[str] = None) -> int:
        """Helper function to count tokens in a string."""
        if not string:
            return 0
        # Use model name from LLM interface if encoding_name is not provided
        model_name = encoding_name if encoding_name else self.llm_interface.model_name
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except KeyError:
            logger.warning(f"Warning: Encoding for model '{model_name}' not found. Using cl100k_base.")
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except Exception as e:
            logger.error(f"Error calculating tokens: {e}. Estimating based on length.")
            # Fallback estimation
            return int(len(string) / 4) # Rough estimation

    def _get_language_name(self, lang_code: str) -> str:
        """언어 코드를 전체 언어 이름으로 변환 (기존 로직 재사용)"""
        lang_map = {
            'ko': 'Korean', 'en': 'English', 'ja': 'Japanese', 'zh': 'Chinese',
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'ru': 'Russian', 'pt': 'Portuguese'
        }
        return lang_map.get(lang_code, 'English')

    def clean_text(self, text: str) -> str:
        """텍스트 전처리 (기존 로직 재사용)"""
        import re
        if not isinstance(text, str): return ""
        text = text.replace('\u200b', '')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def detect_language(self, text: str) -> str:
        """텍스트 언어 감지 (기존 로직 재사용, langdetect 필요)"""
        if not text: return 'en' # 기본값
        try:
            from langdetect import detect, LangDetectException
            return detect(text)
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to 'en'.")
            return 'en'
        except ImportError:
            logger.warning("langdetect library not found. Cannot detect language. Defaulting to 'en'.")
            return 'en'

    async def _translate_summary(self, summary: Dict) -> Dict:
        """요약 결과를 한국어로 번역 (기존 로직 재사용, deep_translator 필요)"""
        if self.output_language != 'ko':
            return summary # 한국어가 아니면 번역 불필요

        try:
            # Check if translation is actually needed
            current_lang_sample = str(summary.get('one_sentence_summary', '') or str(summary.get('full_summary', '')))[:100]
            if not current_lang_sample: return summary # No text to detect/translate

            # Detect language only if needed
            try:
                from langdetect import detect, LangDetectException
                current_lang = detect(current_lang_sample)
                if current_lang == 'ko':
                     logger.info("Summary seems already in Korean, skipping translation.")
                     return summary
                logger.info(f"Detected summary language '{current_lang}', attempting translation to Korean.")
            except LangDetectException:
                 logger.warning("Could not reliably detect summary language, attempting translation to Korean anyway.")
            except ImportError:
                logger.warning("langdetect not found, attempting translation to Korean anyway.")


            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='auto', target='ko')

            # 필드별 번역 (기존 로직과 유사하게)
            fields_to_translate = ['one_sentence_summary', 'full_summary']
            for field in fields_to_translate:
                if summary.get(field):
                    summary[field] = translator.translate(str(summary[field]))

            if isinstance(summary.get('sections'), list):
                for section in summary['sections']:
                    if isinstance(section, dict):
                        if section.get('title'):
                            section['title'] = translator.translate(str(section['title']))
                        if section.get('summary'):
                            if isinstance(section['summary'], list):
                                section['summary'] = [translator.translate(str(item)) for item in section['summary']]
                            elif isinstance(section['summary'], str):
                                section['summary'] = translator.translate(section['summary'])

            if isinstance(summary.get('keywords'), list):
                for keyword_dict in summary.get('keywords', []):
                    if isinstance(keyword_dict, dict) and keyword_dict.get('term'):
                        keyword_dict['term'] = translator.translate(str(keyword_dict['term']))

            return summary

        except ImportError:
            logger.error("deep_translator library not found. Cannot translate summary.")
            return summary # 번역 실패 시 원본 반환
        except Exception as e:
            logger.error(f"Error during summary translation: {str(e)}")
            return summary # 번역 실패 시 원본 반환

    async def _call_llm_with_retry(self, prompt: str, system_content: Optional[str] = None, **kwargs) -> Optional[Any]:
        """
        LLM 인터페이스를 호출하고, 실패 시 설정된 횟수만큼 재시도합니다.
        지수 백오프(exponential backoff)를 사용하여 재시도 간 대기 시간을 늘립니다.

        Args:
            prompt: LLM에 전달할 주 프롬프트
            system_content: LLM에 전달할 시스템 메시지 (선택 사항)
            **kwargs: llm_interface.generate에 전달할 추가 인자 (예: output_format)

        Returns:
            LLM 응답 또는 최대 재시도 실패 시 None
        """
        max_retries = self.cfg.llm.retry.get('max_retries', 3)
        initial_delay = self.cfg.llm.retry.get('initial_delay', 1)
        backoff_factor = self.cfg.llm.retry.get('backoff_factor', 2)
        current_delay = initial_delay

        for attempt in range(max_retries + 1): # 0부터 max_retries까지 시도 (총 max_retries + 1번)
            try:
                response = await self.llm_interface.generate(
                    prompt=prompt,
                    system_content=system_content if system_content is not None else self.system_content,
                    **kwargs # output_format 등 추가 인자 전달
                )
                # 응답이 비어있지 않은지 확인 (문자열 또는 dict 형태 모두 고려)
                if isinstance(response, str) and response.strip():
                    return response
                if isinstance(response, dict) and response:
                    return response
                if response is not None and not isinstance(response, (str, dict)): # 예상치 못한 타입이지만 비어있지 않다면 반환
                     logger.warning(f"LLM returned unexpected type: {type(response)}. Returning as is.")
                     return response


                # 응답이 비어 있는 경우 (None, 빈 문자열, 빈 딕셔너리)
                if attempt < max_retries:
                    logger.warning(f"LLM returned an empty response. Retrying in {current_delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                else:
                     logger.error(f"LLM returned an empty response after {max_retries + 1} attempts. Giving up.")
                     return None # 최종 실패

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Error calling LLM: {e}. Retrying in {current_delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                else:
                    logger.error(f"Error calling LLM after {max_retries + 1} attempts: {e}. Giving up.")
                    return None # 최종 실패

            # 재시도 전 대기
            if attempt < max_retries:
                await asyncio.sleep(current_delay) # 비동기 sleep 사용
                current_delay *= backoff_factor # 다음 대기 시간 증가

        return None # 모든 재시도 실패

    @abstractmethod
    async def summarize(self, text: str, title: Optional[str] = None) -> Dict:
        """
        주어진 텍스트를 요약하는 추상 메소드.
        각 전략 클래스는 이 메소드를 구현해야 합니다.

        Args:
            text: 요약할 원본 텍스트
            title: 콘텐츠 제목 (선택 사항)

        Returns:
            요약 결과 딕셔너리 (src/schema.py의 형식 준수)
        """
        pass

    def _get_default_summary(self, strategy_name: str) -> Dict:
        """전략 이름을 포함한 기본 요약 스키마를 반환합니다."""
        summary = get_default_summary_schema().copy()
        summary['summary_strategy_used'] = strategy_name
        summary['model'] = {
             'provider': self.llm_interface.provider,
             'model': self.llm_interface.model_name,
             'output_language': self.output_language
        }
        return summary
