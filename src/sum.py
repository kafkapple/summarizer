import json
import logging
import re
from typing import List, Dict, Optional, Any, Union
import asyncio
from functools import partial
import os
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from omegaconf import DictConfig
from deep_translator import GoogleTranslator
import tiktoken

from src.utils import Utils
from src.utils import setup_logging
from src.llm_interface import LLMInterface
from src.schema import get_default_summary_schema, get_minimal_summary_schema # 스키마 import 활성화

# from src.schema import get_default_summary_schema, get_minimal_summary_schema # 주석 처리

logger, debug_logger = setup_logging()

class LangChainSummarizer:
    """LangChain 기반 요약기 - BaseSummarizer와 호환됨"""
    
    def __init__(self, cfg: DictConfig, verbose=False, quiet=False):
        """
        LangChain 요약기 초기화
        
        Args:
            cfg: 설정 정보가 담긴 DictConfig 객체
            verbose: 상세 로깅 여부
            quiet: 상세 로깅 제외 여부
        """
        self.cfg = cfg
        self.verbose = verbose
        self.quiet = quiet
        self.debug_llm = cfg.debug_llm
        self.summary_strategy = cfg.summary.get('strategy', 'default')
        self.output_language = cfg.output.get('language', 'ko')
        self.output_language_full = self._get_language_name(self.output_language)
        self.include_full_text = cfg.include_full_text
        self.enable_chapters = cfg.enable_chapters
        self.highlight_keywords = cfg.highlight_keywords
        self.few_shot_enabled = cfg.few_shot.get('enabled', False)
        self.few_shot_examples_str = self._load_few_shot_examples() if self.few_shot_enabled else ""
        
        # 프롬프트 설정
        self.system_content = cfg.prompt.get('system_content', '')
        self.instruction_output = cfg.prompt.get('instruction_output', '')
        
        # 기타 설정
        self.include_keywords = cfg.get('include_keywords', True)
        self.source_lang = 'auto'  # 자동 감지
        
        # 모델 설정
        self.model = cfg.llm.get('model', 'gpt-3.5-turbo')
        self.temperature = float(cfg.llm.get('temperature', 0.3))
        self.api_key = cfg.api_keys.openai
        
        # 토큰 관련 설정
        self.max_token = int(cfg.llm.get('max_tokens', 4000))  # 최대 토큰 수
        self.response_token = int(cfg.llm.get('response_tokens', 600))  # 응답 토큰 수
        self.buffer_token = int(cfg.llm.get('buffer_tokens', 200))  # 버퍼 토큰 수
        self.max_translate_length = int(cfg.llm.get('max_translate_length', 4500))  # 최대 번역 길이
        self.max_chunks_per_chapter = int(cfg.llm.get('max_chunks_per_chapter', 6))  # 챕터당 최대 청크 수
        
        # 청크 사이즈 계산 (시스템 프롬프트 토큰 감안)
        self.system_token = self._num_tokens_from_string(self.system_content)
        self.prompt_token = self.max_token - self.system_token - self.response_token - self.buffer_token
        
        # LangChain 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.prompt_token,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        # LLM 인터페이스 초기화
        self.llm_interface = LLMInterface(cfg)
        
        if self.verbose:
            logger.info('\n' + '#'*7 + ' LangChain Summarizer Initialized ' + '#'*7 + 
                      f"\n모델 = {self.llm_interface.model_name}\n초기화: 대상 언어 = {self.output_language}")
            logger.info(f'Max/System/Response/Prompt Tokens: {self.max_token}/{self.system_token}/{self.response_token}/{self.prompt_token}')
            logger.info(f'Summary Strategy: {self.summary_strategy}')
    
    def _get_language_name(self, lang_code: str) -> str:
        """언어 코드를 전체 언어 이름으로 변환"""
        lang_map = {
            'ko': 'Korean',
            'en': 'English',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'pt': 'Portuguese'
        }
        return lang_map.get(lang_code, 'English')  # 기본값은 영어
    
    def _num_tokens_from_string(self, string: str) -> int:
        """문자열의 토큰 수 계산"""
        try:
            # 모델 이름은 LLMInterface에서 가져옴
            encoding = tiktoken.encoding_for_model(self.llm_interface.model_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except:
            # 단순 추정 (대략 단어당 1.3 토큰)
            return int(len(string.split()) * 1.3)
    
    def detect_language(self, text: str) -> str:
        """텍스트 언어 감지"""
        from langdetect import detect
        try:
            return detect(text)
        except:
            return 'en'
    
    async def summarize(self, text: str, title: str) -> Optional[Dict]:
        """텍스트 요약 (다양한 전략 지원)"""
        actual_strategy_used = self.summary_strategy # Initialize strategy used at the beginning
        try:
            if not text or not isinstance(text, str):
                logger.error("텍스트 전처리 중 오류 발생: 입력은 문자열이어야 합니다")
                return None
            
            # 텍스트 전처리
            text = self.clean_text(text)
            if not text:
                logger.error("요약 처리 중 오류 발생: No features in text.")
                return None
            
            # 소스 언어 감지
            self.source_lang = self.detect_language(text)
            if self.verbose:
                logger.info(f"감지된 언어: {self.source_lang}, 타겟 언어: {self.output_language}")
            
            # 텍스트 길이에 따른 전략 선택
            text_length = len(text)
            if self.verbose:
                logger.info(f"텍스트 길이: {text_length} 자")
            
            # 로깅 추가: 토큰 수 및 청크 수
            total_tokens = self._num_tokens_from_string(text)
            logger.info(f"Total tokens in input text: {total_tokens}")
            
            # 디버깅: 청크 분할 직전 토큰 설정 로깅
            if self.debug_llm or self.verbose:
                logger.info("--- Token settings for chunk splitting ---")
                logger.info(f"Model Max Tokens (max_token): {self.max_token}")
                logger.info(f"System Prompt Tokens (system_token): {self.system_token}")
                logger.info(f"Max Response Tokens (response_token): {self.response_token}")
                logger.info(f"Buffer Tokens (buffer_token): {self.buffer_token}")
                logger.info(f"=> Calculated Max Chunk Size (prompt_token): {self.prompt_token}")
                logger.info("----------------------------------------")
            
            # 텍스트를 청크로 분할 (LangChain 표준 체인 또는 기본 로직 공통 사용)
            docs = self.text_splitter.create_documents([text])
            chunks = [doc.page_content for doc in docs]
            n_chunks = len(chunks)
            
            # 로깅 추가: 토큰 수 및 청크 수
            logger.info(f"총 토큰 수: {total_tokens}, 청크 수: {n_chunks}")
            if self.debug_llm:
                debug_logger.debug(f"Total tokens: {total_tokens}, Number of chunks: {n_chunks}")
            
            if n_chunks == 0:
                logger.error("요약 처리 중 오류 발생: No features in text.")
                return None
            
            # 청크 수에 따른 처리 전략
            if n_chunks == 1:
                # 단일 청크 처리
                logger.info("Processing as a single chunk.")
                prompt_text = text # 전체 텍스트를 프롬프트로
                # 단일 청크 요약 시도
                summary_result = await self._get_chunk_summary(prompt_text)
                # 단일 청크 결과에서 core_summary 생성 추가
                if summary_result and summary_result.get('sections'):
                    summary_result['core_summary'] = self._format_core_summary(summary_result['sections'])
                    # sections_summary도 필요하면 여기서 생성 가능
                    summary_result['sections_summary'] = self._format_sections_summary(summary_result['sections'])
                else: # sections 정보가 없는 경우 대비
                     summary_result = summary_result or {}
                     summary_result.setdefault('core_summary', summary_result.get('full_summary', '')) # fallback
                     summary_result.setdefault('sections_summary', summary_result.get('full_summary', ''))

            elif n_chunks <= self.max_chunks_per_chapter:
                # 중간 길이 문서: 여러 청크 처리
                logger.info(f"Processing with {n_chunks} chunks (<= max_chunks_per_chapter).")
                chunk_summaries = []
                for chunk in chunks:
                    chunk_summary = await self._get_chunk_summary(chunk)
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)
                
                # 요약 병합
                summary_result = await self._merge_chunk_summaries(chunk_summaries, title)
            else:
                # 매우 긴 문서: 챕터 기반 처리
                logger.info(f"Processing with {n_chunks} chunks (> max_chunks_per_chapter), using chapter approach.")
                summary_result = await self._process_large_text(chunks, title)
            
            # 2. Stuff 실패 또는 텍스트가 긴 경우, 설정된 전략 사용
            if summary_result is None: 
                # LangChain 표준 체인 사용
                if self.summary_strategy in ['map_reduce', 'refine']:
                     # ... (체인 실행 로직) ...
                     chain_result = await asyncio.to_thread(chain.run, docs)
                     # 결과를 표준 딕셔너리 형식으로 변환 (스키마 사용)
                     # 임시: schema.py 사용 전 직접 구현 -> 제거
                     # temp_schema = {'full_summary': chain_result} 
                     summary_result = get_minimal_summary_schema().copy() # 최소 스키마 복사본 사용
                     summary_result['full_summary'] = chain_result # 실제 결과 채우기

                     actual_strategy_used = self.summary_strategy
                 # ... (Fallback 및 default 전략 로직) ...

            if not summary_result:
                logger.error("Failed to generate summary using any strategy.")
                # 임시: 스키마 사용 전 하드코딩 -> 스키마 사용
                # return { ... }
                summary_result = get_default_summary_schema().copy() # 스키마 복사본 사용
                summary_result['summary_strategy_used'] = 'failed' # 상태 업데이트
                return summary_result 

            # 메타데이터 추가
            summary_result['metadata'] = {
                'title': title,
                'url': getattr(self, 'url', ''),
                'channel': getattr(self, 'channel', ''),
                'view_count': getattr(self, 'view_count', 0),
                'like_count': getattr(self, 'like_count', 0),
                'comment_count': getattr(self, 'comment_count', 0),
                'published_date': getattr(self, 'published_date', ''),
                'duration': getattr(self, 'duration', ''),
                'category': getattr(self, 'category', ''),
                'tags': getattr(self, 'tags', []),
                'description': getattr(self, 'description', '')
            }
            
            # 언어 변환 필요시 처리
            if self.output_language == 'ko':
                try:
                    current_lang = self.detect_language(str(summary_result.get('one_sentence_summary', '') or str(summary_result.get('full_summary', ''))))
                    if current_lang == 'ko':
                        logger.info("Summary already in Korean, skipping translation.")
                    else:
                        logger.info(f"Detected summary language '{current_lang}', attempting translation to Korean.")
                        summary_result = await self._translate_summary(summary_result)
                except Exception as detect_err:
                    logger.warning(f"Could not reliably detect summary language, attempting translation to Korean anyway: {detect_err}")
                    summary_result = await self._translate_summary(summary_result)

            # 최종 포맷팅
            formatted_result = self._format_summary(summary_result, text)
            summary_result['summary_strategy_used'] = actual_strategy_used # 실제 사용된 전략 기록
            return formatted_result
        
        except Exception as e:
            logger.error(f"요약 처리 중 오류 발생: {str(e)}")
            # 임시: 스키마 사용 전 하드코딩 -> 스키마 사용
            # return { ... }
            summary_result = get_default_summary_schema().copy()
            summary_result['summary_strategy_used'] = 'error'
            return summary_result

    def clean_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 특수 문자 처리
        text = text.replace('\u200b', '')  # 제로 폭 공백 제거
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 일관성을 위해 줄바꿈 정리
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    async def _get_chunk_summary(self, chunk: str) -> Optional[Dict]:
        """단일 청크의 요약 생성 (LLMInterface 사용)"""
        try:
            # LLMInterface를 통해 LLM 호출
            parsed_result = await self.llm_interface.invoke_llm(chunk)
            return parsed_result # LLMInterface가 파싱 및 오류 처리 담당

        except Exception as e:
            logger.error(f"청크 요약 중 오류 발생 (_get_chunk_summary): {str(e)}")
            if self.debug_llm:
                debug_logger.error(f"청크 요약 중 오류 발생 (_get_chunk_summary): {str(e)}", exc_info=True)
            # LLMInterface의 기본 구조 반환 메서드 사용 가능 (선택 사항)
            # return self.llm_interface._default_summary_structure()
            return {
                "keywords": [], "sections": [], "full_summary": [], "one_sentence_summary": ""
            } # 기존 방식 유지
    
    def _process_json_response(self, response: str) -> Optional[Dict]:
        """LLM 응답을 처리하여 요약 결과를 반환 (LLMInterface로 이동)"""
        # 이 로직은 LLMInterface.invoke_llm 내부로 이동되었으므로 여기서는 제거합니다.
        pass # 또는 raise NotImplementedError("Moved to LLMInterface")
    
    async def _process_large_text(self, chunks: List[str], title: str) -> Optional[Dict]:
        """
        대용량 텍스트 처리 (챕터 기반)
        
        Args:
            chunks: 처리할 텍스트 청크 목록
            title: 콘텐츠 제목
            
        Returns:
            최종 요약 결과
        """
        try:
            # 청크를 챕터로 분할
            chapters = self._divide_chunks_into_chapters(chunks, self.max_chunks_per_chapter)
            logger.info(f'챕터 수: {len(chapters)}')
            
            chapter_info = []
            chapter_summaries = []
            total_section_count = 0
            
            # 각 챕터 처리
            for i, chapter_chunks in enumerate(chapters):
                chapter_num = i + 1
                logger.info(f'챕터 {chapter_num} 처리 중... (청크 수: {len(chapter_chunks)})')
                
                # 챕터 내 청크 요약
                chunk_summaries = []
                for chunk in chapter_chunks:
                    chunk_summary = await self._get_chunk_summary(chunk)
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)
                
                # 챕터 내 요약 병합
                chapter_title = f"Chapter {chapter_num}"
                chapter_summary = await self._merge_chunk_summaries(chunk_summaries, chapter_title)
                
                if not chapter_summary:
                    continue
                
                # 섹션 정보 처리
                chapter_sections = chapter_summary.get("sections", [])
                chapter_keywords = chapter_summary.get("keywords", [])
                
                # 섹션 인덱스 계산
                section_start = total_section_count
                total_section_count += len(chapter_sections)
                
                # 챕터 정보 저장
                chapter_info.append({
                    'chapter_number': chapter_num,
                    'chapter_title': chapter_title,
                    'section_indices': {
                        'start': section_start,
                        'end': total_section_count
                    },
                    'sections': chapter_sections,
                    'keywords': chapter_keywords,
                    'summary': chapter_summary.get('full_summary', ''),
                    'one_sentence_summary': chapter_summary.get('one_sentence_summary', '')
                })
                
                chapter_summaries.append(chapter_summary)
            
            # 최종 요약 생성
            final_summary = await self._merge_chunk_summaries(chapter_summaries, title)
            
            if not final_summary:
                return None
            
            # 챕터 정보 추가
            final_summary['chapters'] = chapter_info
            
            return final_summary
            
        except Exception as e:
            logger.error(f"대용량 텍스트 처리 중 오류 발생: {str(e)}")
            return None

    def _divide_chunks_into_chapters(self, chunks: List[str], max_chunks_per_chapter: int) -> List[List[str]]:
        """
        청크들을 가능한 균등한 크기의 챕터로 나눕니다.
        
        Args:
            chunks: 분할할 청크 목록
            max_chunks_per_chapter: 챕터당 최대 청크 수
            
        Returns:
            챕터별로 구분된 청크 리스트
        """
        n_chunks = len(chunks)
        
        if n_chunks <= max_chunks_per_chapter:
            return [chunks]
        
        # 적절한 챕터 수 계산
        n_chapters = (n_chunks + max_chunks_per_chapter - 1) // max_chunks_per_chapter
        
        # 더 균등한 분배를 위해 챕터 수 조정
        if n_chunks / n_chapters < max_chunks_per_chapter * 0.5:  # 챕터당 청크가 너무 적은 경우
            n_chapters = max(2, (n_chunks + 1) // 2)  # 최소 2개의 챕터
        
        # 챕터당 기본 청크 수 계산
        base_chunks_per_chapter = n_chunks // n_chapters
        remaining_chunks = n_chunks % n_chapters
        
        chapters = []
        start_idx = 0
        
        for i in range(n_chapters):
            # 남은 청크를 균등하게 분배
            chapter_size = base_chunks_per_chapter + (1 if i < remaining_chunks else 0)
            end_idx = start_idx + chapter_size
            chapters.append(chunks[start_idx:end_idx])
            start_idx = end_idx
        
        if self.verbose:
            logger.info(f'총 {n_chunks}개 청크를 {n_chapters}개 챕터로 나눔')
            logger.info(f'챕터별 청크 수: {[len(chapter) for chapter in chapters]}')
        
        return chapters

    async def _merge_chunk_summaries(self, chunk_summaries: List[Dict], title: str) -> Optional[Dict]:
        """
        청크별 요약을 병합하고, 최종 LLM 호출을 통해 한 문장/전체 요약을 정제합니다.
        
        Args:
            chunk_summaries: 청크별 요약 리스트
            title: 콘텐츠 제목
            
        Returns:
            병합 및 정제된 요약 결과
        """
        try:
            if not chunk_summaries:
                 logger.warning("No chunk summaries to merge.")
                 return None

            if self.debug_llm:
                debug_logger.debug("===== Starting Chunk Summary Merge =====")
                debug_logger.debug(f"Number of chunks to merge: {len(chunk_summaries)}")
                debug_logger.debug(f"Title: {title}")
            
            # 1. 섹션 정보 및 키워드 병합 (기존 방식 유지)
            all_sections = []
            all_keywords_dict = {} # 중복 제거용 딕셔너리
            
            for i, chunk in enumerate(chunk_summaries):
                if not isinstance(chunk, dict):
                    logger.warning(f"Skipping invalid chunk summary at index {i}: type is {type(chunk)}")
                    continue

                if self.debug_llm:
                    debug_logger.debug(f"\nProcessing chunk {i+1}:")
                    debug_logger.debug(f"Chunk keys: {chunk.keys()}")
                
                sections = chunk.get('sections', [])
                if isinstance(sections, list):
                    all_sections.extend(sections)
                else:
                    logger.warning(f"Chunk {i+1} 'sections' is not a list: {type(sections)}")

                keywords = chunk.get('keywords', [])
                if isinstance(keywords, list):
                    for kw in keywords:
                        if isinstance(kw, dict) and 'term' in kw:
                            term = kw['term']
                            if term not in all_keywords_dict:
                                all_keywords_dict[term] = kw
                        elif isinstance(kw, str): # 키워드가 문자열인 경우 처리
                            if kw not in all_keywords_dict:
                                all_keywords_dict[kw] = {'term': kw, 'frequency': 1} # 기본 빈도 추가

                else:
                     logger.warning(f"Chunk {i+1} 'keywords' is not a list: {type(keywords)}")

            all_keywords = list(all_keywords_dict.values())

            # 2. 섹션별 요약 생성 (기존 방식 유지)
            sections_summary_str = self._format_sections_summary(all_sections)
            
            # 4. 최종 요약 생성 (새로운 LLM 호출)
            final_one_sentence_summary = ""
            final_full_summary = ""

            if sections_summary_str.strip(): # 병합된 내용이 있을 경우에만 최종 요약 시도
                if self.debug_llm:
                    debug_logger.debug("===== Calling Final LLM for Refinement =====")
                    debug_logger.debug(f"Input context (sections_summary length): {len(sections_summary_str)}")

                # 최종 요약을 위한 프롬프트 구성 (LLMInterface가 처리하도록 수정)
                # LLMInterface에 새로운 프롬프트 유형 또는 지시사항 전달 방식 필요
                # 예시: llm_interface에 refinement_prompt_template 추가 또는 instruction 파라미터 활용
                
                # invoke_llm 호출 시, 어떤 종류의 요약을 원하는지 명시 필요
                # 예: refinement=True 또는 task='refine_summary' 파라미터 추가 가정
                try:
                    # LLMInterface에 refinement 작업 요청 (invoke_llm 수정 필요 가정 -> 수정 완료)
                    final_summary_result = await self.llm_interface.invoke_llm(
                        chunk=sections_summary_str, # 'context=' -> 'chunk=' 변경
                        task_type="refine", 
                        title=title
                    )

                    if final_summary_result:
                        final_one_sentence_summary = final_summary_result.get('one_sentence_summary', '')
                        final_full_summary = final_summary_result.get('full_summary', '')
                        # 필요 시, 여기서 생성된 keywords나 sections로 기존 all_keywords/all_sections 업데이트 가능
                        if self.debug_llm:
                             debug_logger.debug("Refinement LLM call successful.")
                             debug_logger.debug(f"Refined One Sentence: {final_one_sentence_summary}")
                             debug_logger.debug(f"Refined Full Summary (first 100 chars): {final_full_summary[:100]}...")
                    else:
                         logger.warning("Final refinement LLM call returned no result. Using concatenated summaries as fallback.")
                         # Fallback 수정: 기존 병합 방식 사용
                         final_full_summary = "\n\n".join([str(chunk.get('full_summary', '')) for chunk in chunk_summaries if chunk])
                         # Fallback 수정: 첫번째 청크의 한 문장 요약 사용
                         if chunk_summaries and isinstance(chunk_summaries[0], dict):
                            final_one_sentence_summary = chunk_summaries[0].get('one_sentence_summary', '')

                except Exception as refine_e:
                    logger.error(f"Error during final summary refinement LLM call: {str(refine_e)}")
                    if self.debug_llm:
                        debug_logger.error(f"Error during final summary refinement LLM call: {str(refine_e)}", exc_info=True)
                    # Fallback 수정: 기존 병합 방식 사용
                    final_full_summary = "\n\n".join([str(chunk.get('full_summary', '')) for chunk in chunk_summaries if chunk])
                    # Fallback 수정: 첫번째 청크의 한 문장 요약 사용
                    if chunk_summaries and isinstance(chunk_summaries[0], dict):
                        final_one_sentence_summary = chunk_summaries[0].get('one_sentence_summary', '')
            else:
                logger.warning("Skipping final refinement LLM call because merged sections summary is empty.")
                # 병합할 내용이 없을 때 fallback 처리 추가
                final_full_summary = ""
                final_one_sentence_summary = ""

            result = {
                'sections_summary': sections_summary_str,
                'full_summary': final_full_summary, # 최종 LLM 결과 사용
                'one_sentence_summary': final_one_sentence_summary, # 최종 LLM 결과 사용
                'keywords': all_keywords,
                'sections': all_sections, # 병합된 섹션 정보 유지
                'model': { # 모델 정보는 초기 모델 정보 유지 또는 최종 호출 모델 정보 반영
                    'provider': self.llm_interface.provider, # LLMInterface에서 provider 정보 가져오기
                    'model': self.llm_interface.model_name, # LLMInterface에서 model_name 정보 가져오기
                    'output_language': self.output_language
                }
            }
            
            if self.debug_llm:
                debug_logger.debug("===== Chunk Summary Merge Completed =====")
                debug_logger.debug(f"Result keys: {result.keys()}")
                debug_logger.debug(f"Number of sections: {len(result['sections'])}")
                debug_logger.debug(f"Number of keywords: {len(result['keywords'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"청크 요약 병합 중 오류 발생: {str(e)}")
            if self.debug_llm:
                debug_logger.error(f"청크 요약 병합 중 오류 발생: {str(e)}", exc_info=True)
            return None

    def _format_sections_summary(self, sections: List[Dict]) -> str:
        """
        섹션별 요약 포맷팅 (가장 상세)
        
        Args:
            sections: 섹션 정보 리스트
            
        Returns:
            포맷팅된 섹션별 요약
        """
        formatted_sections = []
        for section in sections:
            formatted_sections.append(f"제목: {section.get('title', '')}\n내용: {section.get('summary', '')}\n")
        return "\n".join(formatted_sections)

    def _format_full_summary(self, summaries: List[str]) -> str:
        """
        전체 요약 포맷팅 (중간 수준)
        
        Args:
            summaries: 요약 내용 리스트
            
        Returns:
            포맷팅된 전체 요약
        """
        return "\n".join(summaries)

    def _format_core_summary(self, sections: List[Dict]) -> str:
        """
        핵심 주제 포맷팅 (간단한 요약)
        
        Args:
            sections: 섹션 정보 리스트
            
        Returns:
            포맷팅된 핵심 주제
        """
        core_points = []
        for section in sections:
            # 각 섹션의 핵심 내용만 추출
            section_summary = section.get('summary', '')
            if isinstance(section_summary, list):
                # 리스트인 경우, 각 항목을 불릿 포인트로 만들어 합침
                processed_summary = "\n".join([f"- {s.strip()}" for s in section_summary if isinstance(s, str) and s.strip()])
                if processed_summary: # 빈 문자열이 아니면 추가
                     core_points.append(processed_summary)
            elif isinstance(section_summary, str) and section_summary.strip():
                # 문자열이고 내용이 있으면 추가 (앞에 불릿 추가)
                core_points.append(f"- {section_summary.strip()}")
            # 필요시 다른 타입 처리 추가
            
        # 최종적으로 모든 핵심 내용을 하나의 문자열로 결합
        return "\n".join(core_points)

    def _generate_one_sentence_summary(self, summaries: List[str]) -> str:
        """
        한 문장 요약 생성 (가장 간단)
        
        Args:
            summaries: 요약 내용 리스트
            
        Returns:
            한 문장 요약
        """
        # 모든 요약을 하나로 합치고 가장 중요한 내용 추출
        combined_summary = " ".join(summaries)
        return combined_summary[:200] + "..." if len(combined_summary) > 200 else combined_summary

    def _format_summary(self, merged_summary: Dict, processed_text: str) -> Dict:
        """
        최종 요약 포맷팅
        
        Args:
            merged_summary: 병합된 요약 결과
            processed_text: 처리된 원본 텍스트
            
        Returns:
            포맷팅된 최종 요약
        """
        formatted_summary = {
            # 1. 섹션별 요약 (가장 상세)
            'sections_summary': merged_summary.get('sections_summary', ''),
            
            # 2. 전체 요약 (중간 수준)
            'full_summary': merged_summary.get('full_summary', ''),
            
            # 3. 한 문장 요약 (가장 간단)
            'one_sentence_summary': merged_summary.get('one_sentence_summary', ''),
            
            # 4. 키워드
            'keywords': merged_summary.get('keywords', []),
            
            # 5. 섹션 정보
            'sections': merged_summary.get('sections', []),
            
            # 6. 모델 정보
            'model': merged_summary.get('model', {})
        }
        
        # 원본 텍스트 포함 옵션
        if self.include_full_text:
            formatted_summary['full_text'] = processed_text
            
        return formatted_summary

    async def _translate_summary(self, summary: Dict) -> Dict:
        """
        요약 결과를 한국어로 번역
        
        Args:
            summary: 번역할 요약 결과
            
        Returns:
            번역된 요약 결과
        """
        try:
            translator = GoogleTranslator(source='auto', target='ko')
            
            # 한 문장 요약 번역
            if summary.get('one_sentence_summary'):
                summary['one_sentence_summary'] = translator.translate(summary['one_sentence_summary'])
            
            # 전체 요약 번역
            if summary.get('full_summary'):
                summary['full_summary'] = translator.translate(summary['full_summary'])
            
            # 섹션별 요약 번역
            if summary.get('sections'):
                for section in summary['sections']:
                    if section.get('title'):
                        section['title'] = translator.translate(section['title'])
                    if section.get('summary'):
                        if isinstance(section['summary'], list):
                            section['summary'] = [translator.translate(item) for item in section['summary']]
                        else:
                            section['summary'] = translator.translate(section['summary'])
            
            # 핵심 주제 번역
            if summary.get('core_summary'):
                summary['core_summary'] = translator.translate(summary['core_summary'])
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 번역 중 오류 발생: {str(e)}")
            return summary

    # 추가: _load_few_shot_examples 메서드 (필요한 경우)
    def _load_few_shot_examples(self) -> str:
        # 이전에 few-shot 예제를 로드하던 로직을 여기에 구현
        # 예: 파일에서 읽거나, 코드 내에 정의
        if self.few_shot_enabled and self.cfg.few_shot.get('path'):
            try:
                # few_shot_path 가 base path 기준으로 설정되도록 수정
                few_shot_full_path = os.path.join(self.cfg.paths.base, self.cfg.few_shot.path)
                with open(few_shot_full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                logger.warning(f"Few-shot example file not found at: {few_shot_full_path}")
            except Exception as e:
                logger.error(f"Error loading few-shot examples: {e}")
        return ""
