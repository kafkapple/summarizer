# src/summarizer/strategies/default_strategy.py
import logging
from typing import Dict, Optional, List, Any
import asyncio

from omegaconf import DictConfig

from src.summarizer.base import BaseSummarizer
from src.llm_interface import LLMInterface
from src.schema import get_default_summary_schema # 스키마 가져오기

logger = logging.getLogger(__name__)
# llm_debug 로그를 사용하려면 별도 설정 필요 (base나 utils에서 가져오기 고려)
debug_logger = logging.getLogger('llm_debug') # 임시 설정

class DefaultSummarizer(BaseSummarizer):
    """
    기존의 'default' 요약 전략을 구현하는 클래스.
    청크별 요약, 병합, 챕터 처리 등을 포함합니다.
    """

    def __init__(self, cfg: DictConfig, llm_interface: LLMInterface):
        super().__init__(cfg, llm_interface)
        # Default 전략 관련 추가 설정 (기존 LangChainSummarizer에서 가져옴)
        self.enable_chapters = cfg.summary.get('enable_chapters', True)
        self.max_chunks_per_chapter = int(cfg.llm.get('max_chunks_per_chapter', 6))

    async def summarize(self, text: str, title: Optional[str] = None) -> Dict:
        """'default' 전략을 사용하여 텍스트를 요약합니다."""
        strategy_name = 'default'
        logger.info(f"Executing {strategy_name} strategy for: {title or 'Untitled'}")
        default_summary = self._get_default_summary(strategy_name)

        try:
            # 1. 텍스트 정리 및 언어 감지 (Base 클래스 메소드 사용)
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.error("Cleaned text is empty.")
                return default_summary

            source_lang = self.detect_language(cleaned_text)
            logger.info(f"Detected source language: {source_lang}")

            # 2. 텍스트 청킹 (Base 클래스 분할기 사용)
            chunks = self.text_splitter.split_text(cleaned_text)
            n_chunks = len(chunks)
            logger.info(f"Split text into {n_chunks} chunks.")

            if n_chunks == 0:
                logger.error("No chunks generated from text.")
                return default_summary

            # 3. 청크 수에 따른 요약 로직 분기
            summary_result_dict: Optional[Dict] = None
            if n_chunks == 1:
                logger.info("Processing as a single chunk.")
                summary_result_dict = await self._get_chunk_summary(chunks[0], title)
                # 단일 청크 결과 보강 (기존 로직)
                if summary_result_dict:
                    # sections_summary 생성
                    if summary_result_dict.get('sections'):
                         summary_result_dict['sections_summary'] = self._format_sections_summary(summary_result_dict['sections'])
                    else:
                         summary_result_dict.setdefault('sections_summary', summary_result_dict.get('full_summary', ''))
                else:
                     summary_result_dict = default_summary # 실패 시 기본값

            elif self.enable_chapters and n_chunks > self.max_chunks_per_chapter * 1.5: # 챕터 조건 조정 가능
                # 매우 긴 문서: 챕터 기반 처리
                logger.info(f"Processing with {n_chunks} chunks (> adjusted max_chunks_per_chapter), using chapter approach.")
                summary_result_dict = await self._process_large_text(chunks, title)

            else:
                # 중간 길이 문서: 여러 청크 처리
                logger.info(f"Processing with {n_chunks} chunks (<= adjusted max_chunks_per_chapter or chapters disabled).")
                chunk_summaries = []
                # 청크별 요약 비동기 실행
                tasks = [self._get_chunk_summary(chunk, title) for chunk in chunks]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                         logger.error(f"Error summarizing chunk {i}: {result}")
                    elif result:
                         chunk_summaries.append(result)
                    else:
                         logger.warning(f"Chunk {i} summary returned None.")

                if chunk_summaries:
                    summary_result_dict = await self._merge_chunk_summaries(chunk_summaries, title)
                else:
                     logger.error("No chunk summaries generated successfully.")
                     summary_result_dict = default_summary

            if not summary_result_dict:
                 logger.error("Summary generation failed.")
                 summary_result_dict = default_summary

            # 4. 번역 처리 (Base 클래스 메소드 사용)
            translated_summary = await self._translate_summary(summary_result_dict)

            # 5. 최종 결과 정리 (스키마 기반)
            final_result = self._get_default_summary(strategy_name) # 기본 스키마 로드
            final_result.update(translated_summary) # 생성된/번역된 내용 업데이트

            if self.include_full_text:
                final_result['full_text'] = cleaned_text # 원본 텍스트 추가

            final_result['summary_strategy_used'] = strategy_name # 실제 사용된 전략 재확인

            logger.info(f"Successfully completed {strategy_name} summarization.")
            return final_result

        except Exception as e:
            logger.error(f"Error during {strategy_name} summarization: {str(e)}", exc_info=True)
            return default_summary


    # --- Helper Methods (Moved from LangChainSummarizer) ---

    async def _get_chunk_summary(self, chunk: str, title: Optional[str] = None) -> Optional[Dict]:
        """개별 청크 요약 (LLMInterface 호출) - 이제 이 클래스 내에서 사용"""
        try:
            # LLMInterface를 통해 LLM 호출 (task_type 기본값 사용)
            # 'title'은 refine 단계에서 주로 사용되므로, 여기서는 전달하지 않거나 무시될 수 있음
            parsed_result = await self.llm_interface.invoke_llm(chunk=chunk, task_type="summarize_chunk", title=title)
            return parsed_result # LLMInterface가 파싱 및 오류 처리 담당
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            if self.debug_llm:
                debug_logger.error(f"Error summarizing chunk: {str(e)}", exc_info=True)
            return None # 실패 시 None 반환


    async def _merge_chunk_summaries(self, chunk_summaries: List[Dict], title: str) -> Optional[Dict]:
        """청크 요약 병합 및 정제 (기존 로직 유지, LLMInterface 사용)"""
        try:
            if not chunk_summaries:
                 logger.warning("No chunk summaries to merge.")
                 return None

            if self.debug_llm:
                debug_logger.debug("===== Starting Default Chunk Summary Merge =====")
                # ... (로그 추가 가능) ...

            # 1. 섹션 및 키워드 병합
            all_sections = []
            all_keywords_dict = {}
            for i, chunk in enumerate(chunk_summaries):
                # ... (기존 섹션/키워드 병합 로직과 동일) ...
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
                         elif isinstance(kw, str):
                             if kw not in all_keywords_dict:
                                 all_keywords_dict[kw] = {'term': kw, 'frequency': 1}
                 else:
                      logger.warning(f"Chunk {i+1} 'keywords' is not a list: {type(keywords)}")

            all_keywords = list(all_keywords_dict.values())

            # 2. sections_summary 생성
            sections_summary_str = self._format_sections_summary(all_sections)

            # 3. 최종 요약 생성 (Refine LLM 호출)
            final_one_sentence_summary = ""
            final_full_summary = ""
            if sections_summary_str.strip():
                if self.debug_llm:
                    debug_logger.debug("===== Calling Default Final LLM for Refinement =====")
                    # ... (로그 추가 가능) ...
                try:
                    final_summary_result = await self.llm_interface.invoke_llm(
                        chunk=sections_summary_str,
                        task_type="refine",
                        title=title
                    )
                    if final_summary_result:
                        final_one_sentence_summary = final_summary_result.get('one_sentence_summary', '')
                        final_full_summary = final_summary_result.get('full_summary', '')
                        # ... (로그 추가 가능) ...
                    else:
                         logger.warning("Default Refinement LLM call returned no result. Using fallback.")
                         # Fallback 로직 (기존과 동일)
                         final_full_summary = "\n\n".join([str(chunk.get('full_summary', '')) for chunk in chunk_summaries if chunk])
                         if chunk_summaries and isinstance(chunk_summaries[0], dict):
                            final_one_sentence_summary = chunk_summaries[0].get('one_sentence_summary', '')

                except Exception as refine_e:
                    logger.error(f"Error during default final summary refinement LLM call: {str(refine_e)}")
                    # Fallback 로직 (기존과 동일)
                    final_full_summary = "\n\n".join([str(chunk.get('full_summary', '')) for chunk in chunk_summaries if chunk])
                    if chunk_summaries and isinstance(chunk_summaries[0], dict):
                        final_one_sentence_summary = chunk_summaries[0].get('one_sentence_summary', '')
            else:
                 logger.warning("Skipping default final refinement LLM call because merged sections summary is empty.")

            # 4. 결과 조합 (기본 스키마 형식)
            result = self._get_default_summary('default') # 기본 스키마 + 전략 이름 설정
            result['one_sentence_summary'] = final_one_sentence_summary
            result['full_summary'] = final_full_summary
            result['sections'] = all_sections
            result['keywords'] = all_keywords
            result['sections_summary'] = sections_summary_str

            # 모델 정보는 _get_default_summary 에서 이미 설정됨

            if self.debug_llm:
                 debug_logger.debug("===== Default Chunk Summary Merge Completed =====")
                 # ... (로그 추가 가능) ...

            return result

        except Exception as e:
            logger.error(f"Error merging chunk summaries (default strategy): {str(e)}", exc_info=True)
            return None


    async def _process_large_text(self, chunks: List[str], title: str) -> Optional[Dict]:
        """대용량 텍스트 처리 (챕터 기반) (기존 로직 유지)"""
        try:
            chapters = self._divide_chunks_into_chapters(chunks, self.max_chunks_per_chapter)
            logger.info(f'Divided into {len(chapters)} chapters')

            chapter_info = []
            all_chapter_chunk_summaries = [] # 모든 챕터의 병합된 결과를 모음

            for i, chapter_chunks in enumerate(chapters):
                chapter_num = i + 1
                logger.info(f'Processing Chapter {chapter_num} with {len(chapter_chunks)} chunks')

                # 챕터 내 청크 요약 (비동기)
                tasks = [self._get_chunk_summary(chunk, f"{title} - Chapter {chapter_num}") for chunk in chapter_chunks]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                chunk_summaries = []
                for j, result in enumerate(results):
                     if isinstance(result, Exception):
                          logger.error(f"Error summarizing chunk {j} in chapter {chapter_num}: {result}")
                     elif result:
                          chunk_summaries.append(result)

                if not chunk_summaries:
                    logger.warning(f"No successful summaries for Chapter {chapter_num}. Skipping.")
                    continue

                # 챕터 내 요약 병합 (Refine LLM 호출 포함)
                chapter_title_for_merge = f"{title} - Chapter {chapter_num}"
                # 주의: 여기서 _merge_chunk_summaries는 refine 호출을 포함함.
                # 챕터별 refine을 원하지 않으면 _merge_chunk_summaries 로직 수정 또는 별도 함수 필요
                chapter_summary_merged = await self._merge_chunk_summaries(chunk_summaries, chapter_title_for_merge)

                if not chapter_summary_merged:
                    logger.warning(f"Failed to merge summaries for Chapter {chapter_num}. Skipping.")
                    continue

                # 챕터 정보 저장 (병합된 결과 사용)
                chapter_info.append({
                    'chapter_number': chapter_num,
                    'chapter_title': chapter_title_for_merge, # 임시 제목
                    # 'section_indices': {...}, # 필요 시 계산 추가
                    'sections': chapter_summary_merged.get('sections', []),
                    'keywords': chapter_summary_merged.get('keywords', []),
                    'summary': chapter_summary_merged.get('full_summary', ''), # 챕터별 최종 요약
                    'one_sentence_summary': chapter_summary_merged.get('one_sentence_summary', '') # 챕터별 한 문장 요약
                })
                # 최종 병합을 위해 'chapter_summary_merged' 자체를 사용
                all_chapter_chunk_summaries.append(chapter_summary_merged)


            # 모든 챕터 요약을 최종 병합 (여기서 다시 Refine 호출 발생 가능성!)
            # TODO: _merge_chunk_summaries가 refine 호출을 포함하므로, 중복 호출 방지 로직 필요
            # 예를 들어, _merge_chunk_summaries에 refine 호출 여부 플래그 추가하거나,
            # 챕터 결과를 단순히 취합만 하는 별도 함수 사용
            logger.info("Merging summaries from all chapters...")
            # 임시: 현재 구조상 중복 Refine 발생. 최종 refine만 하도록 수정 필요.
            # 우선은 마지막 챕터의 병합 결과를 사용하거나, 모든 full_summary를 결합하는 임시 방식 사용
            if all_chapter_chunk_summaries:
                 # 임시 방편: 모든 챕터의 full_summary를 결합하고, 마지막 챕터의 다른 정보 사용
                 final_summary = all_chapter_chunk_summaries[-1] # 마지막 챕터 결과 기반
                 final_summary['full_summary'] = "\n\n---\n\n".join(
                     [ch_sum.get('full_summary', '') for ch_sum in all_chapter_chunk_summaries if ch_sum.get('full_summary')]
                 )
                 # 첫 챕터의 한 문장 요약 사용
                 final_summary['one_sentence_summary'] = all_chapter_chunk_summaries[0].get('one_sentence_summary', '')
                 # 섹션과 키워드는 모든 챕터에서 취합
                 final_summary['sections'] = [sec for ch_sum in all_chapter_chunk_summaries for sec in ch_sum.get('sections', [])]
                 # 키워드 취합 및 중복 제거
                 all_ch_keywords_dict = {}
                 for ch_sum in all_chapter_chunk_summaries:
                     for kw in ch_sum.get('keywords', []):
                         term = kw.get('term')
                         if term and term not in all_ch_keywords_dict:
                             all_ch_keywords_dict[term] = kw
                 final_summary['keywords'] = list(all_ch_keywords_dict.values())

                 final_summary['chapters'] = chapter_info # 계산된 챕터 정보 추가
                 final_summary['sections_summary'] = self._format_sections_summary(final_summary['sections']) # 최종 섹션 기반으로 생성
            else:
                 logger.error("No chapter summaries available for final merge.")
                 return None

            return final_summary

        except Exception as e:
            logger.error(f"Error processing large text (default strategy): {str(e)}", exc_info=True)
            return None


    def _divide_chunks_into_chapters(self, chunks: List[str], max_chunks_per_chapter: int) -> List[List[str]]:
        """청크를 챕터로 분할 (기존 로직 유지)"""
        n_chunks = len(chunks)
        if n_chunks == 0: return []
        if n_chunks <= max_chunks_per_chapter: return [chunks]

        # 균등 분배 로직 (기존과 동일)
        n_chapters = (n_chunks + max_chunks_per_chapter - 1) // max_chunks_per_chapter
        # ... (나머지 균등 분배 로직) ...
        base_chunks_per_chapter = n_chunks // n_chapters
        remaining_chunks = n_chunks % n_chapters
        chapters = []
        start_idx = 0
        for i in range(n_chapters):
            chapter_size = base_chunks_per_chapter + (1 if i < remaining_chunks else 0)
            end_idx = start_idx + chapter_size
            if start_idx < n_chunks: # Ensure start_idx is valid
                 chapters.append(chunks[start_idx:end_idx])
            start_idx = end_idx

        if self.verbose:
            logger.info(f'Divided {n_chunks} chunks into {len(chapters)} chapters: {[len(c) for c in chapters]}')
        return chapters


    def _format_sections_summary(self, sections: List[Dict]) -> str:
        """섹션별 요약 포맷팅 (기존 로직)"""
        formatted_sections = []
        for section in sections:
            title = section.get('title', 'Untitled Section')
            summary_content = section.get('summary', '')
            if isinstance(summary_content, list):
                summary_text = "\n".join([f"- {s}" for s in summary_content])
            else:
                summary_text = str(summary_content)
            formatted_sections.append(f"### {title}\n{summary_text}") # Use Markdown heading
        return "\n\n".join(formatted_sections)

    def _format_core_summary(self, sections: List[Dict]) -> str:
         """핵심 주제 포맷팅 (core_summary 제거되었으므로, 필요 시 수정 또는 제거)"""
         # 현재 core_summary는 사용되지 않음. 이 함수는 _get_chunk_summary(단일 청크)에서
         # 임시로 호출될 수 있으나, 실제 결과에는 반영되지 않음.
         # 필요하다면 모든 불릿 포인트를 추출하는 용도로 유지 가능.
         core_points = []
         for section in sections:
             section_summary = section.get('summary', '')
             if isinstance(section_summary, list):
                 processed_summary = "\n".join([f"- {s.strip()}" for s in section_summary if isinstance(s, str) and s.strip()])
                 if processed_summary: core_points.append(processed_summary)
             elif isinstance(section_summary, str) and section_summary.strip():
                 core_points.append(f"- {section_summary.strip()}")
         return "\n".join(core_points)

