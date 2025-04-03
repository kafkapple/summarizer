import logging
from typing import Dict, Optional, List, Any
import asyncio
from collections import Counter
import json

from omegaconf import DictConfig
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RefineDocumentsChain
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.summarizer.base import BaseSummarizer
from src.llm_interface import LLMInterface
from src.schema import get_default_summary_schema # 스키마 가져오기

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger('llm_debug') # 임시 설정

class MultiChainSummarizer(BaseSummarizer):
    """다중 LangChain 체인을 조합하여 요약을 생성하는 전략"""

    def __init__(self, cfg: DictConfig, llm_interface: LLMInterface):
        super().__init__(cfg, llm_interface)
        # MultiChain 전략 관련 추가 설정 (필요 시)
        self._load_prompts()

    def _load_prompts(self):
        self.section_prompt_template = self.cfg.prompt.get('multi_chain_section_structured')
        self.refine_intermediate_prompt_template = self.cfg.prompt.get('multi_chain_refine_intermediate')
        self.refine_final_prompt_template = self.cfg.prompt.get('multi_chain_refine_final')
        self.one_sentence_prompt_template = self.cfg.prompt.get('multi_chain_one_sentence')

        if not all([self.section_prompt_template, self.refine_intermediate_prompt_template,
                    self.refine_final_prompt_template, self.one_sentence_prompt_template]):
            raise ValueError("One or more required prompts for multi_chain strategy are missing in the configuration.")

    async def summarize(self, text: str, title: Optional[str] = None) -> Dict:
        """'multi_chain' 전략을 사용하여 텍스트를 요약합니다."""
        strategy_name = 'multi_chain'
        logger.info(f"Executing {strategy_name} strategy for: {title or 'Untitled'}")
        summary_result = self._get_default_summary(strategy_name)

        try:
            # 1. 텍스트 정리 및 언어 감지
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.error("Cleaned text is empty.")
                return summary_result
            source_lang = self.detect_language(cleaned_text)
            logger.info(f"Detected source language: {source_lang}")

            # 2. LLM 및 기본 설정 가져오기
            llm = self.llm_interface.get_llm()
            if not llm:
                logger.error("LLM not available for multi_chain strategy.")
                return summary_result

            # 3. 텍스트 청킹
            docs = self.text_splitter.split_text(cleaned_text)
            docs = [Document(page_content=chunk) for chunk in docs]
            n_chunks = len(docs)
            logger.info(f"Split text into {n_chunks} chunks.")
            if n_chunks == 0:
                return summary_result

            # 4. Section-Level Summarization (Map Only)
            section_data_list, all_keywords_list = await self._run_section_level_summary(docs, llm)
            logger.info(f"Generated {len(section_data_list)} section summaries.")

            # 5. Medium-Level Summarization (Refine)
            medium_summary = await self._run_medium_level_summary(docs, llm)
            logger.info("Generated medium-level summary.")

            # 6. Single-Sentence Summarization (from Medium Summary)
            single_sentence = await self._run_single_sentence_summary(medium_summary, llm)
            if single_sentence:
                logger.info("Generated single-sentence summary from medium summary.")
            else:
                 logger.warning("Could not generate single-sentence summary.")

            # 7. 결과 매핑 및 통합
            final_summary = self._map_results_to_schema(
                single_sentence, medium_summary, section_data_list, all_keywords_list, strategy_name
            )

            # 8. 번역 처리
            translated_summary = await self._translate_summary(final_summary)

            # 9. 최종 결과 정리
            final_result = self._get_default_summary(strategy_name)
            final_result.update(translated_summary)
            if self.include_full_text:
                final_result['full_text'] = cleaned_text
            final_result['summary_strategy_used'] = strategy_name # 재확인

            logger.info(f"Successfully completed {strategy_name} summarization.")
            return final_result

        except Exception as e:
            logger.error(f"Error during {strategy_name} summarization: {str(e)}", exc_info=True)
            logger.info("Attempting fallback to default strategy...")
            try:
                from .default_strategy import DefaultSummarizer # 지연 import
                fallback_summarizer = DefaultSummarizer(self.cfg, self.llm_interface)
                return await fallback_summarizer.summarize(text, title)
            except Exception as fallback_e:
                 logger.error(f"Fallback to default strategy also failed: {fallback_e}", exc_info=True)
                 return self._get_default_summary(f"{strategy_name}_failed_fallback_to_default")

    async def _run_section_level_summary(self, docs: List[Document], llm) -> tuple[List[Dict], List[str]]:
        """각 청크에 대해 섹션 정보(제목, 요약점, 키워드)를 추출합니다."""
        section_data_list = []
        all_keywords_list = []

        section_prompt = PromptTemplate.from_template(self.section_prompt_template)
        section_chain = section_prompt | llm | JsonOutputParser()

        section_tasks = []
        for i, doc in enumerate(docs):
            task = section_chain.ainvoke({'text': doc.page_content, 'language': self._get_language_name(self.output_language)})
            section_tasks.append(task)
        
        section_results = await asyncio.gather(*section_tasks, return_exceptions=True)
        
        logger.debug(f"Processing {len(section_results)} raw results from section summarization.")
        for i, result in enumerate(section_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {i} for section summary: {result}")
                continue
            
            if isinstance(result, dict) and 'title' in result and 'summary' in result and 'keywords' in result:
                section_data = {
                    'title': result.get('title', f'Section {i+1}'),
                    'summary': result.get('summary', [])
                }
                section_data_list.append(section_data)
                
                keywords = result.get('keywords', [])
                if isinstance(keywords, list):
                    all_keywords_list.extend([kw for kw in keywords if isinstance(kw, str)])
                elif isinstance(keywords, str):
                     all_keywords_list.extend([kw.strip() for kw in keywords.split(',') if kw.strip()])
            else:
                logger.warning(f"Chunk {i} result parsing failed, missing keys, or not a dict. Raw type: {type(result)}, Content: {str(result)[:100]}...")
                section_data_list.append({'title': f'Section {i+1}', 'summary': ["Summary generation failed for this section."]})

        return section_data_list, all_keywords_list

    async def _run_medium_level_summary(self, docs: List[Document], llm) -> str:
        """Refine 체인을 사용하여 중간 길이 요약을 생성합니다."""
        medium_initial_template = self.cfg.prompt.get('multi_chain_refine_intermediate')
        medium_refine_template = self.cfg.prompt.get('multi_chain_refine_final')
        if not medium_initial_template or not medium_refine_template:
             logger.error("Prompts for medium summary (initial/refine) not found in config.")
             return ""

        medium_summary_prompt = PromptTemplate(template=medium_initial_template, input_variables=["text", "language"])
        refine_prompt = PromptTemplate(template=medium_refine_template, input_variables=["existing_summary", "text", "language"])

        medium_summary_chain = RefineDocumentsChain(
            initial_llm_chain=LLMChain(llm=llm, prompt=medium_summary_prompt),
            refine_llm_chain=LLMChain(llm=llm, prompt=refine_prompt),
            document_variable_name="text",
            initial_response_name="existing_summary",
        )
        try:
             result = await medium_summary_chain.ainvoke({"input_documents": docs, "language": self._get_language_name(self.output_language)})
             medium_summary = result.get('output_text', '')
             return medium_summary.strip()
        except Exception as e:
             logger.error(f"Error running medium level summary chain: {e}", exc_info=True)
             return ""

    async def _run_single_sentence_summary(self, medium_summary: str, llm) -> str:
        """중간 요약을 기반으로 한 문장 요약을 생성합니다."""
        single_sentence = ""
        if medium_summary:
             single_prompt_template = self.cfg.prompt.get('multi_chain_one_sentence')
             if not single_prompt_template:
                 logger.error("Prompt 'multi_chain_one_sentence' not found in config.")
             else:
                 try:
                     single_prompt = PromptTemplate.from_template(single_prompt_template)
                     single_chain = single_prompt | llm | StrOutputParser()
                     single_sentence = await single_chain.ainvoke({"existing_summary": medium_summary, "language": self._get_language_name(self.output_language)})
                     return single_sentence.strip()
                 except Exception as e:
                     logger.error(f"Error running single sentence summary chain: {e}", exc_info=True)
        return ""

    def _map_results_to_schema(self,
                               single_sentence: str,
                               medium_summary: str,
                               section_data_list: List[Dict],
                               all_keywords_list: List[str],
                               strategy_name: str) -> Dict:
        """multi_chain 결과를 기본 스키마 형식으로 매핑합니다."""
        final_summary = self._get_default_summary(strategy_name)

        final_summary['one_sentence_summary'] = single_sentence
        final_summary['full_summary'] = medium_summary
        final_summary['sections'] = section_data_list

        # 키워드 처리
        unique_keywords = list(set(all_keywords_list))
        final_summary['keywords'] = [{'term': kw} for kw in unique_keywords]

        # sections_summary 생성 (BaseSummarizer의 헬퍼 사용 또는 직접 구현)
        # 예시: 간단한 직접 구현
        formatted_sections = []
        for section in section_data_list:
            title = section.get('title', 'Untitled Section')
            summary_content = section.get('summary', [])
            if isinstance(summary_content, list):
                summary_text = "\n".join([f"- {s}" for s in summary_content])
            else:
                summary_text = str(summary_content)
            formatted_sections.append(f"### {title}\n{summary_text}")
        final_summary['sections_summary'] = "\n\n".join(formatted_sections)

        # chapters는 이 전략에서 생성되지 않음 (기본값 [] 유지)

        return final_summary 