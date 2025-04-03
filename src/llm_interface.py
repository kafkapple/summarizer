import logging
import asyncio # asyncio 추가
import time # time 추가
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain.llms.base import BaseLLM
from omegaconf import DictConfig
import json

from src.schema import get_default_summary_schema, get_refine_summary_schema # Import schema functions

# from src.schema import get_default_summary_schema, get_refine_summary_schema # 주석 처리 (파일 생성 후 활성화)

logger, debug_logger = logging.getLogger(__name__), logging.getLogger('llm_debug') # Assume loggers are configured elsewhere

class LLMInterface:
    """LLM 상호작용을 위한 인터페이스"""

    def __init__(self, cfg: DictConfig):
        """LLM 인터페이스 초기화"""
        self.cfg = cfg
        self.debug_llm = cfg.debug_llm
        self.provider = cfg.llm.get('provider', 'openai').lower() # provider 속성 추가 및 초기화 (소문자로)
        self.model_name = cfg.llm.get('model', 'gpt-3.5-turbo')
        self.temperature = float(cfg.llm.get('temperature', 0.3))
        # API 키는 provider에 따라 선택적으로 사용될 수 있음
        self.api_key = cfg.api_keys.get('openai') if self.provider == 'openai' else None
        self.max_token_response = int(cfg.llm.get('response_tokens', 600))

        self.llm = self._create_llm()
        self.system_content = cfg.prompt.get('system_content', '')
        self.instruction_output = cfg.prompt.get('instruction_output', '') # 기본 instruction
        # Refinement 용 instruction (config에서 가져오거나 기본값 사용)
        self.refine_instruction = cfg.prompt.get('refine_instruction',
            "Based on the provided context, generate a final one-sentence summary and a comprehensive full summary in {language}. Output ONLY in JSON format with keys 'one_sentence_summary' and 'full_summary'.")
        self.output_language_full = self._get_language_name(cfg.output.get('language', 'ko'))

    def _get_language_name(self, lang_code: str) -> str:
        """언어 코드를 전체 언어 이름으로 변환"""
        lang_map = {
            'ko': 'Korean', 'en': 'English', 'ja': 'Japanese', 'zh': 'Chinese',
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'ru': 'Russian', 'pt': 'Portuguese'
        }
        return lang_map.get(lang_code, 'English')

    def _create_llm(self):
        """LLM 객체 생성 (max_token_response 반영)"""
        try:
            if self.provider == 'openai':
                if not self.api_key:
                     raise ValueError("OpenAI API key is missing in config.api_keys.openai")
                return ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    max_tokens=self.max_token_response
                )
            elif self.provider == 'ollama':
                ollama_config = self.cfg.llm.get('ollama', {})
                base_url = ollama_config.get('url', 'http://localhost:11434')
                return ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=base_url
                    # Note: Ollama might not respect max_tokens from ChatOpenAI compatible way
                )
            else:
                logger.warning(f"Unsupported provider: {self.provider}, defaulting to OpenAI if possible.")
                # Fallback to OpenAI if key exists
                if self.api_key:
                    return ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=self.temperature,
                        api_key=self.api_key,
                        max_tokens=self.max_token_response
                    )
                else:
                     raise ValueError(f"Unsupported provider '{self.provider}' and no OpenAI fallback possible without API key.")
        except Exception as e:
            logger.error(f"LLM initialization error: {str(e)}")
            # Fallback 로직: API 키 없이 시도할 수 있는 모델이 현재 없으므로 에러 발생
            raise RuntimeError("Failed to initialize any LLM provider.") from e

    async def invoke_llm(self, chunk: str, task_type: str = "summarize_chunk", title: Optional[str] = None) -> Optional[Dict]:
        """LLM을 호출하여 작업을 수행하고 JSON으로 파싱합니다. (재시도 로직 포함)"""

        max_retries = self.cfg.llm.retry.get('max_retries', 3)
        initial_delay = self.cfg.llm.retry.get('initial_delay', 1)
        backoff_factor = self.cfg.llm.retry.get('backoff_factor', 2)
        current_delay = float(initial_delay) # float 형으로 변환

        response_text = None
        result_obj = None

        formatted_system_content = self.system_content.format(language=self.output_language_full)
        user_content = chunk
        instruction = ""

        # 작업 유형에 따라 instruction 결정
        if task_type == "summarize_chunk":
            instruction = self.instruction_output
        elif task_type == "refine":
            instruction = self.refine_instruction.format(language=self.output_language_full)
            user_content = f"Original Title: {title}\n\nContext to refine:\n{chunk}"
        else:
            logger.warning(f"Unknown task_type: {task_type}. Using default summarization instruction.")
            instruction = self.instruction_output

        # 로깅 (반복문 밖에서 한 번만 수행)
        if self.debug_llm:
            debug_logger.debug(f"===== LLM Invocation Attempt (Task: {task_type}) =====")
            debug_logger.debug(f"Input Content (first 100 chars): {user_content[:100]}...")
            debug_logger.debug(f"System Content: {formatted_system_content}")
            debug_logger.debug(f"Instruction: {instruction}")
            debug_logger.debug(f"Temperature: {self.temperature}")
            debug_logger.debug(f"Max Response Tokens: {self.max_token_response}")
            debug_logger.debug(f"Retry Config: Max={max_retries}, Delay={initial_delay}, Backoff={backoff_factor}")
            print(f"\n[DEBUG] Preparing LLM Request (Task: {task_type})...")

        # --- 재시도 루프 시작 ---
        for attempt in range(max_retries + 1):
            try:
                if self.debug_llm:
                    print(f"[DEBUG] Attempt {attempt + 1}/{max_retries + 1}: Sending request to LLM...")

                # 최종 프롬프트/메시지 구성 및 LLM 호출
                if isinstance(self.llm, ChatOpenAI):
                     messages = [
                         {"role": "system", "content": formatted_system_content},
                         {"role": "user", "content": user_content},
                     ]
                     if instruction:
                         messages.append({"role": "system", "content": instruction})
                     result_obj = await self.llm.ainvoke(messages)
                     response_text = result_obj.content if hasattr(result_obj, 'content') else str(result_obj)

                elif isinstance(self.llm, ChatOllama):
                     prompt = f"""{formatted_system_content}

{user_content}

{instruction}""" # Ensure triple quotes for multi-line f-string
                     result_obj = await self.llm.ainvoke(prompt)
                     if hasattr(result_obj, 'content'):
                          response_text = result_obj.content
                     else:
                          response_text = str(result_obj) # Fallback
                else:
                     prompt = f"""{formatted_system_content}

{user_content}

{instruction}""" # Ensure triple quotes for multi-line f-string
                     result_obj = await self.llm.ainvoke(prompt)
                     response_text = str(result_obj)

                if self.debug_llm:
                    print(f"[DEBUG] Attempt {attempt + 1}: Received response.")
                    debug_logger.debug(f"----- LLM Raw Response (Attempt {attempt + 1}) -----")
                    debug_logger.debug(f"Response type: {type(result_obj)}")
                    debug_logger.debug(f"Raw response text: {response_text}")

                # 응답 유효성 검사
                if response_text and response_text.strip():
                    if self.debug_llm: print(f"[DEBUG] Attempt {attempt + 1}: Success! Response is not empty.")
                    break # 성공 시 루프 탈출
                else:
                    # 응답이 비었지만 오류는 아님
                    logger.warning(f"LLM returned empty response on attempt {attempt + 1}.")
                    if self.debug_llm: print(f"[DEBUG] Attempt {attempt + 1}: Failed! Response is empty.")
                    # 마지막 시도였으면 response_text는 None 또는 빈 문자열 상태 유지
                    if attempt == max_retries:
                         logger.error(f"LLM returned empty response after {max_retries + 1} attempts.")
                         # response_text 는 이미 None 또는 빈 상태이므로 추가 작업 없음

            except Exception as e:
                logger.warning(f"Error during LLM invocation on attempt {attempt + 1}: {str(e)}")
                if self.debug_llm:
                    print(f"[DEBUG] Attempt {attempt + 1}: Failed! Exception: {e}")
                    debug_logger.warning(f"LLM invocation error (Attempt {attempt + 1}): {str(e)}", exc_info=True)
                # 마지막 시도였으면 루프 종료
                if attempt == max_retries:
                    logger.error(f"LLM invocation failed after {max_retries + 1} attempts due to error: {str(e)}")
                    response_text = None # 확실하게 실패 처리
                    break # 루프 탈출

            # 재시도 전 대기 (마지막 시도 제외)
            if attempt < max_retries:
                logger.info(f"Retrying LLM call in {current_delay:.2f} seconds... (Attempt {attempt + 2}/{max_retries + 1})")
                if self.debug_llm: print(f"[DEBUG] Waiting {current_delay:.2f} seconds before next attempt...")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
        # --- 재시도 루프 종료 ---

        # 여기부터는 기존 로직과 동일 (response_text 처리)
        if not response_text or not response_text.strip():
            logger.error("LLM response is empty after all retries.")
            if self.debug_llm:
                 debug_logger.debug("Empty response after retries detected")
            # 작업 유형에 따라 기본 구조 반환
            return self._default_summary_structure(task_type) if task_type != "refine" else {"full_summary": "", "one_sentence_summary": ""}

        # JSON Parsing (Only if the task expected JSON)
        parsed_result = None
        if task_type == "summarize_chunk" or task_type == "refine": # 두 태스크 모두 JSON 필요
            parsed_result = self._process_json_response(response_text)
            if task_type == "refine": # Refine 특정 키 확인/추가
                if parsed_result:
                    parsed_result.setdefault('one_sentence_summary', '')
                    parsed_result.setdefault('full_summary', '')
                else:
                    parsed_result = {'one_sentence_summary': '', 'full_summary': ''}
        else:
            parsed_result = {"full_summary": response_text}

        if not parsed_result:
             # 작업 유형에 따라 기본 구조 반환
             default_structure = self._default_summary_structure(task_type) if task_type != "refine" else {"full_summary": "", "one_sentence_summary": ""}
             logger.warning("Parsing failed or task didn't expect JSON. Returning default/raw structure.")
             return default_structure

        if self.debug_llm:
            debug_logger.debug("----- Final Parsed/Processed Response -----")
            debug_logger.debug(f"Final result for task '{task_type}': {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")

        return parsed_result

    def _process_json_response(self, response: str) -> Optional[Dict]:
        """LLM 응답 문자열을 JSON으로 파싱합니다."""
        try:
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            if not json_str: return None
            result = json.loads(json_str)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            if self.debug_llm: debug_logger.debug(f"Failed to parse JSON: {response}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error processing JSON response: {str(e)}")
             return None

    def _default_summary_structure(self, task_type: str = "summarize_chunk") -> Dict:
         """Provides a default structure for summary results based on task type."""
         # 임시: schema.py 파일 생성 전까지 하드코딩 유지 -> 제거
         # if task_type == "refine":
         #     return {"one_sentence_summary": "", "full_summary": ""}
         # else: # Default structure for chunk summary
         #     return {
         #        "keywords": [],
         #        "sections": [],
         #        "full_summary": [], 
         #        "one_sentence_summary": ""
         #     }
         # 아래 코드는 schema.py 생성 후 활성화 -> 활성화
         if task_type == "refine":
             return get_refine_summary_schema()
         else: # Default structure for chunk summary or other tasks
             # Return a copy to prevent modification of the original schema dict
             return get_default_summary_schema().copy()

    def get_llm(self) -> BaseLLM:
        """Returns the initialized LangChain LLM object."""
        return self.llm 