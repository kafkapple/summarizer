# src/summarizer/factory.py
import logging
from omegaconf import DictConfig

from src.llm_interface import LLMInterface
from src.summarizer.base import BaseSummarizer
# 각 전략 클래스 임포트
from src.summarizer.strategies.default_strategy import DefaultSummarizer
from src.summarizer.strategies.multi_chain_strategy import MultiChainSummarizer
from src.summarizer.strategies.langchain_standard import StandardLangChainSummarizer
from src.summarizer.strategies.stuff_strategy import StuffSummarizer

logger = logging.getLogger(__name__)

def create_summarizer(cfg: DictConfig, llm_interface: LLMInterface) -> BaseSummarizer:
    """
    설정(cfg)에 지정된 요약 전략에 따라 적절한 Summarizer 인스턴스를 생성하고 반환합니다.

    Args:
        cfg: 전체 설정 객체
        llm_interface: 초기화된 LLMInterface 인스턴스

    Returns:
        선택된 전략에 해당하는 BaseSummarizer의 구체적인 인스턴스.
        지원되지 않는 전략이거나 오류 발생 시 DefaultSummarizer를 반환할 수 있음 (또는 에러 발생).
    """
    strategy_name = cfg.summary.get('strategy', 'default').lower()
    logger.info(f"Creating summarizer for strategy: {strategy_name}")

    try:
        if strategy_name == 'default':
            return DefaultSummarizer(cfg, llm_interface)
        elif strategy_name == 'multi_chain':
            return MultiChainSummarizer(cfg, llm_interface)
        elif strategy_name in ['map_reduce', 'refine']:
            # StandardLangChainSummarizer는 생성 시 전략 이름도 받음
            return StandardLangChainSummarizer(cfg, llm_interface, strategy=strategy_name)
        elif strategy_name == 'stuff':
            # Stuff 전략은 짧은 텍스트에만 적합 - main.py 등 호출부에서 길이 체크 후 사용 권장
            # 또는 여기서 길이 체크 후 다른 전략으로 fallback 할 수도 있음
            return StuffSummarizer(cfg, llm_interface)
        else:
            logger.warning(f"Unsupported summary strategy '{strategy_name}'. Falling back to 'default'.")
            # fallback으로 default 전략 사용
            return DefaultSummarizer(cfg, llm_interface)

    except Exception as e:
        logger.error(f"Error creating summarizer for strategy '{strategy_name}': {e}", exc_info=True)
        logger.warning("Falling back to 'default' strategy due to creation error.")
        # 생성 오류 시에도 fallback으로 default 전략 사용
        # 필요하다면 여기서 None을 반환하거나 에러를 다시 발생시킬 수 있음
        return DefaultSummarizer(cfg, llm_interface) 