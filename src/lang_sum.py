from src.utils import setup_logging
from src.sum import LangChainSummarizer
from omegaconf import DictConfig
import re
from typing import List, Dict, Optional, Any, Union
import asyncio
import logging
import httpx

# 로거 초기화
logger, debug_logger = setup_logging()

class LangChainSummarizerSync:
    """Synchronous wrapper for the asynchronous LangChainSummarizer"""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the synchronous wrapper"""
        # Prepare for error logging
        import sys
        import io
        import logging
        
        self.logger = logging.getLogger(__name__)
        
        # Temporarily redirect stderr to capture initialization errors
        error_stream = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = error_stream
        
        try:
            # Initialize async summarizer
            self.async_summarizer = LangChainSummarizer(cfg)
            
            # Copy attributes from async summarizer
            for attr_name in dir(self.async_summarizer):
                if not attr_name.startswith('_') and not callable(getattr(self.async_summarizer, attr_name)):
                    setattr(self, attr_name, getattr(self.async_summarizer, attr_name))
                    
        except Exception as e:
            # Restore original stderr
            sys.stderr = original_stderr
            
            # Log error details
            error_content = error_stream.getvalue()
            self.logger.error(f"LangChainSummarizer initialization error: {str(e)}")
            self.logger.error(f"Error details:\n{error_content}")
            raise
        finally:
            # Restore original stderr
            sys.stderr = original_stderr
    
    def summarize(self, text: str, title: str) -> Optional[Dict]:
        """
        Synchronous version of the summarize method.
        Calls the main asynchronous summarize method from LangChainSummarizer 
        and runs it synchronously using asyncio.run().
        
        Args:
            text: Text to summarize
            title: Content title
            
        Returns:
            Dictionary with summary information or None on failure.
        """
        try:
            # Basic input validation (can be expanded)
            if not isinstance(text, str):
                self.logger.warning(f"Input type {type(text)} is not str. Attempting conversion.")
                text = str(text)
            if not text.strip():
                self.logger.error("Input text is empty or whitespace.")
                return None
                
            if self.async_summarizer.debug_llm:
                print(f"\n[DEBUG] Starting sync summarization wrapper for: {title}")
                debug_logger.debug(f"===== Starting Summarization via Sync Wrapper for: {title} =====")

            # Call the main asynchronous summarize method
            result = asyncio.run(self.async_summarizer.summarize(text, title))

            if self.async_summarizer.debug_llm:
                print(f"[DEBUG] Finished sync summarization wrapper for: {title}")
                debug_logger.debug(f"===== Finished Summarization via Sync Wrapper for: {title} =====")

            return result
            
        except Exception as e:
            self.logger.error(f"Synchronous summarization error for title '{title}': {str(e)}")
            if hasattr(self, 'async_summarizer') and self.async_summarizer.debug_llm:
                debug_logger.error(f"Synchronous summarization error for title '{title}': {str(e)}", exc_info=True)
            return None # Return None on general failure