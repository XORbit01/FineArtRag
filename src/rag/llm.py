"""
LLM model management
"""

import logging
from typing import Optional, List

from langchain_community.chat_models import ChatOllama
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import BaseMessage

from src.config.settings import OllamaConfig

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM model initialization and operations"""
    
    def __init__(self, config: OllamaConfig, callbacks: Optional[List[BaseCallbackHandler]] = None):
        """
        Initialize LLM manager
        
        Args:
            config: Ollama configuration
            callbacks: Optional list of callback handlers
        """
        self.config = config
        self.callbacks = callbacks or []
        self._llm: Optional[ChatOllama] = None
    
    def initialize(self) -> ChatOllama:
        """
        Initialize LLM model
        
        Returns:
            Initialized LLM model instance
        """
        if self._llm is not None:
            return self._llm
        
        logger.info(f"Initializing LLM model: {self.config.llm_model}")
        
        try:
            self._llm = ChatOllama(
                model=self.config.llm_model,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                num_ctx=self.config.num_ctx,
                callbacks=self.callbacks
            )
            logger.info("LLM model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
        
        return self._llm
    
    def get_llm(self) -> ChatOllama:
        """
        Get LLM model instance
        
        Returns:
            LLM model instance
        """
        if self._llm is None:
            return self.initialize()
        return self._llm
