"""
Embedding model management
"""

import logging
from typing import Optional

from langchain_community.embeddings import OllamaEmbeddings
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings

from src.config.settings import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model initialization and operations"""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding manager
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self._embeddings: Optional[Embeddings] = None
    
    def initialize(self) -> Embeddings:
        """
        Initialize embedding model based on configuration
        
        Returns:
            Initialized embedding model instance
        """
        if self._embeddings is not None:
            return self._embeddings
        
        if self.config.provider == "huggingface":
            logger.info(f"Initializing HuggingFace embeddings: {self.config.model_name}")
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name
            )
            logger.info("HuggingFace embeddings initialized successfully")
        
        elif self.config.provider == "ollama":
            logger.info(f"Initializing Ollama embeddings: {self.config.ollama_model}")
            self._embeddings = OllamaEmbeddings(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url
            )
            logger.info("Ollama embeddings initialized successfully")
        
        else:
            raise ValueError(
                f"Unknown embedding provider: {self.config.provider}. "
                f"Must be 'huggingface' or 'ollama'"
            )
        
        return self._embeddings
    
    def get_embeddings(self) -> Embeddings:
        """
        Get embedding model instance
        
        Returns:
            Embedding model instance
        """
        if self._embeddings is None:
            return self.initialize()
        return self._embeddings
