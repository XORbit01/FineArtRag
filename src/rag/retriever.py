"""
Retrieval configuration and management
"""

import logging
from typing import Optional, Dict, Any

from langchain_community.vectorstores import Chroma
try:
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.schema import BaseRetriever

from src.config.settings import RetrievalConfig

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating retrievers"""
    
    @staticmethod
    def create_retriever(
        vectorstore: Chroma,
        config: RetrievalConfig,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Create retriever with specified configuration
        
        Args:
            vectorstore: Chroma vector store instance
            config: Retrieval configuration
        
        Returns:
            Configured retriever instance
        """
        logger.info(
            f"Creating retriever: type={config.search_type}, top_k={config.top_k}, "
            f"filtered={bool(metadata_filter)}"
        )
        base_kwargs: Dict[str, Any] = {}
        if metadata_filter:
            base_kwargs["filter"] = metadata_filter
        
        if config.search_type == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": config.top_k,
                    "fetch_k": config.top_k * config.fetch_k_multiplier,
                    "lambda_mult": config.mmr_lambda,
                    **base_kwargs
                }
            )
        elif config.search_type == "similarity":
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": config.top_k,
                    **base_kwargs
                }
            )
        else:
            logger.warning(f"Unknown search type: {config.search_type}, using similarity")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": config.top_k,
                    **base_kwargs
                }
            )
        
        logger.info("Retriever created successfully")
        return retriever
