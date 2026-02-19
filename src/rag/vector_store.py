"""
Vector store management
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.schema import Document
    from langchain.embeddings.base import Embeddings

from src.config.settings import VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """
        Initialize vector store manager
        
        Args:
            config: Vector store configuration
            embeddings: Embedding model instance
        """
        self.config = config
        self.embeddings = embeddings
        self._vectorstore: Optional[Chroma] = None
    
    def create_from_documents(
        self,
        documents: List[Document],
        force_recreate: bool = False
    ) -> Chroma:
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects
            force_recreate: Force recreation even if store exists
        
        Returns:
            Chroma vector store instance
        """
        persist_dir = Path(self.config.persist_directory)
        
        # If force recreate, remove existing directory
        if force_recreate and persist_dir.exists():
            logger.info(f"Force recreate requested - removing existing vector store")
            import shutil
            shutil.rmtree(persist_dir)
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(persist_dir),
            collection_name=self.config.collection_name
        )
        
        logger.info(f"Vector store created and persisted to {persist_dir}")
        return self._vectorstore
    
    def load_existing(self) -> Optional[Chroma]:
        """
        Load existing vector store
        
        Returns:
            Chroma vector store instance or None if not found
        """
        persist_dir = Path(self.config.persist_directory)
        
        if not persist_dir.exists():
            logger.warning(f"Vector store directory not found: {persist_dir}")
            return None
        
        try:
            # Try loading with specified collection name first
            try:
                self._vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                    collection_name=self.config.collection_name
                )
                # Verify collection exists by checking if we can get collection info
                _ = self._vectorstore._collection.count()
                logger.info(f"Vector store loaded successfully with collection: {self.config.collection_name}")
                return self._vectorstore
            except Exception as e1:
                logger.debug(f"Failed to load with collection name {self.config.collection_name}: {e1}")
                # Try loading without specifying collection name (uses default)
                try:
                    self._vectorstore = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=self.embeddings
                    )
                    _ = self._vectorstore._collection.count()
                    logger.info("Vector store loaded successfully with default collection")
                    return self._vectorstore
                except Exception as e2:
                    logger.error(f"Failed to load vector store: {e2}")
                    raise
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
    
    def get_vectorstore(self) -> Chroma:
        """
        Get vector store instance
        
        Returns:
            Vector store instance
        
        Raises:
            RuntimeError: If vector store not initialized
        """
        if self._vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call create_from_documents() first.")
        return self._vectorstore
