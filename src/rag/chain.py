"""
RAG chain construction and management
"""

import logging
from typing import Dict, Any, Optional, Union

# Try modern LangChain pattern first
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    MODERN_LANGCHAIN = True
except ImportError:
    MODERN_LANGCHAIN = False

# Fallback to classic RetrievalQA
if not MODERN_LANGCHAIN:
    try:
        from langchain_classic.chains import RetrievalQA
        CLASSIC_AVAILABLE = True
    except ImportError:
        try:
            from langchain.chains import RetrievalQA
            CLASSIC_AVAILABLE = True
        except ImportError:
            CLASSIC_AVAILABLE = False
            raise ImportError(
                "Neither modern LangChain retrieval chain nor RetrievalQA is available. "
                "Please install langchain-classic or update langchain."
            )

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
try:
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    try:
        from langchain.schema import BaseRetriever
    except ImportError:
        from langchain_core.retrievers import BaseRetriever

from src.config.settings import AppConfig

logger = logging.getLogger(__name__)


class RAGChainBuilder:
    """Builds RAG chains with custom prompts"""
    
    DEFAULT_PROMPT_TEMPLATE = """You are a factual assistant for the Lebanese University Faculty of Fine Arts & Architecture.

Answer ONLY from the provided context.
Rules:
- Treat document titles/headings as valid evidence.
- If the requested program/term appears in context, do NOT say it is missing.
- Do not speculate or add outside knowledge.
- If a user asks for a person but no person is named in context, provide the available official faculty contact details instead.
- If context is truly insufficient, say: "I don't have sufficient information in the provided documents."
- Keep the answer concise and direct.

Context: {context}

Question: {question}

Answer:"""
    
    def __init__(self, config: AppConfig):
        """
        Initialize RAG chain builder
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def build_chain(
        self,
        llm: ChatOllama,
        retriever: BaseRetriever,
        prompt_template: Optional[str] = None,
        return_source_documents: bool = True
    ) -> Union[Any, "RetrievalQA"]:
        """
        Build RAG chain using modern or classic approach
        
        Args:
            llm: LLM model instance
            retriever: Retriever instance
            prompt_template: Custom prompt template (optional)
            return_source_documents: Whether to return source documents
        
        Returns:
            Configured RAG chain
        """
        logger.info("Building RAG chain")
        
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        if MODERN_LANGCHAIN:
            # Modern LangChain pattern
            logger.info("Using modern LangChain retrieval chain")
            document_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, document_chain)
            chain._return_source_documents = return_source_documents
        elif CLASSIC_AVAILABLE:
            # Classic RetrievalQA pattern
            logger.info("Using classic RetrievalQA chain")
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": prompt}
            )
        else:
            raise RuntimeError("No available RAG chain implementation")
        
        logger.info("RAG chain built successfully")
        return chain
    
    def query(
        self,
        chain: Union[Any, "RetrievalQA"],
        question: str
    ) -> Dict[str, Any]:
        """
        Execute query on RAG chain
        
        Args:
            chain: RAG chain instance
            question: User question
        
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Executing query: {question[:50]}...")
        
        try:
            if MODERN_LANGCHAIN:
                # Modern chain uses "input" key
                result = chain.invoke({"input": question})
                # Normalize response format - modern chain returns answer and context
                answer = result.get("answer", "")
                context = result.get("context", [])
                return {
                    "result": answer,
                    "source_documents": context if isinstance(context, list) else []
                }
            else:
                # Classic RetrievalQA - use invoke method
                if hasattr(chain, 'invoke'):
                    result = chain.invoke({"query": question})
                else:
                    result = chain({"query": question})
                logger.info("Query executed successfully")
                return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
