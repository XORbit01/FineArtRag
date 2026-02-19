"""
Main RAG system orchestrator
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config.settings import AppConfig
from src.rag.document_loader import DocumentLoader, DocumentChunker
from src.rag.embeddings import EmbeddingManager
from src.rag.llm import LLMManager
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import RetrieverFactory
from src.rag.chain import RAGChainBuilder
from src.rag.query_router import QueryRouter
from src.rag.hybrid_retriever import HybridRetriever, KeywordRetriever

logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: AppConfig):
        """
        Initialize RAG system
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize managers
        self.document_loader = DocumentLoader(config.documents)
        self.document_chunker = DocumentChunker(config.documents)
        self.embedding_manager = EmbeddingManager(config.embeddings)
        self.llm_manager = LLMManager(config.ollama)
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.rag_chain: Optional[Any] = None
        self._vectorstore: Optional[Any] = None
        self._llm: Optional[Any] = None
        self._chunks: list = []
        self._source_titles: Dict[str, str] = {}
        self._chat_turns: List[Dict[str, str]] = []
        self._chat_summary: str = ""
        self._memory_enabled: bool = config.memory.enabled
        
        logger.info("RAG system initialized")
    
    def initialize(
        self,
        force_recreate: bool = False
    ) -> None:
        """
        Initialize the complete RAG system
        
        Args:
            force_recreate: Force recreation of vector store
        """
        logger.info("Initializing RAG system")
        
        # Initialize embeddings first (needed for vector store)
        embeddings = self.embedding_manager.initialize()
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(
            self.config.vector_store,
            embeddings
        )
        
        # Try to load existing vector store first
        persist_dir = Path(self.config.vector_store.persist_directory)
        if persist_dir.exists() and not force_recreate:
            logger.info(f"Attempting to load existing vector store from {persist_dir}")
            existing_store = self.vector_store_manager.load_existing()
            
            if existing_store is not None:
                logger.info("Using existing vector store - skipping document processing")
                vectorstore = existing_store
            else:
                logger.info("Failed to load existing store - processing documents")
                # Load and chunk documents only if needed
                documents = self.document_loader.load_documents()
                chunks = self.document_chunker.chunk_documents(documents)
                vectorstore = self.vector_store_manager.create_from_documents(
                    chunks,
                    force_recreate=False
                )
        else:
            # No existing store or force recreate
            if force_recreate:
                logger.info("Force recreate requested - processing documents")
            else:
                logger.info("No existing vector store found - processing documents")
            
            documents = self.document_loader.load_documents()
            chunks = self.document_chunker.chunk_documents(documents)
            vectorstore = self.vector_store_manager.create_from_documents(
                chunks,
                force_recreate=force_recreate
            )
        
        # Keep chunked documents for lexical retrieval/reranking.
        docs_for_lexical = self.document_loader.load_documents()
        self._source_titles = self._extract_source_titles(docs_for_lexical)
        self._chunks = self.document_chunker.chunk_documents(docs_for_lexical)

        # Initialize LLM
        llm = self.llm_manager.initialize()
        self._llm = llm
        self._vectorstore = vectorstore
        
        # Create retriever
        retriever = RetrieverFactory.create_retriever(
            vectorstore,
            self.config.retrieval
        )
        
        # Build RAG chain
        chain_builder = RAGChainBuilder(self.config)
        self.rag_chain = chain_builder.build_chain(
            llm,
            retriever,
            return_source_documents=True
        )
        
        logger.info("RAG system initialization complete")
    
    def query(
        self,
        question: str,
        return_sources: bool = True,
        use_memory: bool = True,
        routing_question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            return_sources: Whether to include source documents
            use_memory: Whether to include session memory context
            routing_question: Optional query text used only for routing/retrieval intent
        
        Returns:
            Dictionary with answer and optionally source documents
        
        Raises:
            RuntimeError: If system not initialized
        """
        if self.rag_chain is None:
            raise RuntimeError(
                "RAG system not initialized. Call initialize() first."
            )
        
        user_question = routing_question or question

        # Catalog mode: return complete majors list from indexed docs.
        if self._is_major_list_query(user_question):
            majors_answer, majors_sources = self._build_majors_catalog_response()
            response = {"answer": majors_answer}
            if return_sources:
                response["sources"] = majors_sources
            if use_memory and self._memory_enabled:
                self._update_memory(question=question, answer=response["answer"])
            return response

        route = QueryRouter.route(user_question)
        retrieval_query = route.retrieval_query or question
        question_with_memory = self._augment_query_with_memory(
            question=retrieval_query,
            use_memory=use_memory
        )

        # Build a query-scoped hybrid retriever (dense + lexical).
        if self._vectorstore is None or self._llm is None:
            raise RuntimeError("RAG system internal state is incomplete after initialization.")

        dense_retriever = RetrieverFactory.create_retriever(
            self._vectorstore,
            self.config.retrieval,
            metadata_filter=route.metadata_filter
        )
        keyword_retriever = KeywordRetriever(
            documents=self._chunks,
            k=self.config.retrieval.top_k,
            metadata_filter=route.metadata_filter
        )
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            keyword_retriever=keyword_retriever,
            k=self.config.retrieval.top_k
        )

        chain_builder = RAGChainBuilder(self.config)
        query_chain = chain_builder.build_chain(
            self._llm,
            hybrid_retriever,
            return_source_documents=True
        )
        result = chain_builder.query(query_chain, question_with_memory)
        first_pass_result = result

        # Fallback pass: if answer is insufficient, try baseline retriever without filters.
        insufficient = "i don't have sufficient information in the provided documents" in (
            result.get("result", "").lower()
        )
        if insufficient and route.metadata_filter and self.rag_chain is not None and "contact" not in route.intents:
            logger.info("Filtered hybrid retrieval insufficient, retrying with baseline retriever")
            retry_result = chain_builder.query(self.rag_chain, question_with_memory)
            if retry_result.get("result"):
                result = retry_result

        # Deterministic fallback for contact-intent questions.
        answer_text = result.get("result", "")
        still_insufficient = "i don't have sufficient information in the provided documents" in (
            answer_text.lower()
        )
        if still_insufficient and "contact" in route.intents:
            # Prefer first-pass filtered sources for contact extraction.
            first_sources = first_pass_result.get("source_documents", [])
            fallback = self._contact_fallback_answer(
                user_question=user_question,
                source_documents=first_sources or result.get("source_documents", []),
            )
            if fallback:
                result["result"] = fallback
        
        response = {
            "answer": result.get("result", ""),
        }
        
        if return_sources:
            response["sources"] = result.get("source_documents", [])

        if use_memory and self._memory_enabled:
            self._update_memory(question=question, answer=response["answer"])
        
        return response

    @staticmethod
    def _contact_fallback_answer(user_question: str, source_documents: List[Any]) -> str:
        """Extract contact fields directly from retrieved text when LLM fails."""
        if not source_documents:
            return ""

        text = "\n".join(getattr(d, "page_content", "") or "" for d in source_documents)
        if not text.strip():
            return ""

        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
        phone_match = re.search(r"\(\+\d{1,4}\)\s*[0-9][0-9\s.-]+", text)
        address_match = re.search(r"Address:\s*(.+)", text, re.IGNORECASE)

        email = email_match.group(0).strip() if email_match else ""
        phone = phone_match.group(0).strip() if phone_match else ""
        address = address_match.group(1).strip() if address_match else ""

        if not any([email, phone, address]):
            return ""

        q = (user_question or "").lower()
        wants_email = "email" in q
        wants_phone = any(k in q for k in ["phone", "tel", "call", "number"])
        wants_address = "address" in q or "where" in q

        lines: List[str] = []
        if wants_email and email:
            lines.append(f"Email: {email}")
        if wants_phone and phone:
            lines.append(f"Phone: {phone}")
        if wants_address and address:
            lines.append(f"Address: {address}")

        # If intent was contact but specific field not found in question, return all available.
        if not lines:
            if email:
                lines.append(f"Email: {email}")
            if phone:
                lines.append(f"Phone: {phone}")
            if address:
                lines.append(f"Address: {address}")

        return "\n".join(lines)

    def _is_major_list_query(self, question: str) -> bool:
        q = (question or "").lower()
        asks_list = any(k in q for k in ["list", "all", "show"])
        asks_domain = any(k in q for k in ["major", "majors", "program", "programs", "specialization", "specializations"])
        return asks_list and asks_domain

    def _build_majors_catalog_response(self) -> tuple[str, List[Any]]:
        titles: List[tuple[str, str]] = []
        for source_file, title in self._source_titles.items():
            upper = title.upper()
            if "BACHELOR IN " in upper or "MASTER IN " in upper:
                titles.append((source_file, title))

        bachelor = sorted(
            [t for t in titles if "BACHELOR IN " in t[1].upper()],
            key=lambda x: x[1]
        )
        master = sorted(
            [t for t in titles if "MASTER IN " in t[1].upper()],
            key=lambda x: x[1]
        )
        ordered = bachelor + master

        if not ordered:
            return ("I don't have sufficient information in the provided documents.", [])

        lines = ["Here are the majors available in the provided Fine Arts documents:"]
        for _, title in ordered:
            lines.append(f"- {title}")
        answer = "\n".join(lines)

        source_docs: List[Any] = []
        target_sources = {s for s, _ in ordered}
        seen = set()
        for chunk in self._chunks:
            src = chunk.metadata.get("source_file")
            if src in target_sources and src not in seen:
                source_docs.append(chunk)
                seen.add(src)
            if len(source_docs) >= 12:
                break
        return answer, source_docs

    @staticmethod
    def _extract_source_titles(documents: List[Any]) -> Dict[str, str]:
        titles: Dict[str, str] = {}
        for doc in documents:
            source_path = (doc.metadata or {}).get("source", "")
            source_file = Path(source_path).name if source_path else ""
            if not source_file:
                continue
            content = getattr(doc, "page_content", "") or ""
            title = ""
            for line in content.splitlines()[:12]:
                if line.strip().lower().startswith("title:"):
                    title = line.split(":", 1)[-1].strip()
                    break
            if title:
                titles[source_file] = title
        return titles

    def clear_memory(self) -> None:
        """Clear in-session conversation memory."""
        self._chat_turns = []
        self._chat_summary = ""
        logger.info("Conversation memory cleared")

    def set_memory_enabled(self, enabled: bool) -> None:
        """Enable or disable in-session memory usage."""
        self._memory_enabled = enabled
        logger.info("Conversation memory enabled=%s", enabled)

    def get_memory_status(self) -> Dict[str, Any]:
        """Return current memory state for CLI status display."""
        turn_chars = sum(
            len(t.get("question", "")) + len(t.get("answer", ""))
            for t in self._chat_turns
        )
        return {
            "enabled": self._memory_enabled,
            "turns": len(self._chat_turns),
            "summary_present": bool(self._chat_summary.strip()),
            "buffer_chars": turn_chars,
        }

    def _augment_query_with_memory(self, question: str, use_memory: bool) -> str:
        """
        Add summary + recent turns to the query to support multi-turn context.
        """
        if not use_memory or not self._memory_enabled:
            return question

        if not self._chat_turns and not self._chat_summary:
            return question

        recent = self._chat_turns[-self.config.memory.recent_turns :]
        recent_lines: List[str] = []
        for t in recent:
            q = t.get("question", "").strip()
            a = t.get("answer", "").strip()
            if q:
                recent_lines.append(f"User: {q}")
            if a:
                recent_lines.append(f"Assistant: {a}")

        sections: List[str] = []
        if self._chat_summary.strip():
            sections.append(f"Conversation summary:\n{self._chat_summary.strip()}")
        if recent_lines:
            sections.append("Recent turns:\n" + "\n".join(recent_lines))

        if not sections:
            return question

        return (
            "Use this conversation context only to resolve references like pronouns "
            "or follow-ups. Facts must still come from retrieved documents.\n\n"
            + "\n\n".join(sections)
            + f"\n\nCurrent user question:\n{question}"
        )

    def _update_memory(self, question: str, answer: str) -> None:
        """Append turn and summarize overflow into rolling memory."""
        self._chat_turns.append({
            "question": question.strip(),
            "answer": (answer or "").strip(),
        })

        if len(self._chat_turns) > self.config.memory.max_turns:
            self._chat_turns = self._chat_turns[-self.config.memory.max_turns :]

        self._maybe_summarize_history()

    def _maybe_summarize_history(self) -> None:
        """Summarize old turns when memory buffer grows too large."""
        total_chars = sum(
            len(t.get("question", "")) + len(t.get("answer", ""))
            for t in self._chat_turns
        )
        if total_chars < self.config.memory.summarize_trigger_chars:
            return

        keep_n = max(1, self.config.memory.recent_turns)
        if len(self._chat_turns) <= keep_n:
            return

        older_turns = self._chat_turns[:-keep_n]
        if not older_turns:
            return

        new_summary = self._summarize_turns(older_turns)
        if new_summary:
            if self._chat_summary:
                self._chat_summary = f"{self._chat_summary}\n{new_summary}".strip()
            else:
                self._chat_summary = new_summary.strip()

        self._chat_turns = self._chat_turns[-keep_n:]

    def _summarize_turns(self, turns: List[Dict[str, str]]) -> str:
        """Use the LLM to summarize conversation turns; fallback to rule-based summary."""
        if not turns:
            return ""

        transcript_lines: List[str] = []
        for t in turns:
            q = t.get("question", "").strip()
            a = t.get("answer", "").strip()
            if q:
                transcript_lines.append(f"User: {q}")
            if a:
                transcript_lines.append(f"Assistant: {a}")
        transcript = "\n".join(transcript_lines)

        if self._llm is None:
            return self._fallback_summary(transcript_lines)

        prompt = (
            "Summarize this chat history briefly for future QA continuity.\n"
            "Include: user goals, key facts mentioned, and unresolved asks.\n"
            "Use 5 short bullet points max.\n\n"
            f"{transcript}"
        )
        try:
            msg = self._llm.invoke(prompt)
            content = getattr(msg, "content", msg)
            if isinstance(content, list):
                text = " ".join(str(x) for x in content)
            else:
                text = str(content)
            return text.strip() if text else self._fallback_summary(transcript_lines)
        except Exception as e:
            logger.warning("Memory summarization failed, using fallback: %s", e)
            return self._fallback_summary(transcript_lines)

    @staticmethod
    def _fallback_summary(lines: List[str]) -> str:
        """Simple deterministic fallback summary when LLM summarize fails."""
        if not lines:
            return ""
        sample = lines[-8:]
        return "Summary (fallback): " + " | ".join(sample)[:1200]
