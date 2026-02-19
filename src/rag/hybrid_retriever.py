"""
Hybrid retrievers combining dense vector retrieval with lightweight lexical scoring.
"""

import re
from typing import Any, Dict, List, Optional

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    from langchain.schema import Document, BaseRetriever
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_PATTERN.findall((text or "").lower()) if len(t) > 2]


def _match_metadata(doc_metadata: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    if not metadata_filter:
        return True
    for key, value in metadata_filter.items():
        doc_value = doc_metadata.get(key)
        if isinstance(value, dict) and "$in" in value:
            if doc_value not in value["$in"]:
                return False
        elif doc_value != value:
            return False
    return True


class KeywordRetriever(BaseRetriever):
    """BM25 lexical retriever with metadata filtering."""

    documents: List[Document]
    k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        candidate_docs: List[Document] = []
        for doc in self.documents:
            if not _match_metadata(doc.metadata, self.metadata_filter):
                continue
            candidate_docs.append(doc)

        if not candidate_docs:
            return []

        tokenized_corpus = [_tokenize(doc.page_content) for doc in candidate_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(q_tokens)

        ranked = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True
        )
        return [doc for _, doc in ranked[: self.k]]


class HybridRetriever(BaseRetriever):
    """Reciprocal-rank fusion over dense and lexical retrievers."""

    dense_retriever: BaseRetriever
    keyword_retriever: KeywordRetriever
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)

        # RRF merge by source/chunk id identity.
        fused: Dict[str, Dict[str, Any]] = {}

        def doc_key(doc: Document) -> str:
            source = doc.metadata.get("source_file", "unknown")
            chunk = doc.metadata.get("chunk_id", -1)
            return f"{source}::{chunk}"

        for rank, doc in enumerate(dense_docs, start=1):
            key = doc_key(doc)
            fused.setdefault(key, {"doc": doc, "score": 0.0})
            fused[key]["score"] += 1.0 / (60 + rank)

        for rank, doc in enumerate(keyword_docs, start=1):
            key = doc_key(doc)
            fused.setdefault(key, {"doc": doc, "score": 0.0})
            fused[key]["score"] += 1.0 / (60 + rank)

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[: self.k]]
