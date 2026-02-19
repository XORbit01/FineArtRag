"""
Service layer for API endpoints using the core RAGSystem interface.
"""

import threading
import uuid
import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.settings import AppConfig
from src.rag.system import RAGSystem


@dataclass
class ChatSessionState:
    """Per-websocket in-memory conversation state."""
    summary: str = ""
    turns: List[Dict[str, str]] = field(default_factory=list)
    recent_turns: int = 4
    summarize_trigger_chars: int = 2800

    def build_prompt(self, message: str) -> str:
        if not self.turns and not self.summary:
            return message

        recent = self.turns[-self.recent_turns :]
        recent_lines: List[str] = []
        for t in recent:
            q = t.get("question", "").strip()
            a = t.get("answer", "").strip()
            if q:
                recent_lines.append(f"User: {q}")
            if a:
                recent_lines.append(f"Assistant: {a}")

        sections: List[str] = []
        if self.summary.strip():
            sections.append(f"Conversation summary:\n{self.summary}")
        if recent_lines:
            sections.append("Recent turns:\n" + "\n".join(recent_lines))

        return (
            "Use chat context only to resolve follow-ups/references. "
            "All factual claims must come from retrieved documents.\n\n"
            + "\n\n".join(sections)
            + f"\n\nCurrent user question:\n{message}"
        )

    def update(self, question: str, answer: str, sources: Optional[List[Dict[str, str]]] = None) -> None:
        self.turns.append(
            {
                "question": question.strip(),
                "answer": (answer or "").strip(),
                "sources": sources or [],
            }
        )
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        chars = sum(len(t["question"]) + len(t["answer"]) for t in self.turns)
        if chars < self.summarize_trigger_chars or len(self.turns) <= self.recent_turns:
            return
        older = self.turns[:-self.recent_turns]
        summary_line = " | ".join(
            f"U:{t['question'][:120]} A:{t['answer'][:160]}" for t in older[-8:]
        )
        merged = f"{self.summary} {summary_line}".strip()
        self.summary = merged[-2500:]
        self.turns = self.turns[-self.recent_turns :]


@dataclass
class ChatRecord:
    chat_id: str
    title: str
    created_at: float
    updated_at: float
    state: ChatSessionState


class RAGApiService:
    """API-facing service that owns and uses a single RAGSystem instance."""

    def __init__(self, config: AppConfig, force_recreate: bool = False):
        self.config = config
        self.force_recreate = force_recreate
        self._rag: Optional[RAGSystem] = None
        self._lock = threading.Lock()
        self._chats: Dict[str, ChatRecord] = {}
        self._starter_questions: List[str] = []
        self._starter_sources: List[Dict[str, str]] = []
        self._source_url_by_file: Dict[str, str] = {}

    def start(self) -> None:
        """Initialize the underlying RAG system once."""
        with self._lock:
            if self._rag is not None:
                return
            rag = RAGSystem(self.config)
            rag.initialize(force_recreate=self.force_recreate)
            self._rag = rag
            self._starter_questions, self._starter_sources = self._build_starter_templates()

    def is_initialized(self) -> bool:
        return self._rag is not None

    def query(
        self,
        question: str,
        return_sources: bool = True,
        use_memory: bool = False,
        routing_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._rag is None:
            self.start()
        assert self._rag is not None
        result = self._rag.query(
            question=question,
            return_sources=return_sources,
            use_memory=use_memory,
            routing_question=routing_question,
        )
        return {
            "answer": result.get("answer", ""),
            "sources": self._serialize_sources(result.get("sources", [])) if return_sources else [],
        }

    def create_chat(self, title: Optional[str] = None) -> Dict[str, Any]:
        chat_id = uuid.uuid4().hex
        now = time.time()
        record = ChatRecord(
            chat_id=chat_id,
            title=(title or "Untitled chat").strip() or "Untitled chat",
            created_at=now,
            updated_at=now,
            state=ChatSessionState(
                recent_turns=self.config.memory.recent_turns,
                summarize_trigger_chars=self.config.memory.summarize_trigger_chars,
            ),
        )
        self._chats[chat_id] = record
        return self._chat_summary(record)

    def create_session(self) -> str:
        """Backward compatible alias used by websocket router."""
        chat = self.create_chat()
        return chat["chat_id"]

    def list_chats(self) -> List[Dict[str, Any]]:
        chats = [self._chat_summary(rec) for rec in self._chats.values()]
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        rec = self._chats.get(chat_id)
        if rec is None:
            rec = self._ensure_chat(chat_id)
        return {
            "chat_id": rec.chat_id,
            "title": rec.title,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "summary": rec.state.summary,
            "turns": list(rec.state.turns),
        }

    def close_session(self, session_id: str) -> None:
        """No-op: chats are persisted in backend memory until process exit."""
        _ = session_id

    def delete_chat(self, chat_id: str) -> bool:
        return self._chats.pop(chat_id, None) is not None

    def reset_session(self, session_id: str) -> None:
        rec = self._ensure_chat(session_id)
        rec.state = ChatSessionState(
            recent_turns=self.config.memory.recent_turns,
            summarize_trigger_chars=self.config.memory.summarize_trigger_chars,
        )
        rec.updated_at = time.time()

    def chat(self, session_id: str, message: str, return_sources: bool = True) -> Dict[str, Any]:
        if self._rag is None:
            self.start()
        assert self._rag is not None

        if self._is_greeting(message):
            greeting = self._build_greeting_answer()
            return {
                "answer": greeting,
                "sources": self._starter_sources if return_sources else [],
            }

        record = self._ensure_chat(session_id)
        if record.title == "Untitled chat" and message.strip():
            record.title = message.strip()[:56]
            record.updated_at = time.time()
        prompt = record.state.build_prompt(message)
        result = self.query(
            prompt,
            return_sources=return_sources,
            use_memory=False,
            routing_question=message,
        )
        record.state.update(message, result["answer"], result.get("sources", []))
        record.updated_at = time.time()
        return result

    def _serialize_sources(self, sources: List[Any]) -> List[Dict[str, str]]:
        serialized: List[Dict[str, str]] = []
        for doc in sources:
            metadata = getattr(doc, "metadata", {}) or {}
            content = getattr(doc, "page_content", "") or ""
            source_file = metadata.get("source_file", "unknown")
            url = metadata.get("url") or self._source_url_by_file.get(source_file)
            serialized.append(
                {
                    "source_file": source_file,
                    "url": url,
                    "preview": content[:160].replace("\n", " "),
                }
            )
        return serialized

    def _ensure_chat(self, chat_id: str) -> ChatRecord:
        rec = self._chats.get(chat_id)
        if rec is not None:
            return rec
        now = time.time()
        rec = ChatRecord(
            chat_id=chat_id,
            title="Untitled chat",
            created_at=now,
            updated_at=now,
            state=ChatSessionState(
                recent_turns=self.config.memory.recent_turns,
                summarize_trigger_chars=self.config.memory.summarize_trigger_chars,
            ),
        )
        self._chats[chat_id] = rec
        return rec

    @staticmethod
    def _chat_summary(rec: ChatRecord) -> Dict[str, Any]:
        return {
            "chat_id": rec.chat_id,
            "title": rec.title,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "turn_count": len(rec.state.turns),
            "summary_present": bool(rec.state.summary.strip()),
        }

    @staticmethod
    def _is_greeting(message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        if len(text) > 40:
            return False
        compact = re.sub(r"[^\w\s]", " ", text)
        compact = re.sub(r"\s+", " ", compact).strip()
        greeting_patterns = [
            r"^(hi|hello|hey|yo|salam|hola)$",
            r"^good (morning|afternoon|evening)$",
            r"^(hi|hello|hey)[ ]+(there|bot|assistant)$",
            r"^(how are you|what s up)$",
        ]
        return any(re.match(p, compact) for p in greeting_patterns)

    def _build_greeting_answer(self) -> str:
        if not self._starter_questions:
            self._starter_questions, self._starter_sources = self._build_starter_templates()

        lines = ["Welcome to Fine Arts Assistant.", "Try one of these questions:"]
        for q in self._starter_questions[:6]:
            lines.append(f"- {q}")
        return "\n".join(lines)

    def _build_starter_templates(self) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Build suggestion templates by reading local docs and extracting program titles.
        """
        pages_dir = Path(self.config.documents.pages_directory)
        bachelor_titles: List[str] = []
        master_titles: List[str] = []
        sources: List[Dict[str, str]] = []

        if pages_dir.exists():
            for path in sorted(pages_dir.glob("*.txt")):
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                lines = content.splitlines()
                source_url = ""
                title = ""
                if lines and lines[0].strip().lower().startswith("document source:"):
                    source_url = lines[0].split(":", 1)[-1].strip()
                for line in lines[:10]:
                    if line.strip().lower().startswith("title:"):
                        title = line.split(":", 1)[-1].strip()
                        break

                if title:
                    title_upper = title.upper()
                    if "BACHELOR IN " in title_upper:
                        bachelor_titles.append(title.replace("FINE ARTS ", "").title())
                    if "MASTER IN " in title_upper:
                        master_titles.append(title.replace("FINE ARTS ", "").title())

                if source_url:
                    self._source_url_by_file[path.name] = source_url
                    sources.append(
                        {
                            "source_file": path.name,
                            "url": source_url,
                            "preview": (title or path.name)[:140],
                        }
                    )

        questions: List[str] = [
            "What documents are required for new student admission?",
            "What is the contact information for the Faculty of Fine Arts?",
            "What are the latest announcements and deadlines?",
            "What are the enrollment and CEE fees?",
        ]

        if bachelor_titles:
            questions.append(f"What are the requirements and job opportunities for {bachelor_titles[0]}?")
        if master_titles:
            questions.append(f"What are the requirements and fees for {master_titles[0]}?")

        # Keep stable order and remove duplicates.
        deduped_questions: List[str] = []
        for q in questions:
            if q not in deduped_questions:
                deduped_questions.append(q)

        deduped_sources: List[Dict[str, str]] = []
        seen = set()
        for s in sources:
            key = (s.get("source_file"), s.get("url"))
            if key in seen:
                continue
            seen.add(key)
            deduped_sources.append(s)

        return deduped_questions[:6], deduped_sources[:5]
