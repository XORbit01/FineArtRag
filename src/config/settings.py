"""
Configuration management for RAG system
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OllamaConfig:
    """Ollama service configuration"""
    base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:1b"
    temperature: float = 0.2
    num_ctx: int = 4096
    timeout: int = 30


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    provider: str = "huggingface"  # "huggingface" or "ollama"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model
    ollama_model: str = "nomic-embed-text"  # Ollama model (if provider is ollama)
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    persist_directory: Path = Path("chroma_db")
    collection_name: str = "fine_arts_knowledge_base"


@dataclass
class DocumentConfig:
    """Document processing configuration"""
    pages_directory: Path = Path("pages")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_extensions: tuple = (".txt",)


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 5
    fetch_k_multiplier: int = 3
    mmr_lambda: float = 0.5
    search_type: str = "similarity"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None
    console: bool = True


@dataclass
class MemoryConfig:
    """Conversation memory configuration"""
    enabled: bool = True
    max_turns: int = 12
    recent_turns: int = 4
    summarize_trigger_chars: int = 3500


@dataclass
class AppConfig:
    """Main application configuration"""
    ollama: OllamaConfig
    embeddings: EmbeddingConfig
    vector_store: VectorStoreConfig
    documents: DocumentConfig
    retrieval: RetrievalConfig
    memory: MemoryConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls(
            ollama=OllamaConfig(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                llm_model=os.getenv("LLM_MODEL", "llama3.2:1b"),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
                num_ctx=int(os.getenv("NUM_CTX", "4096")),
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "30"))
            ),
            embeddings=EmbeddingConfig(
                provider="huggingface",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                ollama_model="nomic-embed-text",
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ),
            vector_store=VectorStoreConfig(
                persist_directory=Path(os.getenv("CHROMA_DB_DIR", "chroma_db")),
                collection_name=os.getenv("COLLECTION_NAME", "fine_arts_knowledge_base")
            ),
            documents=DocumentConfig(
                pages_directory=Path(os.getenv("PAGES_DIR", "pages")),
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
            ),
            retrieval=RetrievalConfig(
                top_k=int(os.getenv("TOP_K", "5")),
                fetch_k_multiplier=int(os.getenv("FETCH_K_MULTIPLIER", "3")),
                mmr_lambda=float(os.getenv("MMR_LAMBDA", "0.5")),
                search_type=os.getenv("SEARCH_TYPE", "similarity")
            ),
            memory=MemoryConfig(
                enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
                max_turns=int(os.getenv("MEMORY_MAX_TURNS", "12")),
                recent_turns=int(os.getenv("MEMORY_RECENT_TURNS", "4")),
                summarize_trigger_chars=int(os.getenv("MEMORY_SUMMARIZE_TRIGGER_CHARS", "3500"))
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None,
                console=os.getenv("LOG_CONSOLE", "true").lower() == "true"
            )
        )


# Global configuration instance
config = AppConfig.from_env()
