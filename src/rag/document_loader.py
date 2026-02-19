"""
Document loading and processing module
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_experimental.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.documents import Document

from src.config.settings import DocumentConfig

logger = logging.getLogger(__name__)
DOCUMENT_SOURCE_PATTERN = re.compile(
    r"^\s*DOCUMENT SOURCE:\s*(https?://\S+)\s*$",
    re.IGNORECASE
)
SECTION_HEADING_PATTERN = re.compile(r"^\s*#{3,}\s*(.*?)\s*#{0,}\s*$", re.IGNORECASE)


class DocumentLoader:
    """Handles document loading from filesystem"""
    
    def __init__(self, config: DocumentConfig):
        """
        Initialize document loader
        
        Args:
            config: Document configuration
        """
        self.config = config
        self.pages_dir = Path(config.pages_directory)
        
        if not self.pages_dir.exists():
            raise FileNotFoundError(
                f"Pages directory not found: {self.pages_dir}"
            )
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from configured directory
        
        Returns:
            List of Document objects
        """
        logger.info(f"Loading documents from {self.pages_dir}")
        
        documents = []
        txt_files = list(self.pages_dir.glob(f"**/*{self.config.supported_extensions[0]}"))
        
        for file_path in txt_files:
            try:
                # Try multiple encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        loader = TextLoader(
                            str(file_path),
                            encoding=encoding
                        )
                        doc = loader.load()[0]
                        source_url = self._extract_source_url(doc.page_content)
                        if source_url:
                            doc.metadata["url"] = source_url
                        documents.append(doc)
                        logger.debug(f"Loaded {file_path.name} with encoding {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading {file_path.name} with encoding {encoding}: {e}")
                        continue
                else:
                    # If all encodings failed, try with errors='ignore'
                    logger.warning(f"Trying to load {file_path.name} with errors='ignore'")
                    loader = TextLoader(
                        str(file_path),
                        encoding='utf-8',
                        autodetect_encoding=True
                    )
                    doc = loader.load()[0]
                    source_url = self._extract_source_url(doc.page_content)
                    if source_url:
                        doc.metadata["url"] = source_url
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {len(txt_files)} files")
        
        return documents

    @staticmethod
    def _extract_source_url(content: str) -> Optional[str]:
        """Extract declared document source URL from the first line."""
        if not content:
            return None
        first_line = content.splitlines()[0] if content.splitlines() else ""
        match = DOCUMENT_SOURCE_PATTERN.match(first_line)
        return match.group(1) if match else None


class DocumentChunker:
    """Handles document chunking"""
    
    def __init__(self, config: DocumentConfig):
        """
        Initialize document chunker
        
        Args:
            config: Document configuration
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces
        
        Args:
            documents: List of Document objects to chunk
        
        Returns:
            List of chunked Document objects with metadata
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.splitter.split_documents(documents)
        
        # Enhance metadata
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            source_file = Path(source).name
            chunk.metadata.update({
                "chunk_id": i,
                "source_file": source_file,
                "chunk_size": len(chunk.page_content),
                "section": self._infer_section(chunk.page_content),
                "doc_type": self._infer_doc_type(source_file),
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _infer_section(text: str) -> str:
        """Infer semantic section label from heading/keywords."""
        if not text:
            return "general"

        lowered = text.lower()
        if "### contact us ###" in lowered or "phone no.:" in lowered or "email:" in lowered:
            return "contact"
        if "### requirements ###" in lowered or "general conditions" in lowered or "documents required" in lowered:
            return "requirements"

        for line in text.splitlines()[:8]:
            match = SECTION_HEADING_PATTERN.match(line)
            if match:
                heading = match.group(1).strip().lower()
                if "contact" in heading:
                    return "contact"
                if "requirement" in heading:
                    return "requirements"
                if "job opportunit" in heading:
                    return "job_opportunities"
                if "description" in heading:
                    return "description"
                if "history" in heading:
                    return "history"
                return heading.replace(" ", "_")

        return "general"

    @staticmethod
    def _infer_doc_type(source_file: str) -> str:
        name = (source_file or "").lower()
        if "admissions" in name:
            return "admissions"
        if "announcement" in name:
            return "announcements"
        if "overview" in name:
            return "overview"
        if "bachelor" in name:
            return "program_bachelor"
        if "master" in name:
            return "program_master"
        return "general"
