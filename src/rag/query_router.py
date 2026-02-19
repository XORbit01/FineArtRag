"""
Lightweight query analysis, query expansion, and metadata filter construction.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RoutingDecision:
    """Routing output used by the retrieval orchestrator."""
    metadata_filter: Optional[Dict]
    normalized_query: str
    retrieval_query: str
    intents: List[str]


class QueryRouter:
    """Builds metadata filters from query terms (NER-like keyword matching)."""

    # Program aliases mapped to canonical source files
    PROGRAM_FILE_MAP: Dict[str, str] = {
        "audiovisual": "fine_arts_Bachelor_in_Audiovisual.txt",
        "graphic design": "fine_arts_Bachelor_in_Graphic_Design.txt",
        "interior design": "fine_arts_Bachelor_in_Interior_Design.txt",
        "architecture": "fine_arts_Bachelor_in_Architecture.txt",
        "plastic arts": "fine_arts_Bachelor_in_Plastic_Arts.txt",
        "theater acting": "fine_arts_Bachelor_in_Theater_Acting.txt",
        "actor training": "fine_arts_Master_in_Actor_Training.txt",
        "master in architecture": "fine_arts_Master_in_Architecture.txt",
        "master in interior design": "fine_arts_Master_in_Interior_Design.txt",
        "master in plastic arts": "fine_arts_Master_in_PlasticArts.txt",
        "landscape and environmental architecture": "fine_arts_Master_in_Landscape_and_Environmental_Architecture.txt",
        "scenography": "fine_arts_Master_in_Scenography.txt",
        "urbanism": "fine_arts_Master_in_Urbanism.txt",
        "announcements": "fine_arts_announcements.txt",
    }
    CONTACT_SOURCE_FILE = "fine_arts_overview.txt"
    CONTACT_KEYWORDS = (
        "contact",
        "phone",
        "email",
        "address",
        "tel",
        "call",
        "reach",
    )
    FACULTY_KEYWORDS = (
        "fine arts",
        "faculty",
        "faculity",  # common typo
        "fbaa",
        "deanship",
        "dean",
    )
    ADMISSIONS_SOURCE_FILES = (
        "fine_arts_admissions_New_Student.txt",
        "fine_arts_admissions_Foreign_Students.txt",
        "fine_arts_admissions_Lebanese_University_Students.txt",
        "fine_arts_admissions_Private_University_Students.txt",
    )
    ANNOUNCEMENTS_SOURCE_FILE = "fine_arts_announcements.txt"
    OVERVIEW_SOURCE_FILE = "fine_arts_overview.txt"

    INTENT_KEYWORDS = {
        "contact": ("contact", "phone", "email", "address", "tel", "call", "reach"),
        "fees": ("fees", "tuition", "cost", "price", "payment", "subscription fee"),
        "admissions": ("admission", "register", "registration", "enroll", "documents required", "requirements"),
        "dates": ("date", "deadline", "when", "schedule"),
        "announcements": ("announcement", "news", "circular"),
    }

    @classmethod
    def _normalize(cls, question: str) -> str:
        q = (question or "").lower().strip()
        q = q.replace("&", " and ")
        q = re.sub(r"[_/\\\-]+", " ", q)
        q = re.sub(r"[^\w\s]", " ", q)

        # Token/phrase normalization for frequent user shorthand + typos.
        for src, dst in cls.NORMALIZATION_REPLACEMENTS.items():
            q = re.sub(rf"\b{re.escape(src)}\b", dst, q)

        # Canonical phrase joins.
        q = q.replace("fine arts faculty", "faculty of fine arts")
        q = q.replace("audio visual", "audiovisual")
        q = q.replace("job opportunity", "job opportunities")

        q = re.sub(r"\s+", " ", q)
        return q

    @classmethod
    def detect_intents(cls, question: str) -> List[str]:
        """Detect high-level query intents by keyword matching."""
        q = cls._normalize(question)
        intents: List[str] = []
        for intent, words in cls.INTENT_KEYWORDS.items():
            if any(w in q for w in words):
                intents.append(intent)
        return intents

    @classmethod
    def extract_program_files(cls, question: str) -> List[str]:
        """Return matched program source files based on keyword aliases."""
        q = cls._normalize(question)
        matches: List[str] = []

        for alias, source_file in cls.PROGRAM_FILE_MAP.items():
            if alias in q and source_file not in matches:
                matches.append(source_file)

        # Common short aliases
        if "audio visual" in q and "fine_arts_Bachelor_in_Audiovisual.txt" not in matches:
            matches.append("fine_arts_Bachelor_in_Audiovisual.txt")

        return matches

    @classmethod
    def build_metadata_filter(cls, question: str) -> Optional[Dict]:
        """
        Build a Chroma-compatible metadata filter.

        Returns None if no confident match is found.
        """
        q = cls._normalize(question)
        intents = cls.detect_intents(q)

        # Intent-first routing for faculty contact queries.
        if "contact" in intents and any(k in q for k in cls.FACULTY_KEYWORDS):
            return {"source_file": cls.CONTACT_SOURCE_FILE}

        # Program-specific routing (single or multi-program compare).
        files = cls.extract_program_files(q)
        if len(files) == 1:
            return {"source_file": files[0]}
        if len(files) > 1:
            return {"source_file": {"$in": files}}

        # Broader intent routes.
        if "admissions" in intents:
            return {"source_file": {"$in": list(cls.ADMISSIONS_SOURCE_FILES)}}
        if "announcements" in intents or ("dates" in intents and "fees" in intents):
            return {"source_file": cls.ANNOUNCEMENTS_SOURCE_FILE}
        if "contact" in intents:
            return {"source_file": cls.OVERVIEW_SOURCE_FILE}

        return None

    @classmethod
    def build_retrieval_query(cls, question: str) -> str:
        """
        Add compact lexical hints to improve semantic recall for vague prompts.
        """
        q = cls._normalize(question)
        intents = cls.detect_intents(q)
        hints: List[str] = []

        if "contact" in intents:
            hints.extend(["contact", "phone", "email", "address"])
        if "admissions" in intents:
            hints.extend(["admission", "registration", "documents required"])
        if "fees" in intents:
            hints.extend(["fees", "payment", "tuition"])
        if "announcements" in intents:
            hints.extend(["announcements", "news"])

        if hints:
            return f"{q}\nretrieval hints: {' '.join(hints)}"
        return q

    @classmethod
    def route(cls, question: str) -> RoutingDecision:
        """Return a complete routing decision for a query."""
        normalized = cls._normalize(question)
        return RoutingDecision(
            metadata_filter=cls.build_metadata_filter(question),
            normalized_query=normalized,
            retrieval_query=cls.build_retrieval_query(question),
            intents=cls.detect_intents(question),
        )
    NORMALIZATION_REPLACEMENTS = {
        "faculity": "faculty",
        "theatre": "theater",
        "uni": "university",
        "dept": "department",
        "info": "information",
        "docs": "documents",
        "reg": "registration",
        "enrol": "enroll",
        "pls": "please",
        "u": "you",
        "r": "are",
    }
