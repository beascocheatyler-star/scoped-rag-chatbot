"""Document lifecycle: ingest, persist, reload, delete.

The corpus is persisted as JSON on disk so a restart does not lose uploads. The
BM25 index is rebuilt from that JSON at boot — cheap, and it avoids shipping a
binary index format that would be hard for a reviewer to inspect.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.chunking import chunk_text
from app.core.extract import extract
from app.core.retrieval import BM25Index, Chunk


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    doc_id: str
    name: str
    size_bytes: int
    chunk_count: int
    char_count: int
    sha256: str
    uploaded_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


class DuplicateDocumentError(ValueError):
    """Raised when the exact same file content is uploaded twice."""

    def __init__(self, existing: Document):
        super().__init__(
            "This file is already in the knowledge base as '{}'.".format(existing.name)
        )
        self.existing = existing


class DocumentStore:
    """Owns the corpus and keeps the BM25 index in sync with it."""

    def __init__(self, data_dir: Path, chunk_size: int = 1200, overlap: int = 200):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = BM25Index()

        self._lock = threading.RLock()
        self._documents: Dict[str, Document] = {}
        self._chunks_by_doc: Dict[str, List[Dict]] = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_path = self.data_dir / "corpus.json"
        self._load()

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self._corpus_path.exists():
            return
        try:
            raw = json.loads(self._corpus_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt corpus file should not prevent the app from starting.
            return

        for entry in raw.get("documents", []):
            try:
                doc = Document(**entry["meta"])
            except (KeyError, TypeError):
                continue
            self._documents[doc.doc_id] = doc
            self._chunks_by_doc[doc.doc_id] = entry.get("chunks", [])
            self.index.add_chunks(
                Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    position=c["position"],
                    text=c["text"],
                )
                for c in entry.get("chunks", [])
            )

    def _save(self) -> None:
        payload = {
            "version": 1,
            "saved_at": _utcnow(),
            "documents": [
                {
                    "meta": self._documents[doc_id].to_dict(),
                    "chunks": self._chunks_by_doc.get(doc_id, []),
                }
                for doc_id in self._documents
            ],
        }
        tmp = self._corpus_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        tmp.replace(self._corpus_path)  # atomic, so a crash cannot truncate the corpus

    # --- queries -------------------------------------------------------

    def list_documents(self) -> List[Document]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.uploaded_at)

    def get(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._documents.get(doc_id)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._documents

    @property
    def total_chunks(self) -> int:
        return len(self.index)

    # --- mutation ------------------------------------------------------

    def add(self, filename: str, data: bytes) -> Document:
        """Extract, chunk, and index a file. Raises ExtractionError on bad input."""
        text = extract(filename, data)
        digest = hashlib.sha256(data).hexdigest()

        with self._lock:
            for existing in self._documents.values():
                if existing.sha256 == digest:
                    raise DuplicateDocumentError(existing)

            pieces = chunk_text(text, self.chunk_size, self.overlap)
            doc_id = uuid.uuid4().hex[:12]
            doc = Document(
                doc_id=doc_id,
                name=filename,
                size_bytes=len(data),
                chunk_count=len(pieces),
                char_count=len(text),
                sha256=digest,
                uploaded_at=_utcnow(),
            )

            serialized = [
                {
                    "chunk_id": "{}-{}".format(doc_id, i),
                    "position": i,
                    "text": piece,
                }
                for i, piece in enumerate(pieces)
            ]

            self._documents[doc_id] = doc
            self._chunks_by_doc[doc_id] = serialized
            self.index.add_chunks(
                Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=doc_id,
                    doc_name=filename,
                    position=c["position"],
                    text=c["text"],
                )
                for c in serialized
            )
            self._save()
            return doc

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._documents:
                return False
            del self._documents[doc_id]
            self._chunks_by_doc.pop(doc_id, None)
            self.index.remove_document(doc_id)
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self._documents = {}
            self._chunks_by_doc = {}
            self.index.clear()
            self._save()
