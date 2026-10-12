"""BM25 retrieval over the uploaded corpus.

Deliberately dependency-free. A vector index would need either an embeddings API
(cost, network, another key to configure) or a local transformer download
(hundreds of megabytes). BM25 is a strong lexical baseline that runs instantly
offline, which matters more here: the retrieval score also drives the
out-of-scope gate, so it must be cheap enough to run on *every* question.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Common words carry no topical signal, so they would otherwise inflate the
# relevance of off-topic questions and defeat the scope gate.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can cannot could couldn't did didn't
    do does doesn't doing don't down during each few for from further had hadn't has
    hasn't have haven't having he her here hers herself him himself his how i if in
    into is isn't it its itself let's me more most mustn't my myself no nor not of off
    on once only or other ought our ours ourselves out over own same shan't she should
    shouldn't so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your yours
    yourself yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")

# Standard BM25 parameters.
_K1 = 1.5
_B = 0.75


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords and 1-character noise."""
    tokens = _TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    """One retrievable passage plus the provenance needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_name: str
    position: int
    text: str
    tokens: List[str] = field(default_factory=list, repr=False)
    term_freqs: Counter = field(default_factory=Counter, repr=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)
        if not self.term_freqs:
            self.term_freqs = Counter(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    # Score normalised to roughly 0..1 so it can be compared against a fixed
    # relevance floor regardless of query length.
    normalized: float
    # Fraction of the query's content words that appear anywhere in the corpus.
    # BM25 alone rewards a single rare-ish term heavily, which lets a question
    # like "Can I return these boots?" score well against a syllabus purely on
    # the word "return". Coverage catches that: one generic verb matching out of
    # three content words is a weak signal no matter how high BM25 rates it.
    coverage: float = 0.0


class BM25Index:
    """An in-memory BM25 index that supports incremental document addition.

    Thread-safe: FastAPI serves requests from a worker pool, and uploads mutate
    the index while chats read from it.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chunks: List[Chunk] = []
        self._doc_freqs: Counter = Counter()
        self._total_length = 0

    # --- introspection -------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def avg_length(self) -> float:
        with self._lock:
            if not self._chunks:
                return 0.0
            return self._total_length / len(self._chunks)

    def doc_ids(self) -> List[str]:
        with self._lock:
            seen: Dict[str, None] = {}
            for chunk in self._chunks:
                seen.setdefault(chunk.doc_id, None)
            return list(seen)

    # --- mutation ------------------------------------------------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Index ``chunks``. Returns the number added."""
        added = 0
        with self._lock:
            for chunk in chunks:
                self._chunks.append(chunk)
                self._total_length += chunk.length
                for term in chunk.term_freqs:
                    self._doc_freqs[term] += 1
                added += 1
        return added

    def remove_document(self, doc_id: str) -> int:
        """Drop every chunk belonging to ``doc_id``. Returns the number removed."""
        with self._lock:
            keep: List[Chunk] = []
            removed = 0
            for chunk in self._chunks:
                if chunk.doc_id == doc_id:
                    removed += 1
                    self._total_length -= chunk.length
                    for term in chunk.term_freqs:
                        self._doc_freqs[term] -= 1
                        if self._doc_freqs[term] <= 0:
                            del self._doc_freqs[term]
                else:
                    keep.append(chunk)
            self._chunks = keep
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._doc_freqs = Counter()
            self._total_length = 0

    # --- query ---------------------------------------------------------

    def _idf(self, term: str, n_docs: int) -> float:
        df = self._doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF with +1 smoothing to keep it strictly positive.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def _coverage(self, terms: set, n_docs: int) -> float:
        """Fraction of the query's content words that occur in the corpus.

        Deliberately presence-based, ignoring how often a term occurs. An
        earlier version down-weighted terms appearing in many chunks, on the
        theory that ubiquitous terms are uninformative. That is true for a
        heterogeneous corpus and badly wrong for this one: in a single-subject
        knowledge base the most widespread term *is* the subject. On a 7-chunk
        syllabus it scored "exam" at 0.29 and "boots" at 1.00 — precisely
        inverted — so questions about exams were refused while questions about
        footwear passed.

        Plain presence is the robust signal, and it is stable at any corpus
        size: words the corpus has never seen are what mark a question as being
        about something else.
        """
        if not terms:
            return 0.0
        present = sum(1 for t in terms if self._doc_freqs.get(t, 0) > 0)
        return present / len(terms)

    def search(
        self,
        query: str,
        top_k: int = 6,
        doc_ids: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Return the ``top_k`` best-matching chunks, highest score first."""
        query_terms = tokenize(query)
        if not query_terms:
            return []

        with self._lock:
            if not self._chunks:
                return []

            n_docs = len(self._chunks)
            avg_len = self._total_length / n_docs if n_docs else 0.0
            allowed = set(doc_ids) if doc_ids else None

            # Precompute IDF once per unique term.
            idfs = {t: self._idf(t, n_docs) for t in set(query_terms)}

            # The maximum score this query could achieve against a perfectly
            # matching chunk. Used to normalise into a comparable 0..1 range.
            max_possible = sum(idfs.values()) * (_K1 + 1)

            scored: List[Tuple[float, Chunk]] = []
            for chunk in self._chunks:
                if allowed is not None and chunk.doc_id not in allowed:
                    continue

                score = 0.0
                for term in query_terms:
                    freq = chunk.term_freqs.get(term)
                    if not freq:
                        continue
                    idf = idfs[term]
                    if idf <= 0:
                        continue
                    denom = freq + _K1 * (
                        1 - _B + _B * (chunk.length / avg_len if avg_len else 1.0)
                    )
                    score += idf * (freq * (_K1 + 1)) / denom

                if score > 0:
                    scored.append((score, chunk))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            top = scored[:top_k]

            # Coverage is a property of the query against the whole corpus, not
            # of any single chunk: a legitimate question may draw its terms from
            # several passages.
            #
            # Weighting by specificity matters. Counting raw presence treats
            # "return" and "exam" alike, so "Can I return these boots?" scores
            # 50% against a syllabus on one incidental verb. Weighting each term
            # by how *informative* it is — rare corpus terms and absent terms
            # both count heavily — separates a question the corpus is about from
            # one that merely shares vocabulary with it.
            coverage = self._coverage(set(query_terms), n_docs)

            return [
                SearchHit(
                    chunk=chunk,
                    score=score,
                    normalized=(score / max_possible) if max_possible > 0 else 0.0,
                    coverage=coverage,
                )
                for score, chunk in top
            ]
