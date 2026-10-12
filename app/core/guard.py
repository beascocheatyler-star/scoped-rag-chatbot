"""The scope guard — the part that makes this a *scoped* chatbot.

Two defences, layered, because either alone is insufficient:

1. **Retrieval gate (pre-LLM).** If the best BM25 match scores below the
   relevance floor, the question is off-topic by construction: nothing in the
   corpus speaks to it. We refuse without spending a token. This is the cheap,
   deterministic, un-jailbreakable layer — no prompt can talk its way past it,
   because the model is never called.

2. **Grounding instruction (in-LLM).** Retrieval can return weak-but-nonzero
   matches for a question the corpus doesn't actually answer (shared vocabulary,
   e.g. asking a refund-policy bot about "the return of the king"). So the system
   prompt also requires the model to answer *only* from the supplied context and
   to say so when the context falls short.

Layer 1 catches the obvious; layer 2 catches the subtle. A prompt-injection
attempt inside an uploaded document is addressed separately — see
``build_system_prompt``, which tells the model that document text is reference
material, never instructions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.retrieval import SearchHit, tokenize

# Greetings and meta-questions are in-scope conversationally even though they
# retrieve nothing. Refusing "hello" makes the bot feel broken.
_SMALL_TALK = re.compile(
    r"^\s*(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|"
    r"ok(ay)?|got\s+it|bye|goodbye|see\s+ya)"
    # Allow a short trailing flourish: "hey there", "thanks so much".
    r"(\s+(there|all|again|you|so\s+much|a\s+lot))?\b[\s!.?]*$",
    re.IGNORECASE,
)

# Questions about the bot's own remit, which it should answer from its persona
# and document list rather than from retrieved passages.
_META_PATTERNS = re.compile(
    r"\b(what (can|do) you (do|help|answer)|who are you|what are you|"
    r"what documents|what do you know|what topics|how do you work|what is your (scope|purpose))\b",
    re.IGNORECASE,
)


# The gate scores a question as sqrt(bm25_normalized * coverage) — the geometric
# mean, which requires *both* signals to be respectable. Neither alone suffices:
#
#   * High BM25, low coverage  -> "What time does the Super Bowl start?" matches
#     a syllabus strongly on the single word "time".
#   * Low BM25, high coverage  -> "Help me write a cover letter" shares several
#     common words with a policy document without being about it.
#
# The geometric mean punishes a near-zero in either factor far harder than an
# average would, which is exactly the desired behaviour.
#
# Measured against the bundled corpora, in-scope and off-topic questions overlap
# in a narrow band (~0.40 to ~0.50): no single threshold is error-free there.
# Rather than pick a cutoff that is wrong in one direction, the band is treated
# as genuinely uncertain and handed to the LLM *with* the retrieved text and an
# explicit instruction to verify that it addresses the question. Lexical scoring
# decides the easy cases for free; the model adjudicates only the hard ones.
#
# See docs/scope-tuning.md for the measured distribution and how to re-tune.
RELEVANCE_FLOOR_DEFAULT = 0.40

# Above this, retrieval is confident enough to answer without hedging.
CONFIDENT_RELEVANCE = 0.50


class ScopeDecision:
    IN_SCOPE = "in_scope"
    # Retrieval is ambiguous: plausible but not convincing. Goes to the LLM with
    # a heightened instruction to refuse if the context does not actually answer.
    BORDERLINE = "borderline"
    OUT_OF_SCOPE = "out_of_scope"
    SMALL_TALK = "small_talk"
    META = "meta"
    NO_CORPUS = "no_corpus"


@dataclass
class ScopeResult:
    decision: str
    hits: List[SearchHit]
    best_score: float
    reason: str
    coverage: float = 0.0

    @property
    def allows_llm(self) -> bool:
        """Whether this decision should proceed to an LLM call."""
        return self.decision in {
            ScopeDecision.IN_SCOPE,
            ScopeDecision.BORDERLINE,
            ScopeDecision.SMALL_TALK,
            ScopeDecision.META,
        }

    @property
    def needs_verification(self) -> bool:
        """Whether the model should be told to double-check scope itself."""
        return self.decision == ScopeDecision.BORDERLINE


def relevance(hit: Optional[SearchHit]) -> float:
    """Combined topical-relevance score in 0..1 for the best hit."""
    if hit is None:
        return 0.0
    return math.sqrt(max(0.0, hit.normalized) * max(0.0, hit.coverage))


def classify(
    question: str,
    hits: List[SearchHit],
    relevance_floor: float,
    corpus_empty: bool,
) -> ScopeResult:
    """Decide whether ``question`` may be answered from this corpus."""
    best_hit = hits[0] if hits else None
    best = relevance(best_hit)
    coverage = best_hit.coverage if best_hit else 0.0

    if corpus_empty:
        return ScopeResult(
            decision=ScopeDecision.NO_CORPUS,
            hits=[],
            best_score=0.0,
            reason="No documents have been uploaded yet.",
        )

    if _SMALL_TALK.match(question):
        return ScopeResult(
            decision=ScopeDecision.SMALL_TALK,
            hits=hits[:2],
            best_score=best,
            reason="Conversational greeting.",
        )

    if _META_PATTERNS.search(question):
        return ScopeResult(
            decision=ScopeDecision.META,
            hits=hits[:3],
            best_score=best,
            reason="Question about the assistant's own scope.",
        )

    # A question with no content words after stopword removal gives retrieval
    # nothing to work with; treat it as unanswerable rather than off-topic.
    if not tokenize(question):
        return ScopeResult(
            decision=ScopeDecision.OUT_OF_SCOPE,
            hits=[],
            best_score=0.0,
            reason="The question contains no searchable terms.",
        )

    if not hits or best < relevance_floor:
        return ScopeResult(
            decision=ScopeDecision.OUT_OF_SCOPE,
            hits=hits,
            best_score=best,
            coverage=coverage,
            reason=(
                "Relevance {:.3f} is below the floor of {:.3f} ({:.0%} of the "
                "question's key terms appear in the knowledge base). Nothing "
                "here addresses this question.".format(
                    best, relevance_floor, coverage
                )
            ),
        )

    if best < CONFIDENT_RELEVANCE:
        return ScopeResult(
            decision=ScopeDecision.BORDERLINE,
            hits=hits,
            best_score=best,
            coverage=coverage,
            reason=(
                "Relevance {:.3f} is in the uncertain band ({:.2f}–{:.2f}); "
                "asking the model to verify the context before answering.".format(
                    best, relevance_floor, CONFIDENT_RELEVANCE
                )
            ),
        )

    return ScopeResult(
        decision=ScopeDecision.IN_SCOPE,
        hits=hits,
        best_score=best,
        coverage=coverage,
        reason="Relevance {:.3f} with {:.0%} term coverage.".format(best, coverage),
    )


def refusal_message(
    result: ScopeResult,
    assistant_name: str,
    topic: str,
    document_names: Optional[List[str]] = None,
) -> str:
    """A deterministic, on-brand refusal. No LLM call required."""
    if result.decision == ScopeDecision.NO_CORPUS:
        return (
            "I don't have any documents yet, so there's nothing I can answer from. "
            "Upload the relevant files and ask me again."
        )

    lines = [
        "That falls outside what I can help with. I'm {}, and I only answer "
        "questions about {}.".format(assistant_name, topic)
    ]
    if document_names:
        shown = document_names[:5]
        suffix = (
            " (and {} more)".format(len(document_names) - len(shown))
            if len(document_names) > len(shown)
            else ""
        )
        lines.append(
            "\nMy knowledge base currently covers: {}{}.".format(
                ", ".join(shown), suffix
            )
        )
    lines.append("\nAsk me something in that area and I'll give you a sourced answer.")
    return "".join(lines)
