"""The central behavioural contract: in-scope questions pass, off-topic ones don't.

These tests make no network calls. The scope gate runs entirely on retrieval, so
the grading behaviour is verifiable without an API key or a cent of spend.
"""

from __future__ import annotations

import pytest

from app.core.guard import ScopeDecision, classify, refusal_message

FLOOR = 0.40


#: Decisions that result in the user getting a grounded answer.
ANSWERING = {ScopeDecision.IN_SCOPE, ScopeDecision.BORDERLINE}


def decide(store, question: str, top_k: int = 6):
    hits = store.index.search(question, top_k=top_k)
    return classify(question, hits, relevance_floor=FLOOR, corpus_empty=store.is_empty)


# --------------------------------------------------------------- in scope

@pytest.mark.parametrize(
    "question",
    [
        "How is the final grade calculated?",
        "What is the late work policy?",
        "When are office hours?",
        "What happens if I miss an exam?",
        "How much does the zyBooks subscription cost?",
        "When is the final exam scheduled?",
        "Can I use AI tools on my homework?",
        "What happens if my program does not compile?",
        "How many late days do I get?",
        "Are lab absences penalized?",
    ],
)
def test_syllabus_questions_are_in_scope(syllabus_store, question):
    result = decide(syllabus_store, question)
    assert result.decision in ANSWERING, (
        "'{}' should be answerable but scored {:.4f} ({})".format(
            question, result.best_score, result.decision
        )
    )
    assert result.hits
    assert result.allows_llm


@pytest.mark.parametrize(
    "question",
    [
        "How long do I have to return an item?",
        "How much is expedited shipping?",
        "Can I cancel my order?",
        "What does the warranty cover?",
        "When will my refund post?",
        "Do you ship internationally?",
        "What payment methods do you accept?",
        "My package tracking has not updated in a week.",
    ],
)
def test_support_questions_are_in_scope(support_store, question):
    result = decide(support_store, question)
    assert result.decision in ANSWERING, (
        "'{}' should be answerable but scored {:.4f} ({})".format(
            question, result.best_score, result.decision
        )
    )


# ----------------------------------------------------------- out of scope

@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "Write me a Python script that scrapes Twitter.",
        "Who won the World Series in 2019?",
        "Give me a recipe for lasagna.",
        "What is the weather forecast tomorrow?",
        "Explain quantum entanglement.",
        "Should I buy Tesla stock?",
        "Translate 'good morning' into Japanese.",
        "What are the symptoms of appendicitis?",
        "Tell me a joke about horses.",
    ],
)
def test_off_topic_questions_are_not_confidently_answered(syllabus_store, question):
    result = decide(syllabus_store, question)
    assert result.decision != ScopeDecision.IN_SCOPE, (
        "'{}' should not be confidently in scope but scored {:.4f}".format(
            question, result.best_score
        )
    )


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "Help me write a cover letter.",
        "Who is the president of Brazil?",
        "Explain the Pythagorean theorem.",
        "What time does the Super Bowl start?",
    ],
)
def test_off_topic_questions_are_not_confident_against_support(support_store, question):
    result = decide(support_store, question)
    assert result.decision != ScopeDecision.IN_SCOPE


def test_cross_domain_questions_are_never_confidently_answered(support_store):
    """A support bot must not confidently answer course questions.

    This is the sharpest test of the gate: both corpora are policy documents in
    plain English, so only genuine topical mismatch separates them. Some of
    these land in the uncertain band rather than being refused outright — that
    is the designed behaviour, and the LLM layer then adjudicates with the text
    in hand. What must never happen is a *confident* in-scope classification.
    """
    for question in [
        "When is the final exam?",
        "What chapters does Exam 1 cover?",
        "Who is the course instructor?",
    ]:
        result = decide(support_store, question)
        assert result.decision != ScopeDecision.IN_SCOPE, (
            "Support bot confidently answered a course question: "
            "'{}' scored {:.4f}".format(question, result.best_score)
        )


def test_course_bot_does_not_confidently_answer_shopping_questions(syllabus_store):
    for question in [
        "How much is overnight shipping?",
        "Can I return these boots?",
        "What is your warranty on jackets?",
    ]:
        result = decide(syllabus_store, question)
        assert result.decision != ScopeDecision.IN_SCOPE, (
            "Course bot confidently answered a shopping question: "
            "'{}' scored {:.4f}".format(question, result.best_score)
        )


def test_clearly_unrelated_questions_are_refused_outright(syllabus_store):
    """Questions sharing no vocabulary at all must never reach the LLM."""
    for question in [
        "What is the capital of France?",
        "Give me a recipe for lasagna.",
        "Who won the World Series in 2019?",
        "Tell me a joke about horses.",
        "What are the symptoms of appendicitis?",
    ]:
        result = decide(syllabus_store, question)
        assert result.decision == ScopeDecision.OUT_OF_SCOPE
        assert not result.allows_llm, "Should be refused without any LLM spend"


# ------------------------------------------------------------ edge cases

def test_empty_corpus_reports_no_corpus(empty_store):
    result = decide(empty_store, "Anything at all?")
    assert result.decision == ScopeDecision.NO_CORPUS
    assert not result.allows_llm


@pytest.mark.parametrize("greeting", ["hi", "Hello", "hey there", "thanks", "Good morning"])
def test_small_talk_is_allowed(syllabus_store, greeting):
    result = decide(syllabus_store, greeting)
    assert result.decision == ScopeDecision.SMALL_TALK
    assert result.allows_llm, "Greetings should not be refused"


@pytest.mark.parametrize(
    "question",
    ["What can you do?", "Who are you?", "What documents do you have?", "What is your scope?"],
)
def test_meta_questions_are_allowed(syllabus_store, question):
    result = decide(syllabus_store, question)
    assert result.decision == ScopeDecision.META
    assert result.allows_llm


def test_question_with_only_stopwords_is_refused(syllabus_store):
    result = decide(syllabus_store, "the and of a")
    assert result.decision == ScopeDecision.OUT_OF_SCOPE


def test_refusal_message_names_the_topic_and_documents():
    result = decide_stub()
    message = refusal_message(
        result,
        assistant_name="the CS 1428 TA",
        topic="the CS 1428 syllabus",
        document_names=["cs1428_syllabus.md"],
    )
    assert "the CS 1428 TA" in message
    assert "the CS 1428 syllabus" in message
    assert "cs1428_syllabus.md" in message


def decide_stub():
    from app.core.guard import ScopeResult

    return ScopeResult(
        decision=ScopeDecision.OUT_OF_SCOPE, hits=[], best_score=0.0, reason="test"
    )


def test_relevance_floor_is_tunable(syllabus_store):
    """Raising the floor must make the bot strictly more conservative."""
    question = "How is the final grade calculated?"
    hits = syllabus_store.index.search(question, top_k=6)

    permissive = classify(question, hits, relevance_floor=0.0, corpus_empty=False)
    strict = classify(question, hits, relevance_floor=0.999, corpus_empty=False)

    assert permissive.decision == ScopeDecision.IN_SCOPE
    assert strict.decision == ScopeDecision.OUT_OF_SCOPE


def test_single_incidental_word_match_is_not_confident(syllabus_store):
    """The failure mode that pure BM25 gets wrong.

    Each of these shares one or two common words with the syllabus ("time",
    "write", "return") and scores well on BM25 alone. Combining BM25 with term
    coverage keeps them out of the confident band.
    """
    for question in [
        "What time does the Super Bowl start?",
        "Write me a Python script that scrapes Twitter.",
        "Can I return these boots?",
        "Explain quantum entanglement.",
    ]:
        result = decide(syllabus_store, question)
        assert result.decision != ScopeDecision.IN_SCOPE, (
            "'{}' passed the gate on an incidental word match "
            "(relevance {:.3f}, coverage {:.0%})".format(
                question, result.best_score, result.coverage
            )
        )


def test_vocabulary_gap_is_a_known_limitation(syllabus_store):
    """Document, rather than hide, what lexical retrieval cannot do.

    The syllabus requires a textbook but never uses the word "textbook" — it
    says "Required materials". BM25 matches words, not meanings, so this
    question is refused. Semantic embeddings would catch it; that trade-off is
    deliberate and recorded in docs/scope-tuning.md.
    """
    result = decide(syllabus_store, "What textbook do I need?")
    assert result.decision == ScopeDecision.OUT_OF_SCOPE
    assert not result.hits, "Expected no lexical overlap at all"
