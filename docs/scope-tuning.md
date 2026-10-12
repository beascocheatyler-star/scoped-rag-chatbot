# Tuning the scope gate

How the relevance threshold was chosen, what the measured numbers are, and how
to re-tune for your own documents.

Reproduce everything here with:

```bash
python scripts/measure_scope.py
```

---

## The score

```
relevance = √(bm25_normalized × term_coverage)
```

| Component | Meaning | Range |
|---|---|---|
| `bm25_normalized` | Best passage's BM25 score, divided by the maximum that query could achieve | 0–1 |
| `term_coverage` | Fraction of the question's content words occurring anywhere in the corpus | 0–1 |

The geometric mean requires **both** signals to be respectable. An arithmetic
mean would let a single strong signal mask a near-zero one, which is the exact
failure this is designed to prevent.

---

## Measured distribution

Against the two bundled corpora, with 26 in-scope and 23 off-topic questions.

### Lowest-scoring in-scope questions (risk of false refusal)

| Score | Question |
|---|---|
| 0.440 | Are lab absences penalized? |
| 0.446 | How much does the zyBooks subscription cost? |
| 0.511 | What happens if I miss an exam? |
| 0.515 | When will my refund post? |
| 0.519 | My package tracking has not updated in a week. |

### Highest-scoring off-topic questions (risk of false answer)

| Score | Question | Why it scores at all |
|---|---|---|
| 0.338 | Write me a Python script that scrapes Twitter. | "write", "program" |
| 0.385 | What time does the Super Bowl start? | "time" |
| 0.446 | Can I return these boots? | "return" |
| 0.451 | When is the final exam? *(asked of the support bot)* | "final" |
| 0.499 | Help me write a cover letter. | "help", "write", "cover" |

### Errors by threshold

| Threshold | False refusals (of 26) | False answers (of 23) |
|---|---|---|
| 0.30 | 0 | 9 |
| 0.35 | 0 | 7 |
| **0.40** | **0** | **4** |
| 0.45 | 2 | 3 |
| 0.50 | 2 | 0 |
| 0.55 | 5 | 0 |
| 0.60 | 11 | 0 |

---

## Why 0.40, and why a band

The two populations **overlap between roughly 0.44 and 0.50**. No single
threshold separates them cleanly:

- Pick 0.40 → 4 off-topic questions get through.
- Pick 0.50 → 2 legitimate questions get refused.

Rather than accept either, the design splits the difference by treating the
overlap as what it actually is — uncertainty:

| Range | Decision | Behaviour |
|---|---|---|
| `< 0.40` | `out_of_scope` | Refused immediately. No LLM call, no cost. |
| `0.40 – 0.50` | `borderline` | Sent to the LLM **with the retrieved text** and an instruction to verify it genuinely answers the question before responding. |
| `≥ 0.50` | `in_scope` | Answered normally. |

The threshold is set where **false refusals are zero**. That asymmetry is
intentional: a wrongly refused question is a visibly broken product, while a
borderline question merely costs one API call to a layer that can read the
passages and judge properly. Cheap lexical scoring handles the easy 90%; the
model adjudicates only the genuinely ambiguous cases.

---

## Re-tuning for your own documents

`RELEVANCE_FLOOR` in `.env` controls the lower bound.

| Symptom | Fix |
|---|---|
| Bot refuses legitimate questions | **Lower** it (try 0.30) |
| Bot answers off-topic questions | **Raise** it (try 0.50) |

To measure properly rather than guess, edit the question lists at the top of
`scripts/measure_scope.py` to reflect your corpus and your users' real questions,
then run it. Pick the threshold where false refusals reach zero.

Corpus-size note: the scores are stable across corpus sizes because coverage is
presence-based. An earlier version weighted terms by how many chunks contained
them, which inverted badly on small corpora — on a 7-chunk syllabus it rated
"exam" as uninformative (0.29) and "boots" as highly informative (1.00), exactly
backwards. In a single-subject knowledge base, the most widespread term *is* the
subject. That approach was removed; see the comment in `retrieval.py::_coverage`.

---

## Other levers

| Setting | Effect on scope behaviour |
|---|---|
| `TOP_K` | More passages give the model more to work with, but dilute precision. Only the *best* hit drives the gate, so this does not change the decision — only the answer quality. |
| `CHUNK_SIZE` | Smaller chunks sharpen BM25 (less dilution per passage) but fragment context. |
| `CHUNK_OVERLAP` | Higher overlap helps when answers straddle a chunk boundary. |
| `STRICT_SCOPE` | Adds an explicit instruction refusing unrelated code/essay/translation requests even when retrieval passes. |

---

## The known limitation

Retrieval is lexical. "What textbook do I need?" is refused against a syllabus
whose corresponding section is headed *Required materials* — the word "textbook"
never appears, so there is no lexical overlap to find.

This is asserted by `test_vocabulary_gap_is_a_known_limitation` so the behaviour
is recorded rather than discovered later.

Closing the gap means semantic embeddings, which cost an API call per chunk plus
one per question, or a few hundred megabytes of local model. Since the retrieval
score runs on *every* question to power the gate, that cost is paid constantly —
which is why lexical matching was chosen for a bot whose job is answering from a
specific, stable corpus.

A practical mitigation that needs no code change: include a glossary or FAQ in
the uploaded corpus. It supplies the vocabulary users actually ask in.
