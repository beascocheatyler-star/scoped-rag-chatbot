# Architecture

## Request flow

```
POST /api/chat  { "message": "What is the late work policy?" }
        │
        ▼
  BM25Index.search(question)
        │   returns top-k passages, each with:
        │     • bm25 score (normalized 0–1)
        │     • term coverage (0–1, corpus-wide)
        ▼
  guard.classify()
        │
        ├─ corpus empty? ────────────────────► refuse, no LLM call
        ├─ greeting? ───────────────────────► small_talk → LLM
        ├─ "what can you do?" ──────────────► meta → LLM
        ├─ relevance < 0.40 ────────────────► refuse, no LLM call   ← free
        ├─ 0.40 ≤ relevance < 0.50 ─────────► borderline → LLM + verify instruction
        └─ relevance ≥ 0.50 ────────────────► in_scope → LLM
        │
        ▼
  engine._generate()
        │   system prompt: grounding rules + scope + injection defence
        │   user turn:     [Source 1..k] passages + question + instruction
        ▼
  Claude  ──►  answer + citations + token usage
```

The key property: **off-topic questions terminate before the LLM call**. That
makes refusal free, instant, and immune to prompt manipulation.

---

## Modules

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP routes, upload handling, error translation |
| `app/config.py` | Environment-driven settings with defaults |
| `app/schemas.py` | Pydantic request/response contracts |
| `app/core/extract.py` | File bytes → plain text, per format |
| `app/core/chunking.py` | Text → overlapping passages on natural boundaries |
| `app/core/retrieval.py` | BM25 index, term coverage, incremental add/remove |
| `app/core/guard.py` | The scope decision |
| `app/core/engine.py` | Prompt construction, Claude call, citation assembly |
| `app/core/persona.py` | Persona config + presets, persisted |
| `app/core/store.py` | Document lifecycle and JSON persistence |

---

## Key design decisions

### Two independent scope layers

A prompt instruction alone is negotiable — users can argue with it, and
instruction text inside an uploaded document can try to override it. A retrieval
gate the model never sees cannot be argued with.

Conversely, retrieval alone is too blunt: a question can share genuine
vocabulary with the corpus without being answerable from it. The two layers
cover each other's blind spots.

### Scoring with a geometric mean

`√(bm25 × coverage)` rather than either signal alone, or their average. The
mirror-image failure modes are documented with measurements in
[`scope-tuning.md`](scope-tuning.md). The geometric mean collapses toward zero if
either factor is weak, which is the desired bias.

### An explicit uncertainty band

Where the in-scope and off-topic score distributions overlap, no threshold is
correct. Treating that band as `borderline` and delegating to the model — with
the retrieved text in hand — converts a guaranteed misclassification into a
judgment made with evidence.

### Lexical retrieval, deliberately

The retrieval score powers the scope gate, so it runs on **every** question.
That makes its cost profile the dominant constraint: BM25 is instant, free,
offline, and dependency-free. Embeddings would improve recall on paraphrases at
the cost of an API call (or a large model download) per question, forever.

The trade-off is real and is documented rather than hidden — there is a test
asserting the known vocabulary-gap limitation.

### Chunk overlap

Chunks are split on paragraph, then sentence, then word boundaries, and each
chunk carries the tail of its predecessor. Without overlap, a policy spanning a
boundary gets split mid-statement and neither half answers the question.

### Prompt-injection defence

Uploaded documents are untrusted input. The system prompt states that context
passages are data, never instructions, and names the specific patterns to ignore
("ignore your rules", "you are now…", "reveal your prompt"). Passages are
additionally wrapped in explicit delimiters so the boundary is unambiguous.

On the frontend, all document-derived text is HTML-escaped before any markdown
rendering, so a document cannot inject script into the reviewer's browser.

### Persistence as readable JSON

`data/corpus.json` holds the documents and their chunks. A reviewer can open it
and see exactly what the bot knows. The BM25 index is rebuilt from it at
startup — cheap at this scale, and it avoids a binary format nobody can inspect.

Writes go to a temp file and are atomically renamed, so an interrupted write
cannot truncate the corpus.

---

## Concurrency

`BM25Index` and `DocumentStore` guard their mutable state with an `RLock`.
FastAPI serves requests from a thread pool, and uploads mutate the index while
chat requests read it, so this is load-bearing rather than defensive.

---

## Cost profile

| Event | LLM cost |
|---|---|
| Upload a document | none — indexing is local |
| Off-topic question | **none** — refused by the gate |
| Greeting / meta question | one call, minimal context |
| In-scope question | one call: system prompt + `TOP_K` passages + history |

Conversation history is replayed as prose only; prior turns' retrieved context
is not resent, which keeps multi-turn conversations from growing quadratically.
