# Scoped Document Chatbot Engine

Upload documents. The chatbot answers questions about them — and **refuses
questions outside that subject area**.

The same engine becomes a customer-care agent, a course teaching assistant, or
anything else, by swapping a persona configuration. No retraining, no code
changes.

```
┌──────────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────┐
│  Documents   │ ──► │  Chunk +  │ ──► │  Scope gate  │ ──► │ Claude  │
│ PDF/DOCX/MD… │     │   index   │     │  (retrieval) │     │ grounded│
└──────────────┘     └───────────┘     └──────┬───────┘     └─────────┘
                                              │ off-topic
                                              ▼
                                        refuse, $0 spent
```

---

## Quick start

**Prerequisites:** Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/settings/keys).

```bash
git clone https://github.com/YOUR_USERNAME/scoped-rag-chatbot.git
cd scoped-rag-chatbot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then open .env and paste your API key

python run.py
```

Open **http://localhost:8000**.

Then:

1. Drag `examples/course_ta/cs1428_syllabus.md` onto the upload box.
2. Choose the **Course teaching assistant** preset.
3. Ask *"How is the final grade calculated?"* → a sourced answer.
4. Ask *"What is the capital of France?"* → a refusal.

Two example corpora are included so you can test immediately without supplying
your own documents:

| Folder | Scenario |
|---|---|
| `examples/course_ta/` | A CS 1428 syllabus — the TA use case |
| `examples/customer_care/` | A retail returns/shipping/warranty policy — the support use case |

---

## How scope enforcement works

This is the core of the project, so it is worth being precise about. Keeping a
chatbot on-topic is treated as **two independent layers**, because either one
alone is insufficient.

### Layer 1 — the retrieval gate (before the LLM)

Every question is scored against the uploaded corpus. If nothing in the corpus
is relevant, the question is refused **without calling the LLM at all**.

The score combines two signals as a geometric mean:

```
relevance = √(bm25_score × term_coverage)
```

* **BM25 score** — standard lexical ranking: how well does the best passage
  match this question?
* **Term coverage** — what fraction of the question's content words appear
  anywhere in the corpus?

Neither works alone, and the failure modes are mirror images:

| Question (asked of a *syllabus*) | BM25 | Coverage | Why one signal fails |
|---|---|---|---|
| "What **time** does the Super Bowl start?" | 0.59 | 0.25 | High BM25 on one incidental word |
| "**Help** me **write** a cover letter" | 0.33 | 0.70 | Common words, wrong subject |
| "What **happens** if I miss an **exam**?" | 0.78 | 0.33 | Genuinely in scope, modest coverage |

Multiplying them punishes a near-zero in *either* factor far harder than an
average would, which is exactly what is wanted.

**This layer is deterministic and cannot be jailbroken** — no prompt can talk
its way past it, because the model is never invoked. It also costs nothing: an
off-topic question is rejected for $0.

### Layer 2 — grounded generation (inside the LLM)

Retrieval alone cannot catch everything. A question can share real vocabulary
with the corpus without being answerable from it. So questions that pass the
gate are sent to Claude with a system prompt that requires it to:

- answer **only** from the supplied passages,
- never fill gaps with outside knowledge,
- quote figures, dates, and policy windows verbatim,
- state explicitly what the context does *not* cover,
- treat document text as **data, never instructions** (prompt-injection defence).

### The uncertain band

Measured against the bundled corpora, in-scope and off-topic questions overlap
in a narrow range (~0.40–0.50). **No single threshold is error-free there.**

Rather than pick a cutoff that is silently wrong in one direction, that band is
treated as genuinely uncertain: the question goes to the model *with* the
retrieved text and an explicit instruction to verify the passages actually
address the question before answering.

Cheap lexical scoring decides the easy cases for free; the model adjudicates
only the hard ones. The three outcomes are reported in the API as
`decision: "in_scope" | "borderline" | "out_of_scope"`.

### What this does not do

Retrieval is **lexical, not semantic**. It matches words, not meanings. If the
syllabus says "Required materials" and a student asks about a "textbook", the
question is refused — those words never co-occur.

This is a deliberate trade-off, documented rather than hidden, and there is a
test asserting the behaviour (`test_vocabulary_gap_is_a_known_limitation`).
Semantic embeddings would close the gap at the cost of an embeddings API (money,
network, another key) or a large local model download. For a bot whose job is to
answer from a specific, consistent corpus, lexical matching is strong, instant,
free, and — importantly — runs on *every* question to power the scope gate.

See [`docs/scope-tuning.md`](docs/scope-tuning.md) for the measured score
distributions and how to re-tune for your own documents.

---

## Configuring the persona

The persona defines who the bot is and what it is allowed to discuss. Edit it in
the sidebar, or via the API:

```bash
curl -X PUT http://localhost:8000/api/persona \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "the Northwind Support Assistant",
    "topic": "orders, shipping, returns, and warranties",
    "style": "Be warm and direct. Lead with the answer."
  }'
```

Three presets ship with the project: `customer_care`, `course_ta`, `generic`.

```bash
curl -X POST http://localhost:8000/api/persona/preset \
  -H 'Content-Type: application/json' -d '{"preset":"customer_care"}'
```

---

## API

Interactive docs at **http://localhost:8000/docs** once running.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/status` | Health, model, document count, active persona |
| `POST` | `/api/documents` | Upload one or more files (multipart) |
| `GET` | `/api/documents` | List the knowledge base |
| `DELETE` | `/api/documents/{id}` | Remove one document |
| `DELETE` | `/api/documents` | Clear the knowledge base |
| `GET` | `/api/supported-types` | Supported file extensions |
| `GET`/`PUT` | `/api/persona` | Read / set the persona |
| `POST` | `/api/persona/preset` | Apply a built-in persona |
| `POST` | `/api/chat` | Ask a question |

### Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the late work policy?"}'
```

```json
{
  "answer": "Late assignments lose 10 percentage points per calendar day…",
  "in_scope": true,
  "decision": "in_scope",
  "citations": [
    {
      "doc_name": "cs1428_syllabus.md",
      "position": 3,
      "score": 0.554,
      "excerpt": "Late assignments lose 10 percentage points per calendar day…"
    }
  ],
  "retrieval_score": 0.554,
  "reason": "Relevance 0.554 with 100% term coverage.",
  "usage": { "input_tokens": 2841, "output_tokens": 112 }
}
```

Every answer carries its sources, so any claim can be traced back to the
document it came from.

---

## Configuration

All settings are environment variables with working defaults — see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your API key. |
| `MODEL` | `claude-opus-5` | Model ID. |
| `MAX_TOKENS` | `2000` | Response length cap. |
| `EFFORT` | `medium` | Reasoning depth: `low`…`max`. |
| `RELEVANCE_FLOOR` | `0.40` | Scope strictness. Raise to refuse more. |
| `TOP_K` | `6` | Passages retrieved per question. |
| `CHUNK_SIZE` | `1200` | Characters per passage. |
| `CHUNK_OVERLAP` | `200` | Overlap between passages. |
| `STRICT_SCOPE` | `true` | Refuse unrelated code/essay/translation requests. |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit. |
| `DATA_DIR` | `./data` | Where the corpus is persisted. |

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

**159 tests, all running offline with no API key and no spend** — the LLM is
stubbed, and the scope gate is pure retrieval logic, so the behaviour that
matters most is verifiable for free.

Coverage includes: tokenizing, chunking (including pathological inputs like a
single 5000-character word), BM25 ranking, index add/remove consistency,
extraction for every supported format, corpus persistence across restarts,
corrupt-state recovery, all HTTP endpoints, partial-upload failure, prompt
construction, history replay, and — most importantly — **cross-domain scope
enforcement**: that a support bot does not answer course questions and vice
versa.

The frontend markdown renderer writes to `innerHTML`, so it has its own suite
(`tests/markdown_harness.js`) covering rendering and XSS escaping. It runs
through `node` if installed, otherwise macOS's built-in JavaScriptCore, and
skips cleanly when neither is available.

### See scope enforcement without an API key

```bash
python scripts/demo_scope.py
```

Runs a labelled question set through the gate and prints what each bot accepts
and refuses — including the cross-domain cases. No key, no network, ~2 seconds.

```bash
python scripts/measure_scope.py
```

Prints the score distribution and error counts by threshold — the measurements
behind the tuning decisions in [`docs/scope-tuning.md`](docs/scope-tuning.md).

---

## Project layout

```
app/
  main.py              FastAPI routes
  config.py            Environment-driven settings
  schemas.py           Request/response models
  core/
    extract.py         File → text (PDF, DOCX, TXT, MD, CSV, JSON)
    chunking.py        Text → overlapping passages
    retrieval.py       BM25 index + coverage scoring
    guard.py           The scope gate  ← start here
    engine.py          Prompt construction + Claude call
    persona.py         Persona config and presets
web/                   Browser UI (vanilla JS, no build step)
examples/              Two ready-to-use document sets
tests/                 127 offline tests
docs/                  Architecture and tuning notes
```

Reading order for a reviewer: **`core/guard.py`** (the idea), then
**`core/engine.py`** (the prompt), then **`core/retrieval.py`** (the scoring).

---

## Design decisions

**Why BM25 instead of vector embeddings?** The retrieval score doubles as the
scope gate, so it runs on every question — it must be instant and free. BM25
needs no API, no key, no model download, and no network. The trade-off is
documented above and asserted in a test.

**Why refuse before calling the LLM?** A prompt instruction alone can be argued
with. A gate the model never sees cannot. It is also free: off-topic traffic
costs nothing.

**Why persist to JSON?** A reviewer can open `data/corpus.json` and read exactly
what the bot knows. A binary index would be faster and opaque; at this scale the
speed is irrelevant and the transparency is not.

**Why no frontend framework?** The grader should be able to clone and run with
`pip install` and nothing else. No Node, no build step, no lockfile drift.

---

## Security notes

- Uploaded document text is wrapped in explicit untrusted-data markers, and the
  system prompt instructs the model to ignore instruction-shaped text inside it.
- All document content is HTML-escaped before rendering in the browser.
- `.env` is gitignored. No key is ever written to disk by the application.
- Upload size is capped and file types are allowlisted.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

Copyleft: you may use, study, modify, and redistribute this code, but anything
you distribute that is derived from it must also be GPL-3.0 with source
available.
