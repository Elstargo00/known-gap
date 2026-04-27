# Known Gap

A knowledge-gap-aware Q&A service. `/ask` accepts a `mode` — **normal**,
**learning**, or **concise** — and tailors the answer to what each user has
already mastered, by combining a pgvector RAG pipeline with a per-user
**typed knowledge graph** on FalkorDB.

Each concept node carries a `known_score` (0–100) that moves with every
exchange: re-asking something you've already mastered nudges it down,
encountering a new concept inside an answer nudges it up. Concepts above a
configurable threshold get masked as **fill-in-the-blank cloze deletions**
in learning mode, so the answer trains recall on what you already know.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the deployment topology, design
patterns, sequence diagrams, and data model.

## Features

- `POST /ingest` — upload `.pdf`, `.md`, or `.txt` files; extracted text is
  chunked, embedded with Voyage, and stored in Postgres + pgvector. The user's
  graph is also seeded with concepts extracted from the document.
- `POST /ask` with three modes:
  - **normal** — standard RAG answer.
  - **learning** — full answer with concepts whose `known_score` is above the
    threshold wrapped in `<cloze concept="…">…</cloze>` tags so the UI can
    render them as click-to-reveal blanks.
  - **concise** — skips re-explaining anything already mastered, told to the
    LLM via the system prompt.

## Prerequisites

- Docker + Docker Compose
- [`uv`](https://github.com/astral-sh/uv) ≥ 0.5 (only needed for running
  outside Docker)
- API keys: [Anthropic](https://console.anthropic.com/),
  [Voyage](https://www.voyageai.com/), and optionally
  [Gemini](https://aistudio.google.com/)
- A `JWT_SECRET` ≥ 32 bytes — both endpoints require a bearer token

## Quickstart (Docker Compose)

```bash
# 1. Clone and enter the repo
git clone <repo-url> no-gap && cd no-gap

# 2. Fill in .env
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY, VOYAGE_API_KEY, JWT_SECRET

# 3. Bring the stack up (FastAPI + Postgres + FalkorDB)
docker compose up --build

# 4. Service is at http://localhost:8001
#    FalkorDB browser UI at http://localhost:3001 (Redis on 6380)
#    Postgres on host port 5433
```

Swagger UI: `http://localhost:8001/docs` — click **Authorize** and paste the
JWT minted below.

## Minting a JWT for testing

```bash
# Fresh user (new graph)
uv run python scripts/mint_jwt.py

# Specific user (reuses their graph across runs)
uv run python scripts/mint_jwt.py --user-id 11111111-1111-1111-1111-111111111111

# Custom lifetime
uv run python scripts/mint_jwt.py --expires-hours 1
```

The script reads `JWT_SECRET` from `.env` and prints `Bearer <token>` to
paste into the Swagger Authorize dialog or a `curl -H` flag.

## Endpoints

### `POST /ingest`

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/sample.pdf"
```

### `POST /ask`

```bash
curl -X POST http://localhost:8001/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I subclass Sequence from collections.abc?", "mode": "learning"}'
```

## Environment Variables

See [`.env.example`](./.env.example) for the full list with defaults. Keys you
**must** set for the stack to come up:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Primary answer LLM (Sonnet) + concept extraction (Haiku) + Graph Expert (Opus) |
| `VOYAGE_API_KEY` | Embeddings (`voyage-3`, 1024-dim) |
| `JWT_SECRET` | HS256 shared secret (≥ 32 bytes) — must match the frontend |

Optional:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Transient-failure fallback for Anthropic |
| `TOP_K` | `10` | pgvector retrieval cutoff |
| `LLM_PRIMARY_MODEL` / `LLM_FALLBACK_MODEL` / `CONCEPT_EXTRACTION_MODEL` / `GRAPH_EXPERT_MODEL` | see `.env.example` | Model overrides |
| `KNOWN_SCORE_THRESHOLD` | `50` | A concept is "known" when `known_score > THRESHOLD` (strictly above). Drives cloze masking. |
| `KNOWN_SCORE_INITIAL` | `60` | First-exposure score for concepts the user has actually been shown (answer-introduced and ingested). Set above the threshold so a topic-of-the-question concept is classified known on the next turn. |
| `KNOWN_SCORE_INCREMENT` | `10` | Bump applied to answer-only concepts. |
| `KNOWN_SCORE_DECREMENT` | `10` | Penalty applied to re-asked already-known concepts. |
| `GRAPH_EXPERT_DEGREE` | `3` | Max relations the Graph Expert proposes per seed concept. |
| `CONCEPT_NEIGHBORHOOD_MAX_HOPS` | `2` | Max hops when the LLM calls `get_concept_neighborhood`. |
| `TOOL_MAX_ITERATIONS` | `6` | Hard cap on the answerer's tool-calling loop. |

## Development Workflow

```bash
uv sync                        # install from uv.lock
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run ty check                # type check
uv run pytest                  # unit tests (57 currently)
```

CI runs the same four checks on every push (`.github/workflows/ci.yml`).

## Project Structure

```text
src/known_gap/
├── api/                       # FastAPI endpoints, schemas, DI
├── application/
│   ├── services/
│   │   ├── chunker.py
│   │   ├── concept_extractor.py
│   │   ├── knowledge_classifier.py     # partitions mentions into known / unknown by known_score
│   │   ├── answer_post_processor.py    # extracts answer concepts → upserts into the graph
│   │   ├── score_updater.py            # +increment / -decrement / clamp [0, 100]
│   │   ├── cloze_processor.py          # wraps mastered concepts in <cloze> tags
│   │   ├── graph_expander.py           # background task: GraphExpert → upsert new edges
│   │   └── tool_enabled_answerer.py    # Anthropic tool-calling loop with fallback
│   ├── tools/
│   │   └── graph_tools.py              # LangChain BaseTools for graph lookup / neighbourhood / path
│   ├── strategies/                     # normal / learning / concise (Strategy pattern)
│   └── use_cases/                      # ask, ingest_document (orchestrators)
├── domain/
│   ├── models/                         # Document, Chunk, Concept (with known_score),
│   │                                   # ConceptMention, ConceptRelation, ConceptNeighborhood, …
│   ├── repositories/                   # ports — no infrastructure imports
│   └── services/                       # LLMProvider, EmbeddingProvider, GraphExpert,
│                                       # DocumentParser (ports)
├── infrastructure/
│   ├── auth/                           # JWT verifier
│   ├── db/                             # Postgres adapters + migrations
│   ├── embeddings/                     # Voyage provider + factory
│   ├── graph/                          # FalkorDB adapter (typed edges, score arithmetic)
│   ├── llm/                            # Anthropic, Gemini, AnthropicGraphExpert, Fallback, Factory
│   └── parsing/                        # PDF / MD / TXT dispatch
├── shared/                             # exceptions, utils
└── config/                             # settings (pydantic-settings)
```

## License

TBD.
