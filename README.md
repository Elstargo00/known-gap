# Known Gap

A knowledge-gap-aware Q&A service. `/ask` accepts a `mode` — **normal**,
**learning**, or **concise** — and tailors the answer to what each user is
known to have seen before by combining a pgvector RAG pipeline with a
per-user FalkorDB knowledge graph.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the deployment topology, design
patterns, and data model.

## Features

- `POST /ingest` — upload `.pdf`, `.md`, or `.txt` files; extracted text is
  chunked, embedded with Voyage, and stored in Postgres + pgvector.
- `POST /ask` with three modes:
  - **normal** — standard RAG answer, no graph lookup.
  - **learning** — full answer with `<known>` inline tags over concepts the
    user already has in their graph, so the UI can de-emphasise them.
  - **concise** — skips re-explaining anything the user already knows.
- Per-user knowledge graph on FalkorDB; every answer post-updates the graph.
- LLM provider **factory** with Anthropic primary and Gemini fallback on
  transient errors only.
- **Strategy** pattern for modes, **Repository** pattern for Postgres and
  FalkorDB, all testable against pure domain ports.

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
#    FalkorDB browser UI at http://localhost:3000
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

Response:
```json
{ "document_id": "…", "filename": "sample.pdf", "chunk_count": 12 }
```

### `POST /ask`

```bash
curl -X POST http://localhost:8001/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is recursion?", "mode": "learning"}'
```

Response (shape):
```json
{
  "answer": "Recursion is a technique where <known concept=\"function\">a function</known> calls itself…",
  "sources": [
    { "chunk_id": "…", "document_id": "…", "filename": "intro.md",
      "content_preview": "…", "similarity_score": 0.87 }
  ],
  "mode": "learning",
  "known_concepts":   [{ "canonical_name": "function", "display_name": "Function" }],
  "unknown_concepts": [{ "canonical_name": "base case", "display_name": "Base case" }]
}
```

`known_concepts` and `unknown_concepts` are empty on `mode="normal"`.

## Environment Variables

See [`.env.example`](./.env.example) for the full list with defaults. Keys you
**must** set for the stack to come up:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Primary LLM (Sonnet 4.6) and concept extraction (Haiku) |
| `VOYAGE_API_KEY` | Embeddings (`voyage-3`, 1024-dim) |
| `JWT_SECRET` | HS256 shared secret (≥ 32 bytes) |

Optional:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Transient-failure fallback for the Anthropic provider |
| `TOP_K` | Retrieval cutoff (default `10`) |
| `LLM_PRIMARY_MODEL` / `LLM_FALLBACK_MODEL` / `CONCEPT_EXTRACTION_MODEL` | Model overrides |

## Development Workflow

```bash
uv sync                        # install from uv.lock
uv run ruff check --fix .      # lint
uv run ruff format .           # format
uv run ty check                # type check
uv run pytest                  # unit tests
```

CI runs the same four checks on every push (`.github/workflows/ci.yml`).

## Project Structure

```text
src/known_gap/
├── api/                  # FastAPI endpoints, schemas, DI
├── application/
│   ├── services/         # chunker, concept_extractor, knowledge_classifier,
│   │                     # answer_post_processor
│   ├── strategies/       # normal / learning / concise (Strategy pattern)
│   └── use_cases/        # ask, ingest_document (orchestrators)
├── domain/
│   ├── models/           # Document, Chunk, Concept, RetrievedChunk, …
│   ├── repositories/     # ports — no infrastructure imports
│   └── services/         # LLMProvider, EmbeddingProvider, DocumentParser (ports)
├── infrastructure/
│   ├── auth/             # JWT verifier
│   ├── db/               # Postgres adapters + migrations
│   ├── embeddings/       # Voyage provider + factory
│   ├── graph/            # FalkorDB adapter
│   ├── llm/              # Anthropic, Gemini, Fallback, Factory
│   └── parsing/          # PDF / MD / TXT dispatch
├── shared/               # exceptions, utils
└── config/               # settings (pydantic-settings)
```

## License

TBD.
