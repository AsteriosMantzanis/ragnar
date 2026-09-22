<p align="center">
  <img width="600" alt="Ragnar" src="https://github.com/user-attachments/assets/386125a7-8027-41bf-96c9-978c3ec6617f" />
</p>

<h1 align="center">Ragnar</h1>

<p align="center">
  A modular, production-ready, bring-your-own-model RAG framework.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/status-active%20development-orange.svg" alt="Active Development" />
</p>

---

Ragnar is a RAG (Retrieval-Augmented Generation) framework built around one idea: every stage of the pipeline is swappable behind an interface. Bring your own embedding models, your own LLM, your own vector store adapter — Ragnar handles orchestration, caching, session management, reranking, grounding, and observability around them.

It ships as a FastAPI service with a Streamlit reference UI, fully containerized, with a hybrid dense+sparse retrieval stack and a 5-step query pipeline (retrieve → dedupe → rerank → generate → ground) that runs on every request.

## Why Ragnar

Most RAG demos hardcode a vector store, an embedding model, and an LLM provider, then call it a framework. Ragnar instead defines an `interfaces/` ABC for every moving part — loader, chunker, embedder, retriever, reranker, generator, grounder, session store, cache — and ships one concrete implementation per interface. Swapping Qdrant for another vector DB, or Ollama for a hosted LLM, means writing one adapter class, not forking the pipeline.

## Features

- **Hybrid retrieval** — dense (`nomic-embed-text`) + sparse (SPLADE) vectors fused via Reciprocal Rank Fusion in Qdrant
- **Two indexing/query strategies** — `flat` (single-collection chunk retrieval) and `hierarchical` (parent section + child subsection collections, linked and retrieved together)
- **Query pipeline with 5 tracked stages** — retrieval, deduplication, cross-encoder reranking, generation, and answer grounding — each stage timed and recorded per request
- **Answer grounding** — every generated claim is checked against retrieved context via an NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`), and a grounded-percentage score is returned with the response
- **Semantic response caching** — a Qdrant-backed semantic cache short-circuits repeat first-turn questions (cosine threshold 0.90, 30-day TTL), skipped for follow-up turns in a conversation
- **Session-aware conversations** — Redis-backed (or in-memory) session store carries the last few exchanges into each new query as context
- **Document ingestion via Docling** — PDF/DOCX parsing with table and layout extraction, chunked at section or subsection granularity
- **Per-request observability** — every query returns step-by-step timing, source count, grounding percentage, and cache hit/score; a `MetricsCollector` aggregates across requests
- **FastAPI + Streamlit** — a documented REST API (`/docs`, `/redoc`) plus a chat UI showing sources, grounding, and per-query performance breakdown
- **Fully containerized** — one `docker compose up` brings up Ollama, Qdrant, Redis, the API, and the UI

## Architecture

Every subsystem follows the same pattern: an ABC in `interfaces/`, one or more concrete implementations beside it, and a factory function in `api/dependencies.py` (memoized with `@lru_cache`) that wires the chosen implementation into the pipeline. This keeps the pipeline classes dependent only on interfaces, never concrete classes, and makes new backends a matter of implementing one ABC.

```
ragnar/
├── api/                 # FastAPI app, request/response models, DI factories
├── pipelines/
│   ├── index/            # FlatIndexPipeline, HierarchicalIndexPipeline
│   ├── query/             # FlatQueryPipeline, HierarchicalQueryPipeline
│   └── interfaces/        # BaseIndexPipeline, BaseQueryPipeline (shared 7-step query flow)
├── loaders/              # DoclingLoader (PDF/DOCX → Document/DocumentElement)
├── chunkers/              # fixed, page, section, subsection chunkers
├── embeddings/
│   ├── dense/              # FastEmbed dense embedder (nomic-embed-text)
│   └── sparse/              # FastEmbed sparse embedder (SPLADE)
├── retrieval/             # dense, sparse, and hybrid (RRF) Qdrant retrievers
├── rerank/                # FastEmbed cross-encoder reranker
├── grounding/              # HF NLI cross-encoder grounding check
├── generation/             # answer generator (LLM + Jinja2 prompt templates)
├── cache/                 # Qdrant-backed semantic response cache
├── session/                # Session model + Redis / in-memory session stores
├── index/                  # QdrantIndexer (embeds + upserts chunks)
├── models/                 # Document, DocumentElement, Chunk data models
├── prompts/                 # Jinja2 (.j2) prompt templates + loader
├── observability/           # QueryTracker (per-step timing), MetricsCollector
└── ui/                      # Streamlit reference client
```

### Query flow

Every call to `BaseQueryPipeline.query()` runs the same five tracked steps, with the retrieval step overridden per strategy (flat vs. hierarchical):

1. **Retrieve** — the query is run against the vector store (single collection for `flat`, linked parent/child collections for `hierarchical`)
2. **Deduplicate** — results are merged by chunk ID
3. **Rerank** — a cross-encoder reranks the deduplicated set and keeps the top-k
4. **Generate** — an LLM answers the query using the reranked context
5. **Ground** — each claim in the answer is checked against the context, producing a grounded-percentage score

A semantic cache lookup runs before step 1 (and a cache write after step 5) only on the first turn of a session, so follow-up questions always execute the full pipeline.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker and Docker Compose (for the full stack: Ollama, Qdrant, Redis)
- A Hugging Face token (`HF_TOKEN`) if pulling gated embedding/grounding models

## Quick start

### Docker Compose (recommended)

Brings up Ollama (LLM), Qdrant (vector store), Redis (sessions/cache), the API, and the UI:

```bash
git clone <repo-url>
cd ragnar
cp .env.example .env   # fill in HF_TOKEN and any overrides
docker compose up --build
```

- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- UI: [http://localhost:8501](http://localhost:8501)

The `ollama` service automatically pulls `LLM_MODEL` (default `qwen2.5:3b`) on first boot.

### Local development

```bash
uv sync

# start dependencies only
docker compose up qdrant redis ollama

# run the API
uv run python -m ragnar.api.main

# in a separate terminal, run the UI
uv run streamlit run ragnar/ui/streamlit_app.py
```

## Configuration

Ragnar is configured entirely through environment variables (see `.env`):

| Variable | Description | Default |
|---|---|---|
| `dense_embed_model` | FastEmbed dense embedding model | `nomic-ai/nomic-embed-text-v1.5-Q` |
| `sparse_embed_model` | FastEmbed sparse embedding model | `prithivida/Splade_PP_en_v1` |
| `rerank_model` | Cross-encoder reranking model | `jinaai/jina-reranker-v2-base-multilingual` |
| `grounding_model` | NLI cross-encoder for answer grounding | `cross-encoder/nli-deberta-v3-base` |
| `LLM_MODEL` | Ollama model tag | `qwen2.5:3b` |
| `OLLAMA_URL` | Ollama base URL | `http://localhost:11434` |
| `QDRANT_URL` | Qdrant base URL | `http://localhost:6333` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `SESSION_STORE` | `redis` or `memory` | `redis` |
| `SESSION_TTL` | Session TTL, seconds | `604800` (7 days) |
| `PARENT_COLLECTION` | Qdrant collection for section-level chunks | `sections` |
| `CHILD_COLLECTION` | Qdrant collection for subsection-level chunks | `subsections` |
| `LINKAGE_ID` | Field linking child chunks to parent chunks | `parent_section_id` |
| `HF_TOKEN` | Hugging Face token for gated models | — |

> **Never commit a populated `.env`.** It's already git-ignored — keep it that way, and rotate any token that ends up in a shared file or archive.

## API

Interactive docs are served at `/docs` (Swagger) and `/redoc` once the API is running. Core endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/index` | Index documents from a filesystem path (`strategy`: `flat` \| `hierarchical`) |
| `POST` | `/query` | Run a query through the pipeline (`strategy`: `flat` \| `hierarchical`) |
| `GET` | `/sessions/{session_id}` | Retrieve a session's message history |
| `DELETE` | `/sessions/{session_id}` | Delete a session |

**Index documents:**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/docs", "strategy": "hierarchical"}'
```

**Query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "strategy": "hierarchical"}'
```

A query response includes the answer, per-chunk grounding results, an overall `grounded_percentage`, the reranked source chunks, the session ID, and a `metrics` block with per-step timings.

## Extending Ragnar

Adding a new backend for any stage is a three-step process:

1. Implement the relevant ABC (e.g. `ragnar/retrieval/interfaces/base_retriever.py`)
2. Add a factory function in `ragnar/api/dependencies.py`, decorated with `@lru_cache` if it should be a singleton
3. Wire it into the `flat`/`hierarchical` factory functions, or add a new strategy branch

Because pipelines depend only on interfaces, no other code needs to change.

## Roadmap

- Populate `ragnar/evals/` with RAGAS-based evaluation harnesses
- OpenTelemetry tracing across pipeline steps (currently custom `QueryTracker` timing only)
- Agent-based (v2) query pipeline, building on the existing ABC/factory pattern
- Automated test suite (none yet)

## License

MIT — see [LICENSE](LICENSE).
