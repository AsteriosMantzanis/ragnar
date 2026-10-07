<p align="center">
  <img width="600" alt="Ragnar" src="https://github.com/user-attachments/assets/386125a7-8027-41bf-96c9-978c3ec6617f" />
</p>

<h1 align="center">Ragnar</h1>

<p align="center">
  A containerized retrieval-augmented generation workbench.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
</p>

Ragnar ingests PDF and DOCX files, stores hybrid dense+sparse vectors in Qdrant,
and exposes a query pipeline through FastAPI and a Streamlit UI. Ollama provides
the local language model; Redis stores sessions and indexing job state.

## Features

- Dense and sparse FastEmbed vectors fused with Qdrant Reciprocal Rank Fusion.
- Flat subsection indexing or hierarchical section-and-subsection indexing.
- Docling conversion isolated from embedding/indexing in separate processes.
- Persistent JSONL spools, per-document progress checkpoints, and retry of
  failed jobs without re-converting completed spools.
- Deterministic Qdrant point IDs so retried upserts overwrite existing points.
- Query reranking, semantic caching, and per-stage metrics.
- FastAPI endpoints and a Streamlit chat/indexing interface.

## Architecture

```text
Host browser
    |
    +-- Streamlit UI :8501 --API_URL--> FastAPI :8000
                                         |       |
                                         |       +-- Redis job queue
                                         |                 |
                                         |             Indexer worker
                                         |              /         \
                                         |       Docling process  Embed/index process
                                         |              \         /
                                         +-------------- Qdrant :6333
                                         +-------------- Ollama :11434
```

For each queued job, the worker runs Docling conversion first and writes one
document per line to a JSONL spool. That subprocess exits before the embedding
process starts. The indexing process streams the spool through the selected
chunkers and writes vectors to Qdrant in bounded batches.

After all writes for a JSONL document succeed, the worker atomically records a
checkpoint. If an indexing child is killed, the worker records the job as
failed, keeps the spool/checkpoint, exits, and is restarted by Compose. Retrying
the job resumes at the first uncheckpointed document. If a process died midway
through a document, that document is replayed; deterministic IDs make its
already-written points overwrite instead of duplicate.

Successful job spools are deleted. Failed job spools remain in the `index_spool`
named volume. `docker compose down` preserves named volumes; `docker compose
down -v` removes them, including retry data.

## Indexing Strategies

| Strategy | Chunks written | Collections | Query behavior |
|---|---|---|---|
| `flat` | Subsection chunks only | `subsections` | Hybrid retrieval from one collection |
| `hierarchical` | Parent sections and child subsections | `sections` and `subsections` | Retrieve children, then load linked parents |

Both strategies currently store dense and sparse vectors. Flat means a single
chunk level, not dense-only. Use the same strategy when indexing and querying.

## Requirements

- Docker Desktop with Docker Compose, or Docker Engine/Compose.
- Python 3.12 and `uv` for local development.
- Internet access on first startup to download Ollama and embedding/Docling
  models.
- A host documents directory mounted into the containers at `/docs`.

## Quick Start

Create a local environment file from the template if needed:

```powershell
Copy-Item dev.env .env
```

On macOS/Linux:

```bash
cp dev.env .env
```

Edit `.env` for model overrides or tokens. The Compose file currently mounts
`C:/Users/aster/Desktop/ragnar_papers` into `/docs`; change that bind mount to
your documents directory if it differs.

Start the stack:

```bash
docker compose up --build -d
```

- Streamlit UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Qdrant dashboard/API: <http://localhost:6333/dashboard>
- Ollama API: <http://localhost:11434>

Ollama pulls the configured model on startup. First-time model downloads can
take several minutes. Model files and indexing spools are stored in Docker
named volumes.

### Run Your First Index

1. In `docker-compose.yaml`, set the `api` and `indexer` document bind mounts
  to a host folder containing your PDFs or DOCX files. The container path is
  `/docs` by default.
2. Open <http://localhost:8501>, select `flat` or `hierarchical`, enter `/docs`,
  and click **Index documents**.
3. Wait for the job to complete. If it fails, the UI offers **Retry from
  checkpoint**; completed document records are not reprocessed.
4. Open the Chat tab, select the same strategy used for indexing, and ask a
  question about the documents.

## Index and Query

In the UI, select `flat` or `hierarchical`, set the path to `/docs`, and submit.
The index endpoint waits for completion and returns the job ID and counts.
If a job fails, the UI offers a retry from its last completed JSONL document.

Equivalent API requests:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path":"/docs","strategy":"hierarchical"}'
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key findings?","strategy":"hierarchical"}'
```

Retry a failed job by its returned `job_id`:

```bash
curl -X POST http://localhost:8000/index/<job_id>/retry
```

The API also exposes `GET /sessions/{session_id}`, `DELETE
/sessions/{session_id}`, and `GET /logs`. Query responses include the answer, sources, and stage metrics.

## Configuration

Compose reads values from the shell or local `.env`; defaults are shown below.
The model variable names are lowercase and are passed to both API and indexer.

| Variable | Purpose | Compose default |
|---|---|---|
| `LLM_MODEL` | Ollama model | `gemma2:2b` |
| `dense_embed_model` | Dense FastEmbed model | `nomic-ai/nomic-embed-text-v1.5-Q` |
| `sparse_embed_model` | Sparse FastEmbed model | `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| `PARENT_COLLECTION` | Hierarchical section collection | `sections` |
| `CHILD_COLLECTION` | Subsection collection | `subsections` |
| `INDEX_BATCH_SIZE` | Chunks embedded/upserted per batch | `1` |
| `DENSE_EMBED_THREADS` | ONNX threads for dense model | `2` |
| `SPARSE_EMBED_THREADS` | ONNX threads for sparse model | `2` |
| `DOCLING_DO_OCR` | Enable OCR for scanned PDFs | `false` |
| `DOCLING_BATCH_SIZE` | Docling page batch size | `1` |
| `DOCLING_NUM_THREADS` | Docling CPU threads | `2` |
| `HF_TOKEN` | Optional Hugging Face token | empty |

`INDEX_SPOOL_DIR` defaults to `/var/lib/ragnar/index-spool` in the indexer and
is backed by the `index_spool` volume. Changing an embedding model requires
reindexing. Changing the dense model to a different vector dimension also
requires new or recreated Qdrant collections.

For a manually restarted worker, use:

```bash
docker compose up -d --force-recreate indexer
```

## Local Development

Install dependencies and start the backing services:

```bash
uv sync
docker compose up -d qdrant redis ollama
```

Run the API and UI in separate terminals:

```bash
uv run python -m ragnar.api.main
```

```bash
uv run streamlit run ragnar/ui/streamlit_app.py
```

The API uses `http://localhost:6333`, `redis://localhost:6379`, and
`http://localhost:11434` by default when run outside Compose.

## Project Layout

```text
ragnar/
├── api/                 FastAPI application and dependency factories
├── indexing/            Redis worker, process stages, JSONL spool, checkpoints
├── loaders/             Docling loader and loader interfaces
├── chunkers/            Fixed, page, section, and subsection chunkers
├── embeddings/           Dense and sparse FastEmbed adapters
├── index/                Qdrant indexing adapter
├── retrieval/            Dense, sparse, and hybrid Qdrant retrievers
├── pipelines/            Flat/hierarchical index and query pipelines
├── rerank/               Cross-encoder reranking
├── generation/            LLM response generation
├── cache/                 Semantic response cache
├── session/               Session models and Redis store
├── observability/         Query logs and metrics
└── ui/                    Streamlit application
```

## Development Checks

Run the configured repository hooks before committing:

```bash
pre-commit run --all-files
```

## License

MIT. See [LICENSE](LICENSE).
