# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MS-RAGS(ALL-IN-ONE) is a terminal CLI (`ms-rags`) that walks a user through a 16-step guided
workflow to configure every layer of a RAG (Retrieval-Augmented Generation) system —
credentials, RAG architecture, document loading, chunking, embeddings, vector DB, query
enhancement, retrieval, reranking, compression, system prompt, and evaluation. At the end it
generates a **standalone `pipeline.py` + `requirements.txt`** that has zero runtime dependency
on this package. It is a CLI workbench and code generator, not a library that gets imported by
downstream projects — see `AGENTS.md` for the full agent-facing handoff doc and core principles
(permission-first flow, no silent fallbacks, never persist secrets to generated code or session
JSON).

## Commands

```bash
# Install for development (editable, full feature set)
pip install -e ".[production]"     # all providers/vector DBs (except Redis/ares-ai)/evaluators
pip install -e ".[dev]"            # pytest, hypothesis, coverage only
pip install -e .                   # minimal core install

# Run the CLI
ms-rags                            # or: python -m ms_rag
ms-rags --load session.json        # resume a saved session

# Tests
pytest tests/ -v                                # full suite
pytest tests/unit/ -v                           # unit tests only
pytest tests/property/ -v                       # Hypothesis property-based tests
pytest tests/integration/ -v                    # includes rebuild_session_runtime
pytest tests/unit/test_credentials.py -v        # single file
pytest tests/unit/test_credentials.py::test_name -v   # single test
pytest tests/ --cov=ms_rag --cov-report=html    # with coverage

# LangChain import audit (AST-based; verifies every import in ms_rag/ resolves
# and flags deprecated import paths)
python scripts/audit_imports.py
```

`tests/smoke/` is marked `smoke` (opt-in live backend smoke tests) and is not part of the
default run. There is no configured linter/formatter (no ruff/black/mypy config in
`pyproject.toml`) — match existing style rather than introducing new tooling.

## Architecture

### The central object: `PipelineConfig`

`ms_rag/models.py` defines `PipelineConfig`, a single dataclass that accumulates every user
choice across all 16 workflow steps (RAG type, chunking config, embedding model, vector DB,
retrieval strategy, reranker, compression, evaluation, etc.). It is:

- The **sole input** to the code generator (`ms_rag/codegen/code_generator.py`), which renders
  Jinja2 templates into a standalone `pipeline.py`.
- **Serialized to JSON** by `ms_rag/session/session_manager.py` for `/save` and `--load`, with
  secrets stripped/sanitized before serialization (see `_sanitized_vector_db_config` and
  `_is_sensitive_connection_field` in `models.py` — any new connection-param field containing
  KEY/SECRET/TOKEN/PASSWORD/URI/URL/ENDPOINT etc. is treated as sensitive by default).

`CredentialStore` and `SessionState` (also in `models.py`) are runtime-only and are **never**
serialized — credentials are re-prompted on `--load` rather than persisted.

### Workflow steps → module map

Each numbered step in the README/AGENTS.md flow maps to a module that owns that step's
questionary prompts and produces a piece of `PipelineConfig`:

| Step | Module |
|------|--------|
| Credentials | `config/credential_manager.py` (12 providers, Fernet+PBKDF2 encryption) |
| RAG architecture | `workflow/rag_type_selector.py`, `workflow/rag_presets.py` |
| Document types / loaders | `ingestion/document_type_selector.py`, `ingestion/loader_selector.py` |
| Chunking | `ingestion/chunking_engine.py`, `workflow/chunking_configurator.py` |
| Embeddings | `ingestion/vectorization_module.py` |
| Vector DB + live ingestion | `ingestion/vectordb_connector.py`, `ingestion/ingestion_orchestrator.py` |
| Keyword store (for hybrid/BM25/TF-IDF on cloud vector DBs) | `ingestion/keyword_store.py` |
| Graph store (GraphRAG) | `ingestion/graph_store.py` |
| Query enhancement | `query/query_enhancer.py` |
| Retrieval strategy | `query/retrieval_strategy.py` |
| Reranking | `query/reranking_module.py`, `query/reranking_retriever.py` |
| Context compression | `query/context_compressor.py`, `query/compression_retriever.py` |
| System prompt | `workflow/system_prompt_configurator.py` |
| Evaluation | `evaluation/evaluation_framework.py`, `evaluation/evaluator_runners.py` |
| Agentic tools (Agentic/Corrective RAG) | `agent/tools.py`, `agent/tool_configurator.py` |
| LLM factory + LCEL/LangGraph chains | `llm/llm_integration.py` |
| CLI wiring + entry point | `cli/main.py` (16-step orchestration), `cli/query_loop.py` (interactive loop) |

Preset RAG types (HyDE, Multi-Query, RAG-Fusion, Parent-Child, Contextual Compression,
GraphRAG) lock their required downstream modules; Advanced/Modular RAG leave module choices
open for composition. `requires_langgraph` on `RAGTypeConfig` marks the RAG types
(Agentic, Self-RAG, Corrective RAG, Adaptive RAG) that build a LangGraph `StateGraph` instead of
a plain LCEL chain in `llm_integration.py`.

### Session rebuild path

`--load` and `/save` don't just restore config — `rebuild_session_runtime()` in
`llm/llm_integration.py` reconstructs the *live* vector store connection, retriever stack, LLM,
and RAG chain from a loaded `PipelineConfig`. Advanced retrieval strategies (Parent-Child,
Multi-Vector, Time-Weighted) depend on runtime state that isn't in the vector DB itself (parent
docs, an in-memory FAISS representation index, `ms_rag_ingested_at` metadata) — original
document sources must remain available for these to rebuild correctly. `tests/integration/`
covers this path.

### Error handling conventions

- All framework exceptions inherit from `MSRAGError` (`utils/exceptions.py`): `ConnectionError`,
  `IngestionError`, `CredentialError`, `SessionLoadError`, `ValidationError` — each carries
  structured context (e.g. `db_type`, `original`, `field_name`) rather than just a message.
- External API calls (embedding, LLM inference, vector DB writes) should go through
  `retry_with_backoff` / `retry_with_user_prompt` in `utils/retry.py`, which implements the
  Retry/Skip/Abort pattern used throughout the CLI on failure.
- Degradation is never silent: when a feature falls back (e.g. an evaluator package is
  incompatible, OCR tooling is missing), the CLI prints a visible Rich notice *and* emits a
  structured JSON log (`event`, `component`, `reason`, `action` fields) via `utils/logging.py`.

### Import discipline (LangChain ecosystem churn)

This codebase was audited to remove deprecated LangChain import paths — `scripts/audit_imports.py`
enforces via AST that every import in `ms_rag/` actually resolves. When adding LangChain-adjacent
code, use current packages, not `langchain_community` equivalents:

```python
# Deprecated — do not introduce
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from trulens_eval import TruChain

# Current
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_ollama import ChatOllama
from langchain_redis import RedisVectorStore
from trulens.apps.langchain import TruChain
from langchain_classic.retrievers import EnsembleRetriever
```

`grpcio` is pinned to `>=1.59.5,<1.80.0` for `weaviate-client` compatibility — don't loosen
without checking Weaviate compatibility.

### Docs site

`docs/` is a static multi-page site (deployed to Vercel, source of truth for public docs at
ms-rags-all-in-one.vercel.app). If a change affects user-visible CLI behavior, keep `docs/` and
`AGENTS.md` in sync per the principle in `AGENTS.md`. Preview locally with
`python -m http.server 3000` from repo root, then open `/docs/index.html`.

## Working conventions (from AGENTS.md)

- Keep the prompt flow permission-first: any feature needing credentials or external services
  must ask explicitly, deny-by-default (this applies especially to agentic tools in
  `agent/tools.py` — web search, file reads, API requests are all opt-in with explicit
  allowlists).
- Never persist secrets into generated `pipeline.py` or session JSON.
- Prefer explicit user choices over silent fallbacks or new defaults that bypass user choice.
- Read existing code before adding abstractions; prefer small, explicit fixes over broad
  rewrites.
