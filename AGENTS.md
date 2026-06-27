# MS_RAG — Agent Handoff Document

This file is for any AI coding agent (Kiro, Codex, Claude Code, Cursor, Gemini CLI, etc.)
picking up this project. Read this fully before touching any code or spec files.

---

## What This Project Is

**MS_RAG** is a production-grade, terminal-based interactive CLI framework for building
complete RAG (Retrieval-Augmented Generation) pipelines. Inspired by OpenClaw's UX, it
guides users step-by-step through every layer of a RAG system and then generates a
standalone, deployable Python file tailored to their choices.

**It is NOT a library. It is a CLI workbench + code generator.**

---

## Spec Status: COMPLETE ✅

All three spec documents are fully written, synchronized, and deprecation-audited.
**Do NOT rewrite or restructure the spec.** Only update it when implementing reveals
a genuine gap, and only update the affected section.

| File | Location | Status |
|------|----------|--------|
| Requirements | `.kiro/specs/ms-rag/requirements.md` | ✅ Complete (20 requirements) |
| Design | `.kiro/specs/ms-rag/design.md` | ✅ Complete (29 correctness properties) |
| Tasks | `.kiro/specs/ms-rag/tasks.md` | ✅ Complete (24 tasks) |

---

## Implementation Status

**ALL 24 TASKS COMPLETE. Full implementation done.**

The next step is **installing dependencies and running the full test suite** with:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Recent Runtime Fix — Vector Store Rebuild Crash

**Issue observed:** Running `ms-rag` with FAISS local vector DB could ingest documents successfully, then crash before the live query loop with:

```text
RuntimeError: FAISS store has no documents yet.
```

**Root cause:** The interactive setup path ingested into a populated vector store, then called `rebuild_session_runtime()`, which created a brand-new vector store wrapper. FAISS made this visible immediately because the fresh wrapper had no in-memory documents, so MMR/dense retrieval failed when building the retriever. The same pattern was risky for Chroma, Pinecone, Qdrant, Weaviate, Milvus, Redis, PGVector, Elasticsearch, OpenSearch, Azure AI Search, and MongoDB Atlas because the live setup should always query the exact store that ingestion just populated.

**Resolution:** The interactive path now calls `build_session_runtime_from_vector_store()` and reuses the already-populated vector store for every backend. `_FAISSFactory` also persists by default to `./faiss_indexes/{collection_name}`, honors `FAISS_INDEX_PATH`, loads an existing local FAISS index on startup, and saves after `add_documents()`, so `--load` can rebuild FAISS-backed sessions when the persisted index directory exists. Chroma accepts both `CHROMA_PERSIST_DIRECTORY` and the legacy `CHROMA_PERSIST_DIR` alias.

**Generated pipeline hardening:** Standalone generated FAISS pipelines now save on ingest and load the persisted index in query-only mode. Standalone generated Qdrant pipelines now connect to an existing collection in query-only mode instead of calling `from_documents()` without chunks.

**Verification run:** `.\.venv\Scripts\python.exe -m pytest tests\property\test_codegen_properties.py tests\integration\test_rebuild_session_runtime.py tests\integration\test_end_to_end.py tests\unit\test_vectordb_connector.py -q` passed with 41 tests.

### Recent Production UX Hardening — Permissions, Embeddings, and Logs

**User permission principle:** Before irreversible or state-changing setup work, keep an explicit confirmation gate. The live setup now reviews embedding model, embedding dimension, vector DB, collection/index, and sources before ingestion, then asks for confirmation before writing vectors.

**Generation model selection:** The CLI now has an explicit Step 2b for selecting the LLM provider/model that will answer user queries. `PipelineConfig.llm_model` stores that selection so live runtime, `/save` + `--load`, and generated standalone pipelines all use the exact selected generation model. Do not silently fall back to OpenAI when no generation model is selected; ask the user to configure providers/models again or fail loudly in non-interactive paths.

**No secret persistence:** `PipelineConfig.to_json()` must not persist vector database secrets or connection strings. Sensitive vector DB `connection_params` are serialized as env-var markers, and saved sessions should re-prompt or resolve them from environment variables on load.

**No silent production degradation:** Query enhancement, retrieval, reranking, compression, and evaluation may keep the query loop alive when a feature fails, but they must emit an explicit warning/log before falling back. Do not add bare `except/pass` paths for user-selected features.

**Prompt cancellation safety:** Interactive steps must re-prompt on `questionary.*().ask()` returning `None`; do not treat cancellation as a real selection, disabled feature, or model/config value. Prefer shared prompt helpers, or local retry loops when tests patch module-level `questionary`.

**Vector DB recovery preserves embedding context:** When vector DB connection recovery re-selects a database, pass the selected embedding model back into `prompt_and_configure()` so `VectorDBConfig.dimension` remains aligned with the embedding model.

**Generated/runtime parity:** If runtime supports an LLM provider, the code generator must generate matching imports, credentials, and constructor code for standalone pipelines. Keep Together AI, Replicate, AWS Bedrock, Ollama cloud/local, HuggingFace endpoint, Azure OpenAI, and OpenAI-compatible providers aligned.

**Embedding/vector DB compatibility:** Step 8 now explains that embedding dimensions must match the vector DB collection/index. Step 9 carries the selected embedding dimension into `VectorDBConfig.dimension` and displays it in the vector DB summary. If users change embedding models, guide them to create a fresh collection/index or re-ingest.

**HuggingFace embedding modes:** HuggingFace embeddings are split into local/downloaded models (`provider="huggingface"`) and hosted token-only endpoint models (`provider="huggingface_endpoint"`). Hosted selections use `HuggingFaceEndpointEmbeddings` with `HUGGINGFACEHUB_API_TOKEN` and model IDs prefixed with `hf-endpoint:` so they do not collide with local model IDs.

**Ollama local/cloud support:** Ollama remains a single provider, but it now supports both local and cloud execution. If `OLLAMA_API_KEY` is present and no explicit `OLLAMA_BASE_URL` is supplied, the runtime and generated code default to `https://ollama.com`; otherwise they fall back to local `http://localhost:11434`. Both chat and embedding paths pass bearer auth headers when needed.

**Recovery path:** Embedding/vector-store/ingestion setup failures now show likely causes and offer retry, change embedding model, change vector DB settings, or abort. Runtime build failures are wrapped with a clearer production-facing message.

**Terminal feedback:** Query mode now shows a Rich status spinner while context retrieval and answer generation are running.

**Structured observability:** The CLI and query loop now emit structured JSON logs plus lightweight telemetry events for session start/load, selection steps, ingestion approval/completion, and query success/failure. This is intentionally small and local-first so it can later be swapped for a real backend.

**OpenTelemetry:** Optional OpenTelemetry tracing now wraps the main workflow phases and is enabled through a startup terminal prompt in the CLI. Env vars are still supported as a fallback for non-interactive runs, and the usual knobs are `OTEL_SERVICE_NAME`, `OTEL_ENVIRONMENT`, `OTEL_EXPORTER_OTLP_ENDPOINT`, and `OTEL_EXPORTER_OTLP_HEADERS`. It is safe to leave off in local runs.

**Smoke tests:** `tests/smoke/test_vector_db_smoke.py` contains opt-in live backend checks for Chroma, FAISS, and Qdrant. They only run when `MS_RAG_SMOKE_VECTOR_DBS` names the target backend and the required connection settings are present.

**Deployable docs:** `docs/` is a public, multi-page static, Vercel-ready documentation site for users. It covers the product pitch, setup, all RAG types, loaders/extractors, chunking, embeddings, vector databases, retrieval, reranking, compression, evaluation, observability, generated code, recommendations, production readiness, and docs deployment. Root `vercel.json` rewrites clean routes such as `/rag-types` and `/pipeline` to the matching docs pages and includes sitemap/robots routes.

**Verification run:** `.\.venv\Scripts\python.exe -m pytest tests\property\test_vectorization_properties.py tests\unit\test_vectordb_connector.py tests\property\test_query_loop_properties.py -q` passed with 66 tests.

### Task Checklist

- [x] Task 1 — Project scaffold + core data models (`ms_rag/models.py`, `pyproject.toml`, exceptions, validation)
- [x] Task 2 — Banner Display Module (`ms_rag/ui/banner.py`)
- [x] Task 3 — Credential Manager (`ms_rag/config/credential_manager.py`)
- [x] Task 4 — RAG Type Selector (`ms_rag/workflow/rag_type_selector.py`)
- [x] Task 5 — Document Type Selector (`ms_rag/ingestion/document_type_selector.py`)
- [x] Task 6 — Loader Selector (`ms_rag/ingestion/loader_selector.py`)
- [x] Task 7 — Chunking Engine (`ms_rag/ingestion/chunking_engine.py`)
- [x] Task 8 — Chunking Parameter Configuration UI (`ms_rag/workflow/chunking_configurator.py`)
- [x] Task 9 — Vectorization Module (`ms_rag/ingestion/vectorization_module.py`)
- [x] Task 10 — Vector DB Connector (`ms_rag/ingestion/vectordb_connector.py`)
- [x] Task 11 — Ingestion Orchestrator (`ms_rag/ingestion/ingestion_orchestrator.py`)
- [x] Task 12 — Checkpoint: all ingestion pipeline tests pass
- [x] Task 13 — Query Input Loop (`ms_rag/cli/query_loop.py`)
- [x] Task 14 — Query Enhancement Module (`ms_rag/query/query_enhancer.py`)
- [x] Task 15 — Retrieval Strategy Module (`ms_rag/query/retrieval_strategy.py`)
- [x] Task 16 — Reranking Module (`ms_rag/query/reranking_module.py`)
- [x] Task 17 — Context Compression Module (`ms_rag/query/context_compressor.py`)
- [x] Task 18 — System Prompt Configurator (`ms_rag/workflow/system_prompt_configurator.py`)
- [x] Task 19 — Evaluation Framework (`ms_rag/evaluation/evaluation_framework.py`)
- [x] Task 20 — Checkpoint: all query pipeline and evaluation tests pass
- [x] Task 21 — LLM Integration Layer (`ms_rag/llm/llm_integration.py`)
- [x] Task 22 — Code Generator (`ms_rag/codegen/code_generator.py`)
- [x] Task 23 — Session Manager (`ms_rag/session/session_manager.py`)
- [x] Task 24 — Error Handling and API Retry (`ms_rag/utils/retry.py`)

**Update this checklist by changing `[ ]` to `[x]` as tasks complete.**

---

## Project Directory Structure (to be created in Task 1)

```
RAG_framework/
├── AGENTS.md                          ← this file
├── vercel.json                         ← static docs deployment config
├── docs/                               ← deployable user documentation site
├── pyproject.toml
├── README.md
├── ms_rag/
│   ├── __init__.py
│   ├── models.py                      ← all shared dataclasses (PipelineConfig etc.)
│   ├── ui/
│   │   ├── __init__.py
│   │   └── banner.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── credential_manager.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── rag_type_selector.py
│   │   ├── chunking_configurator.py
│   │   └── system_prompt_configurator.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_type_selector.py
│   │   ├── loader_selector.py
│   │   ├── chunking_engine.py
│   │   ├── vectorization_module.py
│   │   ├── vectordb_connector.py
│   │   └── ingestion_orchestrator.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── query_loop.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── query_enhancer.py
│   │   ├── retrieval_strategy.py
│   │   ├── reranking_module.py
│   │   └── context_compressor.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_integration.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluation_framework.py
│   ├── codegen/
│   │   ├── __init__.py
│   │   ├── code_generator.py
│   │   └── templates/
│   ├── session/
│   │   ├── __init__.py
│   │   └── session_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── validation.py
│       ├── exceptions.py
│       └── retry.py
└── tests/
    ├── unit/
    ├── property/
    └── integration/
```

---

## Technology Stack (FIXED — do not change)

| Layer | Package | Notes |
|-------|---------|-------|
| Terminal UI | `rich>=13.0`, `questionary>=2.0` | |
| CLI framework | `click` or `typer` | for `--load` flag |
| LangChain | `langchain>=0.3`, `langchain-community>=0.3` | |
| LangGraph | `langgraph>=0.2` | agentic RAG types only |
| OpenAI | `langchain-openai` | |
| HuggingFace | `langchain-huggingface` | ⚠️ NOT langchain-community |
| Ollama | `langchain-ollama` | ⚠️ NOT langchain-community |
| Unstructured | `langchain-unstructured` | ⚠️ NOT langchain-community |
| Cohere | `langchain-cohere` | |
| Google | `langchain-google-genai` | |
| Mistral | `langchain-mistralai` | |
| Groq | `langchain-groq` | |
| Credential encryption | `cryptography>=41.0` | Fernet + PBKDF2 |
| Code generation | `jinja2>=3.1` | |
| Testing | `pytest>=8.0`, `hypothesis>=6.100` | |
| TruLens | `trulens-core`, `trulens-apps-langchain` | ⚠️ old `trulens_eval` is DEPRECATED |

---

## Critical Deprecation Warnings

These were discovered during a web search audit. **Do NOT use the deprecated versions.**

```
❌ from langchain_community.embeddings import HuggingFaceEmbeddings
✅ from langchain_huggingface import HuggingFaceEmbeddings
✅ from langchain_huggingface import HuggingFaceEndpointEmbeddings  # hosted, token-only

❌ from langchain_community.embeddings import OllamaEmbeddings
✅ from langchain_ollama import OllamaEmbeddings

❌ from langchain_community.llms import Ollama
✅ from langchain_ollama import OllamaLLM

❌ from trulens_eval import TruChain
✅ from trulens.apps.langchain import TruChain
   (install: trulens-core + trulens-apps-langchain + trulens-providers-openai)

❌ from ragas.metrics import faithfulness  (lowercase singleton — still works but old)
✅ from ragas.metrics import Faithfulness  (class-based, preferred for ragas>=0.2)

❌ LLMChain(llm=..., prompt=...)
✅ prompt | llm | StrOutputParser()  (LCEL)
```

---

## Key Data Models (defined in `ms_rag/models.py`)

```python
# Central accumulator — passed to Code Generator at the end
@dataclass
class PipelineConfig:
    schema_version: str = "1.0"
    configured_providers: list[str]
    rag_type: RAGTypeConfig | None
    document_types: list[str]
    loader_map: dict[str, str]          # {doc_type: loader_class_name}
    chunking: ChunkingConfig | None
    embedding_model: EmbeddingModelConfig | None
    vector_db: VectorDBConfig | None
    query_enhancement: list[str]
    retrieval: RetrievalConfig | None
    reranking: RerankingConfig | None
    reranking_enabled: bool = False
    compression: CompressionConfig | None
    compression_enabled: bool = False
    system_prompt: str = ""
    evaluation: EvaluationConfig | None
    evaluation_enabled: bool = False
    document_sources: list[str]

# Runtime container — never serialized
@dataclass
class SessionState:
    config: PipelineConfig
    credentials: CredentialStore        # never serialized, in-memory only
    vector_store: VectorStore | None
    retriever: BaseRetriever | None
    llm: BaseLLM | None
    rag_chain: RunnableSequence | CompiledGraph | None
```

---

## Coding Rules for This Project

1. **Python 3.11+** — use `TypedDict`, `dataclasses`, `match` statements freely
2. **No credentials in generated code** — always use `os.getenv("KEY_NAME")`
3. **All numeric inputs validated** via `validate_numeric()` in `ms_rag/utils/validation.py`
4. **All external API calls** wrapped in `retry_with_backoff()` from `ms_rag/utils/retry.py`
5. **LangChain LCEL** for standard RAG chains; **LangGraph StateGraph** only for agentic types
6. **Rich** for all terminal output; **questionary** for all interactive prompts
7. **Never embed credentials** in `PipelineConfig` JSON — only store env var names
8. **Property-based tests** with Hypothesis for all 29 correctness properties (see design.md)
9. **Generated code is standalone** — zero runtime dependency on `ms_rag` package itself
10. **Session Manager** handles `/save` (JSON) and `--load` CLI arg — `schema_version="1.0"`

---

## The 16-Step Workflow at a Glance

```
Step 1  → Banner (ASCII art "MS_RAG")
Optional → OpenTelemetry startup prompt (user can enable tracing for the session)
Step 2  → LLM Provider credentials (12 providers, encrypted persistence)
Step 3  → RAG type selection (15 types; 4 require LangGraph)
Step 4  → Document type selection (16+ types, multi-select)
Step 5  → Loader selection (filtered by doc types, credential-gated for paid loaders)
Step 6  → Chunking strategy (11 strategies)
Step 7  → Chunking parameters (chunk_size, overlap, separators, tokenizer)
Step 8  → Embedding model (20+ models, filtered by configured providers)
Step 9  → Vector DB selection + credentials + connection test + LIVE INGESTION
Step 10 → Query enhancement (7 techniques, HyDE needs LLM selection)
Step 11 → Retrieval strategy (10 strategies including TF-IDF separate from BM25)
Step 12 → Reranking (6 rerankers, local model prompt for cross-encoder/BGE/ColBERT)
Step 13 → Context compression (6 techniques, ordered, LLM-dependency check)
Step 14 → System prompt (5 testable properties, 10k char limit for replace)
Step 15 → Evaluation (12 frameworks including TruLens, RAGAS, DeepEval, LangSmith)
Step 16 → Runtime build + LIVE query loop (/exit confirm, /config structured, /save, unknown cmd error)
→ Code Generator → pipeline.py + requirements.txt (standalone, no MS_RAG dependency)
```

---

## How to Hand Off To Another Agent

1. Point the agent at this file: `AGENTS.md` at project root
2. Tell them: "Read AGENTS.md and the spec files in `.kiro/specs/ms-rag/`. The spec is complete. Start implementing from the first unchecked task in the checklist."
3. After each completed task, update the `[ ]` → `[x]` checkbox in this file
4. If you complete a task partially, note it: `[~] Task 3 — Credential Manager (store/get done, encryption pending)`

---

## Last Updated

**Last Updated**: June 27, 2026  
**Completed by**: Kiro  
**Status**: ✅ FULLY IMPLEMENTED — 423 tests passing, all 24 tasks complete  
**Next action**: Install full dependencies (`pip install -e ".[dev,pinecone,qdrant,ragas,deepeval,langsmith]"`) and run `ms-rag`
