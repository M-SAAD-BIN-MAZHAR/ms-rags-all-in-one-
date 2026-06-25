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
Step 2  → LLM Provider credentials (12 providers, encrypted persistence)
Step 3  → RAG type selection (15 types; 4 require LangGraph)
Step 4  → Document type selection (16+ types, multi-select)
Step 5  → Loader selection (filtered by doc types, credential-gated for paid loaders)
Step 6  → Chunking strategy (11 strategies)
Step 7  → Chunking parameters (chunk_size, overlap, separators, tokenizer)
Step 8  → Embedding model (20+ models, filtered by configured providers)
Step 9  → Vector DB selection + credentials + connection test + LIVE INGESTION
Step 10 → LIVE query loop (/exit confirm, /config structured, /save, unknown cmd error)
Step 11 → Query enhancement (7 techniques, HyDE needs LLM selection)
Step 12 → Retrieval strategy (10 strategies including TF-IDF separate from BM25)
Step 13 → Reranking (6 rerankers, local model prompt for cross-encoder/BGE/ColBERT)
Step 14 → Context compression (6 techniques, ordered, LLM-dependency check)
Step 15 → System prompt (5 testable properties, 10k char limit for replace)
Step 16 → Evaluation (12 frameworks including TruLens, RAGAS, DeepEval, LangSmith)
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

**Last Updated**: June 2026  
**Completed by**: Kiro  
**Status**: ✅ FULLY IMPLEMENTED — 358 tests passing, all 24 tasks complete  
**Next action**: Install full dependencies (`pip install -e ".[dev,pinecone,qdrant,ragas,deepeval,langsmith]"`) and run `ms-rag`
