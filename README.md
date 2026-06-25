# MS_RAG — Production-Grade RAG Framework Builder

```
███╗   ███╗███████╗      ██████╗  █████╗  ██████╗
████╗ ████║██╔════╝      ██╔══██╗██╔══██╗██╔════╝
██╔████╔██║███████╗      ██████╔╝███████║██║  ███╗
██║╚██╔╝██║╚════██║      ██╔══██╗██╔══██║██║   ██║
██║ ╚═╝ ██║███████║      ██║  ██║██║  ██║╚██████╔╝
╚═╝     ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
                      MS_RAG
```

**Production-Grade RAG Framework Builder** — An OpenClaw-inspired terminal CLI that guides you step-by-step through building a complete RAG pipeline, then generates a standalone, deployable Python script.

---

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install
pip install -e .

# Run interactively
ms-rag

# Resume a saved session
ms-rag --load session.json
```

---

## What MS_RAG Does

MS_RAG is a **live RAG workbench + code generator**:

| Phase | Steps | What happens |
|-------|-------|--------------|
| Configuration | 1–9 | Sequential interactive setup (only Step 9 ingestion runs live) |
| Live querying | 10–16 | Configure AND run simultaneously — query answering is live |
| Code generation | Final | Generates a standalone `pipeline.py` + `requirements.txt` |

---

## 16-Step Interactive Workflow

| Step | What you configure |
|------|--------------------|
| 1 | ASCII banner display |
| 2 | LLM provider credentials (12 providers: OpenAI, Anthropic, Cohere, HuggingFace, Google, Mistral, Groq, Together AI, Replicate, Azure OpenAI, AWS Bedrock, Ollama) |
| 3 | RAG architecture (15 types: Naive, Self-RAG, CRAG, GraphRAG, HyDE, Multi-Query, RAG-Fusion, etc.) |
| 4 | Document types (18 types: PDF, DOCX, CSV, HTML, Markdown, JSON, YouTube, images/OCR, code, SQL, MongoDB, etc.) |
| 5 | Document loaders (30+ loaders filtered by your doc types; paid loaders credential-gated) |
| 6–7 | Chunking strategy (11 strategies) + parameters |
| 8 | Embedding model (22 models across all providers + local) |
| 9 | Vector database (12 DBs: Chroma, Pinecone, Qdrant, Weaviate, FAISS, Milvus, Redis, PGVector, etc.) + live ingestion |
| 10 | Query input loop (/exit, /quit, /config, /save) |
| 11 | Query enhancement (7 techniques: HyDE, Multi-Query, RAG-Fusion, Step-Back, etc.) |
| 12 | Retrieval strategy (10 strategies: Dense Vector, BM25, TF-IDF, Hybrid, MMR, Ensemble, Self-Query, etc.) |
| 13 | Reranking (6 rerankers: Cross-Encoder, Cohere, BGE, LLM, ColBERT, FlashRank) |
| 14 | Context compression (6 techniques: LLM Extraction, Embeddings Filter, Redundancy Removal, etc.) |
| 15 | System prompt (production-grade default + inline edit or replace) |
| 16 | Evaluation (12 frameworks: RAGAS, DeepEval, TruLens, LangSmith, Langfuse, Arize Phoenix, ARES, etc.) |

---

## Project Structure

```
ms_rag/
├── ui/banner.py                  # ASCII banner + Rich display
├── config/credential_manager.py # 12 LLM providers, Fernet encryption
├── workflow/
│   ├── rag_type_selector.py      # 15 RAG architecture variants
│   ├── chunking_configurator.py  # Step 6-7 UI
│   └── system_prompt_configurator.py
├── ingestion/
│   ├── document_type_selector.py # 18 document types
│   ├── loader_selector.py        # 30+ loaders, credential gating
│   ├── chunking_engine.py        # 11 chunking strategies
│   ├── vectorization_module.py   # 22 embedding models
│   ├── vectordb_connector.py     # 12 vector databases
│   └── ingestion_orchestrator.py # Full load→chunk→embed→store pipeline
├── cli/
│   ├── main.py                   # Full 16-step wiring
│   └── query_loop.py             # Interactive query loop
├── query/
│   ├── query_enhancer.py         # 7 enhancement techniques
│   ├── retrieval_strategy.py     # 10 retrieval strategies
│   ├── reranking_module.py       # 6 rerankers
│   └── context_compressor.py    # 6 compression techniques
├── llm/llm_integration.py        # LLM factory, LCEL + LangGraph
├── evaluation/evaluation_framework.py  # 12 evaluators
├── codegen/code_generator.py     # Standalone pipeline.py generator
├── session/session_manager.py    # /save + --load
├── utils/
│   ├── validation.py             # Centralised range validation
│   ├── exceptions.py             # Custom exception hierarchy
│   └── retry.py                  # Exponential backoff + Retry/Skip/Abort
└── models.py                     # All shared dataclasses
```

---

## Technology Stack

| Component | Package |
|-----------|---------|
| Terminal UI | `rich`, `questionary` |
| CLI | `click` |
| LangChain | `langchain>=0.3`, `langchain-core`, `langchain-community` |
| LangGraph | `langgraph>=0.2` |
| HuggingFace | `langchain-huggingface` (NOT deprecated community) |
| Ollama | `langchain-ollama` (NOT deprecated community) |
| Credential encryption | `cryptography` (Fernet + PBKDF2) |
| Testing | `pytest`, `hypothesis` |

---

## Running Tests

```bash
# All tests
.venv/Scripts/python.exe -m pytest tests/ -v

# Property-based tests only (Hypothesis)
pytest tests/property/ -v

# Specific task tests
pytest tests/unit/test_banner.py -v
```

**Test count: 358 tests, all passing.**

---

## Generated Code

The final output is a self-contained `pipeline.py` that:
- Has zero runtime dependency on MS_RAG
- Uses `os.getenv()` for all credentials
- Has a `main()` entry point with `--ingest` and `--query` args
- Works with any of the 15 RAG architecture variants
- Includes an embedded `requirements.txt` comment block

---

## Agent Handoff

See `AGENTS.md` for complete handoff instructions for Codex, Claude Code, Cursor, or any other AI coding agent.
