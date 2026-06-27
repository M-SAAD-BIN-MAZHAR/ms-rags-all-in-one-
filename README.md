<div align="center">

# MS\_RAG — Production-Grade RAG Framework Builder

```
███╗   ███╗███████╗      ██████╗  █████╗  ██████╗
████╗ ████║██╔════╝      ██╔══██╗██╔══██╗██╔════╝
██╔████╔██║███████╗      ██████╔╝███████║██║  ███╗
██║╚██╔╝██║╚════██║      ██╔══██╗██╔══██║██║   ██║
██║ ╚═╝ ██║███████║      ██║  ██║██║  ██║╚██████╔╝
╚═╝     ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
                      MS_RAG
```

**The OpenClaw-inspired terminal CLI for building production RAG pipelines — end to end.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain 0.3+](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://langchain.com)
[![LangGraph 0.2+](https://img.shields.io/badge/LangGraph-0.2%2B-purple)](https://langchain-ai.github.io/langgraph/)
[![Tests: 423 passing](https://img.shields.io/badge/Tests-423%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code style: production](https://img.shields.io/badge/code%20style-production-blue)]()

</div>

---

## Overview

MS\_RAG is a **production-grade, terminal-based interactive CLI framework** that guides you step-by-step through configuring every layer of a RAG (Retrieval-Augmented Generation) system — from credential setup and document ingestion through retrieval, reranking, context compression, LLM integration, evaluation, and final code generation.

Inspired by OpenClaw's UX, MS\_RAG acts as a **live RAG workbench + code generator**:

- You configure interactively through 16 guided steps
- You can optionally enable OpenTelemetry tracing from the terminal at startup
- Your documents are ingested and indexed live during Step 9
- The live query loop starts after the full setup flow and runtime build complete
- At the end, MS\_RAG generates a **standalone `pipeline.py`** you own completely — no runtime dependency on MS\_RAG

---

## Key Features

| Category | What's included |
|----------|----------------|
| **RAG Architectures** | 15 types — Naive, Advanced, Modular, Agentic, Self-RAG, CRAG, GraphRAG, HyDE, Multi-Query, RAG-Fusion, Step-Back, Parent-Child, Adaptive, Contextual Compression |
| **LLM Providers** | 12 providers — OpenAI, Anthropic, Cohere, HuggingFace, Google Gemini, Mistral AI, Groq, Together AI, Replicate, Azure OpenAI, AWS Bedrock, Ollama (local or cloud) |
| **Document Types** | 18 types — PDF, DOCX, CSV, Excel, PPTX, HTML, Markdown, JSON, XML, Web URLs, YouTube transcripts, images/OCR, source code, SQL, MongoDB, ePub, RTF, plain text |
| **Document Loaders** | 30+ LangChain loaders filtered by your document types |
| **Chunking Strategies** | 11 strategies — Recursive Character, Fixed Size, Semantic, Sentence, Paragraph, Token-based, Markdown/HTML/Code-aware, Agentic, Document-aware |
| **Embedding Models** | 25+ models — OpenAI, Cohere, HuggingFace local/downloaded, HuggingFace hosted token-only, Google, Mistral, Ollama local or cloud |
| **Vector Databases** | 12 databases — ChromaDB, Pinecone, Qdrant, Weaviate, FAISS, Milvus, Redis (`langchain-redis`), PGVector, Elasticsearch, OpenSearch, Azure AI Search, MongoDB Atlas |
| **Query Enhancement** | 7 techniques — Query Rewriting, Query Expansion, HyDE, Multi-Query, Step-Back, Sub-question Decomposition, RAG-Fusion |
| **Retrieval Strategies** | 10 strategies — Dense Vector, BM25, TF-IDF, Hybrid, MMR, Ensemble, Parent-Child, Multi-Vector, Self-Query, Time-weighted |
| **Rerankers** | 6 — Cross-Encoder, Cohere, BGE, LLM-based, ColBERT, FlashRank |
| **Context Compression** | 6 techniques — LLM Chain Extraction, Embeddings Filter, Redundancy Removal, Document Compressor Pipeline, Contextual Compression, Summary Compression |
| **Evaluation Frameworks** | 12 — RAGAS, DeepEval, TruLens, LangSmith, Langfuse, Arize Phoenix, ARES, RAGBench, RAGEval, CI/CD Gate, LangGraph Trace, Monitoring Export |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MS_RAG CLI                          │
├──────────────┬──────────────────────────────────────────┤
│ Step 1       │ ASCII Banner (MS_RAG)                    │
│ Optional     │ OpenTelemetry tracing prompt             │
│ Step 2       │ LLM Provider Credentials (12 providers)  │
│ Step 3       │ RAG Architecture Selection (15 types)    │
│ Step 4       │ Document Type Selection (18 types)       │
│ Step 5       │ Document Loader Selection (30+ loaders)  │
│ Steps 6–7    │ Chunking Strategy + Parameters           │
│ Step 8       │ Embedding Model Selection (22+ models)   │
│ Step 9       │ Vector DB + LIVE Ingestion (12 databases)│
│ Step 10      │ Query Enhancement (7 techniques)         │
│ Step 11      │ Retrieval Strategy (10 strategies)       │
│ Step 12      │ Reranking (6 rerankers)                  │
│ Step 13      │ Context Compression (6 techniques)       │
│ Step 14      │ System Prompt Configuration              │
│ Step 15      │ Evaluation Frameworks (12 frameworks)    │
│ Step 16      │ Runtime build + interactive query loop   │
├──────────────┴──────────────────────────────────────────┤
│              CODE GENERATOR                              │
│         pipeline.py + requirements.txt                   │
│         (standalone — no MS_RAG dependency)             │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.11+**
- **Git**
- At least one LLM provider API key (e.g. OpenAI) **or** [Ollama](https://ollama.ai) running locally or via Ollama Cloud credentials

---

## Installation

### One-Command Production Install (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/M-SAAD-BIN-MAZHAR/MS-RAG-ALL-IN-ONE-.git
cd MS-RAG-ALL-IN-ONE-

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install EVERYTHING — core + all vector DBs + all evaluators + rerankers
pip install -e ".[production]"

# 4. Run
ms-rag
```

That's it. **No manual extra installs needed.** The `[production]` extra includes:
- All 12 LLM providers
- All 12 vector databases (Redis via `langchain-redis`, not deprecated `langchain-community`)
- All 12 evaluation frameworks with **live runtime scoring** (RAGAS, DeepEval, TruLens, LangSmith, Langfuse, etc.)
- All rerankers (FlashRank, Cohere)
- All document loaders (PDF, DOCX, CSV, HTML, YouTube, images, etc.)
- Aligned `grpcio` pins for Weaviate compatibility (`grpcio>=1.59.5,<1.80.0`)

### Minimal Install (core only)

```bash
pip install -e .
```

### Install specific extras only

```bash
pip install -e ".[pinecone,qdrant,ragas,langsmith,rerankers]"
```

---

## Docker Usage

MS\_RAG can also run inside a Docker container. This is useful when users want a
repeatable CLI environment without installing Python dependencies directly on
their machine.

### Build The Image

The default image installs the full `production` extra, including provider
packages, vector database integrations, rerankers, evaluators, OpenTelemetry,
document loaders, OCR/PDF tooling, and local embedding support.

```bash
docker build -t ms-rag:1.0.0 .
```

To build a smaller image with only the core dependencies:

```bash
docker build --build-arg INSTALL_EXTRAS= -t ms-rag:core .
```

For a custom feature set, pass any optional-extra group from `pyproject.toml`:

```bash
docker build --build-arg INSTALL_EXTRAS=pinecone,qdrant,ragas,rerankers,telemetry -t ms-rag:custom .
```

### Run Interactively

Mount a local workspace so documents, saved sessions, FAISS indexes, Chroma data,
generated pipelines, and other outputs stay on your machine instead of inside the
temporary container filesystem.

```bash
docker run --rm -it \
  --env-file .env \
  -v "%cd%:/workspace" \
  ms-rag:1.0.0
```

Linux/macOS:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/workspace" \
  ms-rag:1.0.0
```

Windows PowerShell:

```powershell
docker run --rm -it `
  --env-file .env `
  -v "${PWD}:/workspace" `
  ms-rag:1.0.0
```

### Resume A Saved Session

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/workspace" \
  ms-rag:1.0.0 --load session.json
```

### Docker Notes

- Do not bake API keys into the image. Use `--env-file .env`, Docker secrets, or
  your deployment platform's secret manager.
- The image runs as a non-root `msrag` user.
- `/workspace` is the working directory for user documents and generated output.
- Local HuggingFace and sentence-transformer caches are stored under
  `/workspace/.cache` when mounted.
- Ollama local on the host may need a reachable base URL from inside Docker, for
  example `OLLAMA_BASE_URL=http://host.docker.internal:11434` on Docker Desktop.

---

## Quick Start

```bash
# Interactive mode — starts the guided workflow
ms-rag

# Or using Python directly
python -m ms_rag

# Resume a previously saved session (re-prompts credentials, rebuilds runtime)
ms-rag --load session.json
```

Saved sessions rebuild the live vector store connection, retriever stack, LLM, and RAG chain via `rebuild_session_runtime()`. Your vector DB data must still exist on disk or at the configured endpoint.

At startup, the CLI also asks whether you want to enable OpenTelemetry tracing for that session. If you decline, the framework continues with normal structured logging only.

---

## Deployable Documentation

This repository includes a public, multi-page static documentation site in `docs/`.
It is designed for Vercel and covers the MS_RAG value proposition, setup, every
RAG type, loaders/extractors, chunking strategies, embedding choices, vector
databases, retrieval, reranking, compression, evaluation, observability,
generated code, production recommendations, and deployment notes.

```bash
# Local preview from the repo root
python -m http.server 3000
```

Then open `http://localhost:3000/docs/index.html`. The root `vercel.json` rewrites
Vercel traffic to the matching static docs pages, clean URLs, sitemap, robots file,
and assets, so importing this repository into Vercel with the `Other` preset is
enough to deploy the docs.

---

## Usage Guide

### Step-by-Step Workflow

When you run `ms-rag`, you will be guided through these steps:

**Step 2 — LLM Provider Credentials**
Select one or more providers and enter your API keys. Keys are stored in memory and optionally encrypted to disk for reuse.

```
Supported providers:
  1. OpenAI (OPENAI_API_KEY, OPENAI_ORG_ID)
  2. Anthropic (ANTHROPIC_API_KEY)
  3. Cohere (COHERE_API_KEY)
  4. HuggingFace Inference API (HUGGINGFACEHUB_API_TOKEN)
  5. Google Gemini (GOOGLE_API_KEY)
  6. Mistral AI (MISTRAL_API_KEY)
  7. Together AI (TOGETHER_API_KEY)
  8. Groq (GROQ_API_KEY)
  9. Replicate (REPLICATE_API_TOKEN)
 10. Azure OpenAI (AZURE_OPENAI_API_KEY, ENDPOINT, API_VERSION)
 11. AWS Bedrock (AWS_ACCESS_KEY_ID, SECRET_ACCESS_KEY, REGION)
 12. Ollama / Local or Cloud (OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, optional OLLAMA_API_KEY)
```

**Step 3 — RAG Architecture**
Choose your RAG variant. Types marked `[LangGraph]` use a StateGraph agentic loop.

```
 1. Naive RAG               9. HyDE RAG
 2. Advanced RAG           10. Multi-Query RAG
 3. Modular RAG            11. RAG-Fusion
 4. Agentic RAG [LangGraph]12. Step-Back RAG
 5. Self-RAG [LangGraph]   13. Parent-Child RAG
 6. Corrective RAG [LangGraph]14. Adaptive RAG [LangGraph]
 7. Speculative RAG        15. Contextual Compression RAG
 8. GraphRAG
```

**Startup Prompt — Optional Tracing**
Right after the banner, MS\_RAG can ask whether you want OpenTelemetry tracing for the current session. This is optional and does not affect normal usage if you decline it.

**Step 9 — Live Ingestion**
After selecting your vector DB and entering credentials, MS\_RAG runs a connection test, shows a final ingestion review, asks for confirmation, then ingests your documents with a real-time progress bar.

**Step 10–15 — Query Pipeline Configuration**
Query enhancement, retrieval, reranking, compression, system prompt, and evaluation are configured before the live runtime starts.

**Step 16 — Live Query Loop**
Once the runtime is built, you can type natural language questions. Available commands:

| Command | Action |
|---------|--------|
| `/config` | Display full pipeline configuration summary |
| `/save` | Save session to JSON for later resumption |
| `/help` | List available query-loop commands |
| `/exit` or `/quit` | Exit with confirmation prompt |

Empty Enter in the query loop re-prompts instead of exiting. Required workflow inputs (providers, document sources, vector DB connection, telemetry choice, etc.) loop until valid.

---

## Session Save & Load

Save your current session at any point:
```
Query > /save
Save config to file path: my_session.json
✓ Session saved to my_session.json
```

Resume it later (re-enters credentials and rebuilds the live pipeline):
```bash
ms-rag --load my_session.json
```

---

## Generated Code

At the end of the workflow, MS\_RAG generates a **complete, standalone Python pipeline**:

```bash
# Saved to your chosen directory:
./ms_rag_output/
├── pipeline.py        # Full RAG pipeline — no MS_RAG dependency
└── requirements.txt   # All required packages
```

The generated `pipeline.py` supports:
```bash
# Ingest documents
python pipeline.py --ingest --sources ./docs/ https://example.com

# Single query
python pipeline.py --query "What is retrieval-augmented generation?"

# Interactive loop
python pipeline.py
```

---

## Environment Variables

MS\_RAG reads credentials from environment variables. You can create a `.env` file:

```bash
# .env (never commit this file)

# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...

# Embedding / Local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3
OLLAMA_API_KEY=

# Vector Database
PINECONE_API_KEY=...
QDRANT_URL=http://localhost:6333

# Evaluation
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ms_rag_pipeline
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...

# Optional OpenTelemetry fallback for non-interactive runs
MS_RAG_OTEL_ENABLED=1
OTEL_SERVICE_NAME=ms-rag
OTEL_ENVIRONMENT=production
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-token
```

---

## Project Structure

```
MS-RAG-ALL-IN-ONE-/
├── ms_rag/
│   ├── cli/
│   │   ├── main.py                   # 16-step wiring + entry point
│   │   └── query_loop.py             # Interactive query loop
│   ├── config/
│   │   └── credential_manager.py     # 12 providers, Fernet encryption
│   ├── codegen/
│   │   └── code_generator.py         # pipeline.py + requirements.txt generator
│   ├── evaluation/
│   │   ├── evaluation_framework.py   # 12 evaluation frameworks + live scoring
│   │   └── evaluator_runners.py      # RAGAS, DeepEval, LangSmith, Langfuse, etc.
│   ├── ingestion/
│   │   ├── chunking_engine.py        # 11 chunking strategies
│   │   ├── document_type_selector.py # 18 document types
│   │   ├── ingestion_orchestrator.py # Full load→chunk→embed→store pipeline
│   │   ├── loader_selector.py        # 30+ loaders with credential gating
│   │   ├── vectordb_connector.py     # 12 vector databases
│   │   └── vectorization_module.py   # 22+ embedding models
│   ├── llm/
│   │   └── llm_integration.py        # LLM factory, LCEL chains, LangGraph
│   ├── models.py                     # All shared dataclasses
│   ├── query/
│   │   ├── context_compressor.py     # 6 compression techniques
│   │   ├── query_enhancer.py         # 7 enhancement techniques
│   │   ├── reranking_module.py       # 6 rerankers
│   │   └── retrieval_strategy.py     # 10 retrieval strategies
│   ├── session/
│   │   └── session_manager.py        # /save + --load
│   ├── ui/
│   │   ├── banner.py                 # ASCII banner
│   │   └── prompts.py                # Shared re-prompt helpers (required inputs)
│   ├── utils/
│   │   ├── credentials.py            # Shared credential + model resolution
│   │   ├── metadata.py               # ChromaDB metadata sanitization
│   │   ├── exceptions.py             # Custom exception hierarchy
│   │   ├── retry.py                  # Exponential backoff + Retry/Skip/Abort
│   │   └── validation.py             # Centralised range validation
│   └── workflow/
│       ├── chunking_configurator.py  # Chunking parameter UI
│       ├── rag_type_selector.py      # RAG architecture selector
│       └── system_prompt_configurator.py
├── tests/
│   ├── property/                     # Hypothesis property-based tests (29 properties)
│   └── unit/                         # Unit tests
├── .kiro/specs/ms-rag/
│   ├── requirements.md               # 20 requirements
│   ├── design.md                     # Architecture + data models
│   └── tasks.md                      # 24 implementation tasks
├── AGENTS.md                         # AI agent handoff document
├── pyproject.toml                    # Package + dependencies
└── README.md
```

---

## Technology Stack

| Layer | Package | Notes |
|-------|---------|-------|
| Terminal UI | `rich>=13.0`, `questionary>=2.0` | Panels, tables, progress bars, interactive prompts |
| CLI | `click>=8.1` | `--load` flag, help generation |
| LangChain | `langchain>=0.3`, `langchain-classic`, `langchain-core`, `langchain-community` | Loaders, splitters, chains, retrievers |
| LangGraph | `langgraph>=0.2` | Agentic RAG: Self-RAG, CRAG, Adaptive RAG |
| Redis vectors | `langchain-redis>=0.2` | ⚠️ NOT deprecated `langchain_community.vectorstores.Redis` |
| HuggingFace | `langchain-huggingface>=0.1` | Local `HuggingFaceEmbeddings` and hosted token-only `HuggingFaceEndpointEmbeddings`; ⚠️ NOT deprecated `langchain-community` |
| Ollama | `langchain-ollama>=0.2` | ⚠️ NOT deprecated `langchain-community` |
| TruLens | `trulens-core`, `trulens-apps-langchain` | ⚠️ NOT deprecated `trulens_eval` |
| gRPC | `grpcio>=1.59.5,<1.80.0` | Pinned for `weaviate-client` compatibility |
| Credential encryption | `cryptography>=41.0` | Fernet symmetric encryption + PBKDF2 |
| Testing | `pytest>=8.0`, `hypothesis>=6.100` | 390 tests, property-based |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Property-based tests only
pytest tests/property/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (includes rebuild_session_runtime)
pytest tests/integration/ -v

# LangChain import audit (AST-based, no false positives)
python scripts/audit_imports.py

# With coverage
pytest tests/ --cov=ms_rag --cov-report=html
```

---

## Evaluation Runtime

When evaluation is enabled in Step 15, metrics are computed **live after each query**:

| Evaluator | Runtime behaviour |
|-----------|-------------------|
| RAGAS / DeepEval / TruLens | Full framework when installed; lexical fallback otherwise |
| LangSmith / Langfuse | Logs trace/run when credentials are configured |
| ARES / RAGBench / RAGEval | Lexical grounding scores (context recall, faithfulness) |
| CI/CD Gate | Checks thresholds against aggregated metrics via `check_cicd_thresholds()` |
| LangGraph Trace | Appends to `MS_RAG_TRACE_LOG` (default: `./ms_rag_traces.jsonl`) |
| Monitoring Export | Appends metrics to `MS_RAG_METRICS_EXPORT` (default: `./ms_rag_metrics.jsonl`) |

---

## Deprecation Notes

These packages were audited and corrected from deprecated versions:

```python
# ❌ DEPRECATED — do NOT use
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from trulens_eval import TruChain

# ✅ CURRENT — what MS_RAG uses
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_ollama import ChatOllama
from langchain_redis import RedisVectorStore
from trulens.apps.langchain import TruChain
from langchain_classic.retrievers import EnsembleRetriever
```

HuggingFace embeddings are intentionally split into two choices in the terminal:
local models download/cache and run on the user's machine, while hosted HuggingFace
endpoint embeddings use `HUGGINGFACEHUB_API_TOKEN` and do not download the model locally.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `pytest tests/ -v`
4. Commit your changes: `git commit -m "feat: add my feature"`
5. Push and open a Pull Request

Please ensure all tests still pass before submitting:

```bash
pytest tests/ -v
python scripts/audit_imports.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**M. Saad Bin Mazhar**
GitHub: [@M-SAAD-BIN-MAZHAR](https://github.com/M-SAAD-BIN-MAZHAR)

---

<div align="center">

Built with ❤️ using LangChain, LangGraph, Rich, and Hypothesis

</div>
