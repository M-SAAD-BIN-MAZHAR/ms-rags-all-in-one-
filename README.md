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
[![Tests: 358 passing](https://img.shields.io/badge/Tests-358%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code style: production](https://img.shields.io/badge/code%20style-production-blue)]()

</div>

---

## Overview

MS\_RAG is a **production-grade, terminal-based interactive CLI framework** that guides you step-by-step through configuring every layer of a RAG (Retrieval-Augmented Generation) system — from credential setup and document ingestion through retrieval, reranking, context compression, LLM integration, evaluation, and final code generation.

Inspired by OpenClaw's UX, MS\_RAG acts as a **live RAG workbench + code generator**:

- You configure interactively through 16 guided steps
- Your documents are ingested and indexed live during Step 9
- You query your pipeline in real-time from Step 10 onwards
- At the end, MS\_RAG generates a **standalone `pipeline.py`** you own completely — no runtime dependency on MS\_RAG

---

## Key Features

| Category | What's included |
|----------|----------------|
| **RAG Architectures** | 15 types — Naive, Advanced, Modular, Agentic, Self-RAG, CRAG, GraphRAG, HyDE, Multi-Query, RAG-Fusion, Step-Back, Parent-Child, Adaptive, Contextual Compression |
| **LLM Providers** | 12 providers — OpenAI, Anthropic, Cohere, HuggingFace, Google Gemini, Mistral AI, Groq, Together AI, Replicate, Azure OpenAI, AWS Bedrock, Ollama (local) |
| **Document Types** | 18 types — PDF, DOCX, CSV, Excel, PPTX, HTML, Markdown, JSON, XML, Web URLs, YouTube transcripts, images/OCR, source code, SQL, MongoDB, ePub, RTF, plain text |
| **Document Loaders** | 30+ LangChain loaders filtered by your document types |
| **Chunking Strategies** | 11 strategies — Recursive Character, Fixed Size, Semantic, Sentence, Paragraph, Token-based, Markdown/HTML/Code-aware, Agentic, Document-aware |
| **Embedding Models** | 22+ models — OpenAI, Cohere, HuggingFace (BGE, E5, Instructor, sentence-transformers), Google, Mistral, Ollama/local |
| **Vector Databases** | 12 databases — ChromaDB, Pinecone, Qdrant, Weaviate, FAISS, Milvus, Redis, PGVector, Elasticsearch, OpenSearch, Azure AI Search, MongoDB Atlas |
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
│ Step 2       │ LLM Provider Credentials (12 providers)  │
│ Step 3       │ RAG Architecture Selection (15 types)    │
│ Step 4       │ Document Type Selection (18 types)       │
│ Step 5       │ Document Loader Selection (30+ loaders)  │
│ Steps 6–7    │ Chunking Strategy + Parameters           │
│ Step 8       │ Embedding Model Selection (22+ models)   │
│ Step 9       │ Vector DB + LIVE Ingestion (12 databases)│
├──────────────┴──────────────────────────────────────────┤
│              LIVE QUERY LOOP                             │
├──────────────┬──────────────────────────────────────────┤
│ Step 10      │ Interactive Query Input                  │
│ Step 11      │ Query Enhancement (7 techniques)         │
│ Step 12      │ Retrieval Strategy (10 strategies)       │
│ Step 13      │ Reranking (6 rerankers)                  │
│ Step 14      │ Context Compression (6 techniques)       │
│ Step 15      │ System Prompt Configuration              │
│ Step 16      │ Evaluation Frameworks (12 frameworks)    │
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
- At least one LLM provider API key (e.g. OpenAI) **or** [Ollama](https://ollama.ai) running locally

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/M-SAAD-BIN-MAZHAR/MS-RAG-ALL-IN-ONE-.git
cd MS-RAG-ALL-IN-ONE-

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 4. Install core dependencies
pip install -e .

# 5. (Optional) Install provider-specific extras
pip install -e ".[pinecone]"       # Pinecone vector DB
pip install -e ".[qdrant]"         # Qdrant vector DB
pip install -e ".[ragas]"          # RAGAS evaluation
pip install -e ".[deepeval]"       # DeepEval evaluation
pip install -e ".[langsmith]"      # LangSmith tracing
pip install -e ".[rerankers]"      # FlashRank + Cohere rerankers

# 6. Install dev/test dependencies
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# Interactive mode — starts the 16-step workflow
ms-rag

# Or using Python directly
python -m ms_rag

# Resume a previously saved session (skips to query loop)
ms-rag --load session.json
```

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
 12. Ollama / Local (OLLAMA_BASE_URL, OLLAMA_MODEL_NAME)
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

**Step 9 — Live Ingestion**
After selecting your vector DB and entering credentials, MS\_RAG runs a connection test, then ingests your documents with a real-time progress bar.

**Step 10 — Query Loop**
Type natural language questions. Available commands:

| Command | Action |
|---------|--------|
| `/config` | Display full pipeline configuration summary |
| `/save` | Save session to JSON for later resumption |
| `/exit` or `/quit` | Exit with confirmation prompt |

---

## Session Save & Load

Save your current session at any point:
```
Query > /save
Save config to file path: my_session.json
✓ Session saved to my_session.json
```

Resume it later (skips all setup steps):
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

# Vector Database
PINECONE_API_KEY=...
QDRANT_URL=http://localhost:6333

# Evaluation
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ms_rag_pipeline
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
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
│   │   └── evaluation_framework.py   # 12 evaluation frameworks
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
│   │   └── banner.py                 # ASCII banner
│   ├── utils/
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
| LangChain | `langchain>=0.3`, `langchain-core`, `langchain-community` | Loaders, splitters, chains, retrievers |
| LangGraph | `langgraph>=0.2` | Agentic RAG: Self-RAG, CRAG, Adaptive RAG |
| HuggingFace | `langchain-huggingface>=0.1` | ⚠️ NOT deprecated `langchain-community` |
| Ollama | `langchain-ollama>=0.2` | ⚠️ NOT deprecated `langchain-community` |
| TruLens | `trulens-core`, `trulens-apps-langchain` | ⚠️ NOT deprecated `trulens_eval` |
| Credential encryption | `cryptography>=41.0` | Fernet symmetric encryption + PBKDF2 |
| Testing | `pytest>=8.0`, `hypothesis>=6.100` | 358 tests, property-based |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Property-based tests only
pytest tests/property/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=ms_rag --cov-report=html
```

**Expected result: 358 passed, 1 skipped, 0 failed**

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
from langchain_ollama import ChatOllama
from trulens.apps.langchain import TruChain
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `pytest tests/ -v`
4. Commit your changes: `git commit -m "feat: add my feature"`
5. Push and open a Pull Request

Please ensure all 358 tests still pass before submitting.

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
