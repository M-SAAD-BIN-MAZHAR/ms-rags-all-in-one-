<div align="center">

# MS-RAGS(ALL-IN-ONE) — Production-Grade RAG Framework Builder

```
███╗   ███╗███████╗      ██████╗  █████╗  ██████╗ ███████╗
████╗ ████║██╔════╝      ██╔══██╗██╔══██╗██╔════╝ ██╔════╝
██╔████╔██║███████╗      ██████╔╝███████║██║  ███╗███████╗
██║╚██╔╝██║╚════██║      ██╔══██╗██╔══██║██║   ██║╚════██║
██║ ╚═╝ ██║███████║      ██║  ██║██║  ██║╚██████╔╝███████║
╚═╝     ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                      MS-RAGS(ALL-IN-ONE)
```

**The OpenClaw-inspired terminal CLI for building production RAG pipelines — end to end.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain 0.3+](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://langchain.com)
[![LangGraph 0.2+](https://img.shields.io/badge/LangGraph-0.2%2B-purple)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code style: production](https://img.shields.io/badge/code%20style-production-blue)]()

[Live Documentation](https://ms-rags-all-in-one.vercel.app/) · [GitHub Repository](https://github.com/M-SAAD-BIN-MAZHAR/ms-rags-all-in-one-)

Created by **Muhammad Saad bin Mazhar**.

</div>

---

## Overview

MS-RAGS(ALL-IN-ONE) is a **production-grade, terminal-based interactive CLI framework** that guides you step-by-step through configuring every layer of a RAG (Retrieval-Augmented Generation) system — from credential setup and document ingestion through retrieval, reranking, context compression, LLM integration, evaluation, and final code generation.

Inspired by OpenClaw's UX, MS-RAGS(ALL-IN-ONE) acts as a **live RAG workbench + code generator**:

- You configure interactively through 16 guided steps
- You can optionally enable OpenTelemetry tracing from the terminal at startup
- Your documents are ingested and indexed live during Step 9
- The live query loop starts after the full setup flow and runtime build complete
- At the end, MS-RAGS(ALL-IN-ONE) generates a **standalone `pipeline.py`** you own completely — no runtime dependency on MS-RAGS(ALL-IN-ONE)

---

## Key Features

| Category | What's included |
|----------|----------------|
| **RAG Architectures** | 15 types — Naive, Advanced, Modular, Agentic, Self-RAG, CRAG, GraphRAG, HyDE, Multi-Query, RAG-Fusion, Step-Back, Parent-Child, Adaptive, Contextual Compression |
| **LLM Providers** | 12 providers — OpenAI, Anthropic, Cohere, HuggingFace, Google Gemini, Mistral AI, Groq, Together AI, Replicate, Azure OpenAI, AWS Bedrock, Ollama (chat: local or cloud) |
| **Document Types** | 18 types — PDF, DOCX, CSV, Excel, PPTX, HTML, Markdown, JSON, XML, Web URLs, YouTube transcripts, images/OCR, source code, SQL, MongoDB, ePub, RTF, plain text |
| **Document Loaders** | 30+ LangChain loaders filtered by your document types |
| **Chunking Strategies** | 11 strategies — Recursive Character, Fixed Size, Semantic, Sentence, Paragraph, Token-based, Markdown/HTML/Code-aware, Agentic, Document-aware |
| **Embedding Models** | 25+ models — OpenAI, Cohere, HuggingFace local/downloaded, HuggingFace hosted token-only, Google, Mistral, Ollama local/self-hosted |
| **Vector Databases** | 12 databases — ChromaDB, Pinecone, Qdrant, Weaviate, FAISS, Milvus, Redis (`langchain-redis`), PGVector, Elasticsearch, OpenSearch, Azure AI Search, MongoDB Atlas |
| **Query Enhancement** | 7 techniques — Query Rewriting, Query Expansion, HyDE, Multi-Query, Step-Back, Sub-question Decomposition, RAG-Fusion |
| **Retrieval Strategies** | 10 strategies — Dense Vector, BM25, TF-IDF, Hybrid, MMR, Ensemble, Parent-Child, Multi-Vector, Self-Query, Time-weighted |
| **Rerankers** | 6 — Cross-Encoder, Cohere, BGE, LLM-based, ColBERT, FlashRank |
| **Context Compression** | 6 techniques — LLM Chain Extraction, Embeddings Filter, Redundancy Removal, Document Compressor Pipeline, Contextual Compression, Summary Compression |
| **Evaluation Options** | 12 — RAGAS, DeepEval, TruLens, LangSmith, Langfuse, Arize Phoenix, ARES, RAGEval, RAGBench, CI/CD Gate, LangGraph Trace, Monitoring Export |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MS-RAGS(ALL-IN-ONE) CLI                          │
├──────────────┬──────────────────────────────────────────┤
│ Step 1       │ ASCII Banner (MS-RAGS(ALL-IN-ONE))                    │
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
│         (standalone — no MS-RAGS(ALL-IN-ONE) dependency)             │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.11+**
- **Git**
- At least one LLM provider API key (e.g. OpenAI) **or** [Ollama](https://ollama.ai) running locally or via Ollama Cloud credentials for chat use

### External Tools Some Loaders Need

Python packages are installed with `pip install -e ".[production]"`, but a few document extractors also depend on system tools. MS-RAGS(ALL-IN-ONE) now checks the selected loaders and document sources before ingestion, shows an **External Tools Needed For Selected Loaders** table, and asks for permission before continuing when a required tool is missing.

| Tool | Needed when | Install note |
|------|-------------|--------------|
| **Poppler** (`pdfinfo`) | You choose `UnstructuredPDFLoader` for PDFs, especially scanned/image PDFs. Without it, page counting can fail before OCR/extraction starts. | Install Poppler and add its `bin` folder to `PATH`. |
| **Tesseract OCR** (`tesseract`) | You want local OCR for scanned PDFs or image files. | Install Tesseract and add it to `PATH`. This improves local OCR quality. |
| **Java** (`java`) | You choose `TabulaLoader` for PDF table extraction. | Install a JRE/JDK and make sure `java` is available in the terminal. |
| **Ghostscript** (`gs`/`gswin64c`) | You choose `CamelotLoader` for PDF table extraction, especially lattice/table-line mode. | Install Ghostscript and add it to `PATH`. |

If you do not want local PDF/OCR dependencies, choose **LlamaParseLoader** for complex PDFs and provide `LLAMA_CLOUD_API_KEY`. LlamaParse is cloud-based, so use it only when sending documents to a managed parser is acceptable for your data policy.

---

## Installation

### Public PyPI Install (Recommended)

```bash
pip install ms-rags-all-in-one==1.0.1
ms-rags
```

Use this when you want the published package without cloning the repository.

### Source Install For Development

```bash
# 1. Clone the repository
git clone https://github.com/M-SAAD-BIN-MAZHAR/ms-rags-all-in-one-.git
cd ms-rags-all-in-one-

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install the production feature set
pip install -e ".[production]"

# 4. Run
ms-rags
```

The source `[production]` extra includes:
- All 12 LLM providers
- ChromaDB, FAISS, Pinecone, Qdrant, Weaviate, Milvus, PGVector, Elasticsearch, OpenSearch, MongoDB Atlas, and AWS/Azure-compatible vector integrations
- All 11 evaluation options are explicit about their runtime mode: package-backed scoring when RAGAS/DeepEval are installed and compatible, visible lexical fallback metrics when optional evaluator packages are incompatible with the installed LangChain family, TruLens-prefixed groundedness scores, LangSmith/Langfuse tracing, Phoenix OpenInference export, ARES-compatible scores, RAGBench dataset-compatible scoring, CI/CD gates, LangGraph trace export, and monitoring export.
- All rerankers (FlashRank, Cohere)
- All document loaders (PDF, DOCX, CSV, Excel, PPTX, HTML, Markdown, JSON, XML, YouTube, images/OCR, ePub, RTF, code, SQL, MongoDB, etc.)
- Production parser extras include Unstructured support for PPTX, Markdown, image OCR, ePub, and RTF, plus `jq`, `msoffcrypto-tool`, `python-pptx`, and `pypandoc-binary`.
- Aligned `grpcio` pins for Weaviate compatibility (`grpcio>=1.59.5,<1.80.0`)

### Important Redis Install Note

Redis vector support is available, but it is **not included in `[production]`**
right now because the latest `langchain-redis==0.2.5` package pins
`hf-xet<1.2.1`, while the modern HuggingFace stack used by MS-RAGS(ALL-IN-ONE) requires
`hf-xet>=1.4.3`. Installing both in the same fresh environment causes pip to
fail with a dependency conflict.

If you need Redis vectors, install it intentionally in a Redis-focused
environment:

```bash
pip install -e ".[redis]"
```

### Important ARES Install Note

ARES-compatible scoring is available in the framework, but the external
`ares-ai` package is **not included in `[production]`** because current
`ares-ai` releases pin old OpenAI/Anthropic SDK versions that conflict with the
modern provider stack used by MS-RAGS(ALL-IN-ONE).

If you need the package-backed ARES path, install it intentionally in a separate
ARES-focused environment:

```bash
pip install -e ".[ares]"
```

For Docker:

```bash
docker build --build-arg USE_CONSTRAINTS=0 --build-arg INSTALL_EXTRAS=redis -t ms-rags-all-in-one:redis .
```

Use this Redis-specific path only when Redis is the vector DB you plan to test.
For most production users, prefer Pinecone, Qdrant, ChromaDB, FAISS, Weaviate,
Milvus, PGVector, Elasticsearch, OpenSearch, or MongoDB Atlas until the upstream
`langchain-redis` / `hf-xet` conflict is resolved.

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

MS-RAGS(ALL-IN-ONE) can also run inside a Docker container. This is useful when users want a
repeatable CLI environment without installing Python dependencies directly on
their machine.

### Public Docker Image

```bash
docker pull saad469/ms-rags-all-in-one
docker run --rm -it --env-file .env -v "${PWD}:/workspace" saad469/ms-rags-all-in-one
```

PowerShell:

```powershell
docker pull saad469/ms-rags-all-in-one
docker run --rm -it `
  --env-file .env `
  -v "${PWD}:/workspace" `
  saad469/ms-rags-all-in-one
```

Use the public image when you want the ready-to-run container. Build locally only if you need a custom dependency set.

### Which Docker Build Should I Use?

| Command | What it installs | Recommended for |
|---------|------------------|-----------------|
| `docker build -t ms-rags-all-in-one:1.0.1 .` | Core MS-RAGS(ALL-IN-ONE) package and required base dependencies | Fast local testing, basic CLI walkthroughs, lightweight images |
| `docker build --build-arg INSTALL_EXTRAS=production -t ms-rags-all-in-one:production .` | Core MS-RAGS(ALL-IN-ONE) plus production providers, vector DB clients except Redis, evaluators, rerankers, telemetry, and loader dependencies | Production-style usage and full feature testing |
| `docker build --build-arg INSTALL_EXTRAS=pinecone,qdrant,ragas,rerankers,telemetry -t ms-rags-all-in-one:custom .` | Only the optional extras you name | Smaller images for teams that support a fixed stack |
| `docker build --build-arg USE_CONSTRAINTS=0 --build-arg INSTALL_EXTRAS=redis -t ms-rags-all-in-one:redis .` | Redis vector support path | Redis-only testing while the upstream Redis/HuggingFace dependency conflict exists |

If you want **all dependencies**, use the production build argument:

```bash
docker build --build-arg INSTALL_EXTRAS=production -t ms-rags-all-in-one:production .
```

The plain `ms-rags-all-in-one:1.0.1` local image is intentionally smaller. It is valid, but it does
not install every optional integration.

Production Docker builds use `constraints-production.txt`. That file pins the
known-compatible dependency family from the tested MS-RAGS(ALL-IN-ONE) environment so pip
does not spend a long time trying incompatible dependency combinations.
It intentionally leaves platform-sensitive scientific wheels to pip so the same
file can resolve cleanly on Windows and Linux Docker builds.

### Build The Image

The default image installs the Core MS-RAGS(ALL-IN-ONE) package. This keeps Docker builds
faster and avoids pulling every optional evaluator/reranker package unless the
user asks for them.

```bash
docker build -t ms-rags-all-in-one:1.0.1 .
```

To build the full production image with every optional provider, vector
database except Redis, evaluator, reranker, OpenTelemetry exporter, document
loader, OCR/PDF tool, and local embedding dependency:

```bash
docker build --build-arg INSTALL_EXTRAS=production -t ms-rags-all-in-one:production .
```

For a custom feature set, pass any optional-extra group from `pyproject.toml`:

```bash
docker build --build-arg INSTALL_EXTRAS=pinecone,qdrant,ragas,rerankers,telemetry -t ms-rags-all-in-one:custom .
```

Redis is the exception. Use the Redis-specific Docker command from the
"Important Redis Install Note" section because the upstream `langchain-redis`
dependency currently conflicts with the modern HuggingFace dependency family.

If Docker fails while resolving or downloading from `files.pythonhosted.org`, it
is a Docker network/DNS issue rather than an MS-RAGS(ALL-IN-ONE) code error. Retry the build
after Docker networking is healthy, or pass your internal package index:

```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple -t ms-rags-all-in-one:1.0.1 .
```

If Docker fails with `resolution-too-deep`, make sure the build context includes
`constraints-production.txt` and rebuild without cache:

```bash
docker build --no-cache --build-arg INSTALL_EXTRAS=production -t ms-rags-all-in-one:production .
```

### Local Equivalent

If you are not using Docker and want to install every supported dependency into a
Python virtual environment, run:

```bash
pip install -e ".[production]"
```

For development and testing tools only:

```bash
pip install -e ".[dev]"
```

### Run Interactively

Mount a local workspace so documents, saved sessions, FAISS indexes, Chroma data,
generated pipelines, and other outputs stay on your machine instead of inside the
temporary container filesystem.

```bash
docker run --rm -it \
  --env-file .env \
  -v "%cd%:/workspace" \
  saad469/ms-rags-all-in-one
```

Linux/macOS:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/workspace" \
  saad469/ms-rags-all-in-one
```

Windows PowerShell:

```powershell
docker run --rm -it `
  --env-file .env `
  -v "${PWD}:/workspace" `
  saad469/ms-rags-all-in-one
```

### Resume A Saved Session

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/workspace" \
  saad469/ms-rags-all-in-one --load session.json
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
ms-rags

# Or using Python directly
python -m ms_rag

# Resume a previously saved session (re-prompts credentials, rebuilds runtime)
ms-rags --load session.json
```

Saved sessions rebuild the live vector store connection, retriever stack, LLM, and RAG chain via `rebuild_session_runtime()`. Your vector DB data must still exist on disk or at the configured endpoint.

At startup, the CLI also asks whether you want to enable OpenTelemetry tracing for that session. If you decline, the framework continues with normal structured logging only.

---

## Deployable Documentation

Public docs are live at: **https://ms-rags-all-in-one.vercel.app/**

This repository includes a public, multi-page static documentation site in `docs/`.
It is designed for Vercel and covers the MS-RAGS(ALL-IN-ONE) value proposition, setup, every
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

For release hardening and public-facing safety rules, see [SECURITY.md](/C:/Users/msaad/OneDrive/Documents/RAG_framework/SECURITY.md).

---

## Usage Guide

### Step-by-Step Workflow

When you run `ms-rags`, you will be guided through these steps:

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
 12. Ollama / Chat Local or Cloud (OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, optional OLLAMA_API_KEY)
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

MS-RAGS(ALL-IN-ONE) does not treat these as labels only. Preset RAG types lock the downstream steps they need: HyDE uses HyDE query generation, Multi-Query uses multi-query retrieval, RAG-Fusion uses reciprocal-rank fusion, Parent-Child uses parent-child retrieval state, Contextual Compression uses contextual compression, and GraphRAG builds a persistent knowledge graph plus hybrid evidence retrieval. Advanced RAG and Modular RAG intentionally leave the module choices open so expert users can compose their own pipeline.

**GraphRAG graph store choices**
When GraphRAG is selected, MS-RAGS(ALL-IN-ONE) asks where to store the extracted knowledge graph:

| Graph store | Use when | Credentials |
|-------------|----------|-------------|
| Local JSON | Local apps, Docker, portable deployments, CI smoke tests. | No credential; provide a mounted graph path for production. |
| Neo4j / Neo4j Aura | Managed production graph database. | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, optional `NEO4J_DATABASE`. |
| Kuzu | Embedded graph database for local/server deployments. | No cloud credential; provide `KUZU_DATABASE_PATH`. |

GraphRAG also asks for a query mode: `local` entity-neighborhood retrieval, `global` community-summary retrieval, or `hybrid` combining both.

**Step 3b — Agentic / Corrective Tools**
If you choose Agentic RAG, MS-RAGS(ALL-IN-ONE) asks whether you want to enable permission-gated tools. If you choose Corrective RAG, MS-RAGS(ALL-IN-ONE) can also ask for Web Search so CRAG has an approved fallback when retrieved context is not relevant. Every tool is opt-in and deny-by-default.

| Tool | What it does | Required user permission |
|------|--------------|--------------------------|
| Web Search | Searches the public web through Tavily or Brave Search. | Provider API key plus max-results limit. |
| Memory Systems | Stores short-term, long-term, semantic, episodic, and user profile memories. | Enabled memory types and a JSON memory path such as `./agent_memory/memory.json`. |
| URL Fetch / Web Page Reader | Reads a specific URL for the agent. | Allowed domains, timeout, and max page size. |
| File System Read | Reads selected local files. | Explicit file/folder allowlist and max file size. |
| Document Summarization | Summarizes large tool results before generation. | Uses the selected LLM; no separate credential. |
| API Request | Calls a safe external REST API. | Allowed base URLs, allowed methods, optional auth env var, and credential prompt. |

Agentic tool usage in the query loop is explicit:

```text
Summarize this page: https://docs.example.com/release-notes
Compare my local notes with retrieval context file:"C:\approved notes\notes.txt"
Check the customer API api GET https://api.example.com/v1/status
```

For deployment, mount/persist `./agent_memory/` if you enable persistent memory, provide the same API keys as environment variables or through the terminal prompts, and keep allowlists narrow. MS-RAGS(ALL-IN-ONE) will fail loudly if a selected tool cannot run because a credential, URL domain, file path, or API method was not approved.

**Startup Prompt — Optional Tracing**
Right after the banner, MS-RAGS(ALL-IN-ONE) can ask whether you want OpenTelemetry tracing for the current session. This is optional and does not affect normal usage if you decline it.

**Step 9 — Live Ingestion**
After selecting your vector DB and entering credentials, MS-RAGS(ALL-IN-ONE) runs a connection test, shows a final ingestion review, asks for confirmation, then ingests your documents with a real-time progress bar.

**Step 10–15 — Query Pipeline Configuration**
Query enhancement, retrieval, reranking, compression, system prompt, and evaluation are configured before the live runtime starts.

**Advanced retrieval guardrails**
When you choose Parent-Child, Multi-Vector, Time-Weighted, or an Ensemble containing them, MS-RAGS(ALL-IN-ONE) shows the required runtime state before continuing and asks for confirmation.

| Strategy | Required state/model | Production behavior |
|----------|----------------------|---------------------|
| Parent-Child | Parent documents + child chunk IDs | Retrieves precise child chunks, then returns larger parent context. Runtime build fails clearly if parent state is missing. |
| Multi-Vector | Selected embedding model + chunk documents + local FAISS representation index | Builds a separate in-memory representation index and returns original chunks. It does not write synthetic vectors into your production vector DB. |
| Time-Weighted | `ms_rag_ingested_at` timestamp metadata | Blends semantic relevance with recency. Runtime fails clearly if timestamp metadata is missing. |

For saved sessions, keep original document sources available so MS-RAGS(ALL-IN-ONE) can rebuild advanced retrieval state during `--load`.

**Persistent keyword stores for cloud vector DBs**
Cloud vector DBs such as Pinecone, Qdrant, Weaviate, Milvus, and MongoDB Atlas are primarily vector stores. When you select Hybrid, BM25, TF-IDF, or an Ensemble containing keyword retrieval, MS-RAGS(ALL-IN-ONE) now asks where to persist raw chunk text for keyword search:

| Keyword store | Use when |
|----------------|----------|
| SQLite | Single-server/local production, easiest default |
| PostgreSQL | Production app server with managed Postgres |
| Elasticsearch | Managed full-text search / enterprise hybrid search |
| OpenSearch | AWS/OpenSearch production search |
| Memory only | Development/testing only |

At runtime, the dense side can use Pinecone while the keyword side loads chunk text from the selected keyword store. If credentials are needed, MS-RAGS(ALL-IN-ONE) prompts for them, tests the connection, and stores only sanitized env-var markers in saved session JSON.

**Runtime notices and logs**
MS-RAGS(ALL-IN-ONE) renders fallback/degradation warnings as visible terminal notices and also emits structured JSON logs with fields such as `event`, `component`, `reason`, and `action`. If a feature has to degrade, the terminal tells you what happened and what to check instead of hiding it.

**Step 16 — Live Query Loop**
Once the runtime is built, you can type natural language questions. Available commands:

| Command | Action |
|---------|--------|
| `/config` | Display full pipeline configuration summary |
| `/settings` or `/edit` | Edit live query enhancement, reranking, or compression settings and rebuild runtime |
| `/save` | Save session to JSON for later resumption |
| `/help` | List available query-loop commands |
| `/exit` or `/quit` | Exit with confirmation prompt |

Empty Enter in the query loop re-prompts instead of exiting. Required workflow inputs (providers, document sources, vector DB connection, telemetry choice, etc.) loop until valid.

When query enhancement is enabled, the live query loop prints a **Query Enhancement Trace** showing the original query, generated/re-written query variants, and the exact query sent to retrieval. This keeps HyDE, multi-query, rewrite, and fusion behavior visible instead of hidden.

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
ms-rags --load my_session.json
```

---

## Generated Code

At the end of the workflow, MS-RAGS(ALL-IN-ONE) generates a **complete, standalone Python pipeline**:

```bash
# Saved to your chosen directory:
./ms_rag_output/
├── pipeline.py        # Full RAG pipeline — no MS-RAGS(ALL-IN-ONE) dependency
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

MS-RAGS(ALL-IN-ONE) reads credentials from environment variables. You can create a `.env` file:

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
OTEL_SERVICE_NAME=ms-rags-all-in-one
OTEL_ENVIRONMENT=production
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-token
```

---

## Project Structure

```
MS-RAGS-ALL-IN-ONE/
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
│   │   └── evaluator_runners.py      # RAGAS, DeepEval, LangSmith, Langfuse, RAGEval, etc.
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
| RAGAS / DeepEval | Package-backed LLM-judge scoring when installed and compatible. Step 15 asks for an OpenAI evaluator key and evaluator model such as `gpt-4o-mini`; lexical fallback is used with a visible warning if the evaluator API/package fails |
| TruLens | Modern TruLens package validation when compatible; TruLens-prefixed groundedness scores with visible fallback warning if its LangChain adapter is incompatible |
| LangSmith / Langfuse | Logs trace/run when credentials are configured |
| Arize Phoenix | OpenInference/Phoenix trace export when `PHOENIX_COLLECTOR_ENDPOINT` is configured; Phoenix-prefixed scores otherwise |
| ARES | ARES-compatible retrieval/generation scores; install the external `ares-ai` package only in a separate ARES-focused environment if you need its package-backed path |
| RAGBench | Uses Hugging Face `datasets` tooling when installed plus RAGBench-compatible single-query scores |
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

# ✅ CURRENT — What MS-RAGS(ALL-IN-ONE) uses
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
For local HuggingFace downloads, MS-RAGS disables the `hf-xet` transfer path by
default. If a download is interrupted and leaves a partial cache, the next
ingestion failure will show the exact model cache folder and ask permission before
cleaning only that model cache and retrying.

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
