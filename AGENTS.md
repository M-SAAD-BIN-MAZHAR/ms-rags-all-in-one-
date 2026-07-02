# MS-RAGS(ALL-IN-ONE) — Agent Handoff Document

This file is for AI coding agents and human maintainers picking up the project.
Read it before changing code, docs, or specs.

## Project Summary

**MS-RAGS(ALL-IN-ONE)** is a production-grade terminal CLI for building complete RAG
pipelines. It guides users through each configuration step and generates a standalone
Python project tailored to those choices.

It is not a library. It is a CLI workbench and code generator.

## Core Principles

1. Keep the user in control. Ask for confirmation before irreversible actions.
2. Never persist secrets directly in generated Python files or session JSON.
3. Prefer explicit user choices over silent fallbacks.
4. Keep the live CLI, generated code, docs, and tests in sync.
5. Do not break the terminal UX with hidden behavior or surprise defaults.

## Repository Structure

```text
RAG_framework/
├── AGENTS.md
├── README.md
├── vercel.json
├── docs/
├── docs-deploy/
├── ms_rag/
├── tests/
└── .kiro/specs/ms-rag/
```

## Runtime Flow

1. Banner and startup checks
2. LLM provider selection and credentials
3. RAG type selection
4. Document type selection
5. Loader selection
6. Chunking strategy and parameters
7. Embedding model selection
8. Vector DB selection, credentials, and connection test
9. Live ingestion confirmation
10. Query enhancement
11. Retrieval strategy
12. Reranking
13. Context compression
14. System prompt
15. Evaluation
16. Runtime build and live query loop
17. Code generation

Keep the prompt flow permission-first. If a feature needs credentials or external
services, the CLI should ask for them explicitly.

## Implementation Rules

- Python 3.11+
- Use `rich` for terminal output and `questionary` for prompts
- Use LangChain LCEL for standard RAG paths
- Use LangGraph only for agentic and graph-based workflows
- Do not write secrets into generated code
- Validate numeric inputs
- Wrap external API calls with retry logic
- Keep generated output standalone and runnable outside the repo
- Keep `docs/` and `docs-deploy/` aligned when docs change

## Public Documentation Notes

The public docs should cover:

- product overview
- setup
- RAG types
- loaders and extractors
- chunking
- embeddings
- vector databases
- retrieval
- reranking
- compression
- evaluation
- observability
- generated code
- production readiness
- deployment

## Working Guidance for Future Agents

1. Read the existing code before adding abstractions.
2. Preserve the project’s permission-first flow.
3. Update docs when user-visible behavior changes.
4. Keep generated code in parity with runtime behavior.
5. Prefer small, explicit fixes over broad rewrites.
6. Avoid introducing new defaults that bypass user choice.

## Last Updated

Keep this file short, stable, and public-safe. If you need detailed audit notes,
place them in a private changelog or internal notes instead of this handoff file.
