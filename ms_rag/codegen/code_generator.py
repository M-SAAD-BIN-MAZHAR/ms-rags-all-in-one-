"""Code Generator for MS_RAG.

Renders a complete, standalone, deployable Python pipeline from PipelineConfig.

Requirement 17:
- Produce single self-contained Python file (17.1)
- Use LangChain for all pipeline components (17.2)
- Use LangGraph for agentic RAG types (17.3)
- Include imports, comments, os.getenv, main(), requirements.txt block (17.4)
- Include all configured components (17.5)
- Display in terminal + offer to save (17.6)
- Require confirmation before writing (17.7)
- Create output dir if not exists (17.8)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

try:
    import questionary
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Syntax = None  # type: ignore[assignment]

from ms_rag.models import GeneratedCode, PipelineConfig
from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES


# ---------------------------------------------------------------------------
# Requirements snippets — package names per component
# ---------------------------------------------------------------------------

_BASE_REQUIREMENTS = [
    "langchain>=0.3.0",
    "langchain-classic>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "python-dotenv>=1.0.0",
]

_PROVIDER_REQUIREMENTS: dict[str, list[str]] = {
    "openai":        ["langchain-openai>=0.2.0"],
    "anthropic":     ["langchain-anthropic>=0.3.0"],
    "cohere":        ["langchain-cohere>=0.3.0"],
    "huggingface":   ["langchain-huggingface>=0.1.0"],
    "huggingface_endpoint": ["langchain-huggingface>=0.1.0"],
    "google_gemini": ["langchain-google-genai>=2.0.0"],
    "mistral":       ["langchain-mistralai>=0.2.0"],
    "groq":          ["langchain-groq>=0.2.0"],
    "together_ai":   ["langchain-openai>=0.2.0"],
    "replicate":     ["replicate>=0.25.0"],
    "azure_openai":  ["langchain-openai>=0.2.0"],
    "aws_bedrock":   ["langchain-aws>=0.2.0", "boto3>=1.34.0"],
    "ollama":        ["langchain-ollama>=0.2.0"],
    "local":         ["langchain-huggingface>=0.1.0", "sentence-transformers>=3.0.0"],
}

_VECTOR_DB_REQUIREMENTS: dict[str, list[str]] = {
    "chroma":        ["langchain-chroma>=0.1.0", "chromadb>=0.5.0"],
    "faiss":         ["faiss-cpu>=1.8.0"],
    "pinecone":      ["langchain-pinecone>=0.2.0", "pinecone-client>=4.0.0"],
    "qdrant":        ["langchain-qdrant>=0.2.0", "qdrant-client>=1.9.0"],
    "weaviate":      ["langchain-weaviate>=0.0.3", "weaviate-client>=4.0.0"],
    "pgvector":      ["langchain-postgres>=0.0.9", "psycopg2-binary>=2.9.0"],
    "milvus":        ["langchain-milvus>=0.1.0", "pymilvus>=2.4.0"],
    "redis":         ["langchain-redis>=0.2.0", "redis>=5.0.0"],
    "elasticsearch": ["langchain-elasticsearch>=0.2.0", "elasticsearch>=8.0.0"],
    "opensearch":    ["opensearch-py>=2.4.0"],
    "azure_ai_search": ["azure-search-documents>=11.6.0"],
    "mongodb_atlas": ["langchain-mongodb>=0.2.0", "pymongo>=4.6.0"],
}

_EVALUATOR_REQUIREMENTS: dict[str, list[str]] = {
    "ragas":         ["ragas>=0.2.0"],
    "deepeval":      ["deepeval>=1.0.0"],
    "trulens":       ["trulens-core>=1.0.0", "trulens-apps-langchain>=1.0.0"],
    "langsmith":     ["langsmith>=0.1.0"],
    "langfuse":      ["langfuse>=3.0.0"],
    "arize_phoenix": ["arize-phoenix>=4.0.0", "openinference-instrumentation-langchain>=0.1.0"],
}

_GRAPH_STORE_REQUIREMENTS: dict[str, list[str]] = {
    "local_json": [],
    "neo4j": ["neo4j>=5.20.0"],
    "kuzu": ["kuzu>=0.9.0"],
}


# ---------------------------------------------------------------------------
# CodeGenerator
# ---------------------------------------------------------------------------


class CodeGenerator:
    """Renders a complete, deployable Python pipeline from PipelineConfig."""

    def generate(self, config: PipelineConfig) -> GeneratedCode:
        """Produce the full pipeline.py and requirements.txt from config."""
        requirements = self._collect_requirements(config)
        python_code = self._render_python(config, requirements)
        requirements_txt = "\n".join(sorted(set(requirements)))
        env_txt = self._render_env_file(config)

        return GeneratedCode(
            python_code=python_code,
            requirements_txt=requirements_txt,
            env_txt=env_txt,
            rag_type=config.rag_type.rag_type if config.rag_type else "naive_rag",
        )

    def save(self, code: GeneratedCode, output_dir: Path) -> None:
        """Write pipeline.py and requirements.txt to output_dir. Req 17.7-17.8."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = output_dir / "pipeline.py"
        requirements_path = output_dir / "requirements.txt"
        env_path = output_dir / ".env"

        pipeline_path.write_text(code.python_code, encoding="utf-8")
        requirements_path.write_text(code.requirements_txt, encoding="utf-8")
        env_path.write_text(code.env_txt, encoding="utf-8")

    def display_and_offer_save(
        self, code: GeneratedCode, console: object | None = None
    ) -> None:
        """Display code in terminal with syntax highlighting and offer to save."""
        c = console or Console()

        # Display with Rich syntax highlighting (Req 17.6)
        try:
            syntax = Syntax(code.python_code, "python", theme="monokai", line_numbers=True)
            c.print(syntax)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            # Fallback: plain text display (Req 17.6 — still offer save)
            c.print(code.python_code)  # type: ignore[union-attr]

        c.print(  # type: ignore[union-attr]
            f"\n[green]  ✓ Code generation complete![/green] "
            f"({len(code.python_code):,} chars | RAG type: {code.rag_type})\n"
        )

        # Offer to save (Req 17.7)
        wants_save: bool = questionary.confirm(
            "  Save generated code to disk?",
            default=True,
        ).ask()

        if not wants_save:
            return

        output_path: str = questionary.text(
            "  Output directory path:",
            default="./ms_rag_output",
        ).ask()

        if not output_path or not output_path.strip():
            c.print("[yellow]  Save cancelled.[/yellow]")  # type: ignore[union-attr]
            return

        # Require explicit confirmation before writing (Req 17.7)
        confirmed: bool = questionary.confirm(
            f"  Write pipeline.py, requirements.txt, and .env to {output_path.strip()}?",
            default=True,
        ).ask()

        if not confirmed:
            c.print("[yellow]  Save cancelled.[/yellow]")  # type: ignore[union-attr]
            return

        try:
            self.save(code, Path(output_path.strip()))
            c.print(  # type: ignore[union-attr]
                f"[green]  ✓ Files saved to {output_path.strip()}/[/green]\n"
                f"  [dim]• pipeline.py[/dim]\n"
                f"  [dim]• requirements.txt[/dim]\n"
                f"  [dim]• .env[/dim]"
            )
        except Exception as exc:  # noqa: BLE001
            c.print(f"[red]  ✗ Save failed: {exc}[/red]")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Requirements collection
    # ------------------------------------------------------------------

    def _collect_requirements(self, config: PipelineConfig) -> list[str]:
        reqs: set[str] = set(_BASE_REQUIREMENTS)

        # LLM providers
        for pid in config.configured_providers:
            reqs.update(_PROVIDER_REQUIREMENTS.get(pid, []))
        if config.llm_model:
            reqs.update(_PROVIDER_REQUIREMENTS.get(config.llm_model.provider, []))

        # Embedding model provider
        if config.embedding_model:
            reqs.update(_PROVIDER_REQUIREMENTS.get(config.embedding_model.provider, []))
            if config.embedding_model.provider in ("huggingface", "local"):
                reqs.add("sentence-transformers>=3.0.0")

        # Vector DB
        if config.vector_db:
            reqs.update(_VECTOR_DB_REQUIREMENTS.get(config.vector_db.db_type, []))

        if config.graph_store:
            reqs.update(_GRAPH_STORE_REQUIREMENTS.get(config.graph_store.store_type, []))

        # LangGraph for agentic types
        if config.rag_type and config.rag_type.requires_langgraph:
            reqs.add("langgraph>=0.2.0")

        # Evaluators
        if config.evaluation_enabled and config.evaluation:
            for eid in config.evaluation.evaluators:
                reqs.update(_EVALUATOR_REQUIREMENTS.get(eid, []))

        # Reranker
        if config.reranking_enabled and config.reranking:
            rr = config.reranking.reranker
            if rr in ("cross_encoder", "bge_reranker"):
                reqs.add("sentence-transformers>=3.0.0")
            elif rr == "colbert":
                # Late-interaction MaxSim uses a transformer token encoder.
                reqs.add("transformers>=4.40.0")
                reqs.add("torch>=2.0.0")
            elif rr == "cohere_reranker":
                reqs.add("cohere>=5.0.0")
                reqs.add("langchain-cohere>=0.3.0")
            elif rr == "flashrank":
                reqs.add("flashrank>=0.2.0")

        # Keyword / hybrid retrieval
        if config.retrieval:
            if config.retrieval.strategy in ("keyword_bm25", "hybrid", "ensemble"):
                reqs.add("rank-bm25>=0.2.2")
            if config.retrieval.strategy in ("tfidf", "ensemble"):
                reqs.add("scikit-learn>=1.4.0")
            retrieval_ids = [config.retrieval.strategy] + (config.retrieval.ensemble_sub_retrievers or [])
            if "multi_vector" in retrieval_ids:
                reqs.add("faiss-cpu>=1.8.0")

        # Context compression (LangChain 1.x retriever compressors live in langchain-classic)
        if config.compression_enabled and config.compression:
            reqs.add("langchain-classic>=0.3.0")

        return sorted(reqs)

    def _render_env_file(self, config: PipelineConfig) -> str:
        """Render a deployable .env template from the exact selected config."""
        from ms_rag.config.credential_manager import PROVIDER_FIELDS  # noqa: PLC0415
        from ms_rag.ingestion.loader_selector import LOADER_MAP  # noqa: PLC0415

        sections: list[tuple[str, list[str]]] = []

        def unique(items: list[str]) -> list[str]:
            return list(dict.fromkeys(item for item in items if item))

        provider_ids = unique(
            list(config.configured_providers or [])
            + ([config.llm_model.provider] if config.llm_model else [])
            + ([config.embedding_model.provider] if config.embedding_model else [])
        )
        provider_ids = ["huggingface" if item in {"huggingface_endpoint", "local"} else item for item in provider_ids]
        provider_fields: list[str] = []
        for provider_id in unique(provider_ids):
            provider_fields.extend(PROVIDER_FIELDS.get(provider_id, []))
        if provider_fields:
            sections.append(("LLM and embedding providers", unique(provider_fields)))

        loader_fields: list[str] = []
        for loader_class in (config.loader_map or {}).values():
            loader_info = LOADER_MAP.get(loader_class)
            if loader_info:
                loader_fields.extend(loader_info.credential_fields)
        if loader_fields:
            sections.append(("Document loaders", unique(loader_fields)))

        if config.vector_db:
            vector_fields = list((config.vector_db.connection_params or {}).keys())
            if config.vector_db.db_type == "faiss":
                vector_fields.append("FAISS_INDEX_PATH")
            sections.append((f"Vector database: {config.vector_db.db_type}", unique(vector_fields)))

        if config.keyword_store:
            keyword_fields = list((config.keyword_store.connection_params or {}).keys())
            sections.append((f"Keyword store: {config.keyword_store.store_type}", unique(keyword_fields)))

        if config.graph_store:
            graph_fields = list((config.graph_store.connection_params or {}).keys())
            if config.graph_store.store_type == "local_json":
                graph_fields.append("GRAPH_STORE_PATH")
            elif config.graph_store.store_type == "kuzu":
                graph_fields.append("KUZU_DATABASE_PATH")
            elif config.graph_store.store_type == "neo4j":
                graph_fields.extend(["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"])
            sections.append((f"Graph store: {config.graph_store.store_type}", unique(graph_fields)))

        if config.reranking_enabled and config.reranking and config.reranking.reranker == "cohere_reranker":
            sections.append(("Reranking", ["COHERE_API_KEY"]))

        evaluator_fields: list[str] = []
        if config.evaluation_enabled and config.evaluation:
            evaluator_ids = set(config.evaluation.evaluators or [])
            if evaluator_ids.intersection({"ragas", "deepeval"}):
                evaluator_fields.extend(["OPENAI_API_KEY", "OPENAI_ORG_ID"])
            if "langsmith" in evaluator_ids:
                evaluator_fields.extend(["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"])
            if "langfuse" in evaluator_ids:
                evaluator_fields.extend(["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"])
            if "arize_phoenix" in evaluator_ids:
                evaluator_fields.extend(["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT"])
            if evaluator_fields:
                sections.append(("Evaluation and monitoring", unique(evaluator_fields)))

        tool_fields: list[str] = []
        if config.agent_tools:
            settings = config.agent_tools.tool_settings or {}
            web_provider = str((settings.get("web_search") or {}).get("provider") or "")
            if "web_search" in (config.agent_tools.enabled_tools or []):
                if web_provider == "brave":
                    tool_fields.append("BRAVE_SEARCH_API_KEY")
                else:
                    tool_fields.append("TAVILY_API_KEY")
            api_auth = str((settings.get("api_request") or {}).get("auth_env_var") or "")
            if api_auth:
                tool_fields.append(api_auth)
            if "memory" in (config.agent_tools.enabled_tools or []):
                tool_fields.append("MS_RAG_AGENT_MEMORY_PATH")
                memory_settings = settings.get("memory") or {}
                connection_env = str(memory_settings.get("connection_env") or "")
                if connection_env:
                    tool_fields.append(connection_env)
            if tool_fields:
                sections.append(("Agentic tools", unique(tool_fields)))

        lines = [
            "# Generated by MS-RAGS(ALL-IN-ONE).",
            "# Fill only the variables required by your selected pipeline.",
            "# Secret values are intentionally not written by the generator.",
            "",
        ]
        seen: set[str] = set()
        for title, fields in sections:
            fields = [field for field in fields if field and field not in seen]
            if not fields:
                continue
            lines.append(f"# {title}")
            for field in fields:
                seen.add(field)
                lines.append(f"{field}=")
            lines.append("")
        if not seen:
            lines.append("# No external credentials are required for this selected pipeline.")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # Python code rendering
    # ------------------------------------------------------------------

    def _render_python(self, config: PipelineConfig, requirements: list[str]) -> str:
        rag_type = config.rag_type.rag_type if config.rag_type else "naive_rag"
        uses_langgraph = config.rag_type.requires_langgraph if config.rag_type else False

        sections = [
            self._render_header(rag_type, requirements),
            self._render_imports(config, uses_langgraph),
            self._render_credentials(config),
            self._render_system_prompt(config),
            self._render_loader_function(config),
            self._render_chunking_function(config),
            self._render_vector_store_function(config),
            self._render_retriever_function(config),
            self._render_agent_tool_helpers(config),
        ]

        if config.rag_type and config.rag_type.rag_type == "graphrag":
            sections.append(self._render_graphrag_functions(config))

        if config.reranking_enabled and config.reranking:
            sections.append(self._render_reranker_function(config))

        if config.compression_enabled and config.compression:
            sections.append(self._render_compressor_function(config))

        if uses_langgraph:
            sections.append(self._render_langgraph_workflow(config))
        else:
            sections.append(self._render_lcel_chain(config))

        if config.evaluation_enabled and config.evaluation:
            sections.append(self._render_evaluation_setup(config))

        sections.append(self._render_main(config))

        return "\n\n".join(s for s in sections if s.strip())

    def _render_header(self, rag_type: str, requirements: list[str]) -> str:
        req_block = "\n".join(f"# {r}" for r in requirements)
        return f'''#!/usr/bin/env python3
"""
MS-RAGS(ALL-IN-ONE) Generated Pipeline
Generated by: MS-RAGS(ALL-IN-ONE) Framework
RAG Type: {rag_type}

# ============================================================
# requirements.txt
{req_block}
# ============================================================
"""

import os
import argparse
import warnings
import json
import re
import math
import contextlib
import io
import requests
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

ADVANCED_PARENT_DOCUMENTS = {{}}
ADVANCED_CHUNK_DOCUMENTS = []
'''

    def _render_imports(self, config: PipelineConfig, uses_langgraph: bool) -> str:
        lines = [
            "# ─── LangChain imports ────────────────────────────────────────",
            "from langchain_core.prompts import ChatPromptTemplate",
            "from langchain_core.output_parsers import StrOutputParser",
            "from langchain_core.runnables import RunnableLambda, RunnablePassthrough",
            "from langchain_core.documents import Document",
        ]

        # Embedding imports
        if config.embedding_model:
            p = config.embedding_model.provider
            if p == "openai":
                lines.append("from langchain_openai import OpenAIEmbeddings")
            elif p == "cohere":
                lines.append("from langchain_cohere import CohereEmbeddings")
            elif p in ("huggingface", "local"):
                lines.append("from langchain_huggingface import HuggingFaceEmbeddings")
            elif p == "huggingface_endpoint":
                lines.append("from langchain_huggingface import HuggingFaceEndpointEmbeddings")
            elif p == "google_gemini":
                lines.append("from langchain_google_genai import GoogleGenerativeAIEmbeddings")
            elif p == "mistral":
                lines.append("from langchain_mistralai import MistralAIEmbeddings")
            elif p == "ollama":
                lines.append("from langchain_ollama import OllamaEmbeddings")

        # Vector store imports
        if config.vector_db:
            db = config.vector_db.db_type
            if db == "chroma":
                lines.append("from langchain_chroma import Chroma")
            elif db == "faiss":
                lines.append("from langchain_community.vectorstores import FAISS")
            elif db == "pinecone":
                lines.append("from langchain_pinecone import PineconeVectorStore")
            elif db == "qdrant":
                lines.append("from langchain_qdrant import QdrantVectorStore")
            elif db == "weaviate":
                lines.append("from langchain_weaviate import WeaviateVectorStore")
                lines.append("import weaviate")
            elif db == "pgvector":
                lines.append("from langchain_postgres import PGVector")
            elif db == "milvus":
                lines.append("from langchain_milvus import Milvus")
            elif db == "elasticsearch":
                lines.append("from langchain_elasticsearch import ElasticsearchStore")
            elif db == "redis":
                lines.append("from langchain_redis import RedisConfig, RedisVectorStore")
            elif db == "opensearch":
                lines.append("from langchain_community.vectorstores import OpenSearchVectorSearch")
            elif db == "azure_ai_search":
                lines.append("from langchain_community.vectorstores import AzureSearch")
            elif db == "mongodb_atlas":
                lines.append("from langchain_mongodb import MongoDBAtlasVectorSearch")
                lines.append("from pymongo import MongoClient")

        # LLM imports
        llm_providers = list(dict.fromkeys(
            list(config.configured_providers)
            + ([config.llm_model.provider] if config.llm_model else [])
        ))
        for pid in llm_providers:
            if pid == "openai":
                lines.append("from langchain_openai import ChatOpenAI")
            elif pid == "anthropic":
                lines.append("from langchain_anthropic import ChatAnthropic")
            elif pid == "cohere":
                lines.append("from langchain_cohere import ChatCohere")
            elif pid == "huggingface":
                lines.append("from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint")
            elif pid == "google_gemini":
                lines.append("from langchain_google_genai import ChatGoogleGenerativeAI")
            elif pid == "mistral":
                lines.append("from langchain_mistralai import ChatMistralAI")
            elif pid == "groq":
                lines.append("from langchain_groq import ChatGroq")
            elif pid == "together_ai":
                lines.append("from langchain_openai import ChatOpenAI")
            elif pid == "replicate":
                lines.append("from langchain_community.llms import Replicate")
            elif pid == "ollama":
                lines.append("from langchain_ollama import ChatOllama")
            elif pid == "azure_openai":
                lines.append("from langchain_openai import AzureChatOpenAI")
            elif pid == "aws_bedrock":
                lines.append("from langchain_aws import ChatBedrock")

        if uses_langgraph:
            lines.append("")
            lines.append("# ─── LangGraph imports ────────────────────────────────────")
            lines.append("from langgraph.graph import StateGraph, END")
            lines.append("from typing import TypedDict")

        return "\n".join(lines)

    def _render_credentials(self, config: PipelineConfig) -> str:
        lines = ["# ─── Credentials (via environment variables) ─────────────────"]
        embedding_provider = config.embedding_model.provider if config.embedding_model else ""
        if embedding_provider in {"huggingface_endpoint", "local"}:
            embedding_provider = "huggingface"
        credential_providers = list(dict.fromkeys(
            list(config.configured_providers)
            + ([config.llm_model.provider] if config.llm_model else [])
            + ([embedding_provider] if embedding_provider else [])
        ))
        for pid in credential_providers:
            from ms_rag.config.credential_manager import PROVIDER_FIELDS  # noqa: PLC0415
            for field in PROVIDER_FIELDS.get(pid, []):
                lines.append(f'{field} = os.getenv("{field}")')
        if config.embedding_model and config.embedding_model.provider == "ollama" and "ollama" not in config.configured_providers:
            from ms_rag.config.credential_manager import PROVIDER_FIELDS  # noqa: PLC0415
            for field in PROVIDER_FIELDS.get("ollama", []):
                if f'{field} = os.getenv("{field}")' not in lines:
                    lines.append(f'{field} = os.getenv("{field}")')

        uses_ollama = "ollama" in credential_providers or (
            config.embedding_model is not None and config.embedding_model.provider == "ollama"
        )
        if uses_ollama:
            lines.extend([
                "",
                "def _normalize_ollama_base_url(base_url: str) -> str:",
                '    normalized = (base_url or "").strip().rstrip("/")',
                '    if normalized.endswith("/v1"):',
                '        normalized = normalized[:-3].rstrip("/")',
                "    return normalized",
                "",
                "def _is_ollama_cloud_url(base_url: str) -> bool:",
                '    return base_url.startswith("https://ollama.com") or base_url.startswith("http://ollama.com")',
                "",
                "def _ollama_base_url(*, usage: str = 'chat'):",
                '    if OLLAMA_BASE_URL:',
                '        base_url = _normalize_ollama_base_url(OLLAMA_BASE_URL)',
                "    elif usage == 'chat' and OLLAMA_API_KEY:",
                '        base_url = "https://ollama.com"',
                "    else:",
                '        base_url = "http://localhost:11434"',
                "    if usage == 'embedding' and _is_ollama_cloud_url(base_url):",
                '        raise ValueError("Ollama Cloud currently supports chat models only. Use a local/self-hosted Ollama base URL for embedding models.")',
                "    return base_url",
                "",
                "def _ollama_client_kwargs(*, usage: str = 'chat'):",
                '    base_url = _ollama_base_url(usage=usage)',
                '    if OLLAMA_API_KEY and (usage == "chat" or not _is_ollama_cloud_url(base_url)):',
                '        return {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}}',
                "    return {}",
            ])
        uses_local_huggingface = (
            config.embedding_model is not None
            and config.embedding_model.provider in {"huggingface", "local"}
        )
        if uses_local_huggingface:
            lines.extend([
                "",
                "def _local_huggingface_embeddings(model_name: str):",
                '    os.environ["HF_HUB_DISABLE_XET"] = "1"',
                '    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"',
                '    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)',
                "    if HUGGINGFACEHUB_API_TOKEN:",
                '        os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN',
                '        os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGINGFACEHUB_API_TOKEN',
                "        return HuggingFaceEmbeddings(",
                "            model_name=model_name,",
                '            model_kwargs={"token": HUGGINGFACEHUB_API_TOKEN},',
                "        )",
                "    return HuggingFaceEmbeddings(model_name=model_name)",
            ])
        if config.vector_db and config.vector_db.connection_params:
            lines.append("")
            lines.append("# Vector DB credentials")
            for key in config.vector_db.connection_params:
                lines.append(f'{key} = os.getenv("{key}")')
        if config.graph_store:
            lines.append("")
            lines.append("# GraphRAG graph store credentials/config")
            for key in config.graph_store.connection_params:
                lines.append(f'{key} = os.getenv("{key}")')
            if config.graph_store.store_type == "neo4j":
                for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
                    line = f'{key} = os.getenv("{key}")'
                    if line not in lines:
                        lines.append(line)
            if config.graph_store.store_type == "local_json":
                if 'GRAPH_STORE_PATH = os.getenv("GRAPH_STORE_PATH")' not in lines:
                    lines.append('GRAPH_STORE_PATH = os.getenv("GRAPH_STORE_PATH")')
            if config.graph_store.store_type == "kuzu":
                if 'KUZU_DATABASE_PATH = os.getenv("KUZU_DATABASE_PATH")' not in lines:
                    lines.append('KUZU_DATABASE_PATH = os.getenv("KUZU_DATABASE_PATH")')
        return "\n".join(lines)

    def _render_system_prompt(self, config: PipelineConfig) -> str:
        prompt = config.system_prompt or "You are a helpful assistant."
        escaped = prompt.replace('"""', '\\"\\"\\"')
        return f'# ─── System Prompt ────────────────────────────────────────────\nSYSTEM_PROMPT = """{escaped}"""'

    def _render_loader_function(self, config: PipelineConfig) -> str:
        loader_map_literal = repr(dict(config.loader_map or {}))
        return f'''# ─── Document Loading ─────────────────────────────────────────
LOADER_BY_DOC_TYPE = {loader_map_literal}

def _infer_doc_type(source: str) -> str:
    lowered = str(source).lower()
    if lowered.startswith(("http://", "https://")):
        if "youtube.com" in lowered or "youtu.be" in lowered:
            return "youtube"
        return "url"
    suffix = Path(source).suffix.lower().lstrip(".")
    return {{
        "md": "markdown",
        "markdown": "markdown",
        "jpg": "image_ocr",
        "jpeg": "image_ocr",
        "png": "image_ocr",
        "tif": "image_ocr",
        "tiff": "image_ocr",
        "doc": "docx",
        "docx": "docx",
        "ppt": "pptx",
        "pptx": "pptx",
        "xls": "xlsx",
        "xlsx": "xlsx",
        "htm": "html",
        "html": "html",
        "txt": "txt",
        "py": "code",
        "js": "code",
        "ts": "code",
        "java": "code",
        "cpp": "code",
        "c": "code",
        "json": "json",
        "xml": "xml",
        "csv": "csv",
        "pdf": "pdf",
        "epub": "epub",
        "rtf": "rtf",
    }}.get(suffix, suffix or "txt")

def _extract_youtube_id(url: str) -> str:
    for pattern in [r"youtube\\.com/watch\\?v=([^&]+)", r"youtu\\.be/([^?]+)", r"youtube\\.com/embed/([^?]+)"]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url

def _invoke_loader(loader_class_name: str, source: str) -> list:
    if loader_class_name == "PyPDFLoader":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(source).load()
    if loader_class_name == "UnstructuredPDFLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "PDFPlumberLoader":
        from langchain_community.document_loaders import PDFPlumberLoader
        return PDFPlumberLoader(source).load()
    if loader_class_name == "CamelotLoader":
        try:
            import camelot
        except ImportError as exc:
            raise ImportError("CamelotLoader requires camelot-py. Install it or choose PyPDFLoader/PDFPlumberLoader.") from exc
        tables = camelot.read_pdf(source, pages="all")
        docs = []
        for index, table in enumerate(tables):
            dataframe = getattr(table, "df", None)
            if dataframe is None or dataframe.empty:
                continue
            docs.append(Document(
                page_content=dataframe.to_csv(index=False),
                metadata={{"source": source, "loader": "CamelotLoader", "table_index": index, "page": int(getattr(table, "page", 0) or 0)}},
            ))
        if docs:
            return docs
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(source).load()
    if loader_class_name == "TabulaLoader":
        from langchain_community.document_loaders import UnstructuredPDFLoader
        return UnstructuredPDFLoader(source, mode="elements", strategy="fast").load()
    if loader_class_name == "LlamaParseLoader":
        try:
            from llama_parse import LlamaParse
        except ImportError as exc:
            raise ImportError("LlamaParseLoader requires `pip install llama-parse` and LLAMA_CLOUD_API_KEY.") from exc
        return LlamaParse(result_type="markdown").load_data(source)
    if loader_class_name == "UnstructuredWordDocumentLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "Docx2txtLoader":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(source).load()
    if loader_class_name == "CSVLoader":
        from langchain_community.document_loaders.csv_loader import CSVLoader
        return CSVLoader(source).load()
    if loader_class_name == "UnstructuredCSVLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "UnstructuredExcelLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "PandasDataFrameLoader":
        import pandas as pd
        from langchain_community.document_loaders import DataFrameLoader
        frame = pd.read_csv(source) if str(source).lower().endswith(".csv") else pd.read_excel(source)
        return DataFrameLoader(frame, page_content_column=str(frame.columns[0])).load()
    if loader_class_name == "UnstructuredPowerPointLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "BSHTMLLoader":
        from langchain_community.document_loaders import BSHTMLLoader
        return BSHTMLLoader(source).load()
    if loader_class_name == "UnstructuredHTMLLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "UnstructuredMarkdownLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "JSONLoader":
        from langchain_community.document_loaders import JSONLoader
        return JSONLoader(file_path=source, jq_schema=".", text_content=False).load()
    if loader_class_name == "UnstructuredXMLLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "WebBaseLoader":
        from langchain_community.document_loaders import WebBaseLoader
        return WebBaseLoader(source).load()
    if loader_class_name == "AsyncHtmlLoader":
        from langchain_community.document_loaders import AsyncHtmlLoader
        return AsyncHtmlLoader([source]).load()
    if loader_class_name == "FireCrawlLoader":
        from langchain_community.document_loaders import FireCrawlLoader
        return FireCrawlLoader(url=source, mode="scrape").load()
    if loader_class_name == "ApifyWebScraper":
        raise RuntimeError("ApifyWebScraper generated pipeline expects an Apify dataset ID/actor integration. Use FireCrawl or WebBaseLoader for direct URLs, or customize this generated branch with your Apify actor.")
    if loader_class_name == "YoutubeLoader":
        from langchain_community.document_loaders import YoutubeLoader
        return YoutubeLoader(video_id=_extract_youtube_id(source), language="en").load()
    if loader_class_name == "UnstructuredImageLoader":
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    if loader_class_name == "GenericLoader":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(source, encoding="utf-8").load()
    if loader_class_name == "TextLoader":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(source, encoding="utf-8").load()
    if loader_class_name == "SQLDatabaseLoader":
        from langchain_community.document_loaders import SQLDatabaseLoader
        from langchain_community.utilities import SQLDatabase
        db = SQLDatabase.from_uri(source)
        return SQLDatabaseLoader(query="SELECT * FROM documents LIMIT 1000", db=db).load()
    if loader_class_name == "MongoDBAtlasLoader":
        from langchain_community.document_loaders import MongodbLoader
        return MongodbLoader(connection_string=source, db_name=os.getenv("MONGODB_DB_NAME", "ms_rag"), collection_name=os.getenv("MONGODB_COLLECTION_NAME", "docs")).load()
    if loader_class_name in {{"UnstructuredEPubLoader", "UnstructuredRTFLoader"}}:
        from langchain_unstructured import UnstructuredLoader
        return UnstructuredLoader(source).load()
    raise ValueError(f"Unsupported generated loader: {{loader_class_name}}")

def load_documents(sources: list[str]) -> list:
    """Load documents from configured sources."""
    all_docs = []
    for source in sources:
        try:
            doc_type = _infer_doc_type(source)
            loader_class = LOADER_BY_DOC_TYPE.get(doc_type) or LOADER_BY_DOC_TYPE.get("txt") or "TextLoader"
            docs = _invoke_loader(loader_class, source)
            for doc in docs:
                metadata = dict(getattr(doc, "metadata", {{}}) or {{}})
                metadata.setdefault("source", source)
                metadata.setdefault("ms_rag_loader", loader_class)
                doc.metadata = metadata
            all_docs.extend(docs)
        except Exception as e:
            raise RuntimeError(f"Failed to load {{source}} with configured loader: {{e}}") from e
    return all_docs'''

    def _render_chunking_function(self, config: PipelineConfig) -> str:
        if not config.chunking:
            size, overlap, strategy = 1000, 200, "recursive_character"
            tokenizer = None
            language = None
            separators = None
        else:
            size = config.chunking.chunk_size
            overlap = config.chunking.chunk_overlap
            strategy = config.chunking.strategy
            tokenizer = config.chunking.tokenizer
            language = config.chunking.language
            separators = config.chunking.separators
        emb_provider = config.embedding_model.provider if config.embedding_model else "openai"
        emb_model = config.embedding_model.model_id if config.embedding_model else "text-embedding-3-small"
        semantic_emb_init = {
            "openai": f'OpenAIEmbeddings(model="{emb_model}")',
            "cohere": f'CohereEmbeddings(model="{emb_model}")',
            "huggingface": f'_local_huggingface_embeddings("{emb_model}")',
            "huggingface_endpoint": (
                "HuggingFaceEndpointEmbeddings("
                f'model="{emb_model.removeprefix("hf-endpoint:")}", '
                'huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))'
            ),
            "local": f'_local_huggingface_embeddings("{emb_model}")',
            "google_gemini": f'GoogleGenerativeAIEmbeddings(model="{emb_model}")',
            "ollama": f'OllamaEmbeddings(model="{emb_model}", base_url=_ollama_base_url(usage="embedding"), client_kwargs=_ollama_client_kwargs(usage="embedding"))',
            "mistral": f'MistralAIEmbeddings(model="{emb_model}")',
        }.get(emb_provider, f'OpenAIEmbeddings(model="{emb_model}")')
        llm_init = self._render_llm_initializer(config)
        separators_literal = repr(separators) if separators else "None"
        tokenizer_literal = repr(tokenizer) if tokenizer else "None"
        language_literal = repr(language or "python")
        return f'''# ─── Chunking ─────────────────────────────────────────────────
def _build_text_splitter():
    strategy = "{strategy}"
    if strategy == "recursive_character":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        kwargs = {{"chunk_size": {size}, "chunk_overlap": {overlap}}}
        if {separators_literal}:
            kwargs["separators"] = {separators_literal}
        return RecursiveCharacterTextSplitter(**kwargs)
    if strategy == "fixed_size":
        from langchain_text_splitters import CharacterTextSplitter
        return CharacterTextSplitter(chunk_size={size}, chunk_overlap={overlap}, separator="")
    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        embeddings = {semantic_emb_init}
        return SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    if strategy == "sentence":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size={size}, chunk_overlap={overlap}, separators=[". ", "? ", "! ", "\\n\\n", "\\n", " ", ""])
    if strategy == "paragraph":
        from langchain_text_splitters import CharacterTextSplitter
        return CharacterTextSplitter(separator="\\n\\n", chunk_size={size}, chunk_overlap={overlap})
    if strategy == "token_based":
        from langchain_text_splitters import TokenTextSplitter
        kwargs = {{"chunk_size": {size}, "chunk_overlap": {overlap}}}
        if {tokenizer_literal}:
            kwargs["encoding_name"] = {tokenizer_literal}
        return TokenTextSplitter(**kwargs)
    if strategy == "markdown_aware":
        from langchain_text_splitters import MarkdownTextSplitter
        return MarkdownTextSplitter(chunk_size={size}, chunk_overlap={overlap})
    if strategy == "html_aware":
        from langchain_text_splitters import HTMLSectionSplitter
        return HTMLSectionSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")])
    if strategy == "code_aware":
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        try:
            language = Language[str({language_literal}).upper()]
        except KeyError:
            language = Language.PYTHON
        return RecursiveCharacterTextSplitter.from_language(language=language, chunk_size={size}, chunk_overlap={overlap})
    if strategy in {{"agentic", "document_aware"}}:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        if strategy == "document_aware":
            return _DocumentAwareGeneratedSplitter(chunk_size={size}, chunk_overlap={overlap})
        return _AgenticGeneratedChunker(chunk_size={size}, llm={llm_init})
    raise ValueError(f"Unsupported chunking strategy in generated pipeline: {{strategy}}")

class _DocumentAwareGeneratedSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list) -> list:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")], strip_headers=False)
        recursive = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = []
        for doc in documents:
            splits = header_splitter.split_text(doc.page_content)
            for split in splits:
                metadata = dict(getattr(split, "metadata", {{}}) or {{}})
                metadata.update(getattr(doc, "metadata", {{}}) or {{}})
                split.metadata = metadata
            chunks.extend(recursive.split_documents(splits))
        return chunks

class _AgenticGeneratedChunker:
    def __init__(self, chunk_size: int, llm) -> None:
        self.chunk_size = chunk_size
        self.llm = llm

    def split_documents(self, documents: list) -> list:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        marker = "<MS_RAG_CHUNK>"
        window_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size * 4, chunk_overlap=min(200, max(0, self.chunk_size // 5)))
        size_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        windows = window_splitter.split_documents(documents)
        chunks = []
        prompt = (
            "You are chunking a document for retrieval. Insert the marker "
            f"{{marker}} between semantically complete sections. Keep all original text, "
            "do not summarize, and do not add commentary. Target chunks should be under "
            f"{{self.chunk_size}} characters when possible.\\n\\nTEXT:\\n"
        )
        for window in windows:
            response = self.llm.invoke(prompt + window.page_content)
            marked = getattr(response, "content", str(response))
            parts = [part.strip() for part in marked.split(marker) if part.strip()]
            if not parts:
                raise RuntimeError("Agentic chunking LLM returned no chunk boundaries/content.")
            for part in parts:
                chunks.extend(size_splitter.split_documents([Document(page_content=part, metadata=dict(getattr(window, "metadata", {{}}) or {{}}))]))
        return chunks

def _coalesce_documents(documents: list, target: int) -> list:
    """Merge adjacent same-source element docs up to `target` chars before splitting.

    Element-based loaders (Unstructured PDF/Word, HTML) return one Document per
    line; TextSplitter.split_documents only splits within a Document and never
    merges across them, which would yield tiny ~100-char chunks and wreck
    retrieval. This coalesces consecutive same-source elements into section-sized
    blocks first.
    """
    from langchain_core.documents import Document
    if not documents:
        return documents
    target = target if isinstance(target, int) and target > 0 else 1500
    merged = []
    buf, buf_len, buf_meta, buf_src = [], 0, None, object()
    def _flush():
        nonlocal buf, buf_len, buf_meta
        if buf:
            merged.append(Document(page_content="\\n\\n".join(buf), metadata=dict(buf_meta or {{}})))
        buf, buf_len, buf_meta = [], 0, None
    for doc in documents:
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        meta = dict(getattr(doc, "metadata", {{}}) or {{}})
        source = meta.get("source")
        if buf and (source != buf_src or buf_len + len(text) + 2 > target):
            _flush()
        if not buf:
            buf_src = source
            buf_meta = meta
        buf.append(text)
        buf_len += len(text) + 2
    _flush()
    return merged or documents

def create_chunks(documents: list) -> list:
    """Split documents into chunks using {strategy} strategy."""
    from datetime import UTC, datetime
    import json
    global ADVANCED_PARENT_DOCUMENTS, ADVANCED_CHUNK_DOCUMENTS
    ADVANCED_PARENT_DOCUMENTS = {{}}
    documents = _coalesce_documents(documents, {size})
    prepared_documents = []
    ingested_at = datetime.now(UTC).isoformat()
    for index, doc in enumerate(documents):
        metadata = dict(getattr(doc, "metadata", {{}}) or {{}})
        source = metadata.get("source", f"document_{{index}}")
        parent_id = metadata.get("ms_rag_parent_id") or f"{{source}}::parent::{{index}}"
        metadata.update({{
            "source": source,
            "ms_rag_parent_id": parent_id,
            "ms_rag_ingested_at": metadata.get("ms_rag_ingested_at", ingested_at),
        }})
        doc.metadata = metadata
        ADVANCED_PARENT_DOCUMENTS[parent_id] = doc
        prepared_documents.append(doc)
    splitter = _build_text_splitter()
    chunks = splitter.split_documents(prepared_documents)
    ADVANCED_CHUNK_DOCUMENTS = []
    for index, chunk in enumerate(chunks):
        metadata = dict(getattr(chunk, "metadata", {{}}) or {{}})
        parent_id = metadata.get("ms_rag_parent_id") or metadata.get("source") or f"unknown_parent::{{index}}"
        child_id = metadata.get("ms_rag_child_id") or f"{{parent_id}}::child::{{index}}"
        metadata.update({{
            "ms_rag_parent_id": parent_id,
            "ms_rag_child_id": child_id,
            "ms_rag_multi_vector_source_id": metadata.get("ms_rag_multi_vector_source_id", child_id),
            "ms_rag_ingested_at": metadata.get("ms_rag_ingested_at", ingested_at),
        }})
        chunk.metadata = metadata
        clean = {{}}
        for key, value in chunk.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list) and value and all(isinstance(v, (str, int, float, bool)) for v in value):
                clean[key] = value
            else:
                clean[key] = json.dumps(value, default=str)
        chunk.metadata = clean
        ADVANCED_CHUNK_DOCUMENTS.append(chunk)
    return chunks'''

    def _render_vector_store_function(self, config: PipelineConfig) -> str:
        db_type = config.vector_db.db_type if config.vector_db else "chroma"
        collection = config.vector_db.collection_name if config.vector_db else "ms_rag_collection"
        emb_provider = config.embedding_model.provider if config.embedding_model else "openai"
        emb_model = config.embedding_model.model_id if config.embedding_model else "text-embedding-3-small"
        params = config.vector_db.connection_params if config.vector_db else {}

        emb_init = {
            "openai": f'OpenAIEmbeddings(model="{emb_model}")',
            "cohere": f'CohereEmbeddings(model="{emb_model}")',
            "huggingface": f'_local_huggingface_embeddings("{emb_model}")',
            "huggingface_endpoint": (
                "HuggingFaceEndpointEmbeddings("
                f'model="{emb_model.removeprefix("hf-endpoint:")}", '
                'huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))'
            ),
            "local": f'_local_huggingface_embeddings("{emb_model}")',
            "google_gemini": f'GoogleGenerativeAIEmbeddings(model="{emb_model}")',
            "ollama": f'OllamaEmbeddings(model="{emb_model}", base_url=_ollama_base_url(usage="embedding"), client_kwargs=_ollama_client_kwargs(usage="embedding"))',
            "mistral": f'MistralAIEmbeddings(model="{emb_model}")',
        }.get(emb_provider, f'OpenAIEmbeddings(model="{emb_model}")')

        if db_type == "faiss":
            default_path = params.get("FAISS_INDEX_PATH") or f"./faiss_indexes/{collection}"
            default_path_literal = repr(str(default_path))
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and FAISS vector store with local persistence."""
    embeddings = {emb_init}
    index_path = Path(os.getenv("FAISS_INDEX_PATH", {default_path_literal}))

    if chunks:
        vector_store = FAISS.from_documents(chunks, embeddings)
        setattr(
            vector_store,
            "_ms_rag_keyword_corpus",
            [chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()],
        )
        setattr(vector_store, "_ms_rag_parent_documents", ADVANCED_PARENT_DOCUMENTS)
        setattr(vector_store, "_ms_rag_chunk_documents", ADVANCED_CHUNK_DOCUMENTS)
        setattr(vector_store, "_ms_rag_embeddings", embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
        return vector_store

    if not index_path.exists():
        raise RuntimeError(
            f"FAISS index not found at {{index_path}}. "
            "Run with --ingest first or set FAISS_INDEX_PATH to an existing index."
        )

    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    setattr(vector_store, "_ms_rag_embeddings", embeddings)
    return vector_store'''

        if db_type == "qdrant":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and Qdrant vector store. Ingest chunks if provided."""
    from qdrant_client import QdrantClient

    embeddings = {emb_init}
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    if chunks:
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url=url,
            api_key=api_key,
            collection_name="{collection}",
        )
        setattr(
            vector_store,
            "_ms_rag_keyword_corpus",
            [chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()],
        )
        setattr(vector_store, "_ms_rag_parent_documents", ADVANCED_PARENT_DOCUMENTS)
        setattr(vector_store, "_ms_rag_chunk_documents", ADVANCED_CHUNK_DOCUMENTS)
        setattr(vector_store, "_ms_rag_embeddings", embeddings)
        return vector_store

    client = QdrantClient(url=url, api_key=api_key)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="{collection}",
        embedding=embeddings,
    )
    setattr(vector_store, "_ms_rag_embeddings", embeddings)
    return vector_store'''

        if db_type == "weaviate":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def _connect_weaviate_client():
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080").strip().rstrip("/")
    api_key = os.getenv("WEAVIATE_API_KEY") or None
    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
    if "weaviate.cloud" in url:
        cluster_url = url if url.startswith(("http://", "https://")) else f"https://{{url}}"
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=auth,
        )
    clean = url.replace("https://", "").replace("http://", "")
    host_port = clean.split("/")[0]
    host = host_port.split(":")[0]
    port = int(host_port.split(":")[-1]) if ":" in host_port else 8080
    secure = url.startswith("https://")
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=os.getenv("WEAVIATE_GRPC_HOST", host),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=secure,
        auth_credentials=auth,
    )


def init_vector_store(chunks: list = None):
    """Initialise embeddings and Weaviate vector store. Ingest chunks if provided."""
    embeddings = {emb_init}
    client = _connect_weaviate_client()
    vector_store = WeaviateVectorStore(
        client=client,
        index_name="{collection}",
        text_key="text",
        embedding=embeddings,
    )
    if chunks:
        vector_store.add_documents(chunks)
        setattr(
            vector_store,
            "_ms_rag_keyword_corpus",
            [chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()],
        )
        setattr(vector_store, "_ms_rag_parent_documents", ADVANCED_PARENT_DOCUMENTS)
        setattr(vector_store, "_ms_rag_chunk_documents", ADVANCED_CHUNK_DOCUMENTS)
    setattr(vector_store, "_ms_rag_embeddings", embeddings)
    return vector_store'''

        if db_type == "mongodb_atlas":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and MongoDB Atlas vector store. Ingest chunks if provided."""
    embeddings = {emb_init}
    mongo_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI") or os.getenv("MONGODB_ATLAS_CONNECTION_STRING", "")
    client = MongoClient(mongo_uri)
    db = client[os.getenv("MONGODB_ATLAS_DB_NAME", "ms_rag_db")]
    collection_obj = db[os.getenv("MONGODB_ATLAS_COLLECTION_NAME", "{collection}")]
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection_obj,
        embedding=embeddings,
        index_name=os.getenv("MONGODB_ATLAS_INDEX_NAME", "vector_index"),
    )
    if chunks:
        vector_store.add_documents(chunks)
        setattr(
            vector_store,
            "_ms_rag_keyword_corpus",
            [chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()],
        )
        setattr(vector_store, "_ms_rag_parent_documents", ADVANCED_PARENT_DOCUMENTS)
        setattr(vector_store, "_ms_rag_chunk_documents", ADVANCED_CHUNK_DOCUMENTS)
    setattr(vector_store, "_ms_rag_embeddings", embeddings)
    return vector_store'''

        chroma_default_path = (
            params.get("CHROMA_PERSIST_DIRECTORY")
            or params.get("CHROMA_PERSIST_DIR")
            or "./chroma_db"
        )
        store_init = {
            "chroma": (
                f'Chroma(collection_name="{collection}", embedding_function=embeddings, '
                f'persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY") '
                f'or os.getenv("CHROMA_PERSIST_DIR") or {chroma_default_path!r})'
            ),
            "pinecone": f'PineconeVectorStore(index_name="{collection}", embedding=embeddings)',
            "pgvector": f'PGVector(embeddings=embeddings, collection_name="{collection}", connection=os.getenv("PGVECTOR_CONNECTION_STRING", ""))',
            "milvus": (
                f'Milvus(embedding_function=embeddings, collection_name="{collection}", '
                f'connection_args={{k: v for k, v in {{"uri": os.getenv("MILVUS_URI", "http://localhost:19530"), '
                f'"token": os.getenv("MILVUS_TOKEN")}}.items() if v}})'
            ),
            "elasticsearch": (
                f'ElasticsearchStore(index_name="{collection}", embedding=embeddings, '
                f'es_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"), '
                f'es_user=os.getenv("ELASTICSEARCH_USERNAME") or None, '
                f'es_password=os.getenv("ELASTICSEARCH_PASSWORD") or None, '
                f'es_api_key=os.getenv("ELASTICSEARCH_API_KEY") or None)'
            ),
            "redis": (
                f'RedisVectorStore(embeddings=embeddings, config=RedisConfig('
                f'index_name=os.getenv("REDIS_INDEX_NAME", "{collection}"), '
                f'redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")))'
            ),
            "opensearch": (
                f'OpenSearchVectorSearch(index_name="{collection}", embedding_function=embeddings, '
                f'opensearch_url=os.getenv("OPENSEARCH_URL", "http://localhost:9200"), '
                f'http_auth=(os.getenv("OPENSEARCH_USERNAME", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin")))'
            ),
            "azure_ai_search": (
                f'AzureSearch(azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""), '
                f'azure_search_key=os.getenv("AZURE_SEARCH_KEY") or os.getenv("AZURE_SEARCH_API_KEY", ""), '
                f'index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "{collection}"), '
                f'embedding_function=embeddings)'
            ),
        }.get(db_type, f'Chroma(collection_name="{collection}", embedding_function=embeddings)')

        return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and vector store. Ingest chunks if provided."""
    embeddings = {emb_init}
    vector_store = {store_init}
    if chunks:
        vector_store.add_documents(chunks)
        setattr(
            vector_store,
            "_ms_rag_keyword_corpus",
            [chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()],
        )
        setattr(vector_store, "_ms_rag_parent_documents", ADVANCED_PARENT_DOCUMENTS)
        setattr(vector_store, "_ms_rag_chunk_documents", ADVANCED_CHUNK_DOCUMENTS)
    setattr(vector_store, "_ms_rag_embeddings", embeddings)
    return vector_store'''

    def _render_retriever_function(self, config: PipelineConfig) -> str:
        strategy = config.retrieval.strategy if config.retrieval else "dense_vector"
        top_k = config.retrieval.top_k if config.retrieval else 5
        alpha = config.retrieval.alpha if config.retrieval and config.retrieval.alpha is not None else 0.5
        lam = (
            config.retrieval.lambda_diversity
            if config.retrieval and config.retrieval.lambda_diversity is not None
            else 0.5
        )
        metadata_fields = config.retrieval.metadata_fields if config.retrieval and config.retrieval.metadata_fields else []
        ensemble_subs = config.retrieval.ensemble_sub_retrievers if config.retrieval and config.retrieval.ensemble_sub_retrievers else ["dense_vector", "keyword_bm25"]
        ensemble_weights = config.retrieval.ensemble_weights if config.retrieval and config.retrieval.ensemble_weights else [1.0 / len(ensemble_subs)] * len(ensemble_subs)

        helper_block = '''def _extract_corpus_texts(vector_store):
    """Extract indexed texts for keyword retrieval."""
    cached = getattr(vector_store, "_ms_rag_keyword_corpus", None)
    if isinstance(cached, list) and cached:
        return [text for text in cached if isinstance(text, str) and text.strip()]

    texts = []
    get_fn = getattr(vector_store, "get", None)
    if callable(get_fn):
        try:
            result = get_fn()
            documents = result.get("documents", []) if isinstance(result, dict) else []
            for doc in documents:
                if isinstance(doc, str) and doc.strip():
                    texts.append(doc)
                elif hasattr(doc, "page_content") and isinstance(doc.page_content, str) and doc.page_content.strip():
                    texts.append(doc.page_content)
        except Exception as exc:
            warnings.warn(
                f"Could not extract documents from vector_store.get(); keyword retrieval may degrade: {exc}",
                stacklevel=2,
            )
    if not texts:
        docstore = getattr(vector_store, "docstore", None)
        raw_docs = getattr(docstore, "_dict", None)
        if isinstance(raw_docs, dict):
            for doc in raw_docs.values():
                if isinstance(doc, str) and doc.strip():
                    texts.append(doc)
                elif hasattr(doc, "page_content") and isinstance(doc.page_content, str) and doc.page_content.strip():
                    texts.append(doc.page_content)
    return texts


def _dense_retriever(vector_store, *, top_k):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def _require_keyword_texts(texts, strategy_name):
    if not texts:
        raise RuntimeError(
            f"{strategy_name} retrieval requires keyword corpus text. "
            "Run this generated pipeline with --ingest first and keep the generated vector/keyword state, "
            "or choose dense_vector retrieval."
        )
    return texts


def _compact_multivector_text(content, metadata):
    source = metadata.get("source", "")
    first_lines = " ".join(line.strip() for line in content.splitlines()[:3] if line.strip())
    snippet = (first_lines or content.strip())[:700]
    return f"Source: {source}\\nSummary representation: {snippet}".strip()


def _recency_score(raw_timestamp):
    from datetime import datetime
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return 0.0
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_seconds = max((datetime.now(parsed.tzinfo) - parsed).total_seconds(), 0.0)
    return 1.0 / (1.0 + (age_seconds / 86400))


def _parent_child_retriever(vector_store, *, top_k):
    from langchain_core.runnables import RunnableLambda
    parent_documents = getattr(vector_store, "_ms_rag_parent_documents", None)
    if not isinstance(parent_documents, dict) or not parent_documents:
        raise RuntimeError(
            "Parent-Child retrieval requires parent document state. Run this generated pipeline with --ingest first."
        )
    dense = vector_store.as_retriever(search_kwargs={"k": top_k})

    def retrieve(query):
        child_docs = dense.invoke(query)
        results = []
        seen = set()
        for child in child_docs:
            parent_id = getattr(child, "metadata", {}).get("ms_rag_parent_id")
            parent_doc = parent_documents.get(parent_id)
            if parent_id and parent_doc is not None and parent_id not in seen:
                results.append(parent_doc)
                seen.add(parent_id)
            elif parent_id not in seen:
                results.append(child)
                if parent_id:
                    seen.add(parent_id)
        return results[:top_k]

    return RunnableLambda(retrieve)


def _multi_vector_retriever(vector_store, *, top_k):
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    from langchain_community.vectorstores import FAISS
    chunk_documents = getattr(vector_store, "_ms_rag_chunk_documents", None)
    embeddings = getattr(vector_store, "_ms_rag_embeddings", None)
    if not isinstance(chunk_documents, list) or not chunk_documents:
        raise RuntimeError(
            "Multi-Vector retrieval requires chunk state. Run this generated pipeline with --ingest first."
        )
    if embeddings is None:
        raise RuntimeError("Multi-Vector retrieval requires embeddings for the local representation index.")
    source_documents = {}
    representation_docs = []
    for doc in chunk_documents:
        content = getattr(doc, "page_content", "")
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source_id = metadata.get("ms_rag_multi_vector_source_id") or metadata.get("ms_rag_child_id")
        if not source_id or not content:
            continue
        source_documents[source_id] = doc
        representation_docs.append(Document(
            page_content=_compact_multivector_text(content, metadata),
            metadata={"ms_rag_multi_vector_source_id": source_id},
        ))
    if not representation_docs:
        raise RuntimeError("Multi-Vector retrieval could not build representation documents from chunks.")
    representation_store = FAISS.from_documents(representation_docs, embeddings)
    representation_retriever = representation_store.as_retriever(search_kwargs={"k": max(top_k * 2, top_k)})

    def retrieve(query):
        hits = representation_retriever.invoke(query)
        results = []
        seen = set()
        for hit in hits:
            source_id = getattr(hit, "metadata", {}).get("ms_rag_multi_vector_source_id")
            source_doc = source_documents.get(source_id)
            if source_id and source_doc is not None and source_id not in seen:
                results.append(source_doc)
                seen.add(source_id)
            if len(results) >= top_k:
                break
        return results

    return RunnableLambda(retrieve)


def _time_weighted_retriever(vector_store, *, top_k):
    from langchain_core.runnables import RunnableLambda
    dense = vector_store.as_retriever(search_kwargs={"k": max(top_k * 4, top_k)})

    def retrieve(query):
        docs = dense.invoke(query)
        if not any(getattr(doc, "metadata", {}).get("ms_rag_ingested_at") for doc in docs):
            raise RuntimeError(
                "Time-Weighted retrieval requires ms_rag_ingested_at metadata. Run this generated pipeline with --ingest first."
            )
        scored = []
        for rank, doc in enumerate(docs):
            dense_score = 1.0 / (rank + 1)
            recency_score = _recency_score(getattr(doc, "metadata", {}).get("ms_rag_ingested_at"))
            scored.append((0.4 * dense_score + 0.6 * recency_score, rank, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[:top_k]]

    return RunnableLambda(retrieve)
'''

        if strategy == "hybrid":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build hybrid retriever (BM25 + dense), top_k={top_k}, alpha={alpha}."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    dense = _dense_retriever(vector_store, top_k={top_k})
    texts = _require_keyword_texts(_extract_corpus_texts(vector_store), "Hybrid")
    bm25 = BM25Retriever.from_texts(texts, k={top_k})
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[{1 - alpha}, {alpha}],
    )'''

        if strategy == "keyword_bm25":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build BM25 retriever, top_k={top_k}."""
    from langchain_community.retrievers import BM25Retriever

    texts = _require_keyword_texts(_extract_corpus_texts(vector_store), "BM25")
    return BM25Retriever.from_texts(texts, k={top_k})'''

        if strategy == "tfidf":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build TF-IDF retriever, top_k={top_k}."""
    from langchain_community.retrievers import TFIDFRetriever

    texts = _require_keyword_texts(_extract_corpus_texts(vector_store), "TF-IDF")
    return TFIDFRetriever.from_texts(texts, k={top_k})'''

        if strategy == "mmr":
            return f'''# ─── Retriever ────────────────────────────────────────────────
def build_retriever(vector_store):
    """Build MMR retriever, top_k={top_k}, lambda={lam}."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={{"k": {top_k}, "lambda_mult": {lam}}},
    )'''

        if strategy == "ensemble":
            sub_ids_literal = repr(ensemble_subs)
            weights_literal = repr(ensemble_weights)
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def _build_single_retriever(vector_store, strategy_id, *, top_k, alpha, lam):
    from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
    texts = _extract_corpus_texts(vector_store)

    if strategy_id == "dense_vector":
        return _dense_retriever(vector_store, top_k=top_k)
    if strategy_id == "keyword_bm25":
        texts = _require_keyword_texts(texts, "BM25 ensemble member")
        return BM25Retriever.from_texts(texts, k=top_k)
    if strategy_id == "tfidf":
        texts = _require_keyword_texts(texts, "TF-IDF ensemble member")
        return TFIDFRetriever.from_texts(texts, k=top_k)
    if strategy_id == "hybrid":
        from langchain_classic.retrievers import EnsembleRetriever
        dense = _dense_retriever(vector_store, top_k=top_k)
        texts = _require_keyword_texts(texts, "Hybrid ensemble member")
        bm25 = BM25Retriever.from_texts(texts, k=top_k)
        return EnsembleRetriever(retrievers=[bm25, dense], weights=[1 - alpha, alpha])
    if strategy_id == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={{"k": top_k, "lambda_mult": lam}},
        )
    if strategy_id == "parent_child":
        return _parent_child_retriever(vector_store, top_k=top_k)
    if strategy_id == "multi_vector":
        return _multi_vector_retriever(vector_store, top_k=top_k)
    if strategy_id == "time_weighted":
        return _time_weighted_retriever(vector_store, top_k=top_k)
    return _dense_retriever(vector_store, top_k=top_k)


def build_retriever(vector_store):
    """Build ensemble retriever, top_k={top_k}."""
    from langchain_classic.retrievers import EnsembleRetriever

    sub_ids = {sub_ids_literal}
    weights = {weights_literal}
    sub_retrievers = [
        _build_single_retriever(vector_store, sub_id, top_k={top_k}, alpha={alpha}, lam={lam})
        for sub_id in sub_ids
    ]
    return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)'''

        if strategy == "parent_child":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build Parent-Child retriever, top_k={top_k}."""
    return _parent_child_retriever(vector_store, top_k={top_k})'''

        if strategy == "multi_vector":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build Multi-Vector retriever, top_k={top_k}."""
    return _multi_vector_retriever(vector_store, top_k={top_k})'''

        if strategy == "time_weighted":
            return f'''# ─── Retriever ────────────────────────────────────────────────
{helper_block}


def build_retriever(vector_store):
    """Build Time-Weighted retriever, top_k={top_k}."""
    return _time_weighted_retriever(vector_store, top_k={top_k})'''

        if strategy == "self_query":
            return f'''# ─── Retriever ────────────────────────────────────────────────
def build_retriever(vector_store):
    """Build retriever using self_query strategy, top_k={top_k}.

    Self-Query needs a live LLM object and vector store metadata translator.
    This generated single-file pipeline uses dense retrieval unless you extend it
    with provider-specific SelfQueryRetriever configuration.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={{"k": {top_k}}},
    )'''

        return f'''# ─── Retriever ────────────────────────────────────────────────
def build_retriever(vector_store):
    """Build retriever using {strategy} strategy, top_k={top_k}."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={{"k": {top_k}}},
    )'''

    def _render_reranker_function(self, config: PipelineConfig) -> str:
        if not config.reranking:
            return ""
        reranker = config.reranking.reranker
        model_id = config.reranking.model_id
        top_k = config.reranking.top_k
        needs_llm = reranker == "llm_reranker"
        llm_init = self._render_llm_initializer(config) if needs_llm else "None"
        return f'''# ─── Reranking ────────────────────────────────────────────────
class _RerankingRetriever:
    """Retriever wrapper that applies the selected reranker before generation."""

    def __init__(self, base_retriever, rerank_fn):
        self.base_retriever = base_retriever
        self.rerank_fn = rerank_fn

    def invoke(self, query: str, *args, **kwargs):
        docs = list(self.base_retriever.invoke(query, *args, **kwargs) or [])
        return self.rerank_fn(query, docs)

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda query: self.invoke(query)) | other


def _build_rerank_llm():
    return {llm_init}


def rerank_documents(query: str, docs: list) -> list:
    """Rerank documents using {reranker} (top_k={top_k})."""
    if not docs:
        return []
    reranker = {reranker!r}
    model_id = {model_id!r}
    try:
        if reranker in {{"cross_encoder", "bge_reranker"}}:
            model = _get_cross_encoder(model_id)
            pairs = [(query, getattr(doc, "page_content", "")) for doc in docs]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
            return [doc for _, doc in ranked[:{top_k}]]

        if reranker == "colbert":
            scorer = _get_colbert_scorer(model_id or "colbert-ir/colbertv2.0")
            scores = scorer.score(query, [getattr(doc, "page_content", "") for doc in docs])
            ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
            return [doc for _, doc in ranked[:{top_k}]]

        if reranker == "cohere_reranker":
            import cohere

            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise RuntimeError("COHERE_API_KEY is required for Cohere reranking.")
            client = cohere.Client(api_key)
            response = client.rerank(
                model=model_id,
                query=query,
                documents=[getattr(doc, "page_content", "") for doc in docs],
                top_n={top_k},
            )
            return [docs[result.index] for result in response.results]

        if reranker == "llm_reranker":
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            llm = _build_rerank_llm()
            if llm is None:
                raise RuntimeError("LLM reranker requires a configured generation LLM.")
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Score relevance from 0 to 10. Reply with only the number."),
                ("human", "Query: {{query}}\\n\\nDocument:\\n{{document}}"),
            ])
            chain = prompt | llm | StrOutputParser()
            scored = []
            for doc in docs:
                raw = chain.invoke({{"query": query, "document": getattr(doc, "page_content", "")}})
                try:
                    score = float(str(raw).strip().split()[0])
                except (ValueError, IndexError):
                    score = 0.0
                scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [doc for _, doc in scored[:{top_k}]]

        if reranker == "flashrank":
            from flashrank import Ranker, RerankRequest

            ranker = Ranker()
            passages = [{{"id": i, "text": getattr(doc, "page_content", "")}} for i, doc in enumerate(docs)]
            results = ranker.rerank(RerankRequest(query=query, passages=passages))
            top_ids = [result["id"] for result in results[:{top_k}]]
            return [docs[i] for i in top_ids]

        raise RuntimeError(f"Unsupported reranker: {{reranker}}")
    except Exception as exc:
        print(f"WARNING: Reranker {{reranker!r}} failed; using original top-k documents: {{exc}}")
        return docs[:{top_k}]


def apply_reranking(retriever):
    return _RerankingRetriever(base_retriever=retriever, rerank_fn=rerank_documents)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


class _ColBERTLateInteractionScorer:
    """ColBERT-style late interaction: token-level MaxSim, not a cross-encoder."""

    def __init__(self, model_id: str, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    def _encode(self, texts):
        torch = self._torch
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        mask = encoded["attention_mask"].bool()
        return [hidden[i][mask[i]] for i in range(hidden.shape[0])]

    def score(self, query: str, documents: list) -> list:
        if not documents:
            return []
        query_tokens = self._encode([query])[0]
        scores = []
        for start in range(0, len(documents), 16):
            for doc_tokens in self._encode(documents[start:start + 16]):
                if doc_tokens.shape[0] == 0 or query_tokens.shape[0] == 0:
                    scores.append(0.0)
                    continue
                similarity = query_tokens @ doc_tokens.T
                scores.append(float(similarity.max(dim=1).values.sum().item()))
        return scores


@lru_cache(maxsize=1)
def _get_colbert_scorer(model_id: str):
    return _ColBERTLateInteractionScorer(model_id or "colbert-ir/colbertv2.0")'''

    def _render_compressor_function(self, config: PipelineConfig) -> str:
        if not config.compression:
            return ""
        techniques = list(config.compression.techniques or [])
        techniques_label = ", ".join(techniques)
        threshold = config.compression.similarity_threshold
        needs_llm = bool({"llm_chain_extraction", "summary_compression"} & set(techniques))
        needs_llm = needs_llm or (
            "contextual_compression" in techniques and not config.embedding_model
        )
        llm_init = self._render_llm_initializer(config) if needs_llm else "None"
        return f'''# ─── Context Compression ──────────────────────────────────────
class _SafeCompressionRetriever:
    """Context compression wrapper that never hides an empty-compression failure."""

    def __init__(self, base_retriever, base_compressor, min_retained_documents: int = 3):
        self.base_retriever = base_retriever
        self.base_compressor = base_compressor
        self.min_retained_documents = max(int(min_retained_documents), 1)

    def invoke(self, query: str, *args, **kwargs):
        docs = list(self.base_retriever.invoke(query, *args, **kwargs) or [])
        if not docs:
            return []
        compressed = list(_compress_or_transform_documents(self.base_compressor, docs, query) or [])
        minimum = min(self.min_retained_documents, len(docs))
        should_guard = not compressed or len(docs) >= self.min_retained_documents
        if compressed and (not should_guard or len(compressed) >= minimum):
            return compressed
        print(
            "WARNING: Context compression returned too little context "
            f"({{len(compressed)}}/{{len(docs)}} documents); using the top {{minimum}} "
            "uncompressed retrieved documents instead. Lower the threshold, increase top_k, "
            "or disable compression."
        )
        return docs[:minimum]

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda query: self.invoke(query)) | other


def _compress_or_transform_documents(compressor, documents: list, query: str) -> list:
    if hasattr(compressor, "compress_documents"):
        return list(compressor.compress_documents(documents, query) or [])
    if hasattr(compressor, "transform_documents"):
        return list(compressor.transform_documents(documents) or [])
    raise TypeError(
        f"Compression component {{type(compressor).__name__}} does not implement "
        "compress_documents() or transform_documents()."
    )


class _LLMSummaryCompressor:
    """Summarize each retrieved document against the question."""

    def __init__(self, llm):
        self.llm = llm

    def compress_documents(self, documents: list, query: str, callbacks=None) -> list:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Summarize the document for answering the question. Keep concrete facts, names, dates, numbers, and source-specific details. Return only the useful compressed context.",
            ),
            ("human", "Question:\\n{{query}}\\n\\nDocument:\\n{{document}}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        summarized = []
        for doc in documents:
            text = str(getattr(doc, "page_content", "") or "")
            if not text.strip():
                continue
            summary = str(chain.invoke({{"query": query, "document": text}})).strip()
            if summary:
                metadata = dict(getattr(doc, "metadata", {{}}) or {{}})
                metadata["ms_rag_compression"] = "summary_compression"
                summarized.append(Document(page_content=summary, metadata=metadata))
        return summarized


def _build_compression_llm():
    return {llm_init}


def compress_context(retriever, embeddings):
    """Apply context compression: {techniques_label}."""
    from langchain_classic.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter,
        LLMChainExtractor,
    )
    from langchain_community.document_transformers import EmbeddingsRedundantFilter

    techniques = {techniques!r}
    compressors = []
    compression_llm = None

    def require_llm():
        nonlocal compression_llm
        if compression_llm is None:
            compression_llm = _build_compression_llm()
        if compression_llm is None:
            raise RuntimeError("Selected compression technique requires an LLM, but no LLM is configured.")
        return compression_llm

    for technique in techniques:
        if technique == "llm_chain_extraction":
            compressors.append(LLMChainExtractor.from_llm(require_llm()))
        elif technique == "embeddings_filter":
            if embeddings is None:
                raise RuntimeError("Embeddings Filter compression requires the generated vector store embeddings.")
            compressors.append(EmbeddingsFilter(embeddings=embeddings, similarity_threshold={threshold}))
        elif technique == "redundancy_removal":
            if embeddings is None:
                raise RuntimeError("Redundancy Removal compression requires the generated vector store embeddings.")
            compressors.append(EmbeddingsRedundantFilter(embeddings=embeddings))
        elif technique == "contextual_compression":
            if embeddings is not None:
                compressors.append(EmbeddingsFilter(embeddings=embeddings, similarity_threshold={threshold}))
            else:
                compressors.append(LLMChainExtractor.from_llm(require_llm()))
        elif technique == "document_compressor_pipeline":
            if len(techniques) > 1:
                continue
            if embeddings is not None:
                compressors.append(DocumentCompressorPipeline(transformers=[
                    EmbeddingsRedundantFilter(embeddings=embeddings),
                    EmbeddingsFilter(embeddings=embeddings, similarity_threshold={threshold}),
                ]))
            else:
                compressors.append(LLMChainExtractor.from_llm(require_llm()))
        elif technique == "summary_compression":
            compressors.append(_LLMSummaryCompressor(require_llm()))
        else:
            raise RuntimeError(f"Unsupported compression technique: {{technique}}")

    if not compressors:
        raise RuntimeError("Compression was enabled, but no compressor could be built.")

    compressor = compressors[0] if len(compressors) == 1 else DocumentCompressorPipeline(transformers=compressors)
    return _SafeCompressionRetriever(base_compressor=compressor, base_retriever=retriever)'''

    def _render_llm_initializer(self, config: PipelineConfig) -> str:
        """Render the selected generation LLM constructor."""
        from ms_rag.utils.credentials import DEFAULT_LLM_MODELS  # noqa: PLC0415

        if config.llm_model:
            provider = config.llm_model.provider
            model_id = config.llm_model.model_id
        elif config.configured_providers:
            provider = config.configured_providers[0]
            model_id = DEFAULT_LLM_MODELS.get(provider, "default")
        else:
            raise ValueError("Cannot generate pipeline without a selected generation LLM model.")

        model = repr(model_id)
        llm_init = {
            "openai": f"ChatOpenAI(model={model}, openai_api_key=OPENAI_API_KEY)",
            "anthropic": f"ChatAnthropic(model={model}, api_key=ANTHROPIC_API_KEY)",
            "cohere": f"ChatCohere(model={model}, cohere_api_key=COHERE_API_KEY)",
            "huggingface": (
                "ChatHuggingFace(llm=HuggingFaceEndpoint("
                f"repo_id={model}, "
                "huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, "
                "task='conversational'), "
                f"model_id={model})"
            ),
            "google_gemini": f"ChatGoogleGenerativeAI(model={model}, google_api_key=GOOGLE_API_KEY)",
            "mistral": f"ChatMistralAI(model={model}, api_key=MISTRAL_API_KEY)",
            "groq": f"ChatGroq(model={model}, groq_api_key=GROQ_API_KEY)",
            "together_ai": (
                f"ChatOpenAI(model={model}, base_url='https://api.together.xyz/v1', "
                "openai_api_key=TOGETHER_API_KEY)"
            ),
            "replicate": (
                f"Replicate(model={model}, replicate_api_token=REPLICATE_API_TOKEN)"
            ),
            "ollama": (
                f"ChatOllama(model={model}, base_url=_ollama_base_url(usage='chat'), "
                "client_kwargs=_ollama_client_kwargs(usage='chat'))"
            ),
            "azure_openai": (
                f"AzureChatOpenAI(azure_deployment={model}, "
                "azure_endpoint=AZURE_OPENAI_ENDPOINT or '', "
                'api_version=AZURE_OPENAI_API_VERSION or "2024-02-01", '
                "openai_api_key=AZURE_OPENAI_API_KEY)"
            ),
            "aws_bedrock": (
                f"ChatBedrock(model_id={model}, region_name=AWS_REGION or 'us-east-1', "
                "aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)"
            ),
        }.get(provider)
        if not llm_init:
            raise ValueError(
                f"Unsupported generation LLM provider for code generation: {provider!r}"
            )
        return llm_init

    def _render_graphrag_functions(self, config: PipelineConfig) -> str:
        graph_store = config.graph_store
        store_type = graph_store.store_type if graph_store else "local_json"
        graph_name = graph_store.graph_name if graph_store else "ms_rag_graph"
        query_mode = graph_store.query_mode if graph_store else "hybrid"
        params = graph_store.connection_params if graph_store else {}
        graph_path = params.get("GRAPH_STORE_PATH") or f"./graph_indexes/{graph_name}.json"
        kuzu_path = params.get("KUZU_DATABASE_PATH") or "./graph_indexes/kuzu"
        return f'''# ─── Full GraphRAG Graph Index ────────────────────────────────
GRAPH_STORE_TYPE = os.getenv("GRAPH_STORE_TYPE", "{store_type}")
GRAPH_NAME = os.getenv("GRAPH_NAME", "{graph_name}")
GRAPH_QUERY_MODE = os.getenv("GRAPH_QUERY_MODE", "{query_mode}")


def _entity_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")[:100] or "entity"


def _fallback_graph_extract(text: str) -> dict:
    candidates = re.findall(r"\\b[A-Z][A-Za-z0-9_.-]*(?:\\s+[A-Z][A-Za-z0-9_.-]*){{0,4}}\\b", text)
    entities = [{{"name": item.strip(), "type": "Concept"}} for item in dict.fromkeys(candidates[:12])]
    relationships = [
        {{"source": left["name"], "target": right["name"], "type": "RELATED_TO", "description": "Co-mentioned in the same chunk."}}
        for left, right in zip(entities, entities[1:])
    ]
    return {{"entities": entities, "relationships": relationships}}


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\\{{.*\\}}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_graph_from_text(text: str, llm=None) -> dict:
    if llm is None:
        return _fallback_graph_extract(text)
    prompt = (
        "Extract a knowledge graph from the text. Return only JSON with keys entities and relationships. "
        "entities: [{{name,type}}]. relationships: [{{source,target,type,description}}]. Text:\\n\\n"
        + text[:6000]
    )
    try:
        result = llm.invoke(prompt)
        parsed = _json_from_text(getattr(result, "content", str(result)))
        if not isinstance(parsed.get("entities", []), list):
            return _fallback_graph_extract(text)
        parsed.setdefault("relationships", [])
        return parsed
    except Exception:
        return _fallback_graph_extract(text)


def _summarize_community(names: list[str], evidence: str, llm=None) -> str:
    if llm is None:
        return f"Entities: {{', '.join(names)}}. Evidence: {{evidence[:700]}}"
    try:
        result = llm.invoke(
            "Summarize this GraphRAG community with key entities, relationships, and facts.\\n\\n"
            f"Entities: {{', '.join(names)}}\\nEvidence:\\n{{evidence}}"
        )
        return str(getattr(result, "content", result))[:1500]
    except Exception:
        return f"Entities: {{', '.join(names)}}. Evidence: {{evidence[:700]}}"


def build_graph_index(chunks: list, llm=None) -> dict:
    nodes, edges, source_chunks = {{}}, {{}}, []
    for index, doc in enumerate(chunks):
        text = str(getattr(doc, "page_content", "") or "")
        metadata = dict(getattr(doc, "metadata", {{}}) or {{}})
        chunk_id = str(metadata.get("ms_rag_child_id") or f"chunk::{{index}}")
        source_chunks.append({{"chunk_id": chunk_id, "text": text[:2000], "source": str(metadata.get("source", ""))}})
        extracted = extract_graph_from_text(text, llm)
        for entity in extracted.get("entities", []):
            name = str(entity.get("name", "")).strip()[:120]
            if not name:
                continue
            node_id = _entity_id(name)
            node = nodes.setdefault(node_id, {{"id": node_id, "name": name, "type": entity.get("type", "Entity"), "chunk_ids": []}})
            if chunk_id not in node["chunk_ids"]:
                node["chunk_ids"].append(chunk_id)
        for rel in extracted.get("relationships", []):
            src, dst = _entity_id(str(rel.get("source", ""))), _entity_id(str(rel.get("target", "")))
            if not src or not dst or src == dst:
                continue
            rel_type = re.sub(r"[^A-Z0-9_]+", "_", str(rel.get("type", "RELATED_TO")).upper()) or "RELATED_TO"
            edge = edges.setdefault((src, dst, rel_type), {{"source": src, "target": dst, "type": rel_type, "descriptions": [], "chunk_ids": []}})
            desc = str(rel.get("description", "")).strip()
            if desc and desc not in edge["descriptions"]:
                edge["descriptions"].append(desc[:500])
            if chunk_id not in edge["chunk_ids"]:
                edge["chunk_ids"].append(chunk_id)
    communities = _build_graph_communities(nodes, list(edges.values()), source_chunks, llm)
    return {{"schema_version": "1.0", "nodes": list(nodes.values()), "edges": list(edges.values()), "communities": communities, "source_chunks": source_chunks}}


def _build_graph_communities(nodes: dict, edges: list, chunks: list, llm=None) -> list:
    adjacency = {{node_id: set() for node_id in nodes}}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    seen, communities, chunk_by_id = set(), [], {{chunk["chunk_id"]: chunk for chunk in chunks}}
    for node_id in nodes:
        if node_id in seen:
            continue
        stack, component = [node_id], []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency.get(current, set()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        evidence_ids = []
        for member in component:
            evidence_ids.extend(nodes[member].get("chunk_ids", [])[:2])
        evidence = "\\n".join(chunk_by_id.get(cid, {{}}).get("text", "")[:500] for cid in evidence_ids[:8])
        names = [nodes[member]["name"] for member in component[:12]]
        communities.append({{"id": f"community_{{len(communities)}}", "node_ids": component, "summary": _summarize_community(names, evidence, llm)}})
    return communities


def persist_graph_index(graph: dict) -> None:
    if GRAPH_STORE_TYPE == "local_json":
        path = Path(os.getenv("GRAPH_STORE_PATH", {repr(str(graph_path))})).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if GRAPH_STORE_TYPE == "kuzu":
        path = Path(os.getenv("KUZU_DATABASE_PATH", {repr(str(kuzu_path))})).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        (path / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if GRAPH_STORE_TYPE == "neo4j":
        from neo4j import GraphDatabase
        if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
            raise RuntimeError("Neo4j GraphRAG requires NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD.")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        try:
            with driver.session(database=NEO4J_DATABASE or None) as session:
                session.run("MERGE (g:MSRAGGraph {{name: $name}})", name=GRAPH_NAME)
                for node in graph.get("nodes", []):
                    session.run("MERGE (e:MSRAGEntity {{graph: $graph, id: $id}}) SET e.name=$name, e.type=$type, e.chunk_ids=$chunk_ids", graph=GRAPH_NAME, **node)
                for edge in graph.get("edges", []):
                    session.run("MATCH (s:MSRAGEntity {{graph: $graph, id: $source}}) MATCH (t:MSRAGEntity {{graph: $graph, id: $target}}) MERGE (s)-[r:MSRAG_RELATED {{graph: $graph, type: $type}}]->(t) SET r.descriptions=$descriptions, r.chunk_ids=$chunk_ids", graph=GRAPH_NAME, **edge)
                for community in graph.get("communities", []):
                    session.run("MERGE (c:MSRAGCommunity {{graph: $graph, id: $id}}) SET c.node_ids=$node_ids, c.summary=$summary", graph=GRAPH_NAME, **community)
        finally:
            driver.close()
        return
    raise RuntimeError(f"Unsupported GRAPH_STORE_TYPE: {{GRAPH_STORE_TYPE}}")


def load_graph_index() -> dict:
    if GRAPH_STORE_TYPE == "local_json":
        path = Path(os.getenv("GRAPH_STORE_PATH", {repr(str(graph_path))})).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"GraphRAG graph index not found at {{path}}. Run --ingest first.")
        return json.loads(path.read_text(encoding="utf-8"))
    if GRAPH_STORE_TYPE == "kuzu":
        path = Path(os.getenv("KUZU_DATABASE_PATH", {repr(str(kuzu_path))})).expanduser().resolve() / "graph.json"
        if not path.exists():
            raise RuntimeError(f"Kuzu GraphRAG graph export not found at {{path}}. Run --ingest first.")
        return json.loads(path.read_text(encoding="utf-8"))
    if GRAPH_STORE_TYPE == "neo4j":
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        try:
            with driver.session(database=NEO4J_DATABASE or None) as session:
                nodes = [dict(r["e"]) | {{"id": r["e"]["id"]}} for r in session.run("MATCH (e:MSRAGEntity {{graph: $graph}}) RETURN e", graph=GRAPH_NAME)]
                edges = [dict(r["r"]) | {{"source": r["source"], "target": r["target"]}} for r in session.run("MATCH (s:MSRAGEntity {{graph: $graph}})-[r:MSRAG_RELATED {{graph: $graph}}]->(t:MSRAGEntity {{graph: $graph}}) RETURN s.id AS source, t.id AS target, r", graph=GRAPH_NAME)]
                communities = [dict(r["c"]) | {{"id": r["c"]["id"]}} for r in session.run("MATCH (c:MSRAGCommunity {{graph: $graph}}) RETURN c", graph=GRAPH_NAME)]
        finally:
            driver.close()
        return {{"schema_version": "1.0", "nodes": nodes, "edges": edges, "communities": communities, "source_chunks": []}}
    raise RuntimeError(f"Unsupported GRAPH_STORE_TYPE: {{GRAPH_STORE_TYPE}}")


def retrieve_graph_context(query: str, llm=None) -> str:
    graph = load_graph_index()
    query_entities = extract_graph_from_text(query, llm).get("entities", [])
    names = {{_entity_id(str(item.get("name", ""))) for item in query_entities}}
    nodes, edges = graph.get("nodes", []), graph.get("edges", [])
    chunks = {{item["chunk_id"]: item for item in graph.get("source_chunks", [])}}
    local_lines = []
    for node in nodes:
        node_id = node.get("id")
        if node_id in names or str(node.get("name", "")).lower() in query.lower():
            local_lines.append(f"Entity: {{node.get('name')}} ({{node.get('type')}})")
            for edge in edges:
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    local_lines.append(f"Relationship: {{edge.get('source')}} -[{{edge.get('type')}}]-> {{edge.get('target')}}; {{'; '.join(edge.get('descriptions', [])[:2])}}")
            for chunk_id in node.get("chunk_ids", [])[:3]:
                if chunk_id in chunks:
                    local_lines.append(f"Evidence chunk {{chunk_id}}: {{chunks[chunk_id].get('text', '')[:800]}}")
    global_lines = [f"Community {{item.get('id')}}: {{item.get('summary')}}" for item in graph.get("communities", [])[:8]]
    if GRAPH_QUERY_MODE == "local":
        return "\\n".join(local_lines)
    if GRAPH_QUERY_MODE == "global":
        return "\\n".join(global_lines)
    return "\\n\\n".join(part for part in ("\\n".join(local_lines), "\\n".join(global_lines)) if part.strip())
'''

    def _render_lcel_chain(self, config: PipelineConfig) -> str:
        llm_init = self._render_llm_initializer(config)
        rag_type = config.rag_type.rag_type if config.rag_type else "naive_rag"
        enhancements = list(config.query_enhancement or [])
        enhancements_literal = repr(enhancements)
        retrieval_strategy = config.retrieval.strategy if config.retrieval else "dense_vector"
        keyword_strategy = retrieval_strategy in {"keyword_bm25", "tfidf", "hybrid", "ensemble"}
        special_chain = ""
        if rag_type == "speculative_rag":
            special_chain = '''
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "Draft a concise tentative answer. It may be incomplete; evidence will be retrieved next."),
        ("human", "{question}"),
    ])
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Draft answer:\\n{draft}\\n\\nEvidence passages:\\n{context}\\n\\nQuestion: {question}\\n\\nVerify the draft against the evidence. Correct it if needed."),
    ])
    draft_chain = draft_prompt | llm | StrOutputParser()
    final_chain = final_prompt | llm | StrOutputParser()

    def speculative(query: str) -> str:
        draft = draft_chain.invoke({"question": query})
        docs = retriever.invoke(f"{query}\\nDraft answer: {draft}")
        return final_chain.invoke({"question": query, "draft": draft, "context": format_docs(docs)})

    return RunnableLambda(speculative)
'''
        elif rag_type == "graphrag":
            special_chain = '''
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key entities, topics, and relationships as a short comma-separated query expansion."),
        ("human", "{question}"),
    ])
    entity_chain = entity_prompt | llm | StrOutputParser()
    final_chain = prompt | llm | StrOutputParser()

    def graph_guided(query: str) -> str:
        entities = entity_chain.invoke({"question": query})
        docs = retriever.invoke(f"{query}\\nEntities and relationships: {entities}")
        graph_context = retrieve_graph_context(query, llm=llm)
        context = "\\n\\n".join(part for part in (graph_context, format_docs(docs)) if part.strip())
        if not context.strip():
            raise RuntimeError("GraphRAG could not retrieve graph or vector context. Run --ingest and verify graph/vector stores.")
        return final_chain.invoke({"question": query, "context": context})

    return RunnableLambda(graph_guided)
'''

        return f'''# ─── RAG Chain (LCEL) ─────────────────────────────────────────
def build_rag_chain(retriever):
    """Assemble the RAG chain using LangChain Expression Language."""

    def format_docs(docs):
        return "\\n\\n".join(
            f"[Source: {{getattr(d, \'metadata\', {{}}).get(\'source\', f\'chunk_{{i}}\')}}]\\n{{d.page_content}}"
            for i, d in enumerate(docs)
        )

    llm = {llm_init}
    query_enhancements = {enhancements_literal}
    keyword_sensitive_retrieval = {keyword_strategy!r}

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context passages:\\n\\n{{context}}\\n\\nQuestion: {{question}}"),
    ])

    def _run_prompt(system_message: str, human_message: str, **values) -> str:
        local_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message),
        ])
        return str((local_prompt | llm | StrOutputParser()).invoke(values)).strip()

    def _split_queries(text: str) -> list[str]:
        queries = []
        for line in str(text or "").splitlines():
            cleaned = line.strip().lstrip("-*0123456789. )").strip()
            if cleaned:
                queries.append(cleaned)
        return queries

    def enhance_queries(question: str) -> list[str]:
        queries = [question]
        for technique in query_enhancements:
            try:
                if technique == "query_rewriting":
                    rewritten = _run_prompt(
                        "Rewrite the user question into a precise search query. Return only the rewritten query.",
                        "{{question}}",
                        question=queries[0],
                    )
                    if rewritten:
                        queries = [rewritten]
                elif technique == "query_expansion":
                    expanded = _run_prompt(
                        "Expand the query with synonyms and related domain terms. Return one concise expanded query.",
                        "{{question}}",
                        question=queries[0],
                    )
                    if expanded:
                        queries = [expanded]
                elif technique == "hyde":
                    if keyword_sensitive_retrieval:
                        continue
                    hypothetical = _run_prompt(
                        "Write a short hypothetical document that would answer the question. Return only the document.",
                        "{{question}}",
                        question=queries[0],
                    )
                    if hypothetical:
                        queries = [hypothetical]
                elif technique == "multi_query":
                    generated = _run_prompt(
                        "Generate 3 alternative retrieval queries for the question, one per line.",
                        "{{question}}",
                        question=question,
                    )
                    queries = [question] + _split_queries(generated)
                elif technique == "step_back_prompting":
                    step_back = _run_prompt(
                        "Write one broader step-back question that helps answer the user question.",
                        "{{question}}",
                        question=question,
                    )
                    queries = [question] + ([step_back] if step_back else [])
                elif technique == "sub_question_decomposition":
                    generated = _run_prompt(
                        "Break the question into 3 focused sub-questions for retrieval, one per line.",
                        "{{question}}",
                        question=question,
                    )
                    queries = [question] + _split_queries(generated)
                elif technique == "rag_fusion":
                    generated = _run_prompt(
                        "Generate 4 diverse retrieval queries for reciprocal-rank fusion, one per line.",
                        "{{question}}",
                        question=question,
                    )
                    queries = [question] + _split_queries(generated)
            except Exception as exc:
                print(f"WARNING: Query enhancement {{technique!r}} failed; using current query set: {{exc}}")
        deduped = []
        seen = set()
        for item in queries:
            key = item.strip()
            if key and key not in seen:
                deduped.append(key)
                seen.add(key)
        if deduped != [question]:
            print("\\nQuery enhancement trace:")
            for idx, item in enumerate(deduped, start=1):
                print(f"  {{idx}}. {{item[:240]}}")
        return deduped or [question]

    def retrieve_docs(question: str) -> list:
        retrieval_queries = enhance_queries(question)
        if len(retrieval_queries) == 1:
            return retriever.invoke(retrieval_queries[0])

        ranked = {{}}
        docs_by_key = {{}}
        use_rrf = "rag_fusion" in set(query_enhancements)
        for query_variant in retrieval_queries:
            docs = retriever.invoke(query_variant)
            for rank, doc in enumerate(docs):
                key = (
                    getattr(doc, "metadata", {{}}).get("source"),
                    getattr(doc, "metadata", {{}}).get("page"),
                    getattr(doc, "page_content", "")[:120],
                )
                docs_by_key.setdefault(key, doc)
                ranked[key] = ranked.get(key, 0.0) + (1.0 / (60 + rank) if use_rrf else 1.0 / (rank + 1))
        return [docs_by_key[key] for key, _ in sorted(ranked.items(), key=lambda item: item[1], reverse=True)]

    def prepare_inputs(question: str) -> dict:
        return {{"context": format_docs(retrieve_docs(question)), "question": question}}
{special_chain}

    # LangChain LCEL chain: enhance → retrieve → format → prompt → LLM → parse
    rag_chain = (
        RunnableLambda(prepare_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )
    setattr(rag_chain, "_ms_rag_retrieve_docs", retrieve_docs)
    return rag_chain'''

    def _render_agent_tool_helpers(self, config: PipelineConfig) -> str:
        payload = {"enabled_tools": [], "tool_settings": {}}
        if config.agent_tools:
            payload = {
                "enabled_tools": list(config.agent_tools.enabled_tools or []),
                "tool_settings": dict(config.agent_tools.tool_settings or {}),
            }
        return f'''# ─── Agentic Tool Helpers ─────────────────────────────────────
AGENT_TOOLS = {json.dumps(payload, indent=2, sort_keys=True)}

def _tool_enabled(name: str) -> bool:
    return name in set(AGENT_TOOLS.get("enabled_tools") or [])

def _tool_settings(name: str) -> dict:
    return dict((AGENT_TOOLS.get("tool_settings") or {{}}).get(name) or {{}})

def _tool_domain_allowed(hostname: str, allowed_domains: list[str]) -> bool:
    host = (hostname or "").lower().strip()
    return any(host == str(item).lower().strip() or host.endswith("." + str(item).lower().strip()) for item in allowed_domains)

def _tool_summarize(text: str, llm=None) -> str:
    text = str(text or "")[:12000]
    if not _tool_enabled("document_summarization") or llm is None or len(text) < 4000:
        return text
    result = llm.invoke("Summarize this tool result for a grounded RAG answer. Keep facts, entities, dates, URLs, and decisions.\\n\\n" + text)
    return getattr(result, "content", str(result))

def _tool_web_search(query: str) -> str:
    settings = _tool_settings("web_search")
    provider = str(settings.get("provider") or "tavily")
    timeout = float(settings.get("timeout_seconds") or 20)
    max_results = int(settings.get("max_results") or 5)
    if provider == "tavily":
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("Missing TAVILY_API_KEY for Web Search tool.")
        response = requests.post("https://api.tavily.com/search", json={{"api_key": key, "query": query, "max_results": max_results}}, timeout=timeout)
    elif provider == "brave":
        key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("Missing BRAVE_SEARCH_API_KEY for Web Search tool.")
        response = requests.get("https://api.search.brave.com/res/v1/web/search", params={{"q": query, "count": max_results}}, headers={{"X-Subscription-Token": key}}, timeout=timeout)
    else:
        raise RuntimeError(f"Unsupported web search provider: {{provider}}")
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)[:12000] if "application/json" in response.headers.get("content-type", "") else response.text[:12000]

def _tool_fetch_url(url: str) -> str:
    settings = _tool_settings("url_fetch")
    parsed = urlparse(url)
    if parsed.scheme not in {{"http", "https"}}:
        raise RuntimeError("URL Fetch only allows http/https URLs.")
    if not _tool_domain_allowed(parsed.hostname or "", list(settings.get("allowed_domains") or [])):
        raise RuntimeError(f"URL domain is not allowlisted: {{parsed.hostname}}")
    max_bytes = int(settings.get("max_bytes") or 250000)
    response = requests.get(url, timeout=float(settings.get("timeout_seconds") or 20), stream=True)
    response.raise_for_status()
    content = response.raw.read(max_bytes + 1, decode_content=True)
    if len(content) > max_bytes:
        raise RuntimeError(f"URL response exceeded max page size of {{max_bytes}} bytes.")
    return content.decode(response.encoding or "utf-8", errors="replace")

def _tool_read_file(path: str) -> str:
    settings = _tool_settings("file_read")
    requested = Path(path).expanduser().resolve()
    allowed = [Path(p).expanduser().resolve() for p in settings.get("allowed_paths") or []]
    if not any(requested == base or requested.is_relative_to(base) for base in allowed):
        raise RuntimeError(f"File path is not allowlisted: {{requested}}")
    if requested.is_dir():
        raise RuntimeError("File System Read Tool reads files only.")
    data = requested.read_bytes()
    max_bytes = int(settings.get("max_bytes") or 250000)
    if len(data) > max_bytes:
        raise RuntimeError(f"File exceeded max read size of {{max_bytes}} bytes.")
    return data.decode("utf-8", errors="replace")

def _tool_api_request(method: str, url: str) -> str:
    settings = _tool_settings("api_request")
    method = method.upper()
    allowed_methods = {{str(item).upper() for item in settings.get("allowed_methods") or ["GET"]}}
    if method not in allowed_methods:
        raise RuntimeError(f"HTTP method is not allowlisted: {{method}}")
    parsed = urlparse(url)
    base_url = f"{{parsed.scheme}}://{{parsed.netloc}}"
    if base_url not in [str(item).rstrip("/") for item in settings.get("allowed_base_urls") or []]:
        raise RuntimeError(f"API base URL is not allowlisted: {{base_url}}")
    headers = {{}}
    auth_env = str(settings.get("auth_env_var") or "")
    if auth_env:
        token = os.getenv(auth_env)
        if not token:
            raise RuntimeError(f"Missing API auth secret: {{auth_env}}")
        headers["Authorization"] = "Bearer " + token
    response = requests.request(method, url, headers=headers, timeout=float(settings.get("timeout_seconds") or 20))
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)[:12000] if "application/json" in response.headers.get("content-type", "") else response.text[:12000]

def _tool_recall_memory(query: str) -> str:
    settings = _tool_settings("memory")
    path = Path(os.getenv("MS_RAG_AGENT_MEMORY_PATH", str(settings.get("path") or "./agent_memory/memory.json"))).expanduser()
    if not path.exists():
        return ""
    records = json.loads(path.read_text(encoding="utf-8"))
    enabled_types = set(settings.get("memory_types") or [])
    terms = {{term.lower() for term in query.split() if len(term) > 2}}
    ranked = sorted(
        [item for item in records if item.get("memory_type") in enabled_types],
        key=lambda item: (len(terms.intersection(str(item.get("text", "")).lower().split())), float(item.get("created_at") or 0)),
        reverse=True,
    )
    return "\\n\\n".join(f"[{{item.get('memory_type')}}] {{item.get('text')}}" for item in ranked[:5])

def run_agent_tools(question: str, documents: list, llm=None) -> list[str]:
    results = []
    if _tool_enabled("memory"):
        memory = _tool_recall_memory(question)
        if memory:
            results.append("Memory recall:\\n" + memory)
    if _tool_enabled("url_fetch"):
        for url in re.findall(r"https?://[^\\s)>\\"]+", question)[:3]:
            clean_url = url.rstrip(".,")
            results.append(f"URL fetch result for {{clean_url}}:\\n" + _tool_summarize(_tool_fetch_url(clean_url), llm))
    if _tool_enabled("file_read"):
        matches = re.findall(r"file:\\"([^\\"]+)\\"", question, flags=re.IGNORECASE) + re.findall(r"file:([^\\s\\"]+)", question, flags=re.IGNORECASE)
        for file_path in matches[:3]:
            results.append(f"File read result for {{file_path}}:\\n" + _tool_summarize(_tool_read_file(file_path.strip()), llm))
    if _tool_enabled("api_request"):
        for method, url in re.findall(r"api\\s+(GET|POST|PUT|PATCH)\\s+(https?://\\S+)", question, flags=re.IGNORECASE)[:2]:
            clean_url = url.rstrip(".,")
            results.append(f"API response for {{method.upper()}} {{clean_url}}:\\n" + _tool_summarize(_tool_api_request(method.upper(), clean_url), llm))
    if _tool_enabled("web_search") and not documents:
        results.append("Web search results:\\n" + _tool_summarize(_tool_web_search(question), llm))
    return results
'''

    def _render_langgraph_workflow(self, config: PipelineConfig) -> str:
        rag_type = config.rag_type.rag_type if config.rag_type else "self_rag"
        llm_init = self._render_llm_initializer(config)

        return f'''# ─── LangGraph Workflow ({rag_type}) ───────────────────────────
class GraphState(TypedDict):
    question: str
    generation: str
    documents: list
    rewrite_count: int
    action: str
    route: str
    tool_results: list
    trace: list[str]


def build_rag_chain(retriever):
    """Build LangGraph agentic workflow for {rag_type}."""
    llm = {llm_init}

    def append_trace(state: dict, message: str) -> list[str]:
        return list(state.get("trace", []) or []) + [message]

    def answer_is_unknown(answer: object) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", str(answer or "").lower()).strip()
        return normalized in {{"i don t know", "i do not know"}}

    def lexical_relevant_docs(question: str, docs: list, limit: int = 3) -> list:
        stop_words = {{"about", "tell", "what", "which", "where", "when", "why", "how", "the", "and", "for", "with", "from", "this", "that"}}
        terms = {{term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2 and term not in stop_words}}
        scored = []
        for doc in docs:
            text = str(getattr(doc, "page_content", "") or "").lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])
        return {{"documents": docs, "trace": append_trace(state, f"retrieve: fetched {{len(docs)}} candidate document(s)")}}

    def direct_answer(state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{{question}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return {{
            "generation": chain.invoke({{"question": state["question"]}}),
            "documents": state.get("documents", []),
            "trace": append_trace(state, "direct_answer: answered without retrieval context"),
        }}

    def generate(state: GraphState) -> dict:
        docs = list(state.get("documents", []) or [])
        context = "\\n\\n".join(d.page_content for d in docs)
        tool_context = "\\n\\n".join(state.get("tool_results", []))
        # Agent memory / tool results are a distinct, explicitly-usable source so
        # conversational/meta questions ("what did I ask earlier?") are answerable
        # from memory. Content is passed as prompt variables (never interpolated
        # into the template) so braces in it cannot corrupt the prompt.
        if tool_context.strip():
            sys_text = SYSTEM_PROMPT + "\\n\\nAGENT MEMORY & TOOLS: In addition to the context passages, you are given results from approved agent tools and agent memory. You MAY use them to answer, including questions about the current conversation, what the user asked earlier, or prior interactions. Information from agent memory or tools is valid evidence and does not require a document source citation."
            human = "Context passages:\\n\\n{{context}}\\n\\nApproved tool results & agent memory:\\n\\n{{tool_context}}\\n\\nQuestion: {{question}}"
            values = {{"context": context or "(no document passages retrieved)", "tool_context": tool_context, "question": state["question"]}}
        else:
            sys_text = SYSTEM_PROMPT
            human = "Context passages:\\n\\n{{context}}\\n\\nQuestion: {{question}}"
            values = {{"context": context, "question": state["question"]}}
        prompt = ChatPromptTemplate.from_messages([("system", sys_text), ("human", human)])
        answer = (prompt | llm | StrOutputParser()).invoke(values)
        trace = append_trace(state, f"generate: generated answer from {{len(docs)}} document(s)")
        if (docs or tool_context) and answer_is_unknown(answer):
            retry_sys = sys_text + "\\n\\nSELF-RAG RETRY: Relevant evidence was provided in the context passages and/or the agent-memory/tool section. Before saying you do not know, extract the directly supported facts. Questions about the conversation itself are answerable from agent memory and do not need a document citation. Do not use outside knowledge."
            retry_prompt = ChatPromptTemplate.from_messages([("system", retry_sys), ("human", human)])
            answer = (retry_prompt | llm | StrOutputParser()).invoke(values)
            trace = trace + ["generate: first answer was 'I don't know' despite retrieved/memory evidence; ran one grounded retry"]
        return {{"generation": answer, "trace": trace}}

    def rewrite_query(state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question to be clearer."),
            ("human", "{{question}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        new_question = chain.invoke({{"question": state["question"]}}).strip()
        return {{
            "question": new_question,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "trace": append_trace(state, f"rewrite_query: {{state['question']}} -> {{new_question}}"),
        }}

    def llm_choice(messages: list[tuple[str, str]], values: dict, allowed: set[str], default: str) -> str:
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke(values).strip().lower()
        for item in allowed:
            if item in raw:
                return item
        return default

    def grade_documents(state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Is this document relevant to the question? Answer yes or no."),
            ("human", "Question: {{question}}\\n\\nDocument: {{document}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        original_docs = list(state.get("documents", []))
        relevant = []
        for doc in original_docs:
            grade = chain.invoke({{"question": state["question"], "document": doc.page_content}}).lower()
            if "yes" in grade:
                relevant.append(doc)
        trace = append_trace(state, f"grade_documents: LLM relevance grader kept {{len(relevant)}}/{{len(original_docs)}} document(s)")
        if not relevant and original_docs:
            lexical = lexical_relevant_docs(state["question"], original_docs)
            if lexical:
                relevant = lexical
                trace = trace + [f"grade_documents: safety kept {{len(lexical)}} document(s) with lexical query overlap after grader removed all evidence"]
            else:
                trace = trace + ["grade_documents: no relevant evidence found; graph may rewrite query"]
        return {{"documents": relevant, "trace": trace}}

    def decide_to_generate(state: GraphState) -> str:
        if "{rag_type}" == "corrective_rag" and not state.get("documents"):
            return "corrective_fallback"
        return "generate" if state["documents"] or state.get("rewrite_count", 0) >= 2 else "rewrite_query"

    def check_support(state: GraphState) -> str:
        if not state.get("documents"):
            return "end"
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Is the answer fully supported by the provided context? Answer yes or no."),
            ("human", "Context:\\n{{context}}\\n\\nAnswer:\\n{{answer}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        context = "\\n\\n".join(d.page_content for d in state.get("documents", []))
        verdict = chain.invoke({{"context": context, "answer": state.get("generation", "")}}).lower()
        if "yes" not in verdict:
            print("WARNING: support check marked the answer as not fully grounded.")
        return "end"

    def self_retrieval_need(state: GraphState) -> str:
        if "{rag_type}" != "self_rag":
            return "retrieve"
        route = llm_choice(
            [
                ("system", "Decide if this question needs the private document corpus. Answer exactly retrieve or direct."),
                ("human", "{{question}}"),
            ],
            {{"question": state["question"]}},
            {{"retrieve", "direct"}},
            "retrieve",
        )
        if route == "direct" and not is_obvious_direct_chat(state["question"]):
            route = "retrieve"
        return route

    def record_retrieval_decision(state: GraphState) -> dict:
        if "{rag_type}" == "self_rag":
            return {{"trace": append_trace(state, "decide_retrieval_need: checking whether the private document corpus is needed")}}
        return {{"trace": append_trace(state, "corrective_rag: retrieval is required before correction fallback")}}

    def corrective_fallback(state: GraphState) -> dict:
        if "{rag_type}" != "corrective_rag" or state.get("documents"):
            return {{"documents": state.get("documents", []), "tool_results": state.get("tool_results", [])}}
        if _tool_enabled("web_search"):
            return {{"documents": [], "tool_results": run_agent_tools(state["question"], [], llm)}}
        print("WARNING: Corrective RAG found no relevant chunks and no Web Search tool is configured.")
        return {{"documents": [], "tool_results": state.get("tool_results", [])}}

    def route_question(state: GraphState) -> str:
        question = state["question"]
        heuristic_deep = any(word in question.lower() for word in ["compare", "analyze", "relationship", "relationships", "across", "multi-hop", "why", "explain"])
        route = llm_choice(
            [
                ("system", "Route this query. Answer exactly direct, retrieve, or deep. direct means no private corpus needed. retrieve means one corpus retrieval pass. deep means rewrite/decompose then retrieve for multi-hop, comparison, analysis, or relationship questions."),
                ("human", "{{question}}"),
            ],
            {{"question": question}},
            {{"direct", "retrieve", "deep"}},
            "deep" if heuristic_deep else "retrieve",
        )
        return "deep" if heuristic_deep and route == "retrieve" else route

    def deep_retrieve(state: GraphState) -> dict:
        original = state["question"]
        rewritten = rewrite_query(state)["question"]
        docs = list(retriever.invoke(original)) + list(retriever.invoke(rewritten))
        seen = set()
        merged = []
        for doc in docs:
            key = (getattr(doc, "page_content", ""), str(getattr(doc, "metadata", {{}}).get("source", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return {{"documents": merged, "question": rewritten, "rewrite_count": state.get("rewrite_count", 0) + 1}}

    def agent_plan(state: GraphState) -> dict:
        action = llm_choice(
            [
                ("system", "Choose exactly one action for a RAG agent: retrieve, rewrite, or answer. Use rewrite for unclear questions, retrieve when private corpus evidence is needed, answer for simple general conversation."),
                ("human", "{{question}}"),
            ],
            {{"question": state["question"]}},
            {{"retrieve", "rewrite", "answer"}},
            "retrieve",
        )
        if action == "answer" and not is_obvious_direct_chat(state["question"]):
            action = "retrieve"
        return {{"action": action, "trace": append_trace(state, f"agentic_rag: planner selected action={{action}}")}}

    def is_obvious_direct_chat(question: str) -> bool:
        cleaned = " ".join(str(question or "").strip().lower().split())
        return cleaned in {{
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "who are you",
        }}

    def run_approved_tools_node(state: GraphState) -> dict:
        results = run_agent_tools(state["question"], state.get("documents", []), llm)
        return {{"tool_results": results, "trace": append_trace(state, f"agentic_rag: approved tools returned {{len(results)}} result block(s)")}}

    def route_agent_action(state: GraphState) -> str:
        action = state.get("action", "retrieve")
        if action == "rewrite" and state.get("rewrite_count", 0) >= 2:
            return "retrieve"
        return action if action in {{"retrieve", "rewrite", "answer"}} else "retrieve"

    workflow = StateGraph(GraphState)
    if "{rag_type}" in ("self_rag", "corrective_rag"):
        workflow.add_node("decide_retrieval_need", record_retrieval_decision)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("direct_answer", direct_answer)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("corrective_fallback", corrective_fallback)
        workflow.set_entry_point("decide_retrieval_need")
        workflow.add_conditional_edges("decide_retrieval_need", self_retrieval_need, {{"retrieve": "retrieve", "direct": "direct_answer"}})
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", decide_to_generate, {{"generate": "generate", "rewrite_query": "rewrite_query", "corrective_fallback": "corrective_fallback"}})
        workflow.add_edge("corrective_fallback", "generate")
        workflow.add_conditional_edges("generate", check_support, {{"end": END}})
        workflow.add_edge("rewrite_query", "retrieve")
    elif "{rag_type}" == "adaptive_rag":
        workflow.add_node("route_question", lambda state: {{}})
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("deep_retrieve", deep_retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("direct_answer", direct_answer)
        workflow.set_entry_point("route_question")
        workflow.add_conditional_edges("route_question", route_question, {{"direct": "direct_answer", "retrieve": "retrieve", "deep": "deep_retrieve"}})
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("deep_retrieve", "generate")
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("generate", END)
    else:
        workflow.add_node("agent_plan", agent_plan)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("direct_answer", direct_answer)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("run_approved_tools", run_approved_tools_node)
        workflow.set_entry_point("agent_plan")
        workflow.add_conditional_edges("agent_plan", route_agent_action, {{"retrieve": "retrieve", "rewrite": "rewrite_query", "answer": "direct_answer"}})
        workflow.add_edge("rewrite_query", "agent_plan")
        workflow.add_edge("retrieve", "run_approved_tools")
        workflow.add_edge("run_approved_tools", "generate")
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("generate", END)
    return workflow.compile()'''

    def _render_evaluation_setup(self, config: PipelineConfig) -> str:
        if not config.evaluation:
            return ""
        evaluators = config.evaluation.evaluators
        thresholds = config.evaluation.cicd_thresholds or {}
        evaluators_literal = repr(list(evaluators))
        evaluator_model = config.evaluation.evaluator_llm_model or "gpt-4o-mini"
        lines = [
            "# ─── Evaluation Setup ─────────────────────────────────────────",
            f"# Enabled evaluators: {', '.join(evaluators)}",
            f"ENABLED_EVALUATORS = {evaluators_literal}",
            f"EVALUATOR_LLM_MODEL = {evaluator_model!r}",
            "",
            "def _eval_context_texts(contexts: list) -> list[str]:",
            "    return [getattr(c, 'page_content', str(c)) for c in contexts if str(getattr(c, 'page_content', c)).strip()]",
            "",
            "def _eval_token_set(text: str) -> set[str]:",
            "    return {t for t in re.findall(r'[a-z0-9]+', text.lower()) if len(t) > 2}",
            "",
            "def _lexical_eval(query: str, answer: str, contexts: list, prefix: str = '') -> dict:",
            "    ctx_tokens = _eval_token_set(' '.join(_eval_context_texts(contexts)))",
            "    ans_tokens = _eval_token_set(answer)",
            "    query_tokens = _eval_token_set(query)",
            "    if not ctx_tokens or not ans_tokens:",
            "        return {}",
            "    key = f'{prefix}_' if prefix else ''",
            "    recall = len(ans_tokens & ctx_tokens) / len(ans_tokens)",
            "    precision = len(ans_tokens & ctx_tokens) / len(ctx_tokens)",
            "    relevancy = len(ans_tokens & query_tokens) / len(ans_tokens) if query_tokens else 0.0",
            "    return {f'{key}context_recall': round(recall, 4), f'{key}context_precision': round(precision, 4), f'{key}answer_relevancy': round(relevancy, 4), f'{key}faithfulness': round(recall, 4)}",
            "",
            "def _finite_scores(scores: dict) -> dict:",
            "    clean = {}",
            "    for key, value in scores.items():",
            "        if isinstance(value, (int, float)) and math.isfinite(float(value)):",
            "            clean[str(key)] = float(value)",
            "    return clean",
        ]
        if "ragas" in evaluators:
            lines += [
                "",
                "def _install_ragas_vertexai_compat_shim() -> None:",
                "    import sys, types",
                "    module_name = 'langchain_community.chat_models.vertexai'",
                "    if module_name in sys.modules:",
                "        return",
                "    try:",
                "        __import__(module_name)",
                "        return",
                "    except ModuleNotFoundError:",
                "        pass",
                "    try:",
                "        from langchain_google_vertexai import ChatVertexAI",
                "    except Exception:",
                "        class ChatVertexAI:",
                "            pass",
                "    shim = types.ModuleType(module_name)",
                "    shim.ChatVertexAI = ChatVertexAI",
                "    sys.modules[module_name] = shim",
                "",
                "def evaluate_with_ragas(query: str, answer: str, contexts: list) -> dict:",
                '    """Evaluate response using RAGAS metrics."""',
                "    try:",
                "        _install_ragas_vertexai_compat_shim()",
                "        from ragas import evaluate",
                "        from ragas import EvaluationDataset",
                "        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference",
                "        from langchain_openai import ChatOpenAI",
                "        dataset = EvaluationDataset.from_list([{",
                "            'user_input': query,",
                "            'response': answer,",
                "            'retrieved_contexts': _eval_context_texts(contexts) or [''],",
                "        }])",
                "        metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference()]",
                "        llm = ChatOpenAI(model=EVALUATOR_LLM_MODEL, temperature=0)",
                "        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):",
                "            result = evaluate(dataset, metrics=metrics, llm=llm, show_progress=False)",
                "        records = result.to_pandas().to_dict('records') if hasattr(result, 'to_pandas') else []",
                "        scores = _finite_scores(records[0] if records else {})",
                "        if not scores:",
                "            print('RAGAS returned no finite scores; using lexical fallback.')",
                "            return _lexical_eval(query, answer, contexts, prefix='ragas')",
                "        return scores",
                "    except Exception as exc:",
                "        print(f'RAGAS evaluation unavailable; using lexical fallback: {exc}')",
                "        return _lexical_eval(query, answer, contexts, prefix='ragas')",
            ]
        lines += [
            "",
            "def evaluate_response(query: str, answer: str, contexts: list) -> dict:",
            "    scores = {}",
            "    if 'ragas' in ENABLED_EVALUATORS:",
            "        scores.update(evaluate_with_ragas(query, answer, contexts) if 'evaluate_with_ragas' in globals() else _lexical_eval(query, answer, contexts, prefix='ragas'))",
            "    if 'deepeval' in ENABLED_EVALUATORS:",
            "        try:",
            "            from deepeval.metrics import AnswerRelevancyMetric",
            "            from deepeval.test_case import LLMTestCase",
            "            metric = AnswerRelevancyMetric(threshold=0.5, async_mode=False, verbose_mode=False, model=EVALUATOR_LLM_MODEL)",
            "            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):",
            "                metric.measure(LLMTestCase(input=query, actual_output=answer, retrieval_context=_eval_context_texts(contexts) or ['']))",
            "            deepeval_scores = _finite_scores({'answer_relevancy': float(metric.score), 'deepeval_answer_relevancy': float(metric.score)})",
            "            if deepeval_scores:",
            "                scores.update(deepeval_scores)",
            "            else:",
            "                print('DeepEval returned no finite scores; using lexical fallback.')",
            "                scores.update(_lexical_eval(query, answer, contexts, prefix='deepeval'))",
            "        except Exception as exc:",
            "            print(f'DeepEval unavailable; using lexical fallback: {exc}')",
            "            scores.update(_lexical_eval(query, answer, contexts, prefix='deepeval'))",
            "    if 'trulens' in ENABLED_EVALUATORS:",
            "        try:",
            "            from trulens.core import Feedback, Select",
            "            _ = (Feedback, Select)",
            "            scores['trulens_core_package_available'] = 1.0",
            "        except Exception as exc:",
            "            print(f'TruLens core package unavailable; using groundedness scores: {exc}')",
            "        scores.update(_lexical_eval(query, answer, contexts, prefix='trulens'))",
            "    if 'langsmith' in ENABLED_EVALUATORS:",
            "        if os.getenv('LANGCHAIN_API_KEY'):",
            "            scores['langsmith_configured'] = 1.0",
            "        else:",
            "            print('LangSmith selected but LANGCHAIN_API_KEY is not configured.')",
            "    if 'langfuse' in ENABLED_EVALUATORS:",
            "        if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):",
            "            scores['langfuse_configured'] = 1.0",
            "        else:",
            "            print('Langfuse selected but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not configured.')",
            "    if 'arize_phoenix' in ENABLED_EVALUATORS:",
            "        if os.getenv('PHOENIX_COLLECTOR_ENDPOINT'):",
            "            scores['phoenix_endpoint_configured'] = 1.0",
            "        else:",
            "            print('Phoenix selected but PHOENIX_COLLECTOR_ENDPOINT is not configured; using Phoenix-prefixed scores.')",
            "        scores.update(_lexical_eval(query, answer, contexts, prefix='phoenix'))",
            "    if 'ares' in ENABLED_EVALUATORS:",
            "        try:",
            "            import ares",
            "            _ = ares",
            "            scores['ares_package_available'] = 1.0",
            "        except Exception as exc:",
            "            print(f'ARES package unavailable; using compatible scores: {exc}')",
            "        scores.update(_lexical_eval(query, answer, contexts, prefix='ares'))",
            "    if 'ragbench' in ENABLED_EVALUATORS:",
            "        try:",
            "            import datasets",
            "            _ = datasets",
            "            scores['ragbench_datasets_package_available'] = 1.0",
            "        except Exception as exc:",
            "            print(f'RAGBench datasets tooling unavailable; using compatible scores: {exc}')",
            "        scores.update(_lexical_eval(query, answer, contexts, prefix='ragbench'))",
            "    if 'langgraph_trace' in ENABLED_EVALUATORS:",
            "        path = Path(os.getenv('MS_RAG_TRACE_LOG', './ms_rag_traces.jsonl'))",
            "        path.parent.mkdir(parents=True, exist_ok=True)",
            "        path.open('a', encoding='utf-8').write(json.dumps({'query': query, 'answer': answer, 'context_count': len(contexts)}) + '\\n')",
            "        scores['langgraph_trace_logged'] = 1.0",
            "    if 'monitoring_export' in ENABLED_EVALUATORS:",
            "        path = Path(os.getenv('MS_RAG_METRICS_EXPORT', './ms_rag_metrics.jsonl'))",
            "        path.parent.mkdir(parents=True, exist_ok=True)",
            "        path.open('a', encoding='utf-8').write(json.dumps({'query': query, 'answer': answer, 'metrics': scores}) + '\\n')",
            "        scores['monitoring_export_logged'] = 1.0",
            "    return scores",
        ]
        if "langsmith" in evaluators:
            lines += [
                "",
                "# LangSmith tracing — enable via environment variables:",
                '# export LANGCHAIN_TRACING_V2=true',
                '# export LANGCHAIN_API_KEY=<your_key>',
                '# export LANGCHAIN_PROJECT=ms_rag_pipeline',
            ]
        if thresholds:
            thresh_str = ", ".join(f'"{k}": {v}' for k, v in thresholds.items())
            lines += [
                "",
                f"CICD_THRESHOLDS = {{{thresh_str}}}",
                "",
                "def check_cicd_gate(scores: dict) -> bool:",
                '    """Return True if all scores meet CI/CD thresholds."""',
                "    for metric, threshold in CICD_THRESHOLDS.items():",
                "        score = scores.get(metric)",
                "        if score is not None and score < threshold:",
                f"            print(f'CICD GATE FAILED: {{metric}}={{score:.3f}} < {{threshold}}')",
                "            return False",
                "    return True",
            ]
        lines += [
            "",
            "def print_evaluation_results(scores: dict) -> None:",
            '    """Print evaluation scores in a readable terminal format."""',
            "    if not scores:",
            "        print('Evaluation Results: no scores returned.')",
            "        return",
            "    print('\\nEvaluation Results')",
            "    print('-' * 72)",
            "    for metric, score in sorted(scores.items()):",
            "        try:",
            "            rendered = f'{float(score):.4f}'",
            "        except (TypeError, ValueError):",
            "            rendered = str(score)",
            "        print(f'{metric:<48} {rendered:>12}')",
            "    print('-' * 72)",
        ]
        return "\n".join(lines)

    def _render_main(self, config: PipelineConfig) -> str:
        has_eval = config.evaluation_enabled and config.evaluation and bool(config.evaluation.evaluators if config.evaluation else [])
        has_cicd_gate = bool(has_eval and config.evaluation and config.evaluation.cicd_thresholds)
        eval_call = """
        # Evaluate response
        scores = evaluate_response(query, answer, context_docs)
        print_evaluation_results(scores)"""
        if has_cicd_gate:
            eval_call += """
        check_cicd_gate(scores)"""
        if not has_eval:
            eval_call = ""
        is_graphrag = bool(config.rag_type and config.rag_type.rag_type == "graphrag")
        graph_ingest = ""
        if is_graphrag:
            graph_llm = self._render_llm_initializer(config)
            graph_ingest = f'''
        print("Building GraphRAG knowledge graph...")
        graph_llm = {graph_llm}
        graph = build_graph_index(chunks, llm=graph_llm)
        persist_graph_index(graph)
        print(f"GraphRAG graph complete: {{len(graph.get('nodes', []))}} entities, {{len(graph.get('edges', []))}} relationships, {{len(graph.get('communities', []))}} communities.")'''
        compression_apply = ""
        reranking_apply = ""
        if config.reranking_enabled and config.reranking:
            reranking_apply = '''
    retriever = apply_reranking(retriever)'''
        if config.compression_enabled and config.compression:
            compression_apply = '''
    retriever = compress_context(
        retriever,
        getattr(vector_store, "_ms_rag_embeddings", None),
    )'''

        return f'''# ─── Main Entry Point ─────────────────────────────────────────
def main():
    """MS-RAGS(ALL-IN-ONE) pipeline entry point.

    Usage:
        python pipeline.py --ingest --sources doc1.pdf doc2.txt
        python pipeline.py --query "What is retrieval-augmented generation?"
    """
    parser = argparse.ArgumentParser(description="MS-RAGS(ALL-IN-ONE) Generated Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--sources", nargs="+", default=[], help="Document paths or URLs")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    args = parser.parse_args()

    # Initialise vector store
    vector_store = None

    if args.ingest:
        print("Loading documents...")
        docs = load_documents(args.sources or [])
        print(f"Loaded {{len(docs)}} documents. Chunking...")
        chunks = create_chunks(docs)
        print(f"Created {{len(chunks)}} chunks. Ingesting into vector store...")
        vector_store = init_vector_store(chunks=chunks)
{graph_ingest}
        print(f"Ingestion complete: {{len(chunks)}} chunks stored.")
    else:
        vector_store = init_vector_store()

    # Build retriever and RAG chain
    retriever = build_retriever(vector_store)
{reranking_apply}
{compression_apply}
    rag_chain = build_rag_chain(retriever)

    if args.query:
        # Single query mode
        print(f"\\nQuery: {{args.query}}")
        answer = rag_chain.invoke(args.query)
        print(f"\\nAnswer: {{answer}}")
        context_docs = getattr(rag_chain, "_ms_rag_retrieve_docs", retriever.invoke)(args.query){eval_call}
        return

    # Interactive query loop
    print("\\nMS-RAGS(ALL-IN-ONE) Pipeline ready. Type your question (Ctrl+C to exit):\\n")
    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue
            answer = rag_chain.invoke(query)
            context_docs = getattr(rag_chain, "_ms_rag_retrieve_docs", retriever.invoke)(query)
            print(f"\\nAnswer: {{answer}}\\n")
        except KeyboardInterrupt:
            print("\\nGoodbye.")
            break


if __name__ == "__main__":
    main()'''
