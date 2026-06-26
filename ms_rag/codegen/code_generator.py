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

        return GeneratedCode(
            python_code=python_code,
            requirements_txt=requirements_txt,
            rag_type=config.rag_type.rag_type if config.rag_type else "naive_rag",
        )

    def save(self, code: GeneratedCode, output_dir: Path) -> None:
        """Write pipeline.py and requirements.txt to output_dir. Req 17.7-17.8."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = output_dir / "pipeline.py"
        requirements_path = output_dir / "requirements.txt"

        pipeline_path.write_text(code.python_code, encoding="utf-8")
        requirements_path.write_text(code.requirements_txt, encoding="utf-8")

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
            f"  Write pipeline.py and requirements.txt to {output_path.strip()}?",
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
                f"  [dim]• requirements.txt[/dim]"
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

        # Embedding model provider
        if config.embedding_model:
            reqs.update(_PROVIDER_REQUIREMENTS.get(config.embedding_model.provider, []))
            if config.embedding_model.provider in ("huggingface", "local"):
                reqs.add("sentence-transformers>=3.0.0")

        # Vector DB
        if config.vector_db:
            reqs.update(_VECTOR_DB_REQUIREMENTS.get(config.vector_db.db_type, []))

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

        # Context compression (LangChain 1.x retriever compressors live in langchain-classic)
        if config.compression_enabled and config.compression:
            reqs.add("langchain-classic>=0.3.0")

        return sorted(reqs)

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
        ]

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
MS_RAG Generated Pipeline
Generated by: MS_RAG Framework
RAG Type: {rag_type}

# ============================================================
# requirements.txt
{req_block}
# ============================================================
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
'''

    def _render_imports(self, config: PipelineConfig, uses_langgraph: bool) -> str:
        lines = [
            "# ─── LangChain imports ────────────────────────────────────────",
            "from langchain_core.prompts import ChatPromptTemplate",
            "from langchain_core.output_parsers import StrOutputParser",
            "from langchain_core.runnables import RunnablePassthrough",
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
        for pid in config.configured_providers:
            if pid == "openai":
                lines.append("from langchain_openai import ChatOpenAI")
            elif pid == "anthropic":
                lines.append("from langchain_anthropic import ChatAnthropic")
            elif pid == "cohere":
                lines.append("from langchain_cohere import ChatCohere")
            elif pid == "google_gemini":
                lines.append("from langchain_google_genai import ChatGoogleGenerativeAI")
            elif pid == "mistral":
                lines.append("from langchain_mistralai import ChatMistralAI")
            elif pid == "groq":
                lines.append("from langchain_groq import ChatGroq")
            elif pid == "ollama":
                lines.append("from langchain_ollama import ChatOllama")
            elif pid == "azure_openai":
                lines.append("from langchain_openai import AzureChatOpenAI")

        if uses_langgraph:
            lines.append("")
            lines.append("# ─── LangGraph imports ────────────────────────────────────")
            lines.append("from langgraph.graph import StateGraph, END")
            lines.append("from typing import TypedDict")

        return "\n".join(lines)

    def _render_credentials(self, config: PipelineConfig) -> str:
        lines = ["# ─── Credentials (via environment variables) ─────────────────"]
        for pid in config.configured_providers:
            from ms_rag.config.credential_manager import PROVIDER_FIELDS  # noqa: PLC0415
            for field in PROVIDER_FIELDS.get(pid, []):
                lines.append(f'{field} = os.getenv("{field}")')
        if config.embedding_model and config.embedding_model.provider == "ollama" and "ollama" not in config.configured_providers:
            from ms_rag.config.credential_manager import PROVIDER_FIELDS  # noqa: PLC0415
            for field in PROVIDER_FIELDS.get("ollama", []):
                if f'{field} = os.getenv("{field}")' not in lines:
                    lines.append(f'{field} = os.getenv("{field}")')

        uses_ollama = "ollama" in config.configured_providers or (
            config.embedding_model is not None and config.embedding_model.provider == "ollama"
        )
        if uses_ollama:
            lines.extend([
                "",
                "def _ollama_base_url():",
                '    return OLLAMA_BASE_URL or ("https://ollama.com" if OLLAMA_API_KEY else "http://localhost:11434")',
                "",
                "def _ollama_client_kwargs():",
                '    return {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}} if OLLAMA_API_KEY else {}',
            ])
        if config.vector_db and config.vector_db.connection_params:
            lines.append("")
            lines.append("# Vector DB credentials")
            for key in config.vector_db.connection_params:
                lines.append(f'{key} = os.getenv("{key}")')
        return "\n".join(lines)

    def _render_system_prompt(self, config: PipelineConfig) -> str:
        prompt = config.system_prompt or "You are a helpful assistant."
        escaped = prompt.replace('"""', '\\"\\"\\"')
        return f'# ─── System Prompt ────────────────────────────────────────────\nSYSTEM_PROMPT = """{escaped}"""'

    def _render_loader_function(self, config: PipelineConfig) -> str:
        loaders = []
        for doc_type, loader_class in config.loader_map.items():
            loaders.append(f'    # {doc_type}: {loader_class}')
        loader_block = "\n".join(loaders) if loaders else "    # No loaders configured"
        return f'''# ─── Document Loading ─────────────────────────────────────────
def load_documents(sources: list[str]) -> list:
    """Load documents from configured sources."""
    from langchain_community.document_loaders import TextLoader
    all_docs = []
{loader_block}
    for source in sources:
        try:
            loader = TextLoader(source, encoding="utf-8")
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {{source}}: {{e}}")
    return all_docs'''

    def _render_chunking_function(self, config: PipelineConfig) -> str:
        if not config.chunking:
            size, overlap, strategy = 1000, 200, "recursive_character"
        else:
            size = config.chunking.chunk_size
            overlap = config.chunking.chunk_overlap
            strategy = config.chunking.strategy
        return f'''# ─── Chunking ─────────────────────────────────────────────────
def create_chunks(documents: list) -> list:
    """Split documents into chunks using {strategy} strategy."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import json
    splitter = RecursiveCharacterTextSplitter(
        chunk_size={size},
        chunk_overlap={overlap},
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
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
            "huggingface": f'HuggingFaceEmbeddings(model_name="{emb_model}")',
            "huggingface_endpoint": (
                "HuggingFaceEndpointEmbeddings("
                f'model="{emb_model.removeprefix("hf-endpoint:")}", '
                'huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))'
            ),
            "local": f'HuggingFaceEmbeddings(model_name="{emb_model}")',
            "google_gemini": f'GoogleGenerativeAIEmbeddings(model="{emb_model}")',
            "ollama": f'OllamaEmbeddings(model="{emb_model}", base_url=_ollama_base_url(), client_kwargs=_ollama_client_kwargs())',
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
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
        return vector_store

    if not index_path.exists():
        raise RuntimeError(
            f"FAISS index not found at {{index_path}}. "
            "Run with --ingest first or set FAISS_INDEX_PATH to an existing index."
        )

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )'''

        if db_type == "qdrant":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and Qdrant vector store. Ingest chunks if provided."""
    from qdrant_client import QdrantClient

    embeddings = {emb_init}
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    if chunks:
        return QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url=url,
            api_key=api_key,
            collection_name="{collection}",
        )

    client = QdrantClient(url=url, api_key=api_key)
    return QdrantVectorStore(
        client=client,
        collection_name="{collection}",
        embedding=embeddings,
    )'''

        if db_type == "weaviate":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and Weaviate vector store. Ingest chunks if provided."""
    embeddings = {emb_init}
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    host = url.replace("https://", "").replace("http://", "").split(":")[0]
    port = int(url.rsplit(":", 1)[-1]) if ":" in url.replace("https://", "").replace("http://", "") else 8080
    api_key = os.getenv("WEAVIATE_API_KEY") or None
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=url.startswith("https://"),
        grpc_host=host,
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=url.startswith("https://"),
        auth_credentials=weaviate.auth.AuthApiKey(api_key) if api_key else None,
    )
    vector_store = WeaviateVectorStore(
        client=client,
        index_name="{collection}",
        text_key="text",
        embedding=embeddings,
    )
    if chunks:
        vector_store.add_documents(chunks)
    return vector_store'''

        if db_type == "mongodb_atlas":
            return f'''# ─── Embedding + Vector Store ─────────────────────────────────
def init_vector_store(chunks: list = None):
    """Initialise embeddings and MongoDB Atlas vector store. Ingest chunks if provided."""
    embeddings = {emb_init}
    client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI", ""))
    db = client[os.getenv("MONGODB_ATLAS_DB_NAME", "ms_rag_db")]
    collection_obj = db[os.getenv("MONGODB_ATLAS_COLLECTION_NAME", "{collection}")]
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection_obj,
        embedding=embeddings,
    )
    if chunks:
        vector_store.add_documents(chunks)
    return vector_store'''

        store_init = {
            "chroma": f'Chroma(collection_name="{collection}", embedding_function=embeddings, persist_directory="./chroma_db")',
            "pinecone": f'PineconeVectorStore(index_name="{collection}", embedding=embeddings)',
            "pgvector": f'PGVector(embeddings=embeddings, collection_name="{collection}", connection=os.getenv("PGVECTOR_CONNECTION_STRING", ""))',
            "milvus": f'Milvus(embedding_function=embeddings, collection_name="{collection}", connection_args={{"uri": os.getenv("MILVUS_URI", "http://localhost:19530")}})',
            "elasticsearch": f'ElasticsearchStore(index_name="{collection}", embedding=embeddings, es_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))',
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
                f'azure_search_key=os.getenv("AZURE_SEARCH_KEY", ""), '
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

        if strategy == "hybrid":
            return f'''# ─── Retriever ────────────────────────────────────────────────
def _extract_corpus_texts(vector_store):
    """Extract indexed texts for BM25 keyword retrieval."""
    texts = []
    get_fn = getattr(vector_store, "get", None)
    if callable(get_fn):
        try:
            result = get_fn()
            documents = result.get("documents", []) if isinstance(result, dict) else []
            texts.extend(doc for doc in documents if isinstance(doc, str) and doc.strip())
        except Exception:
            pass
    return texts


def build_retriever(vector_store):
    """Build hybrid retriever (BM25 + dense), top_k={top_k}, alpha={alpha}."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    dense = vector_store.as_retriever(search_kwargs={{"k": {top_k}}})
    texts = _extract_corpus_texts(vector_store)
    if not texts:
        return dense
    bm25 = BM25Retriever.from_texts(texts, k={top_k})
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[{1 - alpha}, {alpha}],
    )'''

        if strategy == "keyword_bm25":
            return f'''# ─── Retriever ────────────────────────────────────────────────
def build_retriever(vector_store):
    """Build BM25 retriever, top_k={top_k}."""
    from langchain_community.retrievers import BM25Retriever

    texts = []
    get_fn = getattr(vector_store, "get", None)
    if callable(get_fn):
        try:
            result = get_fn()
            documents = result.get("documents", []) if isinstance(result, dict) else []
            texts.extend(doc for doc in documents if isinstance(doc, str) and doc.strip())
        except Exception:
            pass
    if not texts:
        return vector_store.as_retriever(search_kwargs={{"k": {top_k}}})
    return BM25Retriever.from_texts(texts, k={top_k})'''

        if strategy == "mmr":
            return f'''# ─── Retriever ────────────────────────────────────────────────
def build_retriever(vector_store):
    """Build MMR retriever, top_k={top_k}, lambda={lam}."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={{"k": {top_k}, "lambda_mult": {lam}}},
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
        return f'''# ─── Reranking ────────────────────────────────────────────────
def rerank_documents(query: str, docs: list) -> list:
    """Rerank documents using {reranker} (top_k={top_k})."""
    # Reranker: {reranker}, model: {model_id}
    try:
        from flashrank import Ranker, RerankRequest
        ranker = Ranker()
        passages = [{{"id": i, "text": d.page_content}} for i, d in enumerate(docs)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        top_ids = [r["id"] for r in results[:{top_k}]]
        return [docs[i] for i in top_ids]
    except ImportError:
        return docs[:{top_k}]'''

    def _render_compressor_function(self, config: PipelineConfig) -> str:
        if not config.compression:
            return ""
        techniques = ", ".join(config.compression.techniques)
        threshold = config.compression.similarity_threshold
        return f'''# ─── Context Compression ──────────────────────────────────────
def compress_context(retriever, embeddings):
    """Apply context compression: {techniques}."""
    from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
    from langchain_classic.retrievers import ContextualCompressionRetriever
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold={threshold},
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )'''

    def _render_lcel_chain(self, config: PipelineConfig) -> str:
        provider = config.configured_providers[0] if config.configured_providers else "openai"
        llm_init = {
            "openai": 'ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)',
            "anthropic": 'ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY)',
            "cohere": 'ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)',
            "google_gemini": 'ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)',
            "mistral": 'ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)',
            "groq": 'ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)',
            "ollama": 'ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME", "llama3"), base_url=_ollama_base_url(), client_kwargs=_ollama_client_kwargs())',
        }.get(provider, 'ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)')

        return f'''# ─── RAG Chain (LCEL) ─────────────────────────────────────────
def build_rag_chain(retriever):
    """Assemble the RAG chain using LangChain Expression Language."""

    def format_docs(docs):
        return "\\n\\n".join(
            f"[Source: {{getattr(d, \'metadata\', {{}}).get(\'source\', f\'chunk_{{i}}\')}}]\\n{{d.page_content}}"
            for i, d in enumerate(docs)
        )

    llm = {llm_init}

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context passages:\\n\\n{{context}}\\n\\nQuestion: {{question}}"),
    ])

    # LangChain LCEL chain: retrieve → format → prompt → LLM → parse
    rag_chain = (
        {{"context": retriever | format_docs, "question": RunnablePassthrough()}}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain'''

    def _render_langgraph_workflow(self, config: PipelineConfig) -> str:
        rag_type = config.rag_type.rag_type if config.rag_type else "self_rag"
        provider = config.configured_providers[0] if config.configured_providers else "openai"
        llm_init = {
            "openai": 'ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)',
            "anthropic": 'ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY)',
            "ollama": 'ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME", "llama3"), base_url=_ollama_base_url(), client_kwargs=_ollama_client_kwargs())',
        }.get(provider, 'ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)')

        return f'''# ─── LangGraph Workflow ({rag_type}) ───────────────────────────
class GraphState(TypedDict):
    question: str
    generation: str
    documents: list
    rewrite_count: int


def build_rag_chain(retriever):
    """Build LangGraph agentic workflow for {rag_type}."""
    llm = {llm_init}

    def retrieve(state: GraphState) -> dict:
        return {{"documents": retriever.invoke(state["question"])}}

    def generate(state: GraphState) -> dict:
        context = "\\n\\n".join(d.page_content for d in state["documents"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\\n\\n{{context}}\\n\\nQuestion: {{{{question}}}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return {{"generation": chain.invoke({{"question": state["question"]}})}}

    def rewrite_query(state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question to be clearer."),
            ("human", "{{question}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return {{"question": chain.invoke({{"question": state["question"]}}), "rewrite_count": state.get("rewrite_count", 0) + 1}}

    def decide_to_generate(state: GraphState) -> str:
        return "generate" if state["documents"] or state.get("rewrite_count", 0) >= 2 else "rewrite_query"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges("retrieve", decide_to_generate, {{"generate": "generate", "rewrite_query": "rewrite_query"}})
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite_query", "retrieve")
    return workflow.compile()'''

    def _render_evaluation_setup(self, config: PipelineConfig) -> str:
        if not config.evaluation:
            return ""
        evaluators = config.evaluation.evaluators
        thresholds = config.evaluation.cicd_thresholds or {}
        lines = [
            "# ─── Evaluation Setup ─────────────────────────────────────────",
            f"# Enabled evaluators: {', '.join(evaluators)}",
        ]
        if "ragas" in evaluators:
            lines += [
                "",
                "def evaluate_with_ragas(query: str, answer: str, contexts: list) -> dict:",
                '    """Evaluate response using RAGAS metrics."""',
                "    try:",
                "        from ragas import evaluate",
                "        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision",
                "        from datasets import Dataset",
                "        dataset = Dataset.from_dict({",
                '            "question": [query],',
                '            "answer": [answer],',
                '            "contexts": [[c.page_content for c in contexts]],',
                "        })",
                "        result = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()])",
                "        return result.to_pandas().to_dict('records')[0]",
                "    except ImportError:",
                "        print('ragas not installed — skipping evaluation')",
                "        return {}",
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
        return "\n".join(lines)

    def _render_main(self, config: PipelineConfig) -> str:
        has_eval = config.evaluation_enabled and config.evaluation and "ragas" in (config.evaluation.evaluators if config.evaluation else [])
        eval_call = """
        # Evaluate response
        scores = evaluate_with_ragas(query, answer, context_docs)
        print(f"Evaluation scores: {scores}")""" if has_eval else ""

        return f'''# ─── Main Entry Point ─────────────────────────────────────────
def main():
    """MS_RAG pipeline entry point.

    Usage:
        python pipeline.py --ingest --sources doc1.pdf doc2.txt
        python pipeline.py --query "What is retrieval-augmented generation?"
    """
    parser = argparse.ArgumentParser(description="MS_RAG Generated Pipeline")
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
        print(f"Ingestion complete: {{len(chunks)}} chunks stored.")
    else:
        vector_store = init_vector_store()

    # Build retriever and RAG chain
    retriever = build_retriever(vector_store)
    rag_chain = build_rag_chain(retriever)

    if args.query:
        # Single query mode
        print(f"\\nQuery: {{args.query}}")
        answer = rag_chain.invoke(args.query)
        print(f"\\nAnswer: {{answer}}")
        context_docs = retriever.invoke(args.query){eval_call}
        return

    # Interactive query loop
    print("\\nMS_RAG Pipeline ready. Type your question (Ctrl+C to exit):\\n")
    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue
            answer = rag_chain.invoke(query)
            context_docs = retriever.invoke(query)
            print(f"\\nAnswer: {{answer}}\\n")
        except KeyboardInterrupt:
            print("\\nGoodbye.")
            break


if __name__ == "__main__":
    main()'''
