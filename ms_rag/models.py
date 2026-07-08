"""Shared dataclasses for MS_RAG.

PipelineConfig is the central accumulator that collects every user selection
across the 16-step workflow and is the sole input to the Code Generator.

CredentialStore and SessionState are runtime-only objects and are NEVER
serialised to JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


_SAFE_VECTOR_DB_CONFIG_FIELDS = frozenset(
    {
        "CHROMA_PERSIST_DIRECTORY",
        "CHROMA_PERSIST_DIR",
        "FAISS_INDEX_PATH",
        "QDRANT_PORT",
        "WEAVIATE_GRPC_PORT",
        "MONGODB_ATLAS_DB_NAME",
        "MONGODB_ATLAS_COLLECTION_NAME",
        "REDIS_INDEX_NAME",
        "PINECONE_INDEX_NAME",
        "PGVECTOR_COLLECTION_NAME",
        "AZURE_SEARCH_INDEX_NAME",
    }
)


def _is_sensitive_connection_field(field_name: str) -> bool:
    """Return True when a vector DB connection field may contain a secret."""
    upper = field_name.upper()
    if upper in _SAFE_VECTOR_DB_CONFIG_FIELDS:
        return False
    return any(
        token in upper
        for token in (
            "KEY",
            "SECRET",
            "TOKEN",
            "PASSWORD",
            "CONNECTION_STRING",
            "CLUSTER_URI",
            "REDIS_URL",
            "PGVECTOR",
            "URI",
            "URL",
            "ENDPOINT",
        )
    )


def _sanitized_vector_db_config(vector_db: "VectorDBConfig | None") -> dict[str, Any] | None:
    """Serialize vector DB config without leaking connection secrets."""
    if vector_db is None:
        return None
    data = asdict(vector_db)
    data["connection_params"] = {
        key: key if _is_sensitive_connection_field(key) else value
        for key, value in vector_db.connection_params.items()
    }
    return data


# ---------------------------------------------------------------------------
# Supporting config types
# ---------------------------------------------------------------------------


@dataclass
class RAGTypeConfig:
    """Represents a selected RAG architecture variant."""

    rag_type: str            # e.g. "self_rag"
    display_name: str        # e.g. "Self-RAG"
    description: str         # 2-4 sentence explanation shown to user
    requires_langgraph: bool # True for agentic_rag, self_rag, corrective_rag, adaptive_rag


@dataclass
class ChunkingConfig:
    """Parameters for the selected chunking strategy."""

    strategy: str                        # e.g. "recursive_character"
    chunk_size: int                      # in tokens or characters depending on strategy
    chunk_overlap: int                   # must be < chunk_size
    separators: list[str] | None = None  # for recursive_character strategy
    tokenizer: str | None = None         # for token_based strategy (tiktoken model or HF tokenizer)
    language: str | None = None          # for code_aware strategy (e.g. "python", "javascript")


@dataclass
class EmbeddingModelConfig:
    """Identifies the selected embedding model."""

    provider: str            # e.g. "openai", "huggingface", "ollama"
    model_id: str            # e.g. "text-embedding-3-large", "BAAI/bge-m3"
    local_path: str | None = None  # for local/Ollama models — HF model ID or filesystem path


@dataclass
class LLMModelConfig:
    """Identifies the selected generation model for answering queries."""

    provider: str            # e.g. "openai", "huggingface", "ollama"
    model_id: str            # e.g. "gpt-4o", "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass
class MetadataField:
    """A single metadata field definition for Self-Query retrieval."""

    name: str        # e.g. "source", "year", "author"
    data_type: str   # "string" | "integer" | "float" | "date"
    description: str # brief description shown to Self-Query LLM


@dataclass
class VectorDBConfig:
    """Connection parameters for the selected vector database."""

    db_type: str                        # e.g. "chroma", "pinecone", "qdrant"
    connection_params: dict[str, str]   # credential fields specific to the DB type
    collection_name: str                # index / collection / namespace name
    dimension: int | None = None        # embedding dimension (auto-detected if None)


@dataclass
class KeywordStoreConfig:
    """Persistent text/keyword store used by hybrid and keyword retrieval."""

    store_type: str                     # sqlite | postgres | elasticsearch | opensearch | memory
    connection_params: dict[str, str]
    collection_name: str = "ms_rag_keywords"


@dataclass
class GraphStoreConfig:
    """Persistent graph store used by full GraphRAG."""

    store_type: str                     # local_json | neo4j | kuzu
    connection_params: dict[str, str]
    graph_name: str = "ms_rag_graph"
    query_mode: str = "hybrid"          # local | global | hybrid


@dataclass
class RetrievalConfig:
    """Parameters for the selected retrieval strategy."""

    strategy: str                                    # e.g. "hybrid", "dense_vector", "mmr"
    top_k: int = 5                                   # number of chunks to retrieve (1-1000)
    alpha: float | None = None                       # hybrid: vector weight (0.0-1.0)
    lambda_diversity: float | None = None            # mmr: diversity param (0.0-1.0)
    metadata_fields: list[MetadataField] | None = None        # self_query: field definitions
    ensemble_weights: list[float] | None = None               # ensemble: weights per sub-retriever
    ensemble_sub_retrievers: list[str] | None = None          # ensemble: sub-retriever strategy IDs


@dataclass
class RerankingConfig:
    """Configuration for the selected reranker."""

    reranker: str    # e.g. "cohere_reranker", "cross_encoder", "bge_reranker", "flashrank"
    model_id: str    # cloud model name OR HuggingFace model ID OR local path
    top_k: int = 3   # chunks to keep after reranking (must be <= retrieval top_k)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""

    # Ordered list of technique IDs — applied in checklist order
    # Valid values: "llm_chain_extraction", "embeddings_filter",
    # "document_compressor_pipeline", "redundancy_removal",
    # "contextual_compression", "summary_compression"
    techniques: list[str]
    similarity_threshold: float = 0.75  # for embeddings_filter (0.0-1.0)


@dataclass
class EvaluationConfig:
    """Configuration for selected evaluation frameworks."""

    evaluators: list[str]  # e.g. ["ragas", "deepeval", "langsmith"]
    # Metric thresholds for CI/CD gate evaluation (metric_name -> min_score)
    cicd_thresholds: dict[str, float] | None = None
    evaluator_llm_provider: str | None = None
    evaluator_llm_model: str | None = None


@dataclass
class AgentToolConfig:
    """Permission-gated external tools for Agentic RAG."""

    enabled_tools: list[str] = field(default_factory=list)
    tool_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of a completed ingestion run."""

    chunk_count: int
    collection_name: str
    failed_documents: list[tuple[str, str]] = field(default_factory=list)
    # Each tuple: (document_path_or_url, error_message)


@dataclass
class GeneratedCode:
    """Output of the Code Generator."""

    python_code: str      # complete pipeline.py content
    requirements_txt: str # requirements.txt content (also embedded as comment at top of python_code)
    rag_type: str         # e.g. "naive_rag" — used for display and file naming
    env_txt: str = ""     # .env content with selected runtime variables, values intentionally blank


# ---------------------------------------------------------------------------
# PipelineConfig — central accumulator
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Accumulates all user selections across the 16-step workflow.

    This is the SOLE input to the Code Generator.
    It is serialisable to/from JSON via to_json() / from_json().
    CredentialStore is intentionally NOT included here.
    """

    schema_version: str = "1.0"

    # Step 2 — LLM Providers
    configured_providers: list[str] = field(default_factory=list)
    llm_model: LLMModelConfig | None = None

    # Step 3 — RAG Architecture
    rag_type: RAGTypeConfig | None = None

    # Step 4 — Document Types
    document_types: list[str] = field(default_factory=list)

    # Step 5 — Loaders  {doc_type_id: loader_class_name}
    loader_map: dict[str, str] = field(default_factory=dict)

    # Steps 6-7 — Chunking
    chunking: ChunkingConfig | None = None

    # Step 8 — Embedding
    embedding_model: EmbeddingModelConfig | None = None

    # Step 9 — Vector DB + Ingestion
    vector_db: VectorDBConfig | None = None
    ingestion_result: IngestionResult | None = None

    # Document sources (file paths, directories, URLs)
    document_sources: list[str] = field(default_factory=list)

    # Step 10 — Query Enhancement
    query_enhancement: list[str] = field(default_factory=list)
    hyde_llm_provider: str | None = None  # which provider to use for HyDE

    # Step 11 — Retrieval
    retrieval: RetrievalConfig | None = None
    keyword_store: KeywordStoreConfig | None = None
    graph_store: GraphStoreConfig | None = None

    # Step 12 — Reranking
    reranking: RerankingConfig | None = None
    reranking_enabled: bool = False

    # Step 13 — Context Compression
    compression: CompressionConfig | None = None
    compression_enabled: bool = False

    # Step 14 — System Prompt
    system_prompt: str = ""

    # Step 15 — Evaluation
    evaluation: EvaluationConfig | None = None
    evaluation_enabled: bool = False

    # Agentic RAG tools — only configured when the user explicitly opts in.
    agent_tools: AgentToolConfig | None = None

    # ---------------------------------------------------------------------------
    # Serialisation helpers
    # ---------------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise PipelineConfig to a JSON string.

        The output always includes ``schema_version`` for forward compatibility.
        CredentialStore is intentionally excluded — never persist credentials here.

        Returns:
            A formatted JSON string.
        """
        data = asdict(self)
        data["vector_db"] = _sanitized_vector_db_config(self.vector_db)
        if self.keyword_store is not None:
            keyword_store_data = asdict(self.keyword_store)
            keyword_store_data["connection_params"] = {
                key: key if _is_sensitive_connection_field(key) else value
                for key, value in self.keyword_store.connection_params.items()
            }
            data["keyword_store"] = keyword_store_data
        if self.graph_store is not None:
            graph_store_data = asdict(self.graph_store)
            graph_store_data["connection_params"] = {
                key: key if _is_sensitive_connection_field(key) else value
                for key, value in self.graph_store.connection_params.items()
            }
            data["graph_store"] = graph_store_data
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineConfig":
        """Deserialise a PipelineConfig from a JSON string.

        Args:
            json_str: A JSON string previously produced by to_json().

        Returns:
            A reconstructed PipelineConfig.

        Raises:
            ValueError: If json_str is empty or not valid JSON.
            KeyError:   If required top-level keys are missing.
        """
        from ms_rag.utils.exceptions import SessionLoadError  # avoid circular import

        if not json_str or not json_str.strip():
            raise SessionLoadError("Config JSON string is empty.", file_path="<string>")

        try:
            data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise SessionLoadError(
                f"Config JSON is malformed: {exc}",
                file_path="<string>",
                original=exc,
            ) from exc

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Recursively reconstruct a PipelineConfig from a plain dict."""

        def _rag_type(d: dict | None) -> RAGTypeConfig | None:
            if d is None:
                return None
            return RAGTypeConfig(**d)

        def _chunking(d: dict | None) -> ChunkingConfig | None:
            if d is None:
                return None
            return ChunkingConfig(**d)

        def _embedding(d: dict | None) -> EmbeddingModelConfig | None:
            if d is None:
                return None
            return EmbeddingModelConfig(**d)

        def _llm_model(d: dict | None) -> LLMModelConfig | None:
            if d is None:
                return None
            return LLMModelConfig(**d)

        def _vector_db(d: dict | None) -> VectorDBConfig | None:
            if d is None:
                return None
            return VectorDBConfig(**d)

        def _keyword_store(d: dict | None) -> KeywordStoreConfig | None:
            if d is None:
                return None
            return KeywordStoreConfig(**d)

        def _graph_store(d: dict | None) -> GraphStoreConfig | None:
            if d is None:
                return None
            return GraphStoreConfig(**d)

        def _retrieval(d: dict | None) -> RetrievalConfig | None:
            if d is None:
                return None
            mf = d.pop("metadata_fields", None)
            config = RetrievalConfig(**d)
            if mf:
                config.metadata_fields = [MetadataField(**f) for f in mf]
            return config

        def _reranking(d: dict | None) -> RerankingConfig | None:
            if d is None:
                return None
            return RerankingConfig(**d)

        def _compression(d: dict | None) -> CompressionConfig | None:
            if d is None:
                return None
            return CompressionConfig(**d)

        def _evaluation(d: dict | None) -> EvaluationConfig | None:
            if d is None:
                return None
            return EvaluationConfig(**d)

        def _agent_tools(d: dict | None) -> AgentToolConfig | None:
            if d is None:
                return None
            return AgentToolConfig(**d)

        def _ingestion_result(d: dict | None) -> IngestionResult | None:
            if d is None:
                return None
            fd = d.get("failed_documents", [])
            return IngestionResult(
                chunk_count=d["chunk_count"],
                collection_name=d["collection_name"],
                failed_documents=[tuple(item) for item in fd],  # type: ignore[misc]
            )

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            configured_providers=data.get("configured_providers", []),
            llm_model=_llm_model(data.get("llm_model")),
            rag_type=_rag_type(data.get("rag_type")),
            document_types=data.get("document_types", []),
            loader_map=data.get("loader_map", {}),
            chunking=_chunking(data.get("chunking")),
            embedding_model=_embedding(data.get("embedding_model")),
            vector_db=_vector_db(data.get("vector_db")),
            ingestion_result=_ingestion_result(data.get("ingestion_result")),
            document_sources=data.get("document_sources", []),
            query_enhancement=data.get("query_enhancement", []),
            hyde_llm_provider=data.get("hyde_llm_provider"),
            retrieval=_retrieval(data.get("retrieval")),
            keyword_store=_keyword_store(data.get("keyword_store")),
            graph_store=_graph_store(data.get("graph_store")),
            reranking=_reranking(data.get("reranking")),
            reranking_enabled=data.get("reranking_enabled", False),
            compression=_compression(data.get("compression")),
            compression_enabled=data.get("compression_enabled", False),
            system_prompt=data.get("system_prompt", ""),
            evaluation=_evaluation(data.get("evaluation")),
            evaluation_enabled=data.get("evaluation_enabled", False),
            agent_tools=_agent_tools(data.get("agent_tools")),
        )


# ---------------------------------------------------------------------------
# CredentialStore — runtime only, never serialised
# ---------------------------------------------------------------------------


@dataclass
class CredentialStore:
    """In-memory credential store.

    NEVER serialised to PipelineConfig JSON. Credentials are stored
    only for the duration of the Session (and optionally persisted
    encrypted to a separate file by CredentialManager).

    Internal structure:
        { provider_id: { field_name: field_value } }

    Example:
        { "openai": { "OPENAI_API_KEY": "sk-...", "OPENAI_ORG_ID": "org-..." } }
    """

    _store: dict[str, dict[str, str]] = field(default_factory=dict)

    def set(self, provider_id: str, field: str, value: str) -> None:
        """Store a credential value."""
        if provider_id not in self._store:
            self._store[provider_id] = {}
        self._store[provider_id][field] = value

    def get(self, provider_id: str, field: str) -> str | None:
        """Retrieve a credential value, or None if not set."""
        return self._store.get(provider_id, {}).get(field)

    def has_provider(self, provider_id: str) -> bool:
        """Return True if any credentials exist for the given provider."""
        return bool(self._store.get(provider_id))

    def has_field(self, provider_id: str, field: str) -> bool:
        """Return True if a specific credential field has been set."""
        return field in self._store.get(provider_id, {})

    def env_var_names(self, provider_id: str) -> list[str]:
        """Return the list of field names (used as os.getenv keys in generated code)."""
        return list(self._store.get(provider_id, {}).keys())

    def all_providers(self) -> list[str]:
        """Return IDs of all providers that have at least one credential set."""
        return [pid for pid, fields in self._store.items() if fields]

    def summary(self) -> dict[str, list[str]]:
        """Return a dict mapping provider_id to list of configured field names.

        Values are NOT included — only key names, suitable for display.
        """
        return {pid: list(fields.keys()) for pid, fields in self._store.items() if fields}

    def clear(self) -> None:
        """Remove all stored credentials (e.g. when user re-edits provider selection)."""
        self._store.clear()


# ---------------------------------------------------------------------------
# SessionState — runtime container, never serialised
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Runtime container for an active MS_RAG session.

    Holds both the accumulating PipelineConfig and all non-serialisable
    runtime resources (vector store, retriever, LLM, chain).

    NEVER serialised — use SessionManager.save(session.config, path)
    to persist only the PipelineConfig portion.
    """

    config: PipelineConfig
    credentials: CredentialStore

    # Populated progressively as the user completes each step
    vector_store: Any = None      # LangChain VectorStore instance
    retriever: Any = None         # LangChain BaseRetriever instance
    llm: Any = None               # LangChain BaseLLM / BaseChatModel instance
    rag_chain: Any = None         # RunnableSequence (LCEL) or CompiledGraph (LangGraph)
    compression_active: bool = False  # True when a compression layer wraps the retriever
    agent_runtime: Any = None     # Shared AgentToolRuntime — keeps short-term memory alive across turns

    current_step: int = 1
    query_history: list[tuple[str, str]] = field(default_factory=list)
    # Each entry: (query_text, answer_text) — capped at 100 entries in
    # query_loop.py to prevent unbounded memory growth in long sessions.
    last_enhanced_queries: list[str] = field(default_factory=list)
    last_primary_retrieval_query: str = ""
    last_retrieved_context_count: int = 0
    last_retrieved_context_preview: list[str] = field(default_factory=list)
    last_evaluation_scores: dict[str, float] = field(default_factory=dict)
    last_evaluation_warning: str = ""
    last_rag_trace: list[str] = field(default_factory=list)
