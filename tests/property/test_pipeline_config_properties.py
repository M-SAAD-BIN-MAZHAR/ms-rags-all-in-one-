"""Property-based tests for PipelineConfig serialization round-trip.

Property 21: Pipeline Config Serialization Round-Trip
    For any valid PipelineConfig, serializing to JSON and deserializing must
    produce a PipelineConfig that is structurally and semantically equivalent
    to the original, including the schema_version field.

Validates: Requirements 18.1, 18.4
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.models import (
    ChunkingConfig,
    CompressionConfig,
    EmbeddingModelConfig,
    EvaluationConfig,
    IngestionResult,
    KeywordStoreConfig,
    LLMModelConfig,
    MetadataField,
    PipelineConfig,
    RAGTypeConfig,
    RerankingConfig,
    RetrievalConfig,
    VectorDBConfig,
)
from ms_rag.utils.exceptions import SessionLoadError


# ---------------------------------------------------------------------------
# Hypothesis strategies for building valid sub-configs
# ---------------------------------------------------------------------------

RAG_TYPE_IDS = [
    "naive_rag", "advanced_rag", "modular_rag", "agentic_rag",
    "self_rag", "corrective_rag", "speculative_rag", "graphrag",
    "hyde_rag", "multi_query_rag", "rag_fusion", "step_back_rag",
    "parent_child_rag", "adaptive_rag", "contextual_compression_rag",
]

PROVIDER_IDS = [
    "openai", "anthropic", "cohere", "huggingface", "google_gemini",
    "mistral", "together_ai", "groq", "replicate", "azure_openai",
    "aws_bedrock", "ollama",
]

CHUNKING_STRATEGIES = [
    "recursive_character", "fixed_size", "semantic", "sentence",
    "paragraph", "token_based", "markdown_aware", "html_aware",
    "code_aware", "agentic", "document_aware",
]

RETRIEVAL_STRATEGIES = [
    "dense_vector", "keyword_bm25", "tfidf", "hybrid", "mmr",
    "ensemble", "parent_child", "multi_vector", "self_query", "time_weighted",
]

RERANKER_IDS = [
    "cross_encoder", "cohere_reranker", "bge_reranker",
    "llm_reranker", "colbert", "flashrank",
]

COMPRESSION_TECHNIQUES = [
    "llm_chain_extraction", "embeddings_filter", "document_compressor_pipeline",
    "redundancy_removal", "contextual_compression", "summary_compression",
]

EVALUATOR_IDS = [
    "ragas", "deepeval", "trulens", "langsmith", "langfuse",
    "arize_phoenix", "ares", "ragbench",
    "cicd_gate", "langgraph_trace", "monitoring_export",
]

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=50,
)


@st.composite
def rag_type_config_strategy(draw: st.DrawFn) -> RAGTypeConfig:
    rag_id = draw(st.sampled_from(RAG_TYPE_IDS))
    return RAGTypeConfig(
        rag_type=rag_id,
        display_name=draw(_safe_text),
        description=draw(_safe_text),
        requires_langgraph=rag_id in {"agentic_rag", "self_rag", "corrective_rag", "adaptive_rag"},
    )


@st.composite
def chunking_config_strategy(draw: st.DrawFn) -> ChunkingConfig:
    chunk_size = draw(st.integers(min_value=64, max_value=4096))
    overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return ChunkingConfig(
        strategy=draw(st.sampled_from(CHUNKING_STRATEGIES)),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


@st.composite
def embedding_model_strategy(draw: st.DrawFn) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        provider=draw(st.sampled_from(PROVIDER_IDS)),
        model_id=draw(_safe_text),
    )


@st.composite
def llm_model_strategy(draw: st.DrawFn, providers: list[str]) -> LLMModelConfig:
    provider = draw(st.sampled_from(providers or PROVIDER_IDS))
    return LLMModelConfig(
        provider=provider,
        model_id=draw(_safe_text),
    )


@st.composite
def vector_db_strategy(draw: st.DrawFn) -> VectorDBConfig:
    return VectorDBConfig(
        db_type=draw(st.sampled_from([
            "chroma", "pinecone", "weaviate", "qdrant", "faiss",
            "milvus", "redis", "pgvector", "elasticsearch", "opensearch",
            "azure_ai_search", "mongodb_atlas",
        ])),
        connection_params={},
        collection_name=draw(_safe_text),
    )


@st.composite
def retrieval_config_strategy(draw: st.DrawFn) -> RetrievalConfig:
    strategy = draw(st.sampled_from(RETRIEVAL_STRATEGIES))
    top_k = draw(st.integers(min_value=1, max_value=20))
    alpha = draw(st.floats(min_value=0.0, max_value=1.0)) if strategy == "hybrid" else None
    lam = draw(st.floats(min_value=0.0, max_value=1.0)) if strategy == "mmr" else None
    mf: list[MetadataField] | None = None
    if strategy == "self_query":
        mf = [MetadataField(name="source", data_type="string", description="document source")]
    ew: list[float] | None = None
    esr: list[str] | None = None
    if strategy == "ensemble":
        ew = [0.5, 0.5]
        esr = ["dense_vector", "keyword_bm25"]
    return RetrievalConfig(
        strategy=strategy,
        top_k=top_k,
        alpha=alpha,
        lambda_diversity=lam,
        metadata_fields=mf,
        ensemble_weights=ew,
        ensemble_sub_retrievers=esr,
    )


@st.composite
def reranking_config_strategy(draw: st.DrawFn) -> RerankingConfig:
    return RerankingConfig(
        reranker=draw(st.sampled_from(RERANKER_IDS)),
        model_id=draw(_safe_text),
        top_k=draw(st.integers(min_value=1, max_value=5)),
    )


@st.composite
def compression_config_strategy(draw: st.DrawFn) -> CompressionConfig:
    num = draw(st.integers(min_value=1, max_value=3))
    techniques = draw(
        st.lists(st.sampled_from(COMPRESSION_TECHNIQUES), min_size=num, max_size=num, unique=True)
    )
    return CompressionConfig(techniques=techniques)


@st.composite
def evaluation_config_strategy(draw: st.DrawFn) -> EvaluationConfig:
    num = draw(st.integers(min_value=1, max_value=3))
    evaluators = draw(
        st.lists(st.sampled_from(EVALUATOR_IDS), min_size=num, max_size=num, unique=True)
    )
    return EvaluationConfig(evaluators=evaluators)


@st.composite
def pipeline_config_strategy(draw: st.DrawFn) -> PipelineConfig:
    """Build a structurally valid PipelineConfig with arbitrary (but legal) values."""
    providers = draw(st.lists(st.sampled_from(PROVIDER_IDS), min_size=0, max_size=3, unique=True))
    include_rag = draw(st.booleans())
    include_llm_model = bool(providers) and draw(st.booleans())
    include_chunking = draw(st.booleans())
    include_embedding = draw(st.booleans())
    include_vector_db = draw(st.booleans())
    include_retrieval = draw(st.booleans())
    include_reranking = draw(st.booleans())
    include_compression = draw(st.booleans())
    include_evaluation = draw(st.booleans())

    return PipelineConfig(
        schema_version="1.0",
        configured_providers=providers,
        llm_model=draw(llm_model_strategy(providers)) if include_llm_model else None,
        rag_type=draw(rag_type_config_strategy()) if include_rag else None,
        document_types=draw(st.lists(
            st.sampled_from(["pdf", "txt", "docx", "csv", "html", "markdown"]),
            min_size=0, max_size=3, unique=True,
        )),
        loader_map=draw(st.dictionaries(
            st.sampled_from(["pdf", "txt", "docx"]),
            st.sampled_from(["PyPDFLoader", "TextLoader", "UnstructuredWordDocumentLoader"]),
            max_size=3,
        )),
        chunking=draw(chunking_config_strategy()) if include_chunking else None,
        embedding_model=draw(embedding_model_strategy()) if include_embedding else None,
        vector_db=draw(vector_db_strategy()) if include_vector_db else None,
        ingestion_result=None,
        document_sources=draw(st.lists(_safe_text, min_size=0, max_size=2)),
        query_enhancement=draw(st.lists(
            st.sampled_from(["query_rewriting", "hyde", "multi_query", "rag_fusion"]),
            min_size=0, max_size=2, unique=True,
        )),
        hyde_llm_provider=draw(st.sampled_from(PROVIDER_IDS)) if draw(st.booleans()) else None,
        retrieval=draw(retrieval_config_strategy()) if include_retrieval else None,
        reranking=draw(reranking_config_strategy()) if include_reranking else None,
        reranking_enabled=include_reranking,
        compression=draw(compression_config_strategy()) if include_compression else None,
        compression_enabled=include_compression,
        system_prompt=draw(st.text(max_size=200)),
        evaluation=draw(evaluation_config_strategy()) if include_evaluation else None,
        evaluation_enabled=include_evaluation,
    )


# ---------------------------------------------------------------------------
# Property 21: Pipeline Config Serialization Round-Trip
# ---------------------------------------------------------------------------


@given(config=pipeline_config_strategy())
@settings(max_examples=100)
def test_pipeline_config_round_trip(config: PipelineConfig) -> None:
    """Feature: ms-rag, Property 21: Pipeline Config Serialization Round-Trip.

    For any valid PipelineConfig, serializing to JSON and deserializing must
    produce a structurally and semantically equivalent PipelineConfig.
    """
    json_str = config.to_json()

    # Must be valid JSON
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)

    # schema_version must be present
    assert "schema_version" in parsed
    assert parsed["schema_version"] == "1.0"

    # Deserialize back
    restored = PipelineConfig.from_json(json_str)

    # Top-level scalar fields
    assert restored.schema_version == config.schema_version
    assert restored.configured_providers == config.configured_providers
    if config.llm_model is None:
        assert restored.llm_model is None
    else:
        assert restored.llm_model is not None
        assert restored.llm_model.provider == config.llm_model.provider
        assert restored.llm_model.model_id == config.llm_model.model_id
    assert restored.document_types == config.document_types
    assert restored.loader_map == config.loader_map
    assert restored.query_enhancement == config.query_enhancement
    assert restored.hyde_llm_provider == config.hyde_llm_provider
    assert restored.reranking_enabled == config.reranking_enabled
    assert restored.compression_enabled == config.compression_enabled
    assert restored.system_prompt == config.system_prompt
    assert restored.evaluation_enabled == config.evaluation_enabled
    assert restored.document_sources == config.document_sources

    # RAG type
    if config.rag_type is None:
        assert restored.rag_type is None
    else:
        assert restored.rag_type is not None
        assert restored.rag_type.rag_type == config.rag_type.rag_type
        assert restored.rag_type.requires_langgraph == config.rag_type.requires_langgraph

    # Chunking
    if config.chunking is None:
        assert restored.chunking is None
    else:
        assert restored.chunking is not None
        assert restored.chunking.strategy == config.chunking.strategy
        assert restored.chunking.chunk_size == config.chunking.chunk_size
        assert restored.chunking.chunk_overlap == config.chunking.chunk_overlap

    # Embedding model
    if config.embedding_model is None:
        assert restored.embedding_model is None
    else:
        assert restored.embedding_model is not None
        assert restored.embedding_model.provider == config.embedding_model.provider
        assert restored.embedding_model.model_id == config.embedding_model.model_id

    # Retrieval
    if config.retrieval is None:
        assert restored.retrieval is None
    else:
        assert restored.retrieval is not None
        assert restored.retrieval.strategy == config.retrieval.strategy
        assert restored.retrieval.top_k == config.retrieval.top_k

    # Keyword store
    if config.keyword_store is None:
        assert restored.keyword_store is None
    else:
        assert restored.keyword_store is not None
        assert restored.keyword_store.store_type == config.keyword_store.store_type
        assert restored.keyword_store.collection_name == config.keyword_store.collection_name

    # Reranking
    if config.reranking is None:
        assert restored.reranking is None
    else:
        assert restored.reranking is not None
        assert restored.reranking.reranker == config.reranking.reranker
        assert restored.reranking.model_id == config.reranking.model_id
        assert restored.reranking.top_k == config.reranking.top_k

    # Compression
    if config.compression is None:
        assert restored.compression is None
    else:
        assert restored.compression is not None
        assert restored.compression.techniques == config.compression.techniques

    # Evaluation
    if config.evaluation is None:
        assert restored.evaluation is None
    else:
        assert restored.evaluation is not None
        assert restored.evaluation.evaluators == config.evaluation.evaluators


def test_from_json_raises_on_empty_string() -> None:
    """from_json() must raise SessionLoadError for an empty string."""
    with pytest.raises(SessionLoadError):
        PipelineConfig.from_json("")


def test_from_json_raises_on_whitespace() -> None:
    """from_json() must raise SessionLoadError for whitespace-only input."""
    with pytest.raises(SessionLoadError):
        PipelineConfig.from_json("   \n  ")


def test_from_json_raises_on_malformed_json() -> None:
    """from_json() must raise SessionLoadError for malformed JSON."""
    with pytest.raises(SessionLoadError):
        PipelineConfig.from_json("{not valid json}")


def test_schema_version_preserved() -> None:
    """schema_version='1.0' must survive a round-trip."""
    config = PipelineConfig()
    restored = PipelineConfig.from_json(config.to_json())
    assert restored.schema_version == "1.0"


def test_empty_pipeline_config_round_trip() -> None:
    """An empty (default) PipelineConfig must survive a round-trip."""
    config = PipelineConfig()
    restored = PipelineConfig.from_json(config.to_json())
    assert restored.schema_version == config.schema_version
    assert restored.configured_providers == []
    assert restored.rag_type is None
    assert restored.system_prompt == ""


def test_keyword_store_secrets_are_sanitized_in_session_json() -> None:
    config = PipelineConfig(
        keyword_store=KeywordStoreConfig(
            store_type="postgres",
            connection_params={
                "KEYWORD_POSTGRES_CONNECTION_STRING": "postgresql://user:secret@example/db",
            },
            collection_name="chunks",
        )
    )

    data = json.loads(config.to_json())

    assert (
        data["keyword_store"]["connection_params"]["KEYWORD_POSTGRES_CONNECTION_STRING"]
        == "KEYWORD_POSTGRES_CONNECTION_STRING"
    )
