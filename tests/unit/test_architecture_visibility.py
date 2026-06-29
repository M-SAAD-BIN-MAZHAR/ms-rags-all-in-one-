"""Tests for selected architecture visibility rendering."""

from __future__ import annotations

from ms_rag.models import (
    ChunkingConfig,
    CompressionConfig,
    EmbeddingModelConfig,
    EvaluationConfig,
    LLMModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RerankingConfig,
    RetrievalConfig,
    VectorDBConfig,
)
from ms_rag.ui.architecture import build_architecture_flow_steps, build_architecture_flow_text, build_visibility_rows


def _config() -> PipelineConfig:
    return PipelineConfig(
        configured_providers=["openai", "cohere"],
        llm_model=LLMModelConfig(provider="openai", model_id="gpt-4o"),
        rag_type=RAGTypeConfig(
            rag_type="self_rag",
            display_name="Self-RAG",
            description="Test Self-RAG",
            requires_langgraph=True,
        ),
        document_types=["pdf", "txt"],
        loader_map={"pdf": "LlamaParse", "txt": "TextLoader"},
        chunking=ChunkingConfig(strategy="semantic", chunk_size=512, chunk_overlap=64),
        embedding_model=EmbeddingModelConfig(
            provider="openai",
            model_id="text-embedding-3-small",
        ),
        vector_db=VectorDBConfig(db_type="faiss", connection_params={}, collection_name="smoke_index", dimension=1536),
        query_enhancement=["query_rewriting"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        reranking_enabled=True,
        reranking=RerankingConfig(reranker="cohere_reranker", model_id="rerank-english-v3.0", top_k=3),
        compression_enabled=True,
        compression=CompressionConfig(techniques=["redundancy_removal"]),
        evaluation_enabled=True,
        evaluation=EvaluationConfig(evaluators=["deepeval"]),
        document_sources=["SmokeDocs"],
    )


def test_visibility_rows_include_each_major_selection() -> None:
    rows = dict(build_visibility_rows(_config()))

    assert rows["RAG architecture"] == "Self-RAG"
    assert rows["Generation model"] == "openai / gpt-4o"
    assert "pdf:LlamaParse" in rows["Loaders"]
    assert "text-embedding-3-small" in rows["Embedding model"]
    assert rows["Vector database"] == "faiss / smoke_index"
    assert rows["Reranking"] == "Enabled / cohere_reranker / top_k=3"
    assert rows["Compression"] == "Enabled / redundancy_removal"
    assert rows["Evaluation"] == "Enabled / deepeval"


def test_architecture_flow_uses_actual_selected_components() -> None:
    steps = build_architecture_flow_steps(_config())
    text = build_architecture_flow_text(_config(), width=88)

    assert any("LlamaParse" in step for step in steps)
    assert any("semantic" in step for step in steps)
    assert any("faiss:smoke_index" in step for step in steps)
    assert any("Self-RAG orchestration" in step for step in steps)
    assert "01. User documents" in text
    assert "\n" + (" " * 43) + "v\n" in text
    assert "Live query loop + visible traces + standalone code generation" in text
