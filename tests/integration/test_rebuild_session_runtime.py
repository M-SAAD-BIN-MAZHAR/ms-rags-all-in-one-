"""Integration tests for rebuild_session_runtime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.llm.llm_integration import rebuild_session_runtime
from ms_rag.models import (
    ChunkingConfig,
    CredentialStore,
    EmbeddingModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    VectorDBConfig,
)


def _minimal_config(*, langgraph: bool = False) -> PipelineConfig:
    return PipelineConfig(
        configured_providers=["openai"],
        rag_type=RAGTypeConfig(
            rag_type="self_rag" if langgraph else "naive_rag",
            display_name="Self-RAG" if langgraph else "Naive RAG",
            description="Test RAG type for integration tests.",
            requires_langgraph=langgraph,
        ),
        chunking=ChunkingConfig(
            strategy="recursive_character",
            chunk_size=500,
            chunk_overlap=50,
        ),
        embedding_model=EmbeddingModelConfig(
            provider="openai",
            model_id="text-embedding-3-small",
        ),
        vector_db=VectorDBConfig(
            db_type="chroma",
            collection_name="test_collection",
            connection_params={"CHROMA_PERSIST_DIR": "./chroma_test"},
        ),
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=4),
        system_prompt="You are a helpful assistant.",
    )


class TestRebuildSessionRuntime:
    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.vectordb_connector.VectorDBConnector.get_vector_store")
    @patch("ms_rag.ingestion.vectorization_module.VectorizationModule.get_embeddings")
    def test_rebuilds_lcel_runtime(
        self,
        mock_embeddings: MagicMock,
        mock_vector_store: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        mock_embeddings.return_value = MagicMock(name="embeddings")
        mock_store = MagicMock(name="vector_store")
        mock_vector_store.return_value = mock_store
        mock_retriever = MagicMock(name="retriever")
        mock_get_retriever.return_value = mock_retriever
        mock_llm = MagicMock(name="llm")
        mock_get_llm.return_value = mock_llm
        mock_chain = MagicMock(name="rag_chain")
        mock_build_chain.return_value = mock_chain

        config = _minimal_config(langgraph=False)
        store = CredentialStore()

        runtime = rebuild_session_runtime(config, store)

        assert set(runtime.keys()) == {"vector_store", "retriever", "llm", "rag_chain"}
        assert runtime["vector_store"] is mock_store
        assert runtime["retriever"] is mock_retriever
        assert runtime["llm"] is mock_llm
        assert runtime["rag_chain"] is mock_chain
        mock_build_chain.assert_called_once()

    @patch("ms_rag.llm.llm_integration.build_langgraph_workflow")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.vectordb_connector.VectorDBConnector.get_vector_store")
    @patch("ms_rag.ingestion.vectorization_module.VectorizationModule.get_embeddings")
    def test_rebuilds_langgraph_runtime(
        self,
        mock_embeddings: MagicMock,
        mock_vector_store: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_graph: MagicMock,
    ) -> None:
        mock_embeddings.return_value = MagicMock(name="embeddings")
        mock_vector_store.return_value = MagicMock(name="vector_store")
        mock_get_retriever.return_value = MagicMock(name="retriever")
        mock_get_llm.return_value = MagicMock(name="llm")
        mock_graph = MagicMock(name="rag_graph")
        mock_build_graph.return_value = mock_graph

        config = _minimal_config(langgraph=True)
        runtime = rebuild_session_runtime(config, CredentialStore())

        assert runtime["rag_chain"] is mock_graph
        mock_build_graph.assert_called_once()

    def test_raises_when_config_incomplete(self) -> None:
        config = PipelineConfig(configured_providers=["openai"])
        with pytest.raises(ValueError, match="incomplete"):
            rebuild_session_runtime(config, CredentialStore())

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.build_retriever_stack")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.vectordb_connector.VectorDBConnector.get_vector_store")
    @patch("ms_rag.ingestion.vectorization_module.VectorizationModule.get_embeddings")
    def test_applies_reranking_and_compression_stack(
        self,
        mock_embeddings: MagicMock,
        mock_vector_store: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_stack: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        mock_embeddings.return_value = MagicMock()
        mock_vector_store.return_value = MagicMock()
        base_retriever = MagicMock(name="base_retriever")
        mock_get_retriever.return_value = base_retriever
        mock_get_llm.return_value = MagicMock()
        stacked = MagicMock(name="stacked_retriever")
        mock_build_stack.return_value = stacked
        mock_build_chain.return_value = MagicMock()

        config = _minimal_config()
        config.reranking_enabled = True
        config.compression_enabled = True
        from ms_rag.models import CompressionConfig, RerankingConfig  # noqa: PLC0415

        config.reranking = RerankingConfig(reranker="cross_encoder", model_id="cross-encoder", top_k=3)
        config.compression = CompressionConfig(
            techniques=["embeddings_filter"],
            similarity_threshold=0.5,
        )

        runtime = rebuild_session_runtime(config, CredentialStore())

        mock_build_stack.assert_called_once()
        assert runtime["retriever"] is stacked
