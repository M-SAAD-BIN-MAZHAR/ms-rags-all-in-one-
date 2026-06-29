"""Integration tests for rebuild_session_runtime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.llm.llm_integration import (
    build_session_runtime_from_vector_store,
    rebuild_session_runtime,
)
from ms_rag.models import (
    ChunkingConfig,
    CredentialStore,
    EmbeddingModelConfig,
    LLMModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    VectorDBConfig,
)


def _minimal_config(*, langgraph: bool = False) -> PipelineConfig:
    return PipelineConfig(
        configured_providers=["openai"],
        llm_model=LLMModelConfig(provider="openai", model_id="gpt-4o"),
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

        assert set(runtime.keys()) == {
            "vector_store",
            "retriever",
            "llm",
            "rag_chain",
            "compression_active",
        }
        assert runtime["vector_store"] is mock_store
        assert runtime["retriever"] is mock_retriever
        assert runtime["llm"] is mock_llm
        assert runtime["rag_chain"] is mock_chain
        assert runtime["compression_active"] is False
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

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    def test_builds_runtime_from_existing_vector_store(
        self,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        config = _minimal_config()
        existing_store = MagicMock(name="already_populated_vector_store")
        retriever = MagicMock(name="retriever")
        llm = MagicMock(name="llm")
        chain = MagicMock(name="rag_chain")
        mock_get_retriever.return_value = retriever
        mock_get_llm.return_value = llm
        mock_build_chain.return_value = chain

        runtime = build_session_runtime_from_vector_store(
            config,
            CredentialStore(),
            vector_store=existing_store,
            embeddings=MagicMock(name="embeddings"),
        )

        assert runtime["vector_store"] is existing_store
        assert runtime["retriever"] is retriever
        mock_get_retriever.assert_called_once()
        assert mock_get_retriever.call_args.args[1] is existing_store

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    def test_uses_selected_generation_model(
        self,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        config = _minimal_config()
        config.configured_providers = ["huggingface"]
        config.llm_model = LLMModelConfig(
            provider="huggingface",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        mock_get_retriever.return_value = MagicMock(name="retriever")
        mock_get_llm.return_value = MagicMock(name="llm")
        mock_build_chain.return_value = MagicMock(name="rag_chain")

        store = CredentialStore()

        build_session_runtime_from_vector_store(
            config,
            store,
            vector_store=MagicMock(name="vector_store"),
            embeddings=MagicMock(name="embeddings"),
        )

        mock_get_llm.assert_called_once()
        assert mock_get_llm.call_args.args[:2] == (
            "huggingface",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        assert mock_get_llm.call_args.kwargs["credential_store"] is store

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.ingestion_orchestrator.IngestionOrchestrator.build_keyword_corpus")
    def test_rebuilds_keyword_corpus_for_hybrid_when_backend_has_no_texts(
        self,
        mock_build_keyword_corpus: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        config = _minimal_config()
        config.document_sources = ["./docs/resume.docx"]
        config.loader_map = {"docx": "Docx2txtLoader"}
        config.retrieval = RetrievalConfig(strategy="hybrid", top_k=4, alpha=0.5)
        mock_build_keyword_corpus.return_value = ["chunk one", "chunk two"]
        mock_get_retriever.return_value = MagicMock(name="retriever")
        mock_get_llm.return_value = MagicMock(name="llm")
        mock_build_chain.return_value = MagicMock(name="rag_chain")
        vector_store = MagicMock(name="vector_store")
        vector_store._ms_rag_parent_documents = None
        vector_store._ms_rag_chunk_documents = None

        build_session_runtime_from_vector_store(
            config,
            CredentialStore(),
            vector_store=vector_store,
            embeddings=MagicMock(name="embeddings"),
        )

        mock_build_keyword_corpus.assert_called_once()
        assert getattr(vector_store, "_ms_rag_keyword_corpus") == ["chunk one", "chunk two"]
        assert mock_get_retriever.call_args.kwargs["corpus_texts"] == ["chunk one", "chunk two"]

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.keyword_store.KeywordStoreConnector.load_texts")
    def test_rebuild_loads_keyword_corpus_from_persistent_keyword_store(
        self,
        mock_load_texts: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        from ms_rag.models import KeywordStoreConfig

        config = _minimal_config()
        config.retrieval = RetrievalConfig(strategy="hybrid", top_k=4, alpha=0.5)
        config.keyword_store = KeywordStoreConfig(
            store_type="sqlite",
            connection_params={"KEYWORD_SQLITE_PATH": "./keywords.sqlite"},
            collection_name="chunks",
        )
        mock_load_texts.return_value = ["stored keyword chunk"]
        mock_get_retriever.return_value = MagicMock(name="retriever")
        mock_get_llm.return_value = MagicMock(name="llm")
        mock_build_chain.return_value = MagicMock(name="rag_chain")
        vector_store = MagicMock(name="vector_store")
        vector_store._ms_rag_keyword_corpus = None

        build_session_runtime_from_vector_store(
            config,
            CredentialStore(),
            vector_store=vector_store,
            embeddings=MagicMock(name="embeddings"),
        )

        mock_load_texts.assert_called_once()
        assert mock_get_retriever.call_args.kwargs["corpus_texts"] == ["stored keyword chunk"]

    @patch("ms_rag.llm.llm_integration.build_rag_chain")
    @patch("ms_rag.llm.llm_integration.get_llm")
    @patch("ms_rag.query.retrieval_strategy.RetrievalStrategyModule.get_retriever")
    @patch("ms_rag.ingestion.ingestion_orchestrator.IngestionOrchestrator.build_retrieval_state")
    def test_rebuilds_advanced_retrieval_state_for_parent_child(
        self,
        mock_build_retrieval_state: MagicMock,
        mock_get_retriever: MagicMock,
        mock_get_llm: MagicMock,
        mock_build_chain: MagicMock,
    ) -> None:
        config = _minimal_config()
        config.document_sources = ["./docs/resume.docx"]
        config.loader_map = {"docx": "Docx2txtLoader"}
        config.retrieval = RetrievalConfig(strategy="parent_child", top_k=4)
        parent_docs = {"parent-1": MagicMock(page_content="parent")}
        chunk_docs = [MagicMock(page_content="child")]
        mock_build_retrieval_state.return_value = {
            "parent_documents": parent_docs,
            "chunk_documents": chunk_docs,
        }
        mock_get_retriever.return_value = MagicMock(name="retriever")
        mock_get_llm.return_value = MagicMock(name="llm")
        mock_build_chain.return_value = MagicMock(name="rag_chain")
        vector_store = MagicMock(name="vector_store")
        vector_store._ms_rag_parent_documents = None
        vector_store._ms_rag_chunk_documents = None

        build_session_runtime_from_vector_store(
            config,
            CredentialStore(),
            vector_store=vector_store,
            embeddings=MagicMock(name="embeddings"),
        )

        mock_build_retrieval_state.assert_called_once()
        assert getattr(vector_store, "_ms_rag_parent_documents") == parent_docs
        assert getattr(vector_store, "_ms_rag_chunk_documents") == chunk_docs
        assert mock_get_retriever.call_args.kwargs["embeddings"] is not None
