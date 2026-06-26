"""End-to-end integration tests for MS_RAG.

Test that all components work together to load, chunk, embed, and ingest documents,
execute queries via the LLM integration layer, and generate standalone code that compiles.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, AIMessage

from ms_rag.models import (
    ChunkingConfig,
    EmbeddingModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    VectorDBConfig,
    SessionState,
    CredentialStore,
)
from ms_rag.ingestion.ingestion_orchestrator import IngestionOrchestrator
from ms_rag.ingestion.vectordb_connector import VectorDBConnector
from ms_rag.query.retrieval_strategy import RetrievalStrategyModule
from ms_rag.llm.llm_integration import build_rag_chain, process_query
from ms_rag.codegen.code_generator import CodeGenerator


# ---------------------------------------------------------------------------
# Offline Mock Classes
# ---------------------------------------------------------------------------

class OfflineFakeEmbeddings(Embeddings):
    """Fake Embeddings implementation to test vectorization offline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Return a simple mock vector (dimension 128)
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


class OfflineFakeChatModel(SimpleChatModel):
    """Fake Chat Model implementation to test LLM generation offline."""

    def _call(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> str:
        return "This is a grounded mock response from the offline LLM."

    @property
    def _llm_type(self) -> str:
        return "offline_fake_chat_model"


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

def test_complete_rag_lifecycle_end_to_end() -> None:
    # Use temporary directories to avoid contaminating the workspace
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # 1. Create a dummy document source
        doc_source_dir = temp_dir / "sources"
        doc_source_dir.mkdir()
        doc_file = doc_source_dir / "doc1.txt"
        doc_file.write_text(
            "Retrieval-Augmented Generation (RAG) is a technique that coordinates "
            "external database retrieval with language model generation.",
            encoding="utf-8"
        )

        # 2. Define the pipeline config
        config = PipelineConfig(
            schema_version="1.0",
            configured_providers=["openai"],
            rag_type=RAGTypeConfig(
                rag_type="naive_rag",
                display_name="Naive RAG",
                description="Simple retrieve-and-generate pipeline.",
                requires_langgraph=False,
            ),
            document_types=["txt"],
            loader_map={"txt": "TextLoader"},
            chunking=ChunkingConfig(
                strategy="recursive_character",
                chunk_size=100,
                chunk_overlap=20,
            ),
            embedding_model=EmbeddingModelConfig(
                provider="openai",
                model_id="text-embedding-3-small",
            ),
            vector_db=VectorDBConfig(
                db_type="faiss",
                connection_params={"FAISS_INDEX_PATH": str(temp_dir / "faiss_index")},
                collection_name="test_collection",
            ),
            document_sources=[str(doc_file)],
            retrieval=RetrievalConfig(
                strategy="dense_vector",
                top_k=2,
            ),
            system_prompt="Answer only using context.",
        )

        # 3. Initialize offline embeddings & vector store
        embeddings = OfflineFakeEmbeddings()
        db_connector = VectorDBConnector()

        vector_store = db_connector.get_vector_store(config.vector_db, embeddings)

        # 4. Ingest documents through IngestionOrchestrator
        orchestrator = IngestionOrchestrator()
        ingestion_result = orchestrator.ingest(
            sources=config.document_sources,
            loader_map=config.loader_map,
            chunking_config=config.chunking,
            embedding_model=config.embedding_model,
            vector_db=config.vector_db,
            vector_store=vector_store,
        )

        assert ingestion_result.chunk_count > 0, "Should have successfully chunked and ingested document."
        assert len(ingestion_result.failed_documents) == 0, "No documents should have failed ingestion."

        # 5. Build retriever
        retrieval_module = RetrievalStrategyModule()
        retriever = retrieval_module.get_retriever(config.retrieval, vector_store)

        # 6. Build RAG chain and process query
        llm = OfflineFakeChatModel()
        rag_chain = build_rag_chain(retriever, llm, config.system_prompt)

        session = SessionState(
            config=config,
            credentials=CredentialStore(),
            vector_store=vector_store,
            retriever=retriever,
            llm=llm,
            rag_chain=rag_chain,
        )

        answer = process_query("What is RAG?", session)
        assert "mock response" in answer

        # 7. Generate pipeline script and requirements
        generator = CodeGenerator()
        code_result = generator.generate(config)

        # 8. Assert on code generation completeness
        assert len(code_result.python_code) > 0
        assert "def main():" in code_result.python_code
        assert "init_vector_store" in code_result.python_code

        # 9. Verify generated code is syntactically valid
        try:
            ast.parse(code_result.python_code)
        except SyntaxError as exc:
            pytest.fail(f"Generated python code contains syntax errors: {exc}")
