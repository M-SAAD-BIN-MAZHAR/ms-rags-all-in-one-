"""Unit tests for VectorDBConnector.

Tests (Requirement 9.4, 9.5):
- test_connection returns success=True when probe succeeds.
- test_connection returns success=False with error message when probe fails.
- All 12 vector DB types are defined.
- Local DBs always return success.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.ingestion.vectordb_connector import (
    VECTOR_DBS,
    VECTOR_DB_IDS,
    VECTOR_DB_MAP,
    ConnectionResult,
    VectorDBConnector,
)
from ms_rag.models import EmbeddingModelConfig, VectorDBConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(db_type: str, collection: str = "test_collection") -> VectorDBConfig:
    return VectorDBConfig(
        db_type=db_type,
        connection_params={},
        collection_name=collection,
    )


# ---------------------------------------------------------------------------
# Connection test — success path (Requirement 9.4)
# ---------------------------------------------------------------------------


def test_connection_success_for_local_chroma() -> None:
    """ChromaDB local never fails probe — must return success."""
    connector = VectorDBConnector()
    result = connector.test_connection(_make_config("chroma"))
    assert result.success is True
    assert result.error_message is None


def test_connection_success_for_local_faiss() -> None:
    """FAISS local never fails probe — must return success."""
    connector = VectorDBConnector()
    result = connector.test_connection(_make_config("faiss"))
    assert result.success is True
    assert result.error_message is None


def test_connection_success_when_probe_does_not_raise() -> None:
    """Requirement 9.4: success=True when _probe_connection doesn't raise."""
    connector = VectorDBConnector()
    with patch.object(connector, "_probe_connection", return_value=None):
        result = connector.test_connection(_make_config("pinecone"))
    assert result.success is True
    assert result.error_message is None


# ---------------------------------------------------------------------------
# Connection test — failure path (Requirement 9.5)
# ---------------------------------------------------------------------------


def test_connection_failure_returns_success_false() -> None:
    """Requirement 9.5: success=False and error message when probe raises."""
    connector = VectorDBConnector()
    with patch.object(
        connector,
        "_probe_connection",
        side_effect=ConnectionError("Connection refused"),
    ):
        result = connector.test_connection(_make_config("pinecone"))

    assert result.success is False
    assert result.error_message is not None
    assert "Connection refused" in result.error_message


def test_connection_failure_on_auth_error() -> None:
    """Authentication errors must be caught and returned as failure."""
    connector = VectorDBConnector()
    with patch.object(
        connector,
        "_probe_connection",
        side_effect=Exception("401 Unauthorized"),
    ):
        result = connector.test_connection(_make_config("qdrant"))

    assert result.success is False
    assert "401" in result.error_message


def test_connection_failure_on_timeout() -> None:
    """Timeout errors must be caught and returned as failure."""
    connector = VectorDBConnector()
    with patch.object(
        connector,
        "_probe_connection",
        side_effect=TimeoutError("Connection timed out after 5s"),
    ):
        result = connector.test_connection(_make_config("elasticsearch"))

    assert result.success is False
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Structural completeness (Requirement 9.1)
# ---------------------------------------------------------------------------


class TestVectorDBListCompleteness:
    def test_exactly_12_vector_dbs_defined(self) -> None:
        assert len(VECTOR_DBS) == 12

    def test_all_required_dbs_present(self) -> None:
        required = {
            "chroma", "pinecone", "weaviate", "qdrant", "faiss",
            "milvus", "redis", "pgvector", "elasticsearch",
            "opensearch", "azure_ai_search", "mongodb_atlas",
        }
        defined = set(VECTOR_DB_IDS)
        missing = required - defined
        assert not missing, f"Missing vector DBs: {missing}"

    def test_no_duplicate_db_type_ids(self) -> None:
        assert len(VECTOR_DB_IDS) == len(set(VECTOR_DB_IDS))

    def test_vector_db_map_matches_list(self) -> None:
        assert set(VECTOR_DB_MAP.keys()) == set(VECTOR_DB_IDS)

    def test_all_dbs_have_display_names(self) -> None:
        for db in VECTOR_DBS:
            assert len(db.display_name.strip()) > 0

    def test_all_dbs_have_descriptions(self) -> None:
        for db in VECTOR_DBS:
            assert len(db.description.strip()) > 0

    def test_local_dbs_have_no_required_credentials(self) -> None:
        for db in VECTOR_DBS:
            if db.is_local:
                assert db.credential_fields == [], (
                    f"Local DB {db.db_type!r} should have no required credential fields"
                )

    def test_chroma_is_local(self) -> None:
        assert VECTOR_DB_MAP["chroma"].is_local is True

    def test_faiss_is_local(self) -> None:
        assert VECTOR_DB_MAP["faiss"].is_local is True

    def test_pinecone_requires_api_key(self) -> None:
        assert "PINECONE_API_KEY" in VECTOR_DB_MAP["pinecone"].credential_fields

    def test_pgvector_requires_connection_string(self) -> None:
        assert "PGVECTOR_CONNECTION_STRING" in VECTOR_DB_MAP["pgvector"].credential_fields

    def test_azure_requires_endpoint_and_key(self) -> None:
        fields = VECTOR_DB_MAP["azure_ai_search"].credential_fields
        assert "AZURE_SEARCH_ENDPOINT" in fields
        assert "AZURE_SEARCH_KEY" in fields


class TestGetVectorStoreDispatch:
    """Verify the factory raises ValueError for unknown DB types."""

    def test_unknown_db_type_raises_value_error(self) -> None:
        connector = VectorDBConnector()
        config = _make_config("nonexistent_db")
        with pytest.raises(ValueError, match="Unsupported vector DB type"):
            connector.get_vector_store(config, MagicMock())

    def test_known_db_types_dont_raise_value_error(self) -> None:
        """Known DB types raise ImportError (package missing) not ValueError."""
        connector = VectorDBConnector()
        for db_type in VECTOR_DB_IDS:
            config = _make_config(db_type)
            try:
                connector.get_vector_store(config, MagicMock())
            except (ImportError, Exception) as exc:
                # ImportError = package missing (OK), any other = connection error (OK)
                if isinstance(exc, ValueError):
                    if "Unsupported vector DB type" in str(exc):
                        pytest.fail(
                            f"DB type {db_type!r} raised ValueError unexpectedly: {exc}"
                        )

    def test_chroma_accepts_legacy_persist_dir_alias(self) -> None:
        connector = VectorDBConnector()
        config = VectorDBConfig(
            db_type="chroma",
            connection_params={"CHROMA_PERSIST_DIR": "./legacy_chroma"},
            collection_name="test_collection",
        )

        with patch("langchain_chroma.Chroma") as mock_chroma:
            connector.get_vector_store(config, MagicMock())

        assert mock_chroma.call_args is not None
        assert mock_chroma.call_args.kwargs["persist_directory"] == "./legacy_chroma"

    def test_faiss_gets_default_persistence_path(self) -> None:
        connector = VectorDBConnector()
        config = VectorDBConfig(
            db_type="faiss",
            connection_params={},
            collection_name="test_collection",
        )

        store = connector.get_vector_store(config, MagicMock())

        assert store is not None
        assert config.connection_params["FAISS_INDEX_PATH"].endswith(
            "faiss_indexes\\test_collection"
        ) or config.connection_params["FAISS_INDEX_PATH"].endswith(
            "faiss_indexes/test_collection"
        )

    def test_embedding_dimension_is_carried_into_vector_db_config(self) -> None:
        connector = VectorDBConnector()
        embedding = EmbeddingModelConfig(
            provider="openai",
            model_id="text-embedding-3-small",
        )

        with patch("ms_rag.ingestion.vectordb_connector.questionary") as mock_q, \
             patch("ms_rag.ui.prompts.questionary") as mock_prompt_q, \
             patch("ms_rag.ui.prompts.get_console") as mock_console:
            mock_q.Choice = MagicMock(side_effect=lambda title, value: value)
            mock_q.text.return_value.ask.return_value = ""
            mock_prompt_q.select.return_value.ask.return_value = "chroma"
            mock_prompt_q.text.return_value.ask.return_value = "test_collection"
            mock_prompt_q.confirm.return_value.ask.return_value = True
            mock_console.return_value = MagicMock()

            config = connector.prompt_and_configure(embedding)

        assert config.dimension == 1536


class TestIngestDocuments:
    def test_empty_docs_returns_zero(self) -> None:
        connector = VectorDBConnector()
        mock_store = MagicMock()
        result = connector.ingest_documents([], mock_store)
        assert result == 0
        mock_store.add_documents.assert_not_called()

    def test_documents_are_added_in_batches(self) -> None:
        connector = VectorDBConnector()
        mock_store = MagicMock()
        docs = [MagicMock() for _ in range(120)]

        with patch("ms_rag.ingestion.vectordb_connector.Progress") as mock_progress_cls:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.add_task.return_value = 0
            mock_progress_cls.return_value = mock_ctx

            result = connector._ingest_batch(docs, mock_store)

        # 120 docs in batches of 50 → 3 calls (50 + 50 + 20)
        assert mock_store.add_documents.call_count == 3
        assert result == 120

    def test_ingest_documents_returns_zero_for_empty_list(self) -> None:
        connector = VectorDBConnector()
        mock_store = MagicMock()
        result = connector.ingest_documents([], mock_store)
        assert result == 0
        mock_store.add_documents.assert_not_called()
