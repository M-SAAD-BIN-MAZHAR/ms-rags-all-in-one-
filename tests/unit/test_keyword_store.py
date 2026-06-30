"""Tests for persistent keyword store integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import pytest

from ms_rag.ingestion.keyword_store import KeywordStoreConnector, _is_secret, retrieval_needs_keyword_store
from ms_rag.models import CredentialStore, KeywordStoreConfig, RetrievalConfig


def test_retrieval_needs_keyword_store_for_hybrid_and_ensemble() -> None:
    assert retrieval_needs_keyword_store(RetrievalConfig(strategy="hybrid"))
    assert retrieval_needs_keyword_store(RetrievalConfig(strategy="keyword_bm25"))
    assert retrieval_needs_keyword_store(
        RetrievalConfig(strategy="ensemble", ensemble_sub_retrievers=["dense_vector", "tfidf"])
    )
    assert not retrieval_needs_keyword_store(RetrievalConfig(strategy="dense_vector"))


def test_sqlite_keyword_store_persists_and_loads_texts(tmp_path: Path) -> None:
    db_path = tmp_path / "keywords.sqlite"
    config = KeywordStoreConfig(
        store_type="sqlite",
        connection_params={"KEYWORD_SQLITE_PATH": str(db_path)},
        collection_name="chunks",
    )
    docs = [
        Document(page_content="Muhammad Saad is an AI engineer.", metadata={"ms_rag_child_id": "c1"}),
        Document(page_content="MS_RAG supports Pinecone hybrid retrieval.", metadata={"ms_rag_child_id": "c2"}),
    ]

    connector = KeywordStoreConnector()
    connector.test_connection(config)
    texts = connector.persist_documents(config, docs)

    assert db_path.exists()
    assert texts == [
        "Muhammad Saad is an AI engineer.",
        "MS_RAG supports Pinecone hybrid retrieval.",
    ]
    assert connector.load_texts(config) == texts


def test_keyword_sqlite_path_is_not_masked_as_secret() -> None:
    assert not _is_secret("KEYWORD_SQLITE_PATH")
    assert _is_secret("ELASTICSEARCH_API_KEY")
    assert _is_secret("KEYWORD_POSTGRES_CONNECTION_STRING")


def test_elasticsearch_keyword_store_uses_api_key_header_and_collection_filter() -> None:
    config = KeywordStoreConfig(
        store_type="elasticsearch",
        connection_params={
            "ELASTICSEARCH_URL": "https://example.es",
            "ELASTICSEARCH_API_KEY": "encoded-key",
        },
        collection_name="my collection",
    )
    connector = KeywordStoreConnector()
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with patch("requests.post", return_value=response) as mock_post:
        connector.persist_documents(config, [Document(page_content="first"), Document(page_content="second")])

    assert mock_post.call_args_list[0].kwargs["headers"] == {"Authorization": "ApiKey encoded-key"}
    assert mock_post.call_args_list[0].kwargs["auth"] is None
    assert "collection_name.keyword" in str(mock_post.call_args_list[0].kwargs["json"])
    assert mock_post.call_args_list[1].args[0] == "https://example.es/my_collection/_doc/0"


def test_opensearch_keyword_store_uses_basic_auth_when_loading() -> None:
    config = KeywordStoreConfig(
        store_type="opensearch",
        connection_params={
            "OPENSEARCH_URL": "https://example.os",
            "OPENSEARCH_USERNAME": "user",
            "OPENSEARCH_PASSWORD": "pass",
        },
        collection_name="chunks",
    )
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "hits": {"hits": [{"_source": {"text": "stored text"}}]},
    }

    with patch("requests.get", return_value=response) as mock_get:
        texts = KeywordStoreConnector().load_texts(config)

    assert texts == ["stored text"]
    assert mock_get.call_args.kwargs["auth"] == ("user", "pass")


def test_keyword_store_resolves_sanitized_secret_marker_from_store() -> None:
    store = CredentialStore()
    store.set("elasticsearch", "ELASTICSEARCH_API_KEY", "store-key")
    config = KeywordStoreConfig(
        store_type="elasticsearch",
        connection_params={
            "ELASTICSEARCH_URL": "https://example.es",
            "ELASTICSEARCH_API_KEY": "ELASTICSEARCH_API_KEY",
        },
        collection_name="chunks",
    )
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with patch("requests.post", return_value=response) as mock_post:
        KeywordStoreConnector(store).persist_documents(config, [Document(page_content="stored")])

    assert mock_post.call_args_list[0].kwargs["headers"] == {"Authorization": "ApiKey store-key"}


def test_postgres_keyword_store_rejects_unsafe_table_name() -> None:
    config = KeywordStoreConfig(
        store_type="postgres",
        connection_params={
            "KEYWORD_POSTGRES_CONNECTION_STRING": "postgresql://example",
            "KEYWORD_POSTGRES_TABLE": "keywords; DROP TABLE users",
        },
        collection_name="chunks",
    )

    with pytest.raises(ValueError, match="Unsafe SQL table name"):
        KeywordStoreConnector()._persist_postgres(config, ["text"])
