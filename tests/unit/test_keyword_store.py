"""Tests for persistent keyword store integration."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ms_rag.ingestion.keyword_store import KeywordStoreConnector, _is_secret, retrieval_needs_keyword_store
from ms_rag.models import KeywordStoreConfig, RetrievalConfig


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
