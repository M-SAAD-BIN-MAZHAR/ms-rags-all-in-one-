"""Opt-in smoke tests for real vector databases."""

from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ms_rag.ingestion.vectordb_connector import VectorDBConnector
from ms_rag.models import VectorDBConfig


def _enabled_backends() -> set[str]:
    raw = os.getenv("MS_RAG_SMOKE_VECTOR_DBS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _smoke_enabled(db_type: str) -> bool:
    return db_type in _enabled_backends()


class SmokeEmbeddings(Embeddings):
    """Deterministic local embeddings for smoke tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * 8 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * 8


def _offline_embeddings() -> object:
    return SmokeEmbeddings()


@pytest.mark.smoke
def test_chroma_smoke() -> None:
    if not _smoke_enabled("chroma"):
        pytest.skip("Set MS_RAG_SMOKE_VECTOR_DBS=chroma to enable.")

    connector = VectorDBConnector()
    config = VectorDBConfig(
        db_type="chroma",
        connection_params={"CHROMA_PERSIST_DIRECTORY": os.getenv("CHROMA_PERSIST_DIRECTORY", "./.smoke/chroma")},
        collection_name=f"smoke_{uuid.uuid4().hex}",
    )
    embeddings = _offline_embeddings()
    store = connector.get_vector_store(config, embeddings)
    connector.ingest_documents([Document(page_content="ms_rag smoke test")], store)
    assert store.as_retriever(search_kwargs={"k": 1}).invoke("smoke")


@pytest.mark.smoke
def test_faiss_smoke() -> None:
    if not _smoke_enabled("faiss"):
        pytest.skip("Set MS_RAG_SMOKE_VECTOR_DBS=faiss to enable.")

    connector = VectorDBConnector()
    config = VectorDBConfig(
        db_type="faiss",
        connection_params={"FAISS_INDEX_PATH": os.getenv("FAISS_INDEX_PATH", "./.smoke/faiss")},
        collection_name=f"smoke_{uuid.uuid4().hex}",
    )
    embeddings = _offline_embeddings()
    store = connector.get_vector_store(config, embeddings)
    connector.ingest_documents([Document(page_content="ms_rag smoke test")], store)
    assert store.as_retriever(search_kwargs={"k": 1}).invoke("smoke")


@pytest.mark.smoke
def test_qdrant_smoke() -> None:
    if not _smoke_enabled("qdrant"):
        pytest.skip("Set MS_RAG_SMOKE_VECTOR_DBS=qdrant to enable.")

    url = os.getenv("QDRANT_URL")
    if not url:
        pytest.skip("Set QDRANT_URL to enable qdrant smoke testing.")

    connector = VectorDBConnector()
    config = VectorDBConfig(
        db_type="qdrant",
        connection_params={
            "QDRANT_URL": url,
            "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
        },
        collection_name=f"smoke_{uuid.uuid4().hex}",
    )
    embeddings = _offline_embeddings()
    store = connector.get_vector_store(config, embeddings)
    connector.ingest_documents([Document(page_content="ms_rag smoke test")], store)
    assert store.as_retriever(search_kwargs={"k": 1}).invoke("smoke")
