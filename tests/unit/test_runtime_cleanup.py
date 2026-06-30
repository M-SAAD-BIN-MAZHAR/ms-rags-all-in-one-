"""Tests for runtime resource cleanup."""

from __future__ import annotations

from ms_rag.models import PipelineConfig, SessionState
from ms_rag.utils.runtime_cleanup import close_session_runtime


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _VectorStoreWithClient(_Closable):
    def __init__(self) -> None:
        super().__init__()
        self.client = _Closable()


def test_close_session_runtime_closes_vector_store_and_client() -> None:
    vector_store = _VectorStoreWithClient()
    session = SessionState(
        config=PipelineConfig(),
        credentials={},
        vector_store=vector_store,
        retriever=None,
        llm=None,
        rag_chain=None,
    )

    close_session_runtime(session)

    assert vector_store.closed is True
    assert vector_store.client.closed is True
