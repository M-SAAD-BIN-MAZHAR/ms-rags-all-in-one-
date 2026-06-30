"""Runtime resource cleanup helpers.

Vector DB clients often keep sockets open behind LangChain vector stores. The
CLI owns those clients during a live session, so it should close them explicitly
when the query loop exits.
"""

from __future__ import annotations

from typing import Any


def close_session_runtime(session_state: Any) -> None:
    """Close best-effort runtime resources attached to a SessionState."""

    seen: set[int] = set()
    for obj in (
        getattr(session_state, "vector_store", None),
        getattr(session_state, "retriever", None),
        getattr(session_state, "rag_chain", None),
    ):
        _close_object_tree(obj, seen, depth=0)


def _close_object_tree(obj: Any, seen: set[int], *, depth: int) -> None:
    if obj is None or depth > 2:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass

    for attr in (
        "client",
        "_client",
        "_weaviate_client",
        "vectorstore",
        "vector_store",
        "_vectorstore",
        "_vector_store",
    ):
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        _close_object_tree(child, seen, depth=depth + 1)
