"""Document metadata sanitization for vector store compatibility.

Vector stores such as ChromaDB accept only scalar metadata values (str, int,
float, bool) or homogeneous lists of those scalars. Document loaders may attach
rich structures (e.g. hyperlink dicts from Word documents) that must be
serialized before upsert.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

_SCALAR_TYPES = (str, int, float, bool)
_RESERVED_METADATA_KEYS = {"text", "vector", "pk"}


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _safe_key(key: object) -> str:
    normalized = str(key)
    if normalized in _RESERVED_METADATA_KEYS:
        return f"ms_rag_metadata_{normalized}"
    return normalized


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Return a copy of *metadata* safe for ChromaDB and similar vector stores."""
    sanitized: dict[str, str | int | float | bool] = {}

    for raw_key, value in metadata.items():
        if value is None:
            continue

        key = _safe_key(raw_key)

        if isinstance(value, _SCALAR_TYPES):
            sanitized[key] = value
            continue

        if isinstance(value, list):
            if value and all(isinstance(item, _SCALAR_TYPES) for item in value):
                item_types = {type(item) for item in value}
                if len(item_types) == 1:
                    sanitized[key] = value  # type: ignore[assignment]
                    continue
            sanitized[key] = json.dumps(value, default=_json_default)
            continue

        if isinstance(value, (datetime, date)):
            sanitized[key] = value.isoformat()
            continue

        if isinstance(value, dict):
            sanitized[key] = json.dumps(value, default=_json_default)
            continue

        sanitized[key] = str(value)

    return sanitized


def sanitize_documents(docs: list[Any]) -> list[Any]:
    """Sanitize metadata on each LangChain Document in *docs* (in place)."""
    for doc in docs:
        doc.metadata = sanitize_metadata(dict(doc.metadata))
    return docs
