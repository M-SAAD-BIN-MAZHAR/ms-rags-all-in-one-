"""Tests for MS_RAG structured and terminal warning logging."""

from __future__ import annotations

import logging
import warnings
from unittest.mock import MagicMock

from ms_rag.utils.logging import JsonFormatter, install_warning_renderer


def test_json_formatter_includes_warning_context_fields() -> None:
    record = logging.LogRecord(
        name="ms_rag",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Retriever degraded",
        args=(),
        exc_info=None,
    )
    record.event = "runtime.warning"
    record.strategy = "hybrid"
    record.fallback = "dense_vector"
    record.reason = "missing_keyword_corpus"
    record.action = "reingest"

    rendered = JsonFormatter().format(record)

    assert '"event": "runtime.warning"' in rendered
    assert '"strategy": "hybrid"' in rendered
    assert '"fallback": "dense_vector"' in rendered
    assert '"reason": "missing_keyword_corpus"' in rendered
    assert '"action": "reingest"' in rendered


def test_warning_renderer_prints_runtime_notice_panel() -> None:
    original = warnings.showwarning
    if hasattr(install_warning_renderer, "_installed"):
        delattr(install_warning_renderer, "_installed")
    console = MagicMock()

    try:
        install_warning_renderer(console)
        warnings.warn("Hybrid retrieval selected but no keyword corpus texts were available.", stacklevel=1)
    finally:
        warnings.showwarning = original
        if hasattr(install_warning_renderer, "_installed"):
            delattr(install_warning_renderer, "_installed")

    console.print.assert_called()
