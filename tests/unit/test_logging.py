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


def test_warning_renderer_suppresses_dependency_deprecation_panels() -> None:
    original = warnings.showwarning
    if hasattr(install_warning_renderer, "_installed"):
        delattr(install_warning_renderer, "_installed")
    console = MagicMock()

    try:
        install_warning_renderer(console)
        warnings.showwarning(
            "deprecated",
            DeprecationWarning,
            r"C:\project\.venv\Lib\site-packages\docling\pipeline\standard_pdf_pipeline.py",
            588,
        )
    finally:
        warnings.showwarning = original
        if hasattr(install_warning_renderer, "_installed"):
            delattr(install_warning_renderer, "_installed")

    console.print.assert_not_called()


def test_warning_renderer_still_prints_project_deprecation_panels() -> None:
    original = warnings.showwarning
    if hasattr(install_warning_renderer, "_installed"):
        delattr(install_warning_renderer, "_installed")
    console = MagicMock()

    try:
        install_warning_renderer(console)
        warnings.showwarning(
            "project deprecation",
            DeprecationWarning,
            r"C:\repo\ms_rag\query\retrieval_strategy.py",
            149,
        )
    finally:
        warnings.showwarning = original
        if hasattr(install_warning_renderer, "_installed"):
            delattr(install_warning_renderer, "_installed")

    console.print.assert_called()


def test_warning_renderer_suppresses_dependency_pydantic_deprecation_panels() -> None:
    class PydanticDeprecatedSince20(Warning):
        pass

    original = warnings.showwarning
    if hasattr(install_warning_renderer, "_installed"):
        delattr(install_warning_renderer, "_installed")
    console = MagicMock()

    try:
        install_warning_renderer(console)
        warnings.showwarning(
            "The `parse_obj` method is deprecated; use `model_validate` instead.",
            PydanticDeprecatedSince20,
            r"C:\project\.venv\Lib\site-packages\cohere\utils.py",
            236,
        )
    finally:
        warnings.showwarning = original
        if hasattr(install_warning_renderer, "_installed"):
            delattr(install_warning_renderer, "_installed")

    console.print.assert_not_called()
