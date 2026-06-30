"""Regression tests for CLI preset confirmation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ms_rag.cli.main import (
    _confirm_compression_preset,
    _confirm_query_enhancement_preset,
)
from ms_rag.models import CompressionConfig


def test_keep_query_enhancement_preset_does_not_crash() -> None:
    preset = SimpleNamespace(query_enhancement=["query_rewriting"])
    console = MagicMock()
    enhancer = MagicMock()

    with patch("ms_rag.cli.main.prompt_select", return_value="keep", create=True), \
         patch("ms_rag.ui.prompts.prompt_select", return_value="keep"):
        result = _confirm_query_enhancement_preset(
            preset,
            enhancer,
            ["huggingface"],
            console,
        )

    assert result == ["query_rewriting"]
    enhancer.configure.assert_not_called()


def test_keep_compression_preset_does_not_crash() -> None:
    compression = CompressionConfig(techniques=["contextual_compression"])
    preset = SimpleNamespace(compression=compression)
    console = MagicMock()
    compressor = MagicMock()

    with patch("ms_rag.ui.prompts.prompt_select", return_value="keep"):
        result = _confirm_compression_preset(
            preset,
            compressor,
            ["openai"],
            console,
        )

    assert result is compression
    compressor.configure.assert_not_called()
