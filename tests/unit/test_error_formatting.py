"""Tests for concise user-facing provider errors."""

from __future__ import annotations

from ms_rag.utils.error_formatting import format_provider_error


def test_removed_model_error_is_actionable() -> None:
    exc = ValueError(
        "status_code: 404, body: model 'command-r-plus' was removed on September 15, 2025."
    )

    message = format_provider_error(exc)

    assert "command-r-plus" in message
    assert "removed on September 15, 2025" in message
    assert "Choose a current model" in message


def test_task_mismatch_error_is_actionable() -> None:
    exc = ValueError(
        "Model meta-llama/Meta-Llama-3-8B-Instruct is not supported for task text-generation "
        "and provider together. Supported task: conversational."
    )

    message = format_provider_error(exc)

    assert "does not support this task" in message
    assert "Choose a model whose task matches" in message
