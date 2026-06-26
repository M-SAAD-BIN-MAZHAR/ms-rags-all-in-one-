"""Unit tests for shared terminal prompt helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ms_rag.ui.prompts import (
    prompt_checkbox,
    prompt_document_sources,
    prompt_select,
    prompt_text,
)


class TestPromptText:
    def test_required_reprompts_on_empty(self) -> None:
        answers = iter(["", "hello"])

        with patch("ms_rag.ui.prompts.questionary") as mock_q:
            mock_field = MagicMock()
            mock_field.ask.side_effect = lambda: next(answers)
            mock_q.text.return_value = mock_field
            result = prompt_text("Name:", required=True, console=MagicMock())
        assert result == "hello"

    def test_cancel_reprompts(self) -> None:
        answers = iter([None, "value"])

        with patch("ms_rag.ui.prompts.questionary") as mock_q:
            mock_field = MagicMock()
            mock_field.ask.side_effect = lambda: next(answers)
            mock_q.text.return_value = mock_field
            result = prompt_text("Name:", required=True, console=MagicMock())
        assert result == "value"


class TestPromptSelect:
    def test_reprompts_on_cancel(self) -> None:
        with patch("ms_rag.ui.prompts.questionary") as mock_q:
            mock_select = MagicMock()
            mock_select.ask.side_effect = [None, "naive_rag"]
            mock_q.select.return_value = mock_select
            result = prompt_select("Pick:", [], console=MagicMock())
        assert result == "naive_rag"


class TestPromptCheckbox:
    def test_reprompts_when_too_few_selected(self) -> None:
        with patch("ms_rag.ui.prompts.questionary") as mock_q:
            mock_box = MagicMock()
            mock_box.ask.side_effect = [[], ["openai"]]
            mock_q.checkbox.return_value = mock_box
            result = prompt_checkbox("Pick:", [], min_selections=1, console=MagicMock())
        assert result == ["openai"]


class TestPromptDocumentSources:
    def test_requires_at_least_one_source(self) -> None:
        with patch("ms_rag.ui.prompts.prompt_text") as mock_text:
            mock_text.side_effect = ["", "./docs/sample.txt"]
            sources = prompt_document_sources(console=MagicMock())
        assert sources == ["./docs/sample.txt"]
        assert mock_text.call_count == 2
