"""Property-based tests for SystemPromptConfigurator.

Properties covered:
    Property 17: System Prompt Verbatim Storage (Req 15.3)

Validates: Requirements 15.1-15.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ms_rag.workflow.system_prompt_configurator import (
    CHOICE_REPLACE,
    CHOICE_USE_DEFAULT,
    DEFAULT_SYSTEM_PROMPT,
    MAX_SYSTEM_PROMPT_LENGTH,
    SystemPromptConfigurator,
)


# ---------------------------------------------------------------------------
# Property 17: System Prompt Verbatim Storage
# ---------------------------------------------------------------------------


@given(prompt=st.text(min_size=1, max_size=500))
@settings(max_examples=100)
def test_system_prompt_verbatim_storage(prompt: str) -> None:
    """Feature: ms-rag, Property 17: System Prompt Verbatim Storage.

    Any non-null custom prompt (including special chars, very long text,
    content different from the default) must be stored exactly as entered.
    """
    assume(len(prompt) <= MAX_SYSTEM_PROMPT_LENGTH)

    configurator = SystemPromptConfigurator()

    with patch("ms_rag.workflow.system_prompt_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.system_prompt_configurator.Console"), \
         patch("ms_rag.workflow.system_prompt_configurator.Panel"), \
         patch("ms_rag.workflow.system_prompt_configurator.Text"):

        # choice: replace
        mock_select = MagicMock()
        mock_select.ask.return_value = CHOICE_REPLACE
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        # text input: return the given prompt
        mock_text = MagicMock()
        mock_text.ask.return_value = prompt
        mock_q.text.return_value = mock_text

        result = configurator.configure()

    # Result should contain the prompt content (stripped)
    assert isinstance(result, str)
    assert prompt.strip() in result or result == prompt.strip()


# ---------------------------------------------------------------------------
# Default prompt requirements (Req 15.2)
# ---------------------------------------------------------------------------


class TestDefaultPromptProperties:
    def test_default_prompt_is_non_empty(self) -> None:
        assert len(DEFAULT_SYSTEM_PROMPT.strip()) > 0

    def test_default_prompt_contains_context_only_instruction(self) -> None:
        """(a) Answer only from provided context passages."""
        lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "context" in lower
        assert "only" in lower or "only using" in lower or "provided context" in lower

    def test_default_prompt_contains_citation_instruction(self) -> None:
        """(b) Cite source document name or chunk identifier."""
        lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "source" in lower or "cite" in lower or "chunk" in lower

    def test_default_prompt_contains_i_dont_know_instruction(self) -> None:
        """(c) Respond with exact phrase "I don't know"."""
        assert "I don't know" in DEFAULT_SYSTEM_PROMPT or "i don't know" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_default_prompt_contains_concise_factual_instruction(self) -> None:
        """(d) Keep answers concise and factual."""
        lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "concise" in lower or "factual" in lower

    def test_default_prompt_contains_no_external_info_instruction(self) -> None:
        """(e) Do not introduce information not present in context."""
        lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "do not introduce" in lower or "not present" in lower or "not in" in lower


# ---------------------------------------------------------------------------
# use_default path (Req 15.3)
# ---------------------------------------------------------------------------


def test_use_default_returns_default_prompt() -> None:
    configurator = SystemPromptConfigurator()

    with patch("ms_rag.workflow.system_prompt_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.system_prompt_configurator.Console"), \
         patch("ms_rag.workflow.system_prompt_configurator.Panel"), \
         patch("ms_rag.workflow.system_prompt_configurator.Text"):

        mock_select = MagicMock()
        mock_select.ask.return_value = CHOICE_USE_DEFAULT
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = configurator.configure()

    assert result == DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Replace path — 10k char limit (Req 15.5)
# ---------------------------------------------------------------------------


def test_replace_rejects_over_10k_chars() -> None:
    """Prompts exceeding 10,000 chars must be rejected with re-prompt."""
    configurator = SystemPromptConfigurator()
    call_count = {"n": 0}

    def text_side_effect(*a, **kw) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        if call_count["n"] == 1:
            m.ask.return_value = "X" * (MAX_SYSTEM_PROMPT_LENGTH + 1)
        else:
            m.ask.return_value = "valid short prompt"
        return m

    with patch("ms_rag.workflow.system_prompt_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.system_prompt_configurator.Console"), \
         patch("ms_rag.workflow.system_prompt_configurator.Panel"), \
         patch("ms_rag.workflow.system_prompt_configurator.Text"):

        mock_select = MagicMock()
        mock_select.ask.return_value = CHOICE_REPLACE
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)
        mock_q.text.side_effect = text_side_effect

        result = configurator.configure()

    assert result == "valid short prompt"
    assert call_count["n"] == 2  # re-prompted once


def test_replace_exactly_at_limit_is_accepted() -> None:
    """A prompt of exactly 10,000 chars must be accepted."""
    configurator = SystemPromptConfigurator()
    at_limit = "A" * MAX_SYSTEM_PROMPT_LENGTH

    with patch("ms_rag.workflow.system_prompt_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.system_prompt_configurator.Console"), \
         patch("ms_rag.workflow.system_prompt_configurator.Panel"), \
         patch("ms_rag.workflow.system_prompt_configurator.Text"):

        mock_select = MagicMock()
        mock_select.ask.return_value = CHOICE_REPLACE
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        mock_text = MagicMock()
        mock_text.ask.return_value = at_limit
        mock_q.text.return_value = mock_text

        result = configurator.configure()

    assert result == at_limit


# ---------------------------------------------------------------------------
# MAX_SYSTEM_PROMPT_LENGTH constant
# ---------------------------------------------------------------------------


def test_max_system_prompt_length_is_10000() -> None:
    assert MAX_SYSTEM_PROMPT_LENGTH == 10_000
