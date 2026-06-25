"""Property-based tests for QueryLoop.

Properties covered:
    Property 19: Query Length Acceptance (Req 10.2)
    Property 20: Exit Command Confirmation Invariant (Req 10.4)

Validates: Requirements 10.1-10.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.cli.query_loop import (
    MAX_QUERY_LENGTH,
    VALID_COMMANDS,
    QueryLoop,
)
from ms_rag.models import (
    CredentialStore,
    PipelineConfig,
    SessionState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> SessionState:
    return SessionState(
        config=PipelineConfig(),
        credentials=CredentialStore(),
    )


# ---------------------------------------------------------------------------
# Property 19: Query Length Acceptance
# ---------------------------------------------------------------------------


@given(query=st.text(min_size=1, max_size=MAX_QUERY_LENGTH))
@settings(max_examples=200)
def test_valid_query_length_is_accepted(query: str) -> None:
    """Feature: ms-rag, Property 19 (positive): queries 1-4096 chars are accepted."""
    loop = QueryLoop()
    stripped = query.strip()
    if not stripped:
        # Whitespace-only strings are "empty" — skip
        return
    if stripped.startswith("/"):
        # Slash commands are not natural language queries — skip
        return

    classification = loop._classify_input(query)
    assert classification == "query", (
        f"Query of length {len(query)} should be classified as 'query', "
        f"got {classification!r}"
    )


def test_over_limit_query_is_rejected() -> None:
    """Feature: ms-rag, Property 19 (negative): queries > 4096 chars are rejected."""
    loop = QueryLoop()
    # Boundary: exactly 4097 characters
    over_limit = "A" * (MAX_QUERY_LENGTH + 1)
    assert loop._classify_input(over_limit) == "too_long"

    # Well over the limit
    very_long = "B" * 10_000
    assert loop._classify_input(very_long) == "too_long"


def test_query_exactly_at_limit_is_accepted() -> None:
    loop = QueryLoop()
    query = "x" * MAX_QUERY_LENGTH
    assert loop._classify_input(query) == "query"


def test_query_one_over_limit_is_rejected() -> None:
    loop = QueryLoop()
    query = "x" * (MAX_QUERY_LENGTH + 1)
    assert loop._classify_input(query) == "too_long"


# ---------------------------------------------------------------------------
# Empty / whitespace query tests (Req 10.3)
# ---------------------------------------------------------------------------


@given(whitespace=st.text(alphabet=" \t\n\r", min_size=0, max_size=20))
@settings(max_examples=50)
def test_empty_and_whitespace_queries_classified_as_empty(whitespace: str) -> None:
    """Empty or whitespace-only input must be classified as 'empty'."""
    loop = QueryLoop()
    assert loop._classify_input(whitespace) == "empty"


def test_empty_string_is_empty() -> None:
    loop = QueryLoop()
    assert loop._classify_input("") == "empty"


# ---------------------------------------------------------------------------
# Property 20: Exit Command Confirmation Invariant
# ---------------------------------------------------------------------------


@given(cmd=st.sampled_from(["/exit", "/EXIT", "/Exit", "/quit", "/QUIT", "/Quit"]))
@settings(max_examples=12)
def test_exit_command_classified_correctly(cmd: str) -> None:
    """Feature: ms-rag, Property 20: /exit and /quit are always classified correctly."""
    loop = QueryLoop()
    result = loop._classify_input(cmd)
    assert result in ("exit", "quit"), (
        f"Command {cmd!r} should be 'exit' or 'quit', got {result!r}"
    )


@given(cmd=st.sampled_from(["/exit", "/quit"]))
@settings(max_examples=10)
def test_exit_command_always_shows_confirmation_prompt(cmd: str) -> None:
    """Requirement 10.4: _confirm_exit must always be called for /exit and /quit."""
    loop = QueryLoop()

    # Verify classification
    result = loop._classify_input(cmd)
    assert result in ("exit", "quit")

    # Verify that _confirm_exit is the path taken in the loop
    # by checking that confirm() is called when exit command is entered
    confirm_called = {"n": 0}

    original_confirm = loop._confirm_exit

    def patched_confirm(console: object) -> bool:
        confirm_called["n"] += 1
        return True  # confirm exit to terminate loop

    loop._confirm_exit = patched_confirm  # type: ignore[method-assign]

    session = _make_session()
    responses = iter([cmd])

    with patch("ms_rag.cli.query_loop.questionary") as mock_q, \
         patch("ms_rag.cli.query_loop.Console"), \
         patch("ms_rag.cli.query_loop.Panel"), \
         patch("ms_rag.cli.query_loop.Text"):

        mock_text = MagicMock()
        mock_text.ask.side_effect = lambda *a, **kw: next(responses, None)
        mock_q.text.return_value = mock_text
        mock_q.confirm = MagicMock()

        loop.run(session)

    assert confirm_called["n"] == 1, (
        f"_confirm_exit must be called exactly once for {cmd!r}"
    )


def test_confirm_exit_returns_true_on_yes() -> None:
    """Confirming exit returns True."""
    loop = QueryLoop()
    with patch("ms_rag.cli.query_loop.questionary") as mock_q:
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm
        assert loop._confirm_exit(console=MagicMock()) is True


def test_confirm_exit_returns_false_on_no() -> None:
    """Declining exit returns False."""
    loop = QueryLoop()
    with patch("ms_rag.cli.query_loop.questionary") as mock_q:
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_q.confirm.return_value = mock_confirm
        assert loop._confirm_exit(console=MagicMock()) is False


# ---------------------------------------------------------------------------
# Slash command classification (Req 10.5, 10.6)
# ---------------------------------------------------------------------------


class TestSlashCommandClassification:
    def test_config_command(self) -> None:
        loop = QueryLoop()
        assert loop._classify_input("/config") == "config"

    def test_save_command(self) -> None:
        loop = QueryLoop()
        assert loop._classify_input("/save") == "save"

    def test_unknown_slash_command(self) -> None:
        loop = QueryLoop()
        assert loop._classify_input("/unknown") == "unknown_command"
        assert loop._classify_input("/foo") == "unknown_command"
        assert loop._classify_input("/help") == "unknown_command"

    def test_valid_commands_list_is_complete(self) -> None:
        assert "/exit" in VALID_COMMANDS
        assert "/quit" in VALID_COMMANDS
        assert "/config" in VALID_COMMANDS
        assert "/save" in VALID_COMMANDS


# ---------------------------------------------------------------------------
# /config display test (Req 10.5)
# ---------------------------------------------------------------------------


def test_display_config_renders_all_components() -> None:
    """_display_config must show all required pipeline components."""
    from ms_rag.models import (
        RAGTypeConfig, ChunkingConfig, EmbeddingModelConfig,
        VectorDBConfig, RetrievalConfig,
    )

    session = _make_session()
    session.config.configured_providers = ["openai"]
    session.config.rag_type = RAGTypeConfig(
        rag_type="naive_rag", display_name="Naive RAG",
        description="test", requires_langgraph=False,
    )
    session.config.document_types = ["pdf"]
    session.config.chunking = ChunkingConfig(
        strategy="recursive_character", chunk_size=1000, chunk_overlap=200
    )
    session.config.embedding_model = EmbeddingModelConfig(
        provider="openai", model_id="text-embedding-3-small"
    )
    session.config.vector_db = VectorDBConfig(
        db_type="chroma", connection_params={}, collection_name="test_col"
    )
    session.config.retrieval = RetrievalConfig(strategy="dense_vector", top_k=5)

    loop = QueryLoop()
    mock_console = MagicMock()

    with patch("ms_rag.cli.query_loop.Table") as mock_table_cls:
        mock_table = MagicMock()
        mock_table_cls.return_value = mock_table
        loop._display_config(session, mock_console)

    # Table must have been printed
    mock_console.print.assert_called()
    # Multiple rows must have been added
    assert mock_table.add_row.call_count >= 10


# ---------------------------------------------------------------------------
# Query error recovery (Req 19.3)
# ---------------------------------------------------------------------------


def test_query_error_returns_to_prompt() -> None:
    """A query processing error must NOT terminate the session."""
    session = _make_session()
    call_count = {"queries": 0}

    def failing_pipeline(query: str, sess: object) -> str:
        call_count["queries"] += 1
        raise RuntimeError("LLM timeout")

    response_seq = iter(["valid query", "/quit"])

    def text_ask(*args, **kwargs) -> str | None:
        return next(response_seq, None)

    def confirm_ask(*args, **kwargs) -> bool:
        return True  # confirm exit on second call

    with patch("ms_rag.cli.query_loop.questionary") as mock_q, \
         patch("ms_rag.cli.query_loop.Console"), \
         patch("ms_rag.cli.query_loop.Panel"), \
         patch("ms_rag.cli.query_loop.Text"):

        mock_text = MagicMock()
        mock_text.ask.side_effect = text_ask
        mock_q.text.return_value = mock_text

        mock_confirm = MagicMock()
        mock_confirm.ask.side_effect = confirm_ask
        mock_q.confirm.return_value = mock_confirm

        loop = QueryLoop(query_pipeline=failing_pipeline)
        loop.run(session)  # should NOT raise

    # The query was attempted exactly once before /quit terminated the loop
    assert call_count["queries"] == 1
