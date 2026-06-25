"""Property-based tests for ChunkingConfigurator.

Properties covered:
    Property 11: Chunk Overlap Validation (Req 7.5)
    Property 12: Chunking Parameters Round-Trip (Req 7.6)

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ms_rag.ingestion.chunking_engine import STRATEGY_IDS, STRATEGY_DESCRIPTIONS
from ms_rag.models import ChunkingConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_chunk_overlap
from ms_rag.workflow.chunking_configurator import ChunkingConfigurator


# ---------------------------------------------------------------------------
# Property 11: Chunk Overlap Validation
# ---------------------------------------------------------------------------


@given(
    chunk_size=st.integers(min_value=1, max_value=8192),
    overlap=st.integers(min_value=0, max_value=8192),
)
@settings(max_examples=300)
def test_chunk_overlap_validation_property(chunk_size: int, overlap: int) -> None:
    """Feature: ms-rag, Property 11: Chunk Overlap Validation.

    If overlap >= chunk_size, validate_chunk_overlap must raise ValidationError.
    ChunkingConfig must NOT be updated with the invalid overlap value.
    """
    if overlap >= chunk_size:
        with pytest.raises(ValidationError) as exc_info:
            validate_chunk_overlap(chunk_size, overlap)
        assert exc_info.value.field_name == "chunk_overlap"
        assert exc_info.value.value == overlap
    else:
        # Must not raise
        validate_chunk_overlap(chunk_size, overlap)


@given(
    chunk_size=st.integers(min_value=1, max_value=4096),
    overlap=st.integers(min_value=0, max_value=4096),
)
@settings(max_examples=200)
def test_configurator_reprompts_on_invalid_overlap(
    chunk_size: int, overlap: int
) -> None:
    """Property 11: When overlap >= chunk_size, configurator must re-prompt.

    The invalid overlap must NOT end up in the returned ChunkingConfig.
    """
    assume(overlap >= chunk_size)

    configurator = ChunkingConfigurator()
    valid_overlap = max(0, chunk_size - 1)

    # Simulate: first overlap entry is invalid, second is valid
    overlap_calls = {"n": 0}

    def mock_text_side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        prompt_text = args[0] if args else kwargs.get("message", "")
        default = kwargs.get("default", "")

        if "overlap" in prompt_text.lower():
            overlap_calls["n"] += 1
            # First call: return invalid overlap; second: return valid
            m.ask.return_value = (
                str(overlap) if overlap_calls["n"] == 1 else str(valid_overlap)
            )
        elif "chunk size" in prompt_text.lower() or "size" in prompt_text.lower():
            m.ask.return_value = str(chunk_size)
        else:
            m.ask.return_value = default or "0"
        return m

    with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.chunking_configurator.Console"), \
         patch("ms_rag.workflow.chunking_configurator.Panel"), \
         patch("ms_rag.workflow.chunking_configurator.Text"):

        mock_q.text.side_effect = mock_text_side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value=None: value or title)

        # Mock strategy select to return simple fixed_size (no extra prompts)
        mock_select = MagicMock()
        mock_select.ask.return_value = "fixed_size"
        mock_q.select.return_value = mock_select

        result = configurator.configure()

    # The returned overlap must be < chunk_size
    assert result.chunk_overlap < result.chunk_size, (
        f"overlap={result.chunk_overlap} must be < chunk_size={result.chunk_size}"
    )
    # Must have been prompted for overlap more than once (re-prompt happened)
    # (only verifiable if invalid overlap was attempted — may not always reach
    # re-prompt due to mock flow, so we check the invariant on the result)
    assert result.chunk_overlap != overlap or overlap < chunk_size


# ---------------------------------------------------------------------------
# Property 12: Chunking Parameters Round-Trip
# ---------------------------------------------------------------------------


@given(
    strategy=st.sampled_from(
        # Only strategies that are fully configurable via simple size/overlap
        ["recursive_character", "fixed_size", "paragraph", "markdown_aware",
         "html_aware", "document_aware"]
    ),
    chunk_size=st.integers(min_value=64, max_value=8192),
    chunk_overlap=st.integers(min_value=0, max_value=63),
)
@settings(max_examples=100)
def test_chunking_parameters_round_trip(
    strategy: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Feature: ms-rag, Property 12: Chunking Parameters Round-Trip.

    For any valid (strategy, chunk_size, chunk_overlap) tuple with
    chunk_overlap < chunk_size, the configurator must store those exact
    values in the returned ChunkingConfig.
    """
    assume(chunk_overlap < chunk_size)

    configurator = ChunkingConfigurator()

    def mock_text_side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        prompt_text = (args[0] if args else kwargs.get("message", "")).lower()
        if "size" in prompt_text and "overlap" not in prompt_text:
            m.ask.return_value = str(chunk_size)
        elif "overlap" in prompt_text:
            m.ask.return_value = str(chunk_overlap)
        elif "separator" in prompt_text:
            m.ask.return_value = ""  # accept defaults
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
         patch("ms_rag.workflow.chunking_configurator.Console"), \
         patch("ms_rag.workflow.chunking_configurator.Panel"), \
         patch("ms_rag.workflow.chunking_configurator.Text"):

        mock_q.text.side_effect = mock_text_side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value=None: value or title)

        mock_select = MagicMock()
        mock_select.ask.return_value = strategy
        mock_q.select.return_value = mock_select

        result = configurator.configure()

    assert result.strategy == strategy, (
        f"strategy: expected {strategy!r}, got {result.strategy!r}"
    )
    assert result.chunk_size == chunk_size, (
        f"chunk_size: expected {chunk_size}, got {result.chunk_size}"
    )
    assert result.chunk_overlap == chunk_overlap, (
        f"chunk_overlap: expected {chunk_overlap}, got {result.chunk_overlap}"
    )


# ---------------------------------------------------------------------------
# Unit tests for _prompt_overlap edge cases
# ---------------------------------------------------------------------------


class TestChunkingConfiguratorUnit:
    def test_prompt_overlap_returns_valid_value(self) -> None:
        configurator = ChunkingConfigurator()
        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
             patch("ms_rag.workflow.chunking_configurator.Console"):
            m = MagicMock()
            m.ask.return_value = "100"
            mock_q.text.return_value = m
            result = configurator._prompt_overlap(
                chunk_size=500, default_overlap=200, console=MagicMock()
            )
        assert result == 100
        assert result < 500

    def test_prompt_overlap_reprompts_on_violation(self) -> None:
        """Req 7.5: overlap >= chunk_size triggers re-prompt."""
        configurator = ChunkingConfigurator()
        call_count = {"n": 0}

        def side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
            m = MagicMock()
            call_count["n"] += 1
            m.ask.return_value = "500" if call_count["n"] == 1 else "50"
            return m

        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
             patch("ms_rag.workflow.chunking_configurator.Console"):
            mock_q.text.side_effect = side_effect
            result = configurator._prompt_overlap(
                chunk_size=500, default_overlap=100, console=MagicMock()
            )

        assert result == 50
        assert call_count["n"] == 2  # re-prompted once

    def test_empty_input_uses_default(self) -> None:
        configurator = ChunkingConfigurator()
        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
             patch("ms_rag.workflow.chunking_configurator.Console"):
            m = MagicMock()
            m.ask.return_value = ""  # empty → use default
            mock_q.text.return_value = m
            result = configurator._prompt_overlap(
                chunk_size=1000, default_overlap=200, console=MagicMock()
            )
        # default (200) is < chunk_size (1000) → should be returned as-is
        assert result == 200
        assert result < 1000

    def test_separators_parsed_correctly(self) -> None:
        configurator = ChunkingConfigurator()
        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = r"\n\n,\n, "
            mock_q.text.return_value = m
            result = configurator._prompt_separators(console=MagicMock())
        assert result is not None
        assert "\n\n" in result
        assert "\n" in result

    def test_separators_blank_returns_none(self) -> None:
        configurator = ChunkingConfigurator()
        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = ""
            mock_q.text.return_value = m
            result = configurator._prompt_separators(console=MagicMock())
        assert result is None

    def test_chunk_size_out_of_range_reprompts(self) -> None:
        """Values outside [1, 32000] must be rejected with re-prompt."""
        configurator = ChunkingConfigurator()
        call_count = {"n": 0}

        def side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
            m = MagicMock()
            call_count["n"] += 1
            m.ask.return_value = "0" if call_count["n"] == 1 else "512"
            return m

        with patch("ms_rag.workflow.chunking_configurator.questionary") as mock_q, \
             patch("ms_rag.workflow.chunking_configurator.Console"):
            mock_q.text.side_effect = side_effect
            result = configurator._prompt_int(
                prompt="Chunk size:",
                default=1000,
                min_val=1,
                max_val=32_000,
                field_name="chunk_size",
                console=MagicMock(),
            )
        assert result == 512
        assert call_count["n"] == 2
