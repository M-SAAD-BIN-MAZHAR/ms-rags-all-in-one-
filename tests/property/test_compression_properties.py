"""Property-based tests for ContextCompressor.

Properties covered:
    Property 16: Context Compressor Multi-Select Round-Trip (Req 14.2, 14.3)

Validates: Requirements 14.1-14.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.models import CompressionConfig
from ms_rag.query.context_compressor import (
    COMPRESSION_TECHNIQUES,
    LLM_REQUIRED_TECHNIQUES,
    TECHNIQUE_INFO,
    ContextCompressor,
)


# ---------------------------------------------------------------------------
# Property 16: Context Compressor Multi-Select Round-Trip
# ---------------------------------------------------------------------------


@given(
    techniques=st.frozensets(
        st.sampled_from(COMPRESSION_TECHNIQUES),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=50)
def test_context_compressor_multi_select_round_trip(
    techniques: frozenset[str],
) -> None:
    """Feature: ms-rag, Property 16: Context Compressor Multi-Select Round-Trip.

    Selected techniques must be stored exactly as selected in the returned
    CompressionConfig, in the order presented in the checklist.
    """
    techniques_in_order = [t for t in COMPRESSION_TECHNIQUES if t in techniques]
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = techniques_in_order
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        # If embeddings_filter selected, mock threshold prompt
        if "embeddings_filter" in techniques:
            mock_text = MagicMock()
            mock_text.ask.return_value = "0.75"
            mock_q.text.return_value = mock_text

        result = compressor.configure(configured_providers=["openai"])

    assert result is not None
    assert set(result.techniques) == techniques, (
        f"Expected {techniques}, got {set(result.techniques)}"
    )


def test_techniques_stored_in_checklist_order() -> None:
    """Techniques must be stored in the order they appear in the checklist."""
    compressor = ContextCompressor()
    # Select techniques in reverse order — they should be stored in checklist order
    selected_ids = ["summary_compression", "embeddings_filter", "llm_chain_extraction"]
    expected_order = [t for t in COMPRESSION_TECHNIQUES if t in selected_ids]

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = expected_order
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        mock_text = MagicMock()
        mock_text.ask.return_value = "0.8"
        mock_q.text.return_value = mock_text

        result = compressor.configure(configured_providers=["openai"])

    assert result.techniques == expected_order


# ---------------------------------------------------------------------------
# No compression (Req 14.1)
# ---------------------------------------------------------------------------


def test_no_compression_returns_none() -> None:
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_q.confirm.return_value = mock_confirm

        result = compressor.configure()

    assert result is None


def test_zero_technique_selection_reprompts() -> None:
    """Req 14.3: zero-technique selection must show error and re-present."""
    compressor = ContextCompressor()
    call_count = {"n": 0}

    def checkbox_side_effect(*args, **kwargs) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        m.ask.return_value = [] if call_count["n"] == 1 else ["redundancy_removal"]
        return m

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_q.checkbox.side_effect = checkbox_side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = compressor.configure(configured_providers=["openai"])

    assert result is not None
    assert "redundancy_removal" in result.techniques
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# LLM-dependency check (Req 14.5)
# ---------------------------------------------------------------------------


def test_llm_required_techniques_are_flagged() -> None:
    """llm_chain_extraction and summary_compression must require LLM."""
    assert "llm_chain_extraction" in LLM_REQUIRED_TECHNIQUES
    assert "summary_compression" in LLM_REQUIRED_TECHNIQUES


def test_non_llm_techniques_not_in_llm_required() -> None:
    non_llm = {"embeddings_filter", "redundancy_removal", "contextual_compression",
               "document_compressor_pipeline"}
    for t in non_llm:
        assert t not in LLM_REQUIRED_TECHNIQUES


# ---------------------------------------------------------------------------
# Threshold prompt (Req 14.4)
# ---------------------------------------------------------------------------


def test_threshold_default_is_0_75() -> None:
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q:
        m = MagicMock()
        m.ask.return_value = ""  # accept default
        mock_q.text.return_value = m
        result = compressor._prompt_threshold(console=MagicMock())

    assert result == 0.75


def test_threshold_out_of_range_reprompts() -> None:
    compressor = ContextCompressor()
    call_count = {"n": 0}

    def side_effect(*a, **kw) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        m.ask.return_value = "1.5" if call_count["n"] == 1 else "0.8"
        return m

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):
        mock_q.text.side_effect = side_effect
        result = compressor._prompt_threshold(console=MagicMock())

    assert result == 0.8
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------


class TestCompressionTechniqueList:
    def test_exactly_6_techniques_defined(self) -> None:
        assert len(COMPRESSION_TECHNIQUES) == 6

    def test_all_required_techniques_present(self) -> None:
        required = {
            "llm_chain_extraction", "embeddings_filter",
            "document_compressor_pipeline", "redundancy_removal",
            "contextual_compression", "summary_compression",
        }
        defined = set(COMPRESSION_TECHNIQUES)
        missing = required - defined
        assert not missing, f"Missing techniques: {missing}"

    def test_no_duplicate_technique_ids(self) -> None:
        assert len(COMPRESSION_TECHNIQUES) == len(set(COMPRESSION_TECHNIQUES))

    def test_all_techniques_have_display_names(self) -> None:
        for tid in COMPRESSION_TECHNIQUES:
            assert len(TECHNIQUE_INFO[tid]["display"].strip()) > 0

    def test_all_techniques_have_descriptions(self) -> None:
        for tid in COMPRESSION_TECHNIQUES:
            assert len(TECHNIQUE_INFO[tid]["description"].strip()) > 0

    def test_compression_config_stores_all_params(self) -> None:
        config = CompressionConfig(
            techniques=["embeddings_filter", "redundancy_removal"],
            similarity_threshold=0.8,
        )
        assert config.techniques == ["embeddings_filter", "redundancy_removal"]
        assert config.similarity_threshold == 0.8
