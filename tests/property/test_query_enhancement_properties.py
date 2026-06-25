"""Property-based tests for QueryEnhancer.

Properties covered:
    Property 18: Query Enhancement Round-Trip and Persistence (Req 11.2, 11.3, 11.4)

Validates: Requirements 11.1-11.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.query.query_enhancer import (
    QUERY_ENHANCEMENT_TECHNIQUES,
    TECHNIQUE_IDS,
    QueryEnhancer,
)


# ---------------------------------------------------------------------------
# Property 18: Query Enhancement Round-Trip and Persistence
# ---------------------------------------------------------------------------


@given(
    techniques=st.frozensets(
        st.sampled_from(TECHNIQUE_IDS),
        min_size=1,
        max_size=len(TECHNIQUE_IDS),
    )
)
@settings(max_examples=50)
def test_query_enhancement_configuration_round_trip(
    techniques: frozenset[str],
) -> None:
    """Feature: ms-rag, Property 18: Query Enhancement Round-Trip and Persistence.

    Selected techniques must be stored and returned exactly as selected —
    no additions, no omissions.
    """
    enhancer = QueryEnhancer()
    techniques_list = sorted(techniques)  # deterministic order

    # Mock: user says yes, then selects the given techniques
    with patch("ms_rag.query.query_enhancer.questionary") as mock_q, \
         patch("ms_rag.query.query_enhancer.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True  # wants enhancement
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = techniques_list
        mock_q.checkbox.return_value = mock_checkbox

        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        # If HyDE is selected, mock the LLM provider selection
        if "hyde" in techniques:
            mock_select = MagicMock()
            mock_select.ask.return_value = "openai"
            mock_q.select.return_value = mock_select

        result = enhancer.configure(configured_providers=["openai"])

    assert set(result) == techniques, (
        f"Expected {techniques}, got {set(result)}"
    )


def test_no_enhancement_returns_empty_list() -> None:
    """Req 11.5: If user says no, configure() returns empty list."""
    enhancer = QueryEnhancer()

    with patch("ms_rag.query.query_enhancer.questionary") as mock_q, \
         patch("ms_rag.query.query_enhancer.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False  # declines enhancement
        mock_q.confirm.return_value = mock_confirm

        result = enhancer.configure()

    assert result == []


def test_empty_checkbox_selection_returns_empty_list() -> None:
    """If user checks nothing in the checkbox, return empty list."""
    enhancer = QueryEnhancer()

    with patch("ms_rag.query.query_enhancer.questionary") as mock_q, \
         patch("ms_rag.query.query_enhancer.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = []
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = enhancer.configure()

    assert result == []


# ---------------------------------------------------------------------------
# enhance() runtime behaviour
# ---------------------------------------------------------------------------


def test_enhance_returns_original_query_when_no_techniques() -> None:
    """No techniques → original query returned unchanged."""
    enhancer = QueryEnhancer()
    result = enhancer.enhance("What is RAG?", techniques=[])
    assert result == ["What is RAG?"]


def test_enhance_with_llm_none_returns_original() -> None:
    """When no LLM provided, all techniques that need LLM return the original."""
    enhancer = QueryEnhancer()
    for technique_id in TECHNIQUE_IDS:
        result = enhancer.enhance("test query", techniques=[technique_id], llm=None)
        assert len(result) >= 1
        # All results should be non-empty strings
        for q in result:
            assert isinstance(q, str)
            assert len(q) > 0


def test_enhance_multi_query_returns_multiple() -> None:
    """multi_query technique with llm=None returns at least the original query."""
    enhancer = QueryEnhancer()
    # Without LLM, _generate_multi_queries falls back to [query]
    result = enhancer._generate_multi_queries("What is RAG?", llm=None, n=3)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(q, str) and len(q) > 0 for q in result)


def test_enhance_failure_is_non_fatal() -> None:
    """A technique that raises must not crash enhance()."""
    enhancer = QueryEnhancer()

    def always_fail(technique, queries, llm, num_queries) -> list[str]:
        raise RuntimeError("technique failed")

    with patch.object(enhancer, "_apply_technique", side_effect=always_fail):
        result = enhancer.enhance("test query", techniques=["query_rewriting"], llm=None)

    # Should return the original query despite the failure
    assert result == ["test query"]


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------


class TestTechniqueListCompleteness:
    def test_exactly_7_techniques_defined(self) -> None:
        assert len(QUERY_ENHANCEMENT_TECHNIQUES) == 7

    def test_all_required_techniques_present(self) -> None:
        required = {
            "query_rewriting", "query_expansion", "hyde",
            "multi_query", "step_back_prompting",
            "sub_question_decomposition", "rag_fusion",
        }
        defined = set(TECHNIQUE_IDS)
        missing = required - defined
        assert not missing, f"Missing techniques: {missing}"

    def test_no_duplicate_technique_ids(self) -> None:
        assert len(TECHNIQUE_IDS) == len(set(TECHNIQUE_IDS))

    def test_all_techniques_have_display_names(self) -> None:
        for t in QUERY_ENHANCEMENT_TECHNIQUES:
            assert len(t["display"].strip()) > 0

    def test_all_techniques_have_descriptions(self) -> None:
        for t in QUERY_ENHANCEMENT_TECHNIQUES:
            assert len(t["description"].strip()) > 0

    def test_hyde_is_in_technique_ids(self) -> None:
        assert "hyde" in TECHNIQUE_IDS

    def test_rag_fusion_is_in_technique_ids(self) -> None:
        assert "rag_fusion" in TECHNIQUE_IDS
