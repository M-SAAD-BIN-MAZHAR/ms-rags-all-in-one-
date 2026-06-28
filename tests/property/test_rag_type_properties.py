"""Property-based tests for RAGTypeSelector.

Properties covered:
    Property 4: RAG Type Description Presence (Req 3.2)
    Property 5: LangGraph Note Conditional Display (Req 3.4)

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.workflow.rag_type_selector import (
    LANGGRAPH_TYPES,
    RAG_TYPES,
    RAG_TYPE_MAP,
)


# ---------------------------------------------------------------------------
# Property 4: RAG Type Description Presence
# ---------------------------------------------------------------------------


@given(rag_id=st.sampled_from([r.rag_type for r in RAG_TYPES]))
@settings(max_examples=15)  # exactly 15 types — exhaustive
def test_rag_type_description_presence(rag_id: str) -> None:
    """Feature: ms-rag, Property 4: RAG Type Description Presence.

    For any valid RAG type, the description must be a non-empty string
    containing between 2 and 4 sentences.
    """
    config = RAG_TYPE_MAP[rag_id]

    assert isinstance(config.description, str)
    assert len(config.description.strip()) > 0

    # Count sentences by splitting on ". " — description must have 2-4
    # We split on ". " and count parts; last part may not end with "."
    sentences = [s.strip() for s in config.description.replace(".\n", ". ").split(". ") if s.strip()]
    sentence_count = len(sentences)
    assert 2 <= sentence_count <= 6, (
        f"RAG type '{rag_id}' description has {sentence_count} sentence parts, "
        f"expected 2-4. Description: {config.description!r}"
    )


# ---------------------------------------------------------------------------
# Property 5: LangGraph Note Conditional Display
# ---------------------------------------------------------------------------


@given(rag_id=st.sampled_from([r.rag_type for r in RAG_TYPES]))
@settings(max_examples=15)  # exhaustive
def test_langgraph_note_conditional(rag_id: str) -> None:
    """Feature: ms-rag, Property 5: LangGraph Note Conditional Display.

    requires_langgraph must be True if and only if rag_id is in LANGGRAPH_TYPES.
    """
    config = RAG_TYPE_MAP[rag_id]
    expected = rag_id in LANGGRAPH_TYPES
    assert config.requires_langgraph == expected, (
        f"RAG type '{rag_id}': requires_langgraph={config.requires_langgraph}, "
        f"expected {expected} (LANGGRAPH_TYPES={LANGGRAPH_TYPES})"
    )


# ---------------------------------------------------------------------------
# Structural completeness tests (Requirement 3.1)
# ---------------------------------------------------------------------------


class TestRAGTypeListCompleteness:
    def test_exactly_15_rag_types_defined(self) -> None:
        """Requirement 3.1: at least 15 RAG types must be listed."""
        assert len(RAG_TYPES) >= 15

    def test_all_required_rag_types_present(self) -> None:
        """Requirement 3.1: all required RAG type IDs must be present."""
        required = {
            "naive_rag", "advanced_rag", "modular_rag", "agentic_rag",
            "self_rag", "corrective_rag", "speculative_rag", "graphrag",
            "hyde_rag", "multi_query_rag", "rag_fusion", "step_back_rag",
            "parent_child_rag", "adaptive_rag", "contextual_compression_rag",
        }
        defined = {r.rag_type for r in RAG_TYPES}
        missing = required - defined
        assert not missing, f"Missing RAG types: {missing}"

    def test_no_duplicate_rag_type_ids(self) -> None:
        ids = [r.rag_type for r in RAG_TYPES]
        assert len(ids) == len(set(ids)), "Duplicate RAG type IDs found"

    def test_all_rag_types_have_display_names(self) -> None:
        for r in RAG_TYPES:
            assert isinstance(r.display_name, str)
            assert len(r.display_name.strip()) > 0

    def test_rag_type_map_matches_list(self) -> None:
        """RAG_TYPE_MAP must contain the same entries as RAG_TYPES."""
        assert set(RAG_TYPE_MAP.keys()) == {r.rag_type for r in RAG_TYPES}

    def test_exactly_4_langgraph_types(self) -> None:
        """Requirement 3.4: exactly 4 types require LangGraph."""
        langgraph_count = sum(1 for r in RAG_TYPES if r.requires_langgraph)
        assert langgraph_count == 4, f"Expected 4 LangGraph types, got {langgraph_count}"

    def test_langgraph_type_ids_match_constant(self) -> None:
        """The LANGGRAPH_TYPES frozenset must match requires_langgraph=True types."""
        from_list = frozenset(r.rag_type for r in RAG_TYPES if r.requires_langgraph)
        assert from_list == LANGGRAPH_TYPES


class TestRAGTypeContent:
    def test_naive_rag_does_not_require_langgraph(self) -> None:
        assert not RAG_TYPE_MAP["naive_rag"].requires_langgraph

    def test_self_rag_requires_langgraph(self) -> None:
        assert RAG_TYPE_MAP["self_rag"].requires_langgraph

    def test_corrective_rag_requires_langgraph(self) -> None:
        assert RAG_TYPE_MAP["corrective_rag"].requires_langgraph

    def test_agentic_rag_requires_langgraph(self) -> None:
        assert RAG_TYPE_MAP["agentic_rag"].requires_langgraph

    def test_adaptive_rag_requires_langgraph(self) -> None:
        assert RAG_TYPE_MAP["adaptive_rag"].requires_langgraph

    def test_graphrag_does_not_require_langgraph(self) -> None:
        assert not RAG_TYPE_MAP["graphrag"].requires_langgraph

    def test_all_descriptions_mention_best_for(self) -> None:
        """Every description should guide the user on when to use it."""
        for r in RAG_TYPES:
            lower = r.description.lower()
            assert "best for" in lower, (
                f"RAG type '{r.rag_type}' description missing 'Best for' guidance"
            )

    def test_advanced_rag_descriptions_match_runtime_claims(self) -> None:
        """Advanced RAG labels must describe implemented workflows, not paper-only variants."""
        agentic = RAG_TYPE_MAP["agentic_rag"].description.lower()
        assert "plan whether to retrieve" in agentic
        assert "rewrite the query" in agentic
        assert "call approved tools" in agentic
        assert "permission-gated" in agentic

        self_rag = RAG_TYPE_MAP["self_rag"].description.lower()
        assert "retrieval need" in self_rag
        assert "grades retrieved evidence" in self_rag
        assert "supported" in self_rag

        corrective = RAG_TYPE_MAP["corrective_rag"].description.lower()
        assert "rewrites weak queries" in corrective
        assert "approved web search" in corrective
        assert "corpus is insufficient" in corrective

        graphrag = RAG_TYPE_MAP["graphrag"].description.lower()
        assert "during ingestion" in graphrag
        assert "persistent graph" in graphrag
        assert "community summaries" in graphrag
        assert "without requiring users to operate a separate graph database" not in graphrag

        adaptive = RAG_TYPE_MAP["adaptive_rag"].description.lower()
        assert "query complexity" in adaptive
        assert "directly to generation" in adaptive
        assert "use retrieval" in adaptive
        assert "rewrite-plus-retrieval" in adaptive
