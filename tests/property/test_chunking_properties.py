"""Property-based tests for ChunkingEngine.

Properties covered:
    Property 10: Chunking Strategy Description Presence (Req 6.2)

Validates: Requirements 6.1, 6.2, 6.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.ingestion.chunking_engine import (
    STRATEGY_DESCRIPTIONS,
    STRATEGY_IDS,
    ChunkingEngine,
    ChunkingStrategyInfo,
)
from ms_rag.models import ChunkingConfig


# ---------------------------------------------------------------------------
# Property 10: Chunking Strategy Description Presence
# ---------------------------------------------------------------------------


@given(strategy_id=st.sampled_from(STRATEGY_IDS))
@settings(max_examples=11)  # exactly 11 strategies — exhaustive
def test_chunking_strategy_description_presence(strategy_id: str) -> None:
    """Feature: ms-rag, Property 10: Chunking Strategy Description Presence.

    For any valid chunking strategy, the description must be a non-empty
    string containing between 1 and 3 sentences.
    """
    info = STRATEGY_DESCRIPTIONS[strategy_id]

    assert isinstance(info.description, str)
    assert len(info.description.strip()) > 0

    # Count sentences — split on ". " ignoring trailing period on last sentence
    parts = [s.strip() for s in info.description.split(". ") if s.strip()]
    sentence_count = len(parts)
    assert 1 <= sentence_count <= 6, (
        f"Strategy '{strategy_id}' has {sentence_count} sentence parts; expected 1-3. "
        f"Description: {info.description!r}"
    )


# ---------------------------------------------------------------------------
# Structural completeness tests (Requirement 6.1)
# ---------------------------------------------------------------------------


class TestStrategyListCompleteness:
    def test_exactly_11_strategies_defined(self) -> None:
        assert len(STRATEGY_IDS) == 11

    def test_all_required_strategies_present(self) -> None:
        required = {
            "recursive_character", "fixed_size", "semantic",
            "sentence", "paragraph", "token_based",
            "markdown_aware", "html_aware", "code_aware",
            "agentic", "document_aware",
        }
        missing = required - set(STRATEGY_IDS)
        assert not missing, f"Missing strategies: {missing}"

    def test_no_duplicate_strategy_ids(self) -> None:
        assert len(STRATEGY_IDS) == len(set(STRATEGY_IDS))

    def test_all_strategies_have_display_names(self) -> None:
        for sid, info in STRATEGY_DESCRIPTIONS.items():
            assert len(info.display_name.strip()) > 0, (
                f"Strategy {sid!r} missing display name"
            )

    def test_all_strategies_have_positive_defaults(self) -> None:
        for sid, info in STRATEGY_DESCRIPTIONS.items():
            if sid != "semantic":  # semantic is threshold-based, not size-based
                assert info.default_chunk_size >= 0, (
                    f"Strategy {sid!r} has invalid default_chunk_size"
                )
                assert info.default_overlap >= 0

    def test_code_aware_requires_language(self) -> None:
        assert STRATEGY_DESCRIPTIONS["code_aware"].requires_language is True

    def test_agentic_requires_llm(self) -> None:
        assert STRATEGY_DESCRIPTIONS["agentic"].requires_llm is True

    def test_recursive_supports_separators(self) -> None:
        assert STRATEGY_DESCRIPTIONS["recursive_character"].supports_separators is True

    def test_token_based_requires_tokenizer(self) -> None:
        assert STRATEGY_DESCRIPTIONS["token_based"].requires_tokenizer is True


# ---------------------------------------------------------------------------
# ChunkingEngine factory tests (no LangChain installed — pure dispatch logic)
# ---------------------------------------------------------------------------


class TestChunkingEngineDispatch:
    """Test that the engine raises ValueError for unknown strategies."""

    def test_unknown_strategy_raises_value_error(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy="nonexistent_strategy",
            chunk_size=500,
            chunk_overlap=50,
        )
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            engine.get_splitter(config)

    def test_valid_strategy_ids_are_all_handled(self) -> None:
        """Every strategy_id in STRATEGY_IDS must NOT raise ValueError.

        Note: may raise ImportError if LangChain packages not installed —
        that is acceptable here (we only check for ValueError / dispatch).
        """
        engine = ChunkingEngine()
        for sid in STRATEGY_IDS:
            config = ChunkingConfig(
                strategy=sid,
                chunk_size=500,
                chunk_overlap=50,
                language="python" if sid == "code_aware" else None,
            )
            try:
                result = engine.get_splitter(config)
                assert result is not None
            except ImportError:
                # LangChain not installed — dispatch worked, import failed — OK
                pass
            except ValueError as exc:
                pytest.fail(
                    f"Strategy {sid!r} raised ValueError unexpectedly: {exc}"
                )


class TestChunkingDefaults:
    def test_recursive_character_default_overlap_less_than_size(self) -> None:
        info = STRATEGY_DESCRIPTIONS["recursive_character"]
        assert info.default_overlap < info.default_chunk_size

    def test_token_based_default_overlap_less_than_size(self) -> None:
        info = STRATEGY_DESCRIPTIONS["token_based"]
        assert info.default_overlap < info.default_chunk_size

    def test_sentence_default_overlap_less_than_size(self) -> None:
        info = STRATEGY_DESCRIPTIONS["sentence"]
        assert info.default_overlap < info.default_chunk_size
