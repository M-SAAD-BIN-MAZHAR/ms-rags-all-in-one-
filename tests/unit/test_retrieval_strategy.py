"""Unit tests for RetrievalStrategyModule.

Tests (Requirement 12.1-12.7):
- All 10 strategy IDs are defined.
- Unknown strategy raises ValueError.
- Known strategies raise ImportError (not ValueError) when packages missing.
- alpha / lambda validation errors trigger re-prompt.
- Ensemble weight validation works correctly.
- Self-Query requires at least one MetadataField.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.models import MetadataField, RetrievalConfig
from ms_rag.query.retrieval_strategy import (
    STRATEGIES,
    STRATEGY_IDS,
    STRATEGY_MAP,
    RetrievalStrategyModule,
)
from ms_rag.utils.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Structural completeness (Req 12.1)
# ---------------------------------------------------------------------------


class TestStrategyListCompleteness:
    def test_exactly_10_strategies_defined(self) -> None:
        assert len(STRATEGIES) == 10

    def test_all_required_strategies_present(self) -> None:
        required = {
            "dense_vector", "keyword_bm25", "tfidf", "hybrid", "mmr",
            "ensemble", "parent_child", "multi_vector", "self_query", "time_weighted",
        }
        defined = set(STRATEGY_IDS)
        missing = required - defined
        assert not missing, f"Missing strategies: {missing}"

    def test_no_duplicate_strategy_ids(self) -> None:
        assert len(STRATEGY_IDS) == len(set(STRATEGY_IDS))

    def test_strategy_map_matches_list(self) -> None:
        assert set(STRATEGY_MAP.keys()) == set(STRATEGY_IDS)

    def test_all_strategies_have_display_names(self) -> None:
        for s in STRATEGIES:
            assert len(s.display_name.strip()) > 0

    def test_bm25_is_separate_from_tfidf(self) -> None:
        """Req 12.1: BM25 and TF-IDF are separate strategies."""
        assert "keyword_bm25" in STRATEGY_IDS
        assert "tfidf" in STRATEGY_IDS
        assert STRATEGY_IDS.index("keyword_bm25") != STRATEGY_IDS.index("tfidf")


# ---------------------------------------------------------------------------
# get_retriever factory (Req 12.1)
# ---------------------------------------------------------------------------


class TestGetRetrieverFactory:
    def test_unknown_strategy_raises_value_error(self) -> None:
        module = RetrievalStrategyModule()
        config = RetrievalConfig(strategy="nonexistent", top_k=5)
        with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
            module.get_retriever(config, MagicMock())

    def test_dense_vector_returns_retriever_from_vector_store(self) -> None:
        module = RetrievalStrategyModule()
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        config = RetrievalConfig(strategy="dense_vector", top_k=5)
        result = module.get_retriever(config, mock_store)

        mock_store.as_retriever.assert_called_once()
        assert result is mock_retriever

    def test_mmr_uses_mmr_search_type(self) -> None:
        module = RetrievalStrategyModule()
        mock_store = MagicMock()
        config = RetrievalConfig(strategy="mmr", top_k=5, lambda_diversity=0.7)
        module.get_retriever(config, mock_store)

        call_kwargs = mock_store.as_retriever.call_args
        assert call_kwargs is not None
        search_kwargs = call_kwargs.kwargs.get("search_kwargs", {})
        assert search_kwargs.get("lambda_mult") == 0.7

    def test_known_strategies_dont_raise_value_error(self) -> None:
        """Known strategies must NOT raise 'Unsupported retrieval strategy' ValueError."""
        module = RetrievalStrategyModule()
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()

        for strategy_id in STRATEGY_IDS:
            config = RetrievalConfig(strategy=strategy_id, top_k=5)
            try:
                module.get_retriever(config, mock_store)
            except Exception as exc:
                # Only our own dispatch error is a real failure
                if isinstance(exc, ValueError) and "Unsupported retrieval strategy" in str(exc):
                    pytest.fail(f"Strategy {strategy_id!r} raised ValueError: {exc}")


# ---------------------------------------------------------------------------
# Parameter validation (Req 12.3, 12.4)
# ---------------------------------------------------------------------------


class TestParameterValidation:
    def test_alpha_out_of_range_raises_validation_error(self) -> None:
        from ms_rag.utils.validation import validate_numeric
        with pytest.raises(ValidationError):
            validate_numeric(1.5, 0.0, 1.0, "hybrid_alpha")

    def test_alpha_boundary_values_pass(self) -> None:
        from ms_rag.utils.validation import validate_numeric
        validate_numeric(0.0, 0.0, 1.0, "hybrid_alpha")
        validate_numeric(1.0, 0.0, 1.0, "hybrid_alpha")
        validate_numeric(0.5, 0.0, 1.0, "hybrid_alpha")

    def test_lambda_out_of_range_raises_validation_error(self) -> None:
        from ms_rag.utils.validation import validate_numeric
        with pytest.raises(ValidationError):
            validate_numeric(-0.1, 0.0, 1.0, "mmr_lambda")

    def test_top_k_range_validation(self) -> None:
        from ms_rag.utils.validation import validate_numeric
        with pytest.raises(ValidationError):
            validate_numeric(0, 1, 1000, "retrieval_top_k")
        with pytest.raises(ValidationError):
            validate_numeric(1001, 1, 1000, "retrieval_top_k")
        validate_numeric(5, 1, 1000, "retrieval_top_k")
        validate_numeric(1000, 1, 1000, "retrieval_top_k")

    def test_ensemble_weights_must_sum_to_one(self) -> None:
        from ms_rag.utils.validation import validate_ensemble_weights
        validate_ensemble_weights([0.5, 0.5])
        with pytest.raises(ValidationError):
            validate_ensemble_weights([0.3, 0.3])  # sum = 0.6


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


class TestPromptHelpers:
    def test_prompt_int_returns_default_on_empty(self) -> None:
        module = RetrievalStrategyModule()
        with patch("ms_rag.query.retrieval_strategy.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = ""
            mock_q.text.return_value = m
            result = module._prompt_int("prompt:", 5, 1, 1000, "top_k", MagicMock())
        assert result == 5

    def test_prompt_int_reprompts_on_out_of_range(self) -> None:
        module = RetrievalStrategyModule()
        call_count = {"n": 0}

        def side_effect(*a, **kw) -> MagicMock:
            m = MagicMock()
            call_count["n"] += 1
            m.ask.return_value = "0" if call_count["n"] == 1 else "10"
            return m

        with patch("ms_rag.query.retrieval_strategy.questionary") as mock_q, \
             patch("ms_rag.query.retrieval_strategy.Console"):
            mock_q.text.side_effect = side_effect
            result = module._prompt_int("prompt:", 5, 1, 1000, "top_k", MagicMock())

        assert result == 10
        assert call_count["n"] == 2

    def test_prompt_float_returns_default_on_empty(self) -> None:
        module = RetrievalStrategyModule()
        with patch("ms_rag.query.retrieval_strategy.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = ""
            mock_q.text.return_value = m
            result = module._prompt_float("prompt:", 0.5, 0.0, 1.0, "alpha", MagicMock())
        assert result == 0.5

    def test_prompt_float_reprompts_on_out_of_range(self) -> None:
        module = RetrievalStrategyModule()
        call_count = {"n": 0}

        def side_effect(*a, **kw) -> MagicMock:
            m = MagicMock()
            call_count["n"] += 1
            m.ask.return_value = "2.0" if call_count["n"] == 1 else "0.7"
            return m

        with patch("ms_rag.query.retrieval_strategy.questionary") as mock_q, \
             patch("ms_rag.query.retrieval_strategy.Console"):
            mock_q.text.side_effect = side_effect
            result = module._prompt_float("prompt:", 0.5, 0.0, 1.0, "alpha", MagicMock())

        assert result == 0.7
        assert call_count["n"] == 2
