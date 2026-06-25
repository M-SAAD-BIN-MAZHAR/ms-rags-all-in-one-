"""Property-based tests for RerankingModule.

Properties covered:
    Property 15: Reranking Top-K Constraint (Req 13.4/13.5)

Validates: Requirements 13.1-13.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ms_rag.models import CredentialStore, RerankingConfig
from ms_rag.query.reranking_module import (
    DEFAULT_MODEL_IDS,
    RERANKER_IDS,
    RERANKER_MAP,
    RERANKERS,
    RerankingModule,
)
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric


# ---------------------------------------------------------------------------
# Property 15: Reranking Top-K Constraint
# ---------------------------------------------------------------------------


@given(
    ret_k=st.integers(min_value=1, max_value=100),
    rerank_k=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=200)
def test_reranking_top_k_constraint(ret_k: int, rerank_k: int) -> None:
    """Feature: ms-rag, Property 15: Reranking Top-K Constraint.

    If rerank_k > retrieval_top_k, validation must fail.
    If rerank_k ≤ retrieval_top_k, validation must pass.
    """
    if rerank_k > ret_k:
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric(rerank_k, 1, ret_k, "reranking_top_k")
        assert exc_info.value.field_name == "reranking_top_k"
    else:
        # Must not raise
        validate_numeric(rerank_k, 1, ret_k, "reranking_top_k")


@given(
    ret_k=st.integers(min_value=1, max_value=50),
    rerank_k=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_prompt_rerank_top_k_reprompts_on_violation(
    ret_k: int, rerank_k: int
) -> None:
    """Property 15 (configurator): prompt re-prompts when rerank_k > ret_k."""
    assume(rerank_k > ret_k)

    module = RerankingModule()
    valid_k = max(1, ret_k - 1) if ret_k > 1 else 1
    call_count = {"n": 0}

    def text_side_effect(*args, **kwargs) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        m.ask.return_value = str(rerank_k) if call_count["n"] == 1 else str(valid_k)
        return m

    with patch("ms_rag.query.reranking_module.questionary") as mock_q:
        mock_q.text.side_effect = text_side_effect
        result = module._prompt_rerank_top_k(ret_k, console=MagicMock())

    assert result <= ret_k, f"result {result} must be ≤ retrieval_top_k {ret_k}"
    assert call_count["n"] >= 2, "Must have prompted at least twice"


# ---------------------------------------------------------------------------
# Structural completeness (Req 13.2)
# ---------------------------------------------------------------------------


class TestRerankerListCompleteness:
    def test_exactly_6_rerankers_defined(self) -> None:
        assert len(RERANKERS) == 6

    def test_all_required_rerankers_present(self) -> None:
        required = {
            "cross_encoder", "cohere_reranker", "bge_reranker",
            "llm_reranker", "colbert", "flashrank",
        }
        defined = set(RERANKER_IDS)
        missing = required - defined
        assert not missing, f"Missing rerankers: {missing}"

    def test_no_duplicate_reranker_ids(self) -> None:
        assert len(RERANKER_IDS) == len(set(RERANKER_IDS))

    def test_all_rerankers_have_display_names(self) -> None:
        for r in RERANKERS:
            assert len(r.display_name.strip()) > 0

    def test_all_rerankers_have_descriptions(self) -> None:
        for r in RERANKERS:
            assert len(r.description.strip()) > 0

    def test_cohere_requires_credentials(self) -> None:
        assert RERANKER_MAP["cohere_reranker"].requires_credentials is True
        assert RERANKER_MAP["cohere_reranker"].credential_field == "COHERE_API_KEY"

    def test_cross_encoder_requires_local_model(self) -> None:
        assert RERANKER_MAP["cross_encoder"].requires_local_model is True

    def test_bge_reranker_requires_local_model(self) -> None:
        assert RERANKER_MAP["bge_reranker"].requires_local_model is True

    def test_colbert_requires_local_model(self) -> None:
        assert RERANKER_MAP["colbert"].requires_local_model is True

    def test_flashrank_requires_no_credentials_or_model(self) -> None:
        info = RERANKER_MAP["flashrank"]
        assert info.requires_credentials is False
        assert info.requires_local_model is False

    def test_llm_reranker_requires_no_credentials_or_model(self) -> None:
        info = RERANKER_MAP["llm_reranker"]
        assert info.requires_credentials is False
        assert info.requires_local_model is False


# ---------------------------------------------------------------------------
# Credential gating (Req 13.3)
# ---------------------------------------------------------------------------


class TestCredentialGating:
    def test_ensure_credentials_returns_true_when_stored(self) -> None:
        store = CredentialStore()
        store.set("cohere", "COHERE_API_KEY", "co-test-key")
        module = RerankingModule(credential_store=store)
        info = RERANKER_MAP["cohere_reranker"]
        result = module._ensure_credentials(info, console=MagicMock())
        assert result is True

    def test_ensure_credentials_prompts_when_absent(self) -> None:
        store = CredentialStore()
        module = RerankingModule(credential_store=store)
        info = RERANKER_MAP["cohere_reranker"]

        with patch("ms_rag.query.reranking_module.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = "test-key"
            mock_q.password.return_value = m

            result = module._ensure_credentials(info, console=MagicMock())

        assert result is True
        assert store.get("cohere", "COHERE_API_KEY") == "test-key"

    def test_ensure_credentials_returns_false_on_cancel(self) -> None:
        store = CredentialStore()
        module = RerankingModule(credential_store=store)
        info = RERANKER_MAP["cohere_reranker"]

        with patch("ms_rag.query.reranking_module.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = ""  # cancelled / empty
            mock_q.password.return_value = m

            result = module._ensure_credentials(info, console=MagicMock())

        assert result is False
        # Key must NOT be stored
        assert store.get("cohere", "COHERE_API_KEY") is None


# ---------------------------------------------------------------------------
# rerank() method
# ---------------------------------------------------------------------------


class TestRerankMethod:
    def test_empty_docs_returns_empty(self) -> None:
        module = RerankingModule()
        config = RerankingConfig(reranker="flashrank", model_id="", top_k=3)
        result = module.rerank("query", [], config)
        assert result == []

    def test_rerank_falls_back_on_import_error(self) -> None:
        """If reranker package is not installed, returns top-k from original."""
        module = RerankingModule()
        docs = [MagicMock() for _ in range(10)]
        config = RerankingConfig(reranker="flashrank", model_id="", top_k=3)

        with patch.object(module, "_rerank_flashrank", side_effect=ImportError("no flashrank")):
            result = module.rerank("query", docs, config)

        # Fallback: returns first top_k from original list
        assert len(result) == 3

    def test_rerank_returns_at_most_top_k(self) -> None:
        module = RerankingModule()
        docs = [MagicMock() for _ in range(10)]
        config = RerankingConfig(reranker="llm_reranker", model_id="", top_k=4)
        result = module.rerank("query", docs, config)
        assert len(result) <= 4


# ---------------------------------------------------------------------------
# Default model IDs
# ---------------------------------------------------------------------------


class TestDefaultModelIDs:
    def test_cross_encoder_has_default(self) -> None:
        assert "cross_encoder" in DEFAULT_MODEL_IDS
        assert len(DEFAULT_MODEL_IDS["cross_encoder"]) > 0

    def test_bge_reranker_has_default(self) -> None:
        assert "bge_reranker" in DEFAULT_MODEL_IDS

    def test_colbert_has_default(self) -> None:
        assert "colbert" in DEFAULT_MODEL_IDS
