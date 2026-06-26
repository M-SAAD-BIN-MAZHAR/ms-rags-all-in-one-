"""Property-based tests for VectorizationModule.

Properties covered:
    Property 13: Embedding Model Provider Filtering (Req 8.1)
    Property 14: Embedding Model Selection Round-Trip (Req 8.3)

Validates: Requirements 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.config.credential_manager import PROVIDER_IDS
from ms_rag.ingestion.vectorization_module import (
    EMBEDDING_MODELS,
    _LOCAL_PROVIDERS,
    EmbeddingModelInfo,
    VectorizationModule,
    get_displayable_models,
)
from ms_rag.models import EmbeddingModelConfig


# ---------------------------------------------------------------------------
# Property 13: Embedding Model Provider Filtering
# ---------------------------------------------------------------------------


@given(
    providers=st.frozensets(
        st.sampled_from(PROVIDER_IDS),
        min_size=0,
        max_size=len(PROVIDER_IDS),
    )
)
@settings(max_examples=100)
def test_embedding_model_provider_filtering(providers: frozenset[str]) -> None:
    """Feature: ms-rag, Property 13: Embedding Model Provider Filtering.

    Displayed models must be exactly those whose provider is in the
    configured set OR whose provider is in _LOCAL_PROVIDERS.
    No models from unconfigured non-local providers should appear.
    """
    provider_list = list(providers)
    displayable = get_displayable_models(provider_list)

    # Build expected set
    provider_set = set(provider_list)
    expected = {
        m.model_id for m in EMBEDDING_MODELS
        if m.provider in provider_set or m.provider in _LOCAL_PROVIDERS
    }

    actual = {m.model_id for m in displayable}
    assert actual == expected, (
        f"For providers {provider_set}:\n"
        f"  Expected: {sorted(expected)}\n"
        f"  Actual:   {sorted(actual)}\n"
        f"  Extra:    {sorted(actual - expected)}\n"
        f"  Missing:  {sorted(expected - actual)}"
    )


def test_local_models_always_available_with_empty_providers() -> None:
    """Local models must be shown even when no providers are configured."""
    displayable = get_displayable_models([])
    local_model_ids = {m.model_id for m in EMBEDDING_MODELS if m.is_local}
    shown_ids = {m.model_id for m in displayable}
    assert local_model_ids.issubset(shown_ids)


def test_openai_models_shown_when_openai_configured() -> None:
    openai_models = {m.model_id for m in EMBEDDING_MODELS if m.provider == "openai"}
    displayable_ids = {m.model_id for m in get_displayable_models(["openai"])}
    assert openai_models.issubset(displayable_ids)


def test_openai_models_not_shown_without_credentials() -> None:
    """OpenAI models must NOT appear when openai is not in configured_providers."""
    displayable_ids = {m.model_id for m in get_displayable_models([])}
    openai_model_ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "openai"}
    # None of the OpenAI-specific models should appear
    overlap = displayable_ids & openai_model_ids
    assert not overlap, f"OpenAI models shown without credentials: {overlap}"


# ---------------------------------------------------------------------------
# Property 14: Embedding Model Selection Round-Trip
# ---------------------------------------------------------------------------


@given(
    model=st.sampled_from([
        m for m in EMBEDDING_MODELS
        if m.model_id != "__user_specified__"  # skip the Ollama placeholder
    ])
)
@settings(max_examples=len(EMBEDDING_MODELS))
def test_embedding_model_selection_round_trip(model: EmbeddingModelInfo) -> None:
    """Feature: ms-rag, Property 14: Embedding Model Selection Round-Trip.

    After display_and_select, the returned EmbeddingModelConfig must have
    provider and model_id equal to the selected model's values.
    """
    module = VectorizationModule()

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"):
        mock_select = MagicMock()
        mock_select.ask.return_value = model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(
            configured_providers=list(
                {model.provider} | _LOCAL_PROVIDERS
            )
        )

    assert result.provider == model.provider, (
        f"provider: expected {model.provider!r}, got {result.provider!r}"
    )
    assert result.model_id == model.model_id, (
        f"model_id: expected {model.model_id!r}, got {result.model_id!r}"
    )


def test_ollama_user_specified_prompts_for_model_name() -> None:
    """Requirement 8.4: Ollama selection must prompt for model name."""
    module = VectorizationModule()

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"):
        mock_select = MagicMock()
        mock_select.ask.return_value = "__user_specified__"
        mock_q.select.return_value = mock_select

        mock_text = MagicMock()
        mock_text.ask.return_value = "nomic-embed-text"
        mock_q.text.return_value = mock_text
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(configured_providers=["ollama"])

    assert result.provider == "ollama"
    assert result.model_id == "nomic-embed-text"
    assert result.local_path == "nomic-embed-text"


# ---------------------------------------------------------------------------
# Structural completeness (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestEmbeddingModelCatalogue:
    def test_at_least_20_models_defined(self) -> None:
        assert len(EMBEDDING_MODELS) >= 20

    def test_openai_models_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "openai"}
        assert "text-embedding-3-large" in ids
        assert "text-embedding-3-small" in ids
        assert "text-embedding-ada-002" in ids

    def test_cohere_models_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "cohere"}
        assert "embed-english-v3.0" in ids
        assert "embed-multilingual-v3.0" in ids

    def test_bge_models_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS}
        assert "BAAI/bge-m3" in ids
        assert "BAAI/bge-large-en-v1.5" in ids

    def test_e5_models_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS}
        assert "intfloat/e5-large-v2" in ids

    def test_instructor_models_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS}
        assert "hkunlp/instructor-xl" in ids

    def test_google_model_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "google_gemini"}
        assert len(ids) >= 1

    def test_mistral_model_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "mistral"}
        assert "mistral-embed" in ids

    def test_ollama_placeholder_present(self) -> None:
        ids = {m.model_id for m in EMBEDDING_MODELS if m.provider == "ollama"}
        assert "__user_specified__" in ids

    def test_all_models_have_display_names(self) -> None:
        for m in EMBEDDING_MODELS:
            assert len(m.display_name.strip()) > 0

    def test_all_models_have_providers(self) -> None:
        for m in EMBEDDING_MODELS:
            assert len(m.provider.strip()) > 0

    def test_local_models_flagged(self) -> None:
        hf_models = [m for m in EMBEDDING_MODELS if m.provider == "huggingface"]
        for m in hf_models:
            assert m.is_local is True, (
                f"HuggingFace model {m.model_id!r} should be flagged is_local=True"
            )

    def test_no_duplicate_model_ids_per_provider(self) -> None:
        from collections import Counter
        counts = Counter(m.model_id for m in EMBEDDING_MODELS)
        dupes = {mid: c for mid, c in counts.items() if c > 1}
        assert not dupes, f"Duplicate model IDs: {dupes}"


class TestGetEmbeddingsDispatch:
    """Verify the factory raises for unknown providers."""

    def test_unknown_provider_raises_value_error(self) -> None:
        module = VectorizationModule()
        config = EmbeddingModelConfig(
            provider="nonexistent_provider",
            model_id="some-model",
        )
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            module.get_embeddings(config)

    def test_known_providers_do_not_raise_value_error(self) -> None:
        """Known providers raise ImportError (package missing) not ValueError."""
        module = VectorizationModule()
        known_providers = ["openai", "cohere", "huggingface", "local",
                           "google_gemini", "mistral", "ollama"]
        for provider in known_providers:
            config = EmbeddingModelConfig(provider=provider, model_id="test-model")
            try:
                module.get_embeddings(config)
            except (ImportError, Exception) as exc:
                if isinstance(exc, ValueError):
                    if "Unsupported embedding provider" in str(exc):
                        pytest.fail(
                            f"Provider {provider!r} raised ValueError unexpectedly: {exc}"
                        )

