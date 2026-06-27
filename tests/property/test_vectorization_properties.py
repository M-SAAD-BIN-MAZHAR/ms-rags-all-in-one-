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
    _prepare_local_huggingface_download,
    EmbeddingModelInfo,
    VectorizationModule,
    get_displayable_models,
    get_embedding_dimension,
    get_embedding_model_info,
)
from ms_rag.models import CredentialStore, EmbeddingModelConfig


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

    Displayed models must include every embedding provider. Credentials are
    requested only after the user selects a hosted/API embedding model.
    """
    provider_list = list(providers)
    displayable = get_displayable_models(provider_list)

    expected = {m.model_id for m in EMBEDDING_MODELS}

    actual = {m.model_id for m in displayable}
    assert actual == expected, (
        f"For providers {set(provider_list)}:\n"
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


def test_openai_models_shown_even_when_openai_not_chat_provider() -> None:
    openai_models = {m.model_id for m in EMBEDDING_MODELS if m.provider == "openai"}
    displayable_ids = {m.model_id for m in get_displayable_models(["mistral"])}
    assert openai_models.issubset(displayable_ids)


def test_embedding_dimension_lookup_returns_catalogue_dimension() -> None:
    config = EmbeddingModelConfig(
        provider="openai",
        model_id="text-embedding-3-small",
    )

    assert get_embedding_dimension(config) == 1536


def test_unknown_embedding_dimension_returns_none() -> None:
    config = EmbeddingModelConfig(
        provider="ollama",
        model_id="custom-local-model",
    )

    assert get_embedding_model_info(config.model_id) is None
    assert get_embedding_dimension(config) is None


def test_hosted_embedding_models_are_visible_without_chat_provider_credentials() -> None:
    """Hosted embedding models appear first, then Step 8 asks for credentials if selected."""
    displayable_ids = {m.model_id for m in get_displayable_models([])}
    hosted_model_ids = {
        m.model_id
        for m in EMBEDDING_MODELS
        if m.provider in {"openai", "cohere", "huggingface_endpoint", "google_gemini", "mistral"}
    }
    assert hosted_model_ids.issubset(displayable_ids)


def test_huggingface_endpoint_models_available_for_embedding_only_token_flow() -> None:
    """Hosted HuggingFace embeddings should be selectable even with a different chat provider."""
    with_no_chat_hf = {m.model_id for m in get_displayable_models([])}
    with_hf_chat = {m.model_id for m in get_displayable_models(["huggingface"])}
    hosted_ids = {
        m.model_id for m in EMBEDDING_MODELS
        if m.provider == "huggingface_endpoint"
    }

    assert hosted_ids
    assert hosted_ids.issubset(with_no_chat_hf)
    assert hosted_ids.issubset(with_hf_chat)


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


def test_hosted_huggingface_embedding_prompts_for_token_when_missing() -> None:
    """Hosted HF embeddings selected after another chat provider must request a token."""
    module = VectorizationModule()
    store = CredentialStore()
    hosted_model = next(m for m in EMBEDDING_MODELS if m.provider == "huggingface_endpoint")

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"), \
         patch("ms_rag.ui.prompts.prompt_text", return_value="hf_test_token") as mock_prompt:
        mock_select = MagicMock()
        mock_select.ask.return_value = hosted_model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(
            configured_providers=["openai", "huggingface"],
            credential_store=store,
        )

    assert result.provider == "huggingface_endpoint"
    assert store.get("huggingface", "HUGGINGFACEHUB_API_TOKEN") == "hf_test_token"
    mock_prompt.assert_called_once()


def test_hosted_huggingface_embedding_does_not_reprompt_when_token_exists() -> None:
    module = VectorizationModule()
    store = CredentialStore()
    store.set("huggingface", "HUGGINGFACEHUB_API_TOKEN", "hf_existing")
    hosted_model = next(m for m in EMBEDDING_MODELS if m.provider == "huggingface_endpoint")

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"), \
         patch("ms_rag.ui.prompts.prompt_text") as mock_prompt:
        mock_select = MagicMock()
        mock_select.ask.return_value = hosted_model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        module.display_and_select(
            configured_providers=["openai", "huggingface"],
            credential_store=store,
        )

    mock_prompt.assert_not_called()


def test_openai_embedding_prompts_for_key_when_chat_provider_is_mistral() -> None:
    module = VectorizationModule()
    store = CredentialStore()
    openai_model = next(m for m in EMBEDDING_MODELS if m.provider == "openai")

    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), \
         patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"), \
         patch("ms_rag.ui.prompts.prompt_text", return_value="sk_test") as mock_prompt:
        mock_select = MagicMock()
        mock_select.ask.return_value = openai_model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(
            configured_providers=["mistral"],
            credential_store=store,
        )

    assert result.provider == "openai"
    assert store.get("openai", "OPENAI_API_KEY") == "sk_test"
    mock_prompt.assert_called_once()


def test_local_huggingface_embedding_offers_optional_token_when_missing() -> None:
    module = VectorizationModule()
    store = CredentialStore()
    local_model = next(m for m in EMBEDDING_MODELS if m.provider == "huggingface")

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"), \
         patch("ms_rag.ui.prompts.prompt_text", return_value="hf_optional") as mock_prompt, \
         patch("ms_rag.ui.prompts.prompt_confirm", return_value=False):
        mock_select = MagicMock()
        mock_select.ask.return_value = local_model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(
            configured_providers=["mistral"],
            credential_store=store,
        )

    assert result.provider == "huggingface"
    assert store.get("huggingface", "HUGGINGFACEHUB_API_TOKEN") == "hf_optional"
    mock_prompt.assert_called_once()


def test_local_huggingface_with_token_can_switch_to_hosted_equivalent() -> None:
    module = VectorizationModule()
    store = CredentialStore()
    local_model = next(
        m
        for m in EMBEDDING_MODELS
        if m.provider == "huggingface" and m.model_id == "sentence-transformers/all-mpnet-base-v2"
    )

    with patch("ms_rag.ingestion.vectorization_module.questionary") as mock_q, \
         patch("ms_rag.ingestion.vectorization_module.Console"), \
         patch("ms_rag.ui.prompts.prompt_text", return_value="hf_optional"), \
         patch("ms_rag.ui.prompts.prompt_confirm", return_value=True):
        mock_select = MagicMock()
        mock_select.ask.return_value = local_model.model_id
        mock_q.select.return_value = mock_select
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = module.display_and_select(
            configured_providers=["mistral"],
            credential_store=store,
        )

    assert result.provider == "huggingface_endpoint"
    assert result.model_id == "hf-endpoint:sentence-transformers/all-mpnet-base-v2"


def test_local_huggingface_download_disables_xet_and_exports_token() -> None:
    with patch.dict(
        "os.environ",
        {
            "HF_HUB_DISABLE_XET": "",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "",
            "HF_TOKEN": "",
            "HUGGING_FACE_HUB_TOKEN": "",
        },
        clear=False,
    ):
        import os

        for key in [
            "HF_HUB_DISABLE_XET",
            "HF_HUB_DISABLE_SYMLINKS_WARNING",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        ]:
            os.environ.pop(key, None)

        _prepare_local_huggingface_download("hf_test")

        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
        assert os.environ["HF_TOKEN"] == "hf_test"
        assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test"


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

    def test_huggingface_endpoint_models_are_hosted(self) -> None:
        hosted = [m for m in EMBEDDING_MODELS if m.provider == "huggingface_endpoint"]
        assert hosted
        for m in hosted:
            assert m.is_local is False
            assert m.model_id.startswith("hf-endpoint:")

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
        known_providers = ["openai", "cohere", "huggingface", "huggingface_endpoint",
                           "local", "google_gemini", "mistral", "ollama"]
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

    def test_huggingface_endpoint_requires_token(self) -> None:
        module = VectorizationModule()
        config = EmbeddingModelConfig(
            provider="huggingface_endpoint",
            model_id="hf-endpoint:sentence-transformers/all-MiniLM-L6-v2",
        )
        with patch("ms_rag.ingestion.vectorization_module.resolve_credential", return_value=None):
            with pytest.raises(ValueError, match="HUGGINGFACEHUB_API_TOKEN"):
                module.get_embeddings(config)

    def test_huggingface_endpoint_uses_hosted_embedding_class(self) -> None:
        module = VectorizationModule()
        config = EmbeddingModelConfig(
            provider="huggingface_endpoint",
            model_id="hf-endpoint:sentence-transformers/all-MiniLM-L6-v2",
        )
        with patch("ms_rag.ingestion.vectorization_module.resolve_credential", return_value="hf_token"), \
             patch("langchain_huggingface.HuggingFaceEndpointEmbeddings") as mock_endpoint:
            module.get_embeddings(config)

        mock_endpoint.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token="hf_token",
        )

    def test_ollama_embeddings_default_to_local_base_when_api_key_present(self) -> None:
        module = VectorizationModule()
        store = MagicMock()
        store.get.side_effect = lambda provider, field: {
            ("ollama", "OLLAMA_API_KEY"): "ollama-token",
            ("ollama", "OLLAMA_BASE_URL"): None,
        }.get((provider, field))
        store.all_providers.return_value = ["ollama"]

        config = EmbeddingModelConfig(
            provider="ollama",
            model_id="gpt-oss:120b",
            local_path="gpt-oss:120b",
        )

        with patch("langchain_ollama.OllamaEmbeddings") as mock_embeddings:
            module.get_embeddings(config, credential_store=store)

        mock_embeddings.assert_called_once_with(
            model="gpt-oss:120b",
            base_url="http://localhost:11434",
            client_kwargs={"headers": {"Authorization": "Bearer ollama-token"}},
        )

    def test_ollama_embeddings_reject_cloud_base_url(self) -> None:
        module = VectorizationModule()
        store = MagicMock()
        store.get.side_effect = lambda provider, field: {
            ("ollama", "OLLAMA_API_KEY"): "ollama-token",
            ("ollama", "OLLAMA_BASE_URL"): "https://ollama.com/v1",
        }.get((provider, field))
        store.all_providers.return_value = ["ollama"]

        config = EmbeddingModelConfig(
            provider="ollama",
            model_id="nomic-embed-text",
            local_path="nomic-embed-text",
        )

        with pytest.raises(ValueError, match="chat models only"):
            module.get_embeddings(config, credential_store=store)
