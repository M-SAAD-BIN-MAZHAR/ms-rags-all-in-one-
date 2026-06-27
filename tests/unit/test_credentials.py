"""Unit tests for shared credential resolution utilities."""

from __future__ import annotations

from ms_rag.models import CredentialStore
from ms_rag.utils.credentials import (
    DEFAULT_LLM_MODELS,
    resolve_credential,
    resolve_model_id,
    resolve_ollama_connection,
)


class TestResolveCredential:
    def test_reads_from_credential_store_by_provider(self) -> None:
        store = CredentialStore()
        store.set("openai", "OPENAI_API_KEY", "sk-from-store")
        assert resolve_credential("OPENAI_API_KEY", store, "openai") == "sk-from-store"

    def test_falls_back_to_any_provider_with_matching_field(self) -> None:
        store = CredentialStore()
        store.set("openai", "OPENAI_API_KEY", "sk-cross-provider")
        assert resolve_credential("OPENAI_API_KEY", store, "cohere") == "sk-cross-provider"

    def test_returns_none_when_missing(self) -> None:
        store = CredentialStore()
        assert resolve_credential("MS_RAG_NONEXISTENT_FIELD_XYZ", store) is None


class TestResolveModelId:
    def test_default_openai_uses_catalog_model(self) -> None:
        assert resolve_model_id("openai", "default") == DEFAULT_LLM_MODELS["openai"]

    def test_default_cohere_does_not_use_removed_alias(self) -> None:
        assert resolve_model_id("cohere", "default") == "command-a-03-2025"
        assert DEFAULT_LLM_MODELS["cohere"] != "command-r-plus"

    def test_explicit_model_id_preserved(self) -> None:
        assert resolve_model_id("openai", "gpt-4o-mini") == "gpt-4o-mini"

    def test_removed_cohere_alias_fails_early(self) -> None:
        try:
            resolve_model_id("cohere", "command-r-plus")
        except ValueError as exc:
            assert "removed" in str(exc)
            assert "command-a-03-2025" in str(exc)
        else:
            raise AssertionError("Expected removed Cohere model to fail early")

    def test_azure_default_uses_deployment_from_store(self) -> None:
        store = CredentialStore()
        store.set("azure_openai", "AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy")
        assert resolve_model_id("azure_openai", "default", store) == "my-deploy"

    def test_ollama_default_uses_model_from_store(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_MODEL_NAME", "llama3.2")
        assert resolve_model_id("ollama", "default", store) == "llama3.2"


class TestResolveOllamaConnection:
    def test_local_defaults_without_api_key(self) -> None:
        base_url, client_kwargs = resolve_ollama_connection()
        assert base_url == "http://localhost:11434"
        assert client_kwargs == {}

    def test_cloud_defaults_when_api_key_exists(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_API_KEY", "ollama-token")
        base_url, client_kwargs = resolve_ollama_connection(store)
        assert base_url == "https://ollama.com"
        assert client_kwargs == {
            "headers": {"Authorization": "Bearer ollama-token"},
        }

    def test_store_base_url_overrides_default(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_API_KEY", "ollama-token")
        store.set("ollama", "OLLAMA_BASE_URL", "https://ollama.com/v1")
        base_url, client_kwargs = resolve_ollama_connection(store)
        assert base_url == "https://ollama.com"
        assert client_kwargs == {
            "headers": {"Authorization": "Bearer ollama-token"},
        }

    def test_embeddings_default_to_local_even_with_api_key(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_API_KEY", "ollama-token")
        base_url, client_kwargs = resolve_ollama_connection(store, usage="embedding")
        assert base_url == "http://localhost:11434"
        assert client_kwargs == {
            "headers": {"Authorization": "Bearer ollama-token"},
        }

    def test_embeddings_reject_ollama_cloud_base_url(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_API_KEY", "ollama-token")
        store.set("ollama", "OLLAMA_BASE_URL", "https://ollama.com/v1")
        try:
            resolve_ollama_connection(store, usage="embedding")
        except ValueError as exc:
            assert "chat models only" in str(exc)
        else:
            raise AssertionError("Expected ValueError for Ollama Cloud embeddings")
