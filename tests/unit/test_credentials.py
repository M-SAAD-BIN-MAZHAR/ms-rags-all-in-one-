"""Unit tests for shared credential resolution utilities."""

from __future__ import annotations

from ms_rag.models import CredentialStore
from ms_rag.utils.credentials import (
    DEFAULT_LLM_MODELS,
    resolve_credential,
    resolve_model_id,
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

    def test_explicit_model_id_preserved(self) -> None:
        assert resolve_model_id("openai", "gpt-4o-mini") == "gpt-4o-mini"

    def test_azure_default_uses_deployment_from_store(self) -> None:
        store = CredentialStore()
        store.set("azure_openai", "AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy")
        assert resolve_model_id("azure_openai", "default", store) == "my-deploy"

    def test_ollama_default_uses_model_from_store(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_MODEL_NAME", "llama3.2")
        assert resolve_model_id("ollama", "default", store) == "llama3.2"
