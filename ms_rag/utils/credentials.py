"""Shared credential resolution for MS_RAG runtime modules.

Credentials are stored in CredentialStore keyed by provider ID and field name
(e.g. openai → OPENAI_API_KEY). Runtime factories also fall back to os.environ
and can search across all configured providers when a field name is unique.
"""

from __future__ import annotations

import os
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ms_rag.models import CredentialStore

# Default chat model per provider when the CLI passes model_id="default"
DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "cohere": "command-r-plus",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "google_gemini": "gemini-1.5-pro",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
    "together_ai": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "replicate": "meta/meta-llama-3-8b-instruct",
    "azure_openai": "gpt-4o",
    "aws_bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "ollama": "llama3",
}


def resolve_credential(
    field: str,
    credential_store: object | None = None,
    provider: str | None = None,
) -> str | None:
    """Return a credential value from the store, then os.environ."""
    if credential_store is not None:
        store: CredentialStore = credential_store  # type: ignore[assignment]
        if provider:
            val = store.get(provider, field)
            if val:
                return val
        for pid in store.all_providers():
            val = store.get(pid, field)
            if val:
                return val
    env_val = os.getenv(field)
    return env_val if env_val else None


def resolve_model_id(
    provider: str,
    model_id: str,
    credential_store: object | None = None,
) -> str:
    """Resolve model_id='default' to a concrete model for the provider."""
    if model_id and model_id != "default":
        return model_id

    if provider == "azure_openai":
        deployment = resolve_credential(
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            credential_store,
            "azure_openai",
        )
        if deployment:
            return deployment

    if provider == "ollama":
        ollama_model = resolve_credential(
            "OLLAMA_MODEL_NAME",
            credential_store,
            "ollama",
        )
        if ollama_model:
            return ollama_model

    return DEFAULT_LLM_MODELS.get(provider, model_id)


def resolve_ollama_connection(
    credential_store: object | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve Ollama base URL and optional auth headers for local or cloud use."""
    api_key = resolve_credential("OLLAMA_API_KEY", credential_store, "ollama")
    base_url = resolve_credential("OLLAMA_BASE_URL", credential_store, "ollama")

    if not base_url:
        base_url = "https://ollama.com" if api_key else "http://localhost:11434"

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return base_url, client_kwargs
