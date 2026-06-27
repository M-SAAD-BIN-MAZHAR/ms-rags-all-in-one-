"""Shared credential resolution for MS_RAG runtime modules.

Credentials are stored in CredentialStore keyed by provider ID and field name
(e.g. openai → OPENAI_API_KEY). Runtime factories also fall back to os.environ
and can search across all configured providers when a field name is unique.
"""

from __future__ import annotations

import os
from typing import Any
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ms_rag.models import CredentialStore

# Default chat model per provider when the CLI passes model_id="default"
DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "cohere": "command-a-03-2025",
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

REMOVED_LLM_MODELS: dict[str, dict[str, str]] = {
    "cohere": {
        "command-r-plus": (
            "Cohere removed the unversioned 'command-r-plus' alias on September 15, 2025. "
            "Use a live model such as 'command-a-03-2025' or 'command-r-plus-08-2024'."
        ),
        "command-r": (
            "Cohere removed the unversioned 'command-r' alias on September 15, 2025. "
            "Use a live model such as 'command-r-08-2024'."
        ),
        "command": (
            "Cohere removed the legacy 'command' model on September 15, 2025. "
            "Use a live Command model such as 'command-a-03-2025'."
        ),
        "command-light": (
            "Cohere removed the legacy 'command-light' model on September 15, 2025. "
            "Use a live Command model such as 'command-r7b-12-2024'."
        ),
    }
}


def validate_llm_model(provider: str, model_id: str) -> None:
    """Fail early for known removed provider models."""
    message = REMOVED_LLM_MODELS.get(provider, {}).get(model_id)
    if message:
        raise ValueError(message)


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
        validate_llm_model(provider, model_id)
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

    resolved = DEFAULT_LLM_MODELS.get(provider, model_id)
    validate_llm_model(provider, resolved)
    return resolved


def _normalize_ollama_base_url(base_url: str) -> str:
    """Normalize common Ollama base URL variants for SDK callers."""
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


def _is_ollama_cloud_url(base_url: str) -> bool:
    """Return True when the base URL points at Ollama Cloud."""
    parsed = urlparse(base_url)
    host = (parsed.netloc or parsed.path).lower()
    return host.startswith("ollama.com") or host.startswith("www.ollama.com")


def resolve_ollama_connection(
    credential_store: object | None = None,
    *,
    usage: str = "chat",
) -> tuple[str, dict[str, Any]]:
    """Resolve Ollama base URL and auth headers for chat or embedding use."""
    api_key = resolve_credential("OLLAMA_API_KEY", credential_store, "ollama")
    base_url = resolve_credential("OLLAMA_BASE_URL", credential_store, "ollama")

    if not base_url:
        if usage == "chat":
            base_url = "https://ollama.com" if api_key else "http://localhost:11434"
        else:
            base_url = "http://localhost:11434"
    else:
        base_url = _normalize_ollama_base_url(base_url)

    if usage == "embedding" and _is_ollama_cloud_url(base_url):
        raise ValueError(
            "Ollama Cloud currently supports chat models only. "
            "Use a local/self-hosted Ollama base URL for embedding models."
        )

    client_kwargs: dict[str, Any] = {}
    if api_key and (usage == "chat" or not _is_ollama_cloud_url(base_url)):
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return base_url, client_kwargs
