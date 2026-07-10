"""Vectorization Module for MS_RAG.

Defines all supported embedding models, filters them by configured
providers, and returns the correct LangChain Embeddings instance.

- Display models filtered by configured providers + always-available local (8.1)
- Include all 20+ required embedding models (8.2)
- Store model identifier and provider in PipelineConfig (8.3)
- Prompt for local model path/name for Ollama/local models (8.4)

Deprecation note (from audit):
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from pathlib import Path

try:
    import questionary
    from rich.console import Console
    from rich.table import Table
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from ms_rag.models import EmbeddingModelConfig
from ms_rag.utils.credentials import resolve_credential, resolve_ollama_connection


# ---------------------------------------------------------------------------
# Embedding model catalogue
# ---------------------------------------------------------------------------

# Providers that are always available regardless of credential config
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"local", "ollama", "huggingface"})
_HOSTED_HF_PROVIDER = "huggingface_endpoint"
_HOSTED_HF_PREFIX = "hf-endpoint:"


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Metadata for a single embedding model."""

    provider: str        # must match a PROVIDER_ID or be in _LOCAL_PROVIDERS
    model_id: str        # identifier passed to the LangChain Embeddings class
    display_name: str    # shown in the selection list
    dimensions: int      # output vector dimensions
    is_local: bool = False   # True for models that run locally (no API needed)
    description: str = ""


EMBEDDING_MODELS: list[EmbeddingModelInfo] = [
    # ── OpenAI ────────────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="openai",
        model_id="text-embedding-3-large",
        display_name="OpenAI text-embedding-3-large",
        dimensions=3072,
        description="Best OpenAI embedding; 3072-dim, supports dimension reduction",
    ),
    EmbeddingModelInfo(
        provider="openai",
        model_id="text-embedding-3-small",
        display_name="OpenAI text-embedding-3-small",
        dimensions=1536,
        description="Faster, cheaper OpenAI embedding with competitive quality",
    ),
    EmbeddingModelInfo(
        provider="openai",
        model_id="text-embedding-ada-002",
        display_name="OpenAI text-embedding-ada-002",
        dimensions=1536,
        description="Legacy OpenAI embedding — still widely used",
    ),
    # ── Cohere ────────────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="cohere",
        model_id="embed-english-v3.0",
        display_name="Cohere embed-english-v3.0",
        dimensions=1024,
        description="Cohere's best English embedding model",
    ),
    EmbeddingModelInfo(
        provider="cohere",
        model_id="embed-multilingual-v3.0",
        display_name="Cohere embed-multilingual-v3.0",
        dimensions=1024,
        description="Cohere multilingual embedding — 100+ languages",
    ),
    # ── HuggingFace / sentence-transformers ───────────────────────────────
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        display_name="all-MiniLM-L6-v2 (HuggingFace, local)",
        dimensions=384,
        is_local=True,
        description="Fast, lightweight — ideal for CPU-only setups",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="sentence-transformers/all-mpnet-base-v2",
        display_name="all-mpnet-base-v2 (HuggingFace, local)",
        dimensions=768,
        is_local=True,
        description="High-quality general-purpose sentence embeddings",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        display_name="multi-qa-mpnet-base-dot-v1 (HuggingFace, local)",
        dimensions=768,
        is_local=True,
        description="Optimised for semantic search / Q&A retrieval",
    ),
    # ── BGE (BAAI) ────────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="BAAI/bge-small-en-v1.5",
        display_name="BGE-Small-EN-v1.5 (local)",
        dimensions=384,
        is_local=True,
        description="Small, fast BGE model — good baseline for English",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="BAAI/bge-base-en-v1.5",
        display_name="BGE-Base-EN-v1.5 (local)",
        dimensions=768,
        is_local=True,
        description="Balanced BGE model — strong MTEB benchmark scores",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="BAAI/bge-large-en-v1.5",
        display_name="BGE-Large-EN-v1.5 (local)",
        dimensions=1024,
        is_local=True,
        description="Largest BGE English model — best quality",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="BAAI/bge-m3",
        display_name="BGE-M3 (local, multilingual)",
        dimensions=1024,
        is_local=True,
        description="BGE multilingual model — dense + sparse + multi-vector",
    ),
    # ── E5 (Microsoft) ────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="intfloat/e5-small-v2",
        display_name="E5-Small-v2 (local)",
        dimensions=384,
        is_local=True,
        description="Efficient E5 model for resource-constrained environments",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="intfloat/e5-base-v2",
        display_name="E5-Base-v2 (local)",
        dimensions=768,
        is_local=True,
        description="Strong general-purpose E5 embedding model",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="intfloat/e5-large-v2",
        display_name="E5-Large-v2 (local)",
        dimensions=1024,
        is_local=True,
        description="Best E5 quality — SOTA on many retrieval benchmarks",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="intfloat/e5-mistral-7b-instruct",
        display_name="E5-Mistral-7B-Instruct (local, GPU recommended)",
        dimensions=4096,
        is_local=True,
        description="LLM-based embedding using Mistral-7B — highest quality locally",
    ),
    # ── Instructor ────────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="hkunlp/instructor-base",
        display_name="Instructor-Base (local)",
        dimensions=768,
        is_local=True,
        description="Instruction-tuned embeddings — provide task instructions",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="hkunlp/instructor-large",
        display_name="Instructor-Large (local)",
        dimensions=768,
        is_local=True,
        description="Larger instruction-tuned embedding model",
    ),
    EmbeddingModelInfo(
        provider="huggingface",
        model_id="hkunlp/instructor-xl",
        display_name="Instructor-XL (local, GPU recommended)",
        dimensions=768,
        is_local=True,
        description="Largest Instructor model — best task-specific quality",
    ),
    # ── HuggingFace hosted embeddings (token/API, no local model download) ──
    EmbeddingModelInfo(
        provider=_HOSTED_HF_PROVIDER,
        model_id=f"{_HOSTED_HF_PREFIX}sentence-transformers/all-MiniLM-L6-v2",
        display_name="Hosted HuggingFace Inference API: all-MiniLM-L6-v2",
        dimensions=384,
        description="Hosted HuggingFace embedding endpoint; requires token, no local model download",
    ),
    EmbeddingModelInfo(
        provider=_HOSTED_HF_PROVIDER,
        model_id=f"{_HOSTED_HF_PREFIX}sentence-transformers/all-mpnet-base-v2",
        display_name="Hosted HuggingFace Inference API: all-mpnet-base-v2",
        dimensions=768,
        description="Hosted HuggingFace embedding endpoint; stronger quality, no local model download",
    ),
    EmbeddingModelInfo(
        provider=_HOSTED_HF_PROVIDER,
        model_id=f"{_HOSTED_HF_PREFIX}sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        display_name="Hosted HuggingFace Inference API: multilingual MiniLM",
        dimensions=384,
        description="Hosted multilingual HuggingFace embeddings; requires token, no local model download",
    ),
    # ── Google Gemini ─────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="google_gemini",
        model_id="models/text-embedding-004",
        display_name="Google text-embedding-004",
        dimensions=768,
        description="Google's latest embedding model with task-type support",
    ),
    # ── Mistral AI ────────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="mistral",
        model_id="mistral-embed",
        display_name="Mistral Embed",
        dimensions=1024,
        description="Mistral's embedding model — strong multilingual performance",
    ),
    # ── Ollama (local) ────────────────────────────────────────────────────
    EmbeddingModelInfo(
        provider="ollama",
        model_id="__user_specified__",
        display_name="Ollama embeddings (local/self-hosted only — enter model name)",
        dimensions=0,
        is_local=True,
        description="Any local/self-hosted Ollama embedding model (Ollama Cloud chat only; embeddings are not supported there)",
    ),
]

# ── Lookup helpers ────────────────────────────────────────────────────────

EMBEDDING_MODEL_IDS: list[str] = [m.model_id for m in EMBEDDING_MODELS]


def get_embedding_model_info(model_id: str) -> EmbeddingModelInfo | None:
    """Return catalogue metadata for a known embedding model id."""
    return next((m for m in EMBEDDING_MODELS if m.model_id == model_id), None)


def get_embedding_dimension(config: EmbeddingModelConfig | None) -> int | None:
    """Return the selected embedding dimension when the catalogue knows it."""
    if config is None:
        return None
    info = get_embedding_model_info(config.model_id)
    if info and info.dimensions > 0:
        return info.dimensions
    return None


def get_displayable_models(configured_providers: list[str]) -> list[EmbeddingModelInfo]:
    """Return all embedding models visible in Step 8.

    Chat providers and embedding providers are intentionally independent.
    If the selected embedding provider needs credentials that were not entered
    in Step 2, Step 8 prompts for them before ingestion starts.

    Args:
        configured_providers: Provider IDs selected for chat. Kept for API
            compatibility and future ranking/grouping behavior.

    Returns:
        Full list of EmbeddingModelInfo instances.
    """
    return list(EMBEDDING_MODELS)


def _hf_endpoint_model_id(model_id: str) -> str:
    """Return the HuggingFace repo id for a hosted endpoint model selection."""
    return model_id.removeprefix(_HOSTED_HF_PREFIX)


def _credential_provider_for_embedding(provider: str) -> str:
    """Map embedding provider IDs to CredentialStore provider IDs."""
    if provider == _HOSTED_HF_PROVIDER:
        return "huggingface"
    return provider


def _embedding_required_fields(provider: str) -> list[str]:
    """Return required credential fields for hosted embedding providers."""
    return {
        "openai": ["OPENAI_API_KEY"],
        "cohere": ["COHERE_API_KEY"],
        _HOSTED_HF_PROVIDER: ["HUGGINGFACEHUB_API_TOKEN"],
        "google_gemini": ["GOOGLE_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
    }.get(provider, [])


def _embedding_optional_fields(provider: str) -> list[str]:
    """Return optional credential fields that improve local embedding setup."""
    return {
        "huggingface": ["HUGGINGFACEHUB_API_TOKEN"],
        "local": ["HUGGINGFACEHUB_API_TOKEN"],
    }.get(provider, [])


def _hosted_hf_equivalent(model_id: str) -> EmbeddingModelInfo | None:
    """Return the hosted HF catalogue entry for a local HF model when available."""
    hosted_id = f"{_HOSTED_HF_PREFIX}{model_id}"
    return next(
        (
            model
            for model in EMBEDDING_MODELS
            if model.provider == _HOSTED_HF_PROVIDER and model.model_id == hosted_id
        ),
        None,
    )


def _prepare_local_huggingface_download(hf_token: str | None) -> None:
    """Make local HuggingFace model downloads reliable on Windows/older stacks."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


def local_huggingface_cache_path(model_id: str) -> Path | None:
    """Return the HuggingFace hub cache folder for a repo-style model id."""
    if not model_id or model_id.startswith(_HOSTED_HF_PREFIX):
        return None
    model_path = Path(model_id)
    if model_path.exists() or model_path.is_absolute() or "/" not in model_id:
        return None

    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache:
        hub_cache = Path(env_hub_cache)
    elif os.environ.get("HF_HOME"):
        hub_cache = Path(str(os.environ["HF_HOME"])) / "hub"
    else:
        try:
            from huggingface_hub import constants as hf_constants  # noqa: PLC0415

            hub_cache = Path(str(hf_constants.HF_HUB_CACHE))
        except Exception:  # noqa: BLE001
            hub_cache = Path.home() / ".cache" / "huggingface" / "hub"

    return hub_cache / f"models--{model_id.replace('/', '--')}"


def remove_local_huggingface_cache(model_id: str) -> Path | None:
    """Delete only the cache folder for one HuggingFace model id."""
    cache_path = local_huggingface_cache_path(model_id)
    if cache_path is None or not cache_path.exists():
        return None
    shutil.rmtree(cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# VectorizationModule
# ---------------------------------------------------------------------------


class VectorizationModule:
    """Interactive embedding model selector and LangChain Embeddings factory.

    Usage::

        module = VectorizationModule()
        config = module.display_and_select(configured_providers=["openai"])
        embeddings = module.get_embeddings(config)
    """

    def display_and_select(
        self,
        configured_providers: list[str],
        credential_store: object | None = None,
    ) -> EmbeddingModelConfig:
        """Show filtered model list and return selected EmbeddingModelConfig.

        filter by provider credentials.
        prompt for local model path/name.

        Args:
            configured_providers: Provider IDs with credentials available.
            credential_store: Optional CredentialStore used to collect embedding-only
                credentials when the embedding provider was not selected for chat.

        Returns:
            EmbeddingModelConfig with provider, model_id, and optional local_path.
        """
        console = Console()
        console.print("\n[bold cyan]Step 8 — Select Embedding Model[/bold cyan]\n")
        console.print(
            "[dim]  Pick the embedding model before the vector database. "
            "The vector dimension must match the collection/index you use; "
            "changing models later usually means creating a new collection or re-indexing.[/dim]\n"
        )

        displayable = get_displayable_models(configured_providers)
        if not displayable:
            # Fallback: show all local models if nothing configured
            displayable = [m for m in EMBEDDING_MODELS if m.is_local]
            console.print(
                "[yellow]  No provider credentials configured. "
                "Showing local models only.[/yellow]\n"
            )

        choices = [
            questionary.Choice(
                title=(
                    f"{m.display_name}  "
                    f"({m.dimensions if m.dimensions else 'custom'} dimensions)  —  "
                    f"{m.description}"
                ),
                value=m.model_id,
            )
            for m in displayable
        ]

        while True:
            selected_id = questionary.select(
                "Select embedding model:",
                choices=choices,
            ).ask()
            if selected_id is None:
                console.print(
                    "[yellow]  Selection cancelled — please choose a model.[/yellow]"
                )
                continue
            break

        # Find the selected model info
        selected_info = next(
            (m for m in displayable if m.model_id == selected_id), None
        )

        # Prompt for local path/name for Ollama
        local_path: str | None = None
        if selected_info and selected_info.model_id == "__user_specified__":
            console.print(
                "[dim]  Ollama embeddings require a local or self-hosted Ollama server. "
                "Ollama Cloud currently supports chat models only.[/dim]"
            )
            while True:
                local_path = questionary.text(
                    "  Enter Ollama model name (e.g. nomic-embed-text, mxbai-embed-large):",
                ).ask()
                if local_path is None:
                    console.print(
                        "[yellow]  Please enter an Ollama model name.[/yellow]"
                    )
                    continue
                local_path = local_path.strip()
                if not local_path:
                    console.print("[red]  ✗ Ollama model name is required.[/red]")
                    continue
                selected_id = local_path
                break

        # Warn about memory-intensive local HuggingFace embedding models
        if selected_info and selected_info.is_local and selected_info.dimensions >= 768:
            estimated_gb = ""
            if selected_info.model_id in ("intfloat/e5-mistral-7b-instruct",):
                estimated_gb = " (~14GB RAM for model weights)"
            elif selected_info.dimensions >= 1024:
                estimated_gb = " (~1-2GB RAM)"
            elif selected_info.dimensions >= 768:
                estimated_gb = " (~0.5-1GB RAM)"
            console.print(
                f"[yellow]  ⚠ Memory notice: this local model loads model weights into RAM{estimated_gb}. "
                "Ensure you have enough available memory before proceeding. "
                "Choose a 'Hosted HuggingFace Inference API' option if memory is constrained.[/yellow]"
            )

        provider = selected_info.provider if selected_info else "huggingface"
        if selected_info is not None:
            self._ensure_embedding_credentials(selected_info, credential_store, console)
            selected_info = self._maybe_switch_local_hf_to_hosted(
                selected_info,
                credential_store,
                console,
            )
            if selected_info.model_id != "__user_specified__":
                selected_id = selected_info.model_id
            provider = selected_info.provider

        config = EmbeddingModelConfig(
            provider=provider,
            model_id=selected_id,
            local_path=local_path,
        )

        console.print(
            f"[green]  ✓ Embedding model: [bold]{selected_id}[/bold] "
            f"(provider: {provider})[/green]"
        )
        if selected_info and selected_info.dimensions:
            console.print(
                f"[dim]  Vector dimension: {selected_info.dimensions}. "
                "Use a fresh collection/index if your existing database was built with a different dimension.[/dim]"
            )
        elif provider == "ollama":
            console.print(
                "[yellow]  ⚠ Custom Ollama dimension is model-dependent. "
                "Use a local/self-hosted Ollama embedding model and check its dimension before reusing an existing vector index.[/yellow]"
            )

        return config

    def _maybe_switch_local_hf_to_hosted(
        self,
        selected_info: EmbeddingModelInfo,
        credential_store: object | None,
        console: object,
    ) -> EmbeddingModelInfo:
        """Offer hosted HF when a user selected local HF but supplied a token."""
        if selected_info.provider not in {"huggingface", "local"}:
            return selected_info
        hosted = _hosted_hf_equivalent(selected_info.model_id)
        if hosted is None:
            return selected_info
        token = resolve_credential("HUGGINGFACEHUB_API_TOKEN", credential_store, "huggingface")
        if not token:
            console.print(
                "[yellow]  Note: this local HuggingFace option downloads model weights. "
                "Choose a 'Hosted HuggingFace Inference API' option if you want token-only embeddings.[/yellow]"
            )
            return selected_info

        from ms_rag.ui.prompts import prompt_confirm  # noqa: PLC0415

        console.print(
            "[yellow]  You selected a local HuggingFace embedding model. "
            "The token only authenticates the download; it does not make this hosted.[/yellow]"
        )
        use_hosted = prompt_confirm(
            f"  Use {hosted.display_name} instead to avoid local model download?",
            default=True,
            console=console,
        )
        if use_hosted:
            console.print(
                f"[green]  ✓ Switched to {hosted.display_name}[/green]"
            )
            return hosted
        return selected_info

    def _ensure_embedding_credentials(
        self,
        selected_info: EmbeddingModelInfo,
        credential_store: object | None,
        console: object,
    ) -> None:
        """Prompt for credentials required by the selected embedding provider."""
        if credential_store is None:
            return
        required_fields = _embedding_required_fields(selected_info.provider)
        optional_fields = _embedding_optional_fields(selected_info.provider)
        if not required_fields and not optional_fields:
            return

        from ms_rag.config.credential_manager import _is_secret_field  # noqa: PLC0415
        from ms_rag.ui.prompts import prompt_text  # noqa: PLC0415

        provider_id = _credential_provider_for_embedding(selected_info.provider)
        missing_fields = [
            field
            for field in required_fields
            if not resolve_credential(field, credential_store, provider_id)
        ]
        optional_missing = [
            field
            for field in optional_fields
            if not resolve_credential(field, credential_store, provider_id)
        ]
        if not missing_fields and not optional_missing:
            return

        console.print(
            f"\n[bold cyan]Credentials for {selected_info.display_name}:[/bold cyan]"
        )
        for field in missing_fields:
            value = prompt_text(
                f"  {field}:",
                secret=_is_secret_field(field),
                required=True,
                console=console,
            )
            credential_store.set(provider_id, field, str(value).strip())  # type: ignore[union-attr]
        for field in optional_missing:
            value = prompt_text(
                f"  {field} (optional, press Enter to skip):",
                secret=_is_secret_field(field),
                required=False,
                console=console,
            )
            if str(value).strip():
                credential_store.set(provider_id, field, str(value).strip())  # type: ignore[union-attr]

    def get_embeddings(
        self,
        config: EmbeddingModelConfig,
        credential_store: object | None = None,
    ) -> object:
        """Return the appropriate LangChain Embeddings instance for *config*.

        Package routing (current as of 2025):
            openai        → langchain_openai.OpenAIEmbeddings
            cohere        → langchain_cohere.CohereEmbeddings
            huggingface   → langchain_huggingface.HuggingFaceEmbeddings
            huggingface_endpoint → langchain_huggingface.HuggingFaceEndpointEmbeddings
            google_gemini → langchain_google_genai.GoogleGenerativeAIEmbeddings
            mistral       → langchain_mistralai.MistralAIEmbeddings
            ollama        → langchain_ollama.OllamaEmbeddings
            local         → langchain_huggingface.HuggingFaceEmbeddings

        Args:
            config:           The selected embedding model configuration.
            credential_store: Optional CredentialStore for API key lookup.

        Returns:
            A LangChain Embeddings instance.

        Raises:
            ImportError: If the required integration package is not installed.
            ValueError:  If the provider is not recognised.
        """
        provider = config.provider
        model_id = config.model_id

        def _env(field: str, provider_id: str | None = None) -> str | None:
            return resolve_credential(
                field,
                credential_store,
                provider_id or provider,
            )

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            openai_kwargs: dict[str, str] = {"model": model_id}
            api_key = _env("OPENAI_API_KEY")
            if api_key:
                openai_kwargs["openai_api_key"] = api_key
            org_id = _env("OPENAI_ORG_ID")
            if org_id:
                openai_kwargs["openai_organization"] = org_id
            return OpenAIEmbeddings(**openai_kwargs)  # type: ignore[arg-type]

        if provider == "cohere":
            from langchain_cohere import CohereEmbeddings  # noqa: PLC0415
            return CohereEmbeddings(
                model=model_id,
                cohere_api_key=_env("COHERE_API_KEY"),  # type: ignore[arg-type]
            )

        if provider in ("huggingface", "local"):
            # Use langchain-huggingface (NOT deprecated langchain-community)
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
            hf_token = _env("HUGGINGFACEHUB_API_TOKEN", "huggingface")
            _prepare_local_huggingface_download(hf_token)
            if hf_token:
                return HuggingFaceEmbeddings(
                    model_name=config.local_path or model_id,
                    model_kwargs={"token": hf_token},
                )
            return HuggingFaceEmbeddings(
                model_name=config.local_path or model_id,
            )

        if provider == _HOSTED_HF_PROVIDER:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings  # noqa: PLC0415

            hf_token = _env("HUGGINGFACEHUB_API_TOKEN", "huggingface")
            if not hf_token:
                raise ValueError(
                    "HUGGINGFACEHUB_API_TOKEN is required for hosted HuggingFace embeddings."
                )
            return HuggingFaceEndpointEmbeddings(
                model=_hf_endpoint_model_id(model_id),
                huggingfacehub_api_token=hf_token,
            )

        if provider == "google_gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415
            return GoogleGenerativeAIEmbeddings(
                model=model_id,
                google_api_key=_env("GOOGLE_API_KEY"),  # type: ignore[arg-type]
            )

        if provider == "mistral":
            from langchain_mistralai import MistralAIEmbeddings  # noqa: PLC0415
            return MistralAIEmbeddings(
                model=model_id,
                api_key=_env("MISTRAL_API_KEY"),  # type: ignore[arg-type]
            )

        if provider == "ollama":
            # Use langchain-ollama (NOT deprecated langchain-community)
            from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
            base_url, client_kwargs = resolve_ollama_connection(
                credential_store,
                usage="embedding",
            )
            return OllamaEmbeddings(
                model=config.local_path or model_id,
                base_url=base_url,
                client_kwargs=client_kwargs,
            )

        raise ValueError(
            f"Unsupported embedding provider: {provider!r}. "
            f"Supported: openai, cohere, huggingface, huggingface_endpoint, local, "
            f"google_gemini, mistral, ollama"
        )
