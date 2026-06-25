"""Vectorization Module for MS_RAG.

Defines all supported embedding models, filters them by configured
providers, and returns the correct LangChain Embeddings instance.

Requirement 8:
- Display models filtered by configured providers + always-available local (8.1)
- Include all 20+ required embedding models (8.2)
- Store model identifier and provider in PipelineConfig (8.3)
- Prompt for local model path/name for Ollama/local models (8.4)

Deprecation note (from audit):
    HuggingFace embeddings: use langchain-huggingface, NOT langchain-community.
    Ollama embeddings:      use langchain-ollama, NOT langchain-community.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import questionary
    from rich.console import Console
    from rich.table import Table
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from ms_rag.models import EmbeddingModelConfig


# ---------------------------------------------------------------------------
# Embedding model catalogue
# ---------------------------------------------------------------------------

# Providers that are always available regardless of credential config
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"local", "ollama", "huggingface"})


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
        display_name="Ollama (local — enter model name)",
        dimensions=0,
        is_local=True,
        description="Any Ollama embedding model running locally (e.g. nomic-embed-text)",
    ),
]

# ── Lookup helpers ────────────────────────────────────────────────────────

EMBEDDING_MODEL_IDS: list[str] = [m.model_id for m in EMBEDDING_MODELS]


def get_displayable_models(configured_providers: list[str]) -> list[EmbeddingModelInfo]:
    """Return models whose provider is configured, plus all local/Ollama/HF models.

    Requirement 8.1: filtered by configured providers + always-available local options.

    Args:
        configured_providers: Provider IDs that the user has credentials for.

    Returns:
        Filtered list of EmbeddingModelInfo instances.
    """
    provider_set = set(configured_providers)
    return [
        m for m in EMBEDDING_MODELS
        if m.provider in provider_set or m.provider in _LOCAL_PROVIDERS
    ]


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
    ) -> EmbeddingModelConfig:
        """Show filtered model list and return selected EmbeddingModelConfig.

        Requirement 8.1 — filter by provider credentials.
        Requirement 8.4 — prompt for local model path/name.

        Args:
            configured_providers: Provider IDs with credentials available.

        Returns:
            EmbeddingModelConfig with provider, model_id, and optional local_path.
        """
        console = Console()
        console.print("\n[bold cyan]Step 8 — Select Embedding Model[/bold cyan]\n")

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
                title=f"{m.display_name}  ({m.dimensions}d)  —  {m.description}",
                value=m.model_id,
            )
            for m in displayable
        ]

        selected_id: str = questionary.select(
            "Select embedding model:",
            choices=choices,
        ).ask()

        # Find the selected model info
        selected_info = next(
            (m for m in displayable if m.model_id == selected_id), None
        )

        # Prompt for local path/name for Ollama (Requirement 8.4)
        local_path: str | None = None
        if selected_info and selected_info.model_id == "__user_specified__":
            local_path = questionary.text(
                "  Enter Ollama model name (e.g. nomic-embed-text, mxbai-embed-large):",
            ).ask()
            if local_path:
                local_path = local_path.strip()
                selected_id = local_path  # use model name as the ID

        provider = selected_info.provider if selected_info else "huggingface"

        config = EmbeddingModelConfig(
            provider=provider,
            model_id=selected_id,
            local_path=local_path,
        )

        console.print(
            f"[green]  ✓ Embedding model: [bold]{selected_id}[/bold] "
            f"(provider: {provider})[/green]"
        )

        return config

    def get_embeddings(self, config: EmbeddingModelConfig) -> object:
        """Return the appropriate LangChain Embeddings instance for *config*.

        Package routing (current as of 2025):
            openai        → langchain_openai.OpenAIEmbeddings
            cohere        → langchain_cohere.CohereEmbeddings
            huggingface   → langchain_huggingface.HuggingFaceEmbeddings
            google_gemini → langchain_google_genai.GoogleGenerativeAIEmbeddings
            mistral       → langchain_mistralai.MistralAIEmbeddings
            ollama        → langchain_ollama.OllamaEmbeddings
            local         → langchain_huggingface.HuggingFaceEmbeddings

        Args:
            config: The selected embedding model configuration.

        Returns:
            A LangChain Embeddings instance.

        Raises:
            ImportError: If the required integration package is not installed.
            ValueError:  If the provider is not recognised.
        """
        provider = config.provider
        model_id = config.model_id

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            return OpenAIEmbeddings(model=model_id)

        if provider == "cohere":
            from langchain_cohere import CohereEmbeddings  # noqa: PLC0415
            return CohereEmbeddings(model=model_id)

        if provider in ("huggingface", "local"):
            # Use langchain-huggingface (NOT deprecated langchain-community)
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
            return HuggingFaceEmbeddings(
                model_name=config.local_path or model_id
            )

        if provider == "google_gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415
            return GoogleGenerativeAIEmbeddings(model=model_id)

        if provider == "mistral":
            from langchain_mistralai import MistralAIEmbeddings  # noqa: PLC0415
            return MistralAIEmbeddings(model=model_id)

        if provider == "ollama":
            # Use langchain-ollama (NOT deprecated langchain-community)
            from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
            return OllamaEmbeddings(
                model=config.local_path or model_id
            )

        raise ValueError(
            f"Unsupported embedding provider: {provider!r}. "
            f"Supported: openai, cohere, huggingface, local, "
            f"google_gemini, mistral, ollama"
        )
