"""Context Compression Module for MS_RAG.

Interactive configuration and LangChain document compressor factory for
all 6 supported context compression techniques.

Requirement 14:
- Ask yes/no for compression (14.1)
- Display all 6 techniques as a numbered checkbox (14.2)
- Reject zero-technique selection; allow 1-6 techniques (14.3)
- Prompt similarity threshold for Embeddings Filter (14.4)
- Block LLM-requiring techniques if no LLM provider configured (14.5)
- Store compression_enabled=True with ordered techniques + params (14.6)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

from ms_rag.models import CompressionConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric

# Techniques that require a configured LLM (Req 14.5)
LLM_REQUIRED_TECHNIQUES: frozenset[str] = frozenset(
    {"llm_chain_extraction", "summary_compression"}
)

COMPRESSION_TECHNIQUES: list[str] = [
    "llm_chain_extraction",
    "embeddings_filter",
    "document_compressor_pipeline",
    "redundancy_removal",
    "contextual_compression",
    "summary_compression",
]

# Display metadata for each technique
TECHNIQUE_INFO: dict[str, dict[str, str]] = {
    "llm_chain_extraction": {
        "display": "LLM Chain Extraction",
        "description": "Extracts only the relevant sentences from each chunk using an LLM [requires LLM]",
    },
    "embeddings_filter": {
        "display": "Embeddings Filter",
        "description": "Removes chunks whose cosine similarity to the query falls below a threshold",
    },
    "document_compressor_pipeline": {
        "display": "Document Compressor Pipeline",
        "description": "Sequential application of multiple compressors in user-defined order",
    },
    "redundancy_removal": {
        "display": "Redundancy Removal",
        "description": "Deduplicates chunks by pairwise embedding similarity",
    },
    "contextual_compression": {
        "display": "Contextual Compression (LangChain ContextualCompressionRetriever)",
        "description": "Wraps the retriever with LangChain's built-in contextual compression",
    },
    "summary_compression": {
        "display": "Summary-Based Compression",
        "description": "Replaces each chunk with an LLM-generated summary [requires LLM]",
    },
}


class ContextCompressor:
    """Interactive configuration and LangChain compressor factory.

    Usage::

        compressor = ContextCompressor()
        config = compressor.configure(configured_providers=["openai"])
        if config:
            compressor_obj = compressor.get_compressor(config, llm, embeddings)
    """

    def configure(
        self,
        configured_providers: list[str] | None = None,
    ) -> CompressionConfig | None:
        """Interactive yes/no → checkbox → params → return config.

        Requirement 14.1-14.6.

        Args:
            configured_providers: Provider IDs that have credentials configured.
                                  Used to block LLM-requiring techniques when no
                                  LLM is available.

        Returns:
            CompressionConfig if enabled, None if user declines.
        """
        console = Console()
        has_llm = bool(configured_providers)

        console.print("\n[bold cyan]Step 13 — Context Compression[/bold cyan]\n")

        wants_compression: bool = questionary.confirm(
            "  Do you want to enable context compression?",
            default=False,
        ).ask()

        if not wants_compression:
            console.print("  [dim]Context compression disabled.[/dim]")
            return None

        choices = []
        for tid in COMPRESSION_TECHNIQUES:
            info = TECHNIQUE_INFO[tid]
            is_llm_required = tid in LLM_REQUIRED_TECHNIQUES
            blocked = is_llm_required and not has_llm

            title = f"{info['display']}  —  {info['description']}"
            if blocked:
                title += "  [BLOCKED: no LLM configured]"

            choices.append(
                questionary.Choice(
                    title=title,
                    value=tid if not blocked else f"__blocked_{tid}__",
                )
            )

        selected: list[str] | None = None
        while True:
            raw: list[str] = questionary.checkbox(
                "  Select compression techniques (1-6, applied in checklist order):",
                choices=choices,
            ).ask()

            if raw is None:
                raw = []

            # Filter out blocked selections
            valid = [t for t in raw if not t.startswith("__blocked_")]

            if not valid:
                console.print(
                    "[red]  ✗ Please select at least one technique.[/red]"
                )
                continue

            if len(valid) > 6:
                console.print(
                    "[red]  ✗ Maximum 6 techniques allowed.[/red]"
                )
                continue

            selected = valid
            break

        # Embeddings filter threshold (Req 14.4)
        similarity_threshold = 0.75
        if "embeddings_filter" in selected:
            similarity_threshold = self._prompt_threshold(console)

        config = CompressionConfig(
            techniques=selected,
            similarity_threshold=similarity_threshold,
        )

        console.print(
            f"[green]  ✓ Compression: [bold]{', '.join(selected)}[/bold] "
            f"| threshold={similarity_threshold}[/green]"
        )
        return config

    def get_compressor(
        self,
        config: CompressionConfig,
        llm: object | None,
        embeddings: object | None,
        base_retriever: object | None = None,
    ) -> object:
        """Build and return a chained LangChain document compressor.

        Techniques are applied in the order stored in config.techniques.

        Args:
            config:         CompressionConfig with ordered techniques list.
            llm:            LangChain BaseChatModel (needed for LLM techniques).
            embeddings:     LangChain Embeddings instance (for embeddings_filter).
            base_retriever: Optional base retriever (for contextual_compression).

        Returns:
            A LangChain BaseDocumentCompressor or ContextualCompressionRetriever.

        Raises:
            ImportError: If required LangChain packages are not installed.
        """
        compressors = []

        for technique in config.techniques:
            comp = self._build_compressor(technique, config, llm, embeddings)
            if comp is not None:
                compressors.append(comp)

        if not compressors:
            return None

        if len(compressors) == 1:
            single = compressors[0]
        else:
            from langchain_classic.retrievers.document_compressors import (  # noqa: PLC0415
                DocumentCompressorPipeline,
            )
            single = DocumentCompressorPipeline(transformers=compressors)

        # Wrap with ContextualCompressionRetriever if base_retriever provided
        if base_retriever is not None:
            from langchain_classic.retrievers import ContextualCompressionRetriever  # noqa: PLC0415
            return ContextualCompressionRetriever(
                base_compressor=single,
                base_retriever=base_retriever,
            )

        return single

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_compressor(
        self,
        technique: str,
        config: CompressionConfig,
        llm: object | None,
        embeddings: object | None,
    ) -> object | None:
        """Build a single LangChain compressor for the given technique."""

        if technique == "llm_chain_extraction":
            if llm is None:
                return None
            from langchain_classic.retrievers.document_compressors import (  # noqa: PLC0415
                LLMChainExtractor,
            )
            return LLMChainExtractor.from_llm(llm)  # type: ignore[arg-type]

        if technique == "embeddings_filter":
            if embeddings is None:
                return None
            from langchain_classic.retrievers.document_compressors import (  # noqa: PLC0415
                EmbeddingsFilter,
            )
            return EmbeddingsFilter(
                embeddings=embeddings,  # type: ignore[arg-type]
                similarity_threshold=config.similarity_threshold,
            )

        if technique == "redundancy_removal":
            if embeddings is None:
                return None
            from langchain_community.document_transformers import (  # noqa: PLC0415
                EmbeddingsRedundantFilter,
            )
            return EmbeddingsRedundantFilter(embeddings=embeddings)  # type: ignore[arg-type]

        if technique == "contextual_compression":
            # This is handled at the get_compressor level with base_retriever
            return None

        if technique == "document_compressor_pipeline":
            # Meta-technique: pipeline is assembled at the end
            return None

        if technique == "summary_compression":
            if llm is None:
                return None
            from langchain_classic.retrievers.document_compressors import (  # noqa: PLC0415
                LLMChainFilter,
            )
            return LLMChainFilter.from_llm(llm)  # type: ignore[arg-type]

        return None

    def _prompt_threshold(self, console: object) -> float:
        """Prompt for Embeddings Filter similarity threshold (Req 14.4)."""
        while True:
            raw: str = questionary.text(
                "  Embeddings filter similarity threshold (0.0-1.0, default 0.75):",
                default="0.75",
            ).ask()

            if not raw or not raw.strip():
                return 0.75

            try:
                value = float(raw.strip())
                validate_numeric(value, 0.0, 1.0, "embeddings_threshold")
                return value
            except ValueError:
                console.print("[red]  ✗ Please enter a decimal number.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]
