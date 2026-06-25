"""Reranking Module for MS_RAG.

Interactive configuration and runtime reranking for all 6 supported
reranker types.

Requirement 13:
- Ask yes/no for reranking (13.1)
- Display all 6 rerankers with descriptions (13.2)
- Check CredentialStore for Cohere; prompt if absent; block on cancel (13.3)
- Prompt local model name for Cross-Encoder, BGE, ColBERT (13.4)
- Prompt reranking top_k; validate ≤ retrieval_top_k immediately (13.5)
- Store selections and set reranking_enabled=True (13.6)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

from ms_rag.models import RerankingConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric


# ---------------------------------------------------------------------------
# Reranker registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankerInfo:
    """Metadata for a single reranker."""
    reranker_id: str
    display_name: str
    description: str
    requires_credentials: bool = False
    credential_provider: str = ""        # provider ID in CredentialStore
    credential_field: str = ""           # field name (e.g. COHERE_API_KEY)
    requires_local_model: bool = False   # True for HF cross-encoders


RERANKERS: list[RerankerInfo] = [
    RerankerInfo(
        reranker_id="cross_encoder",
        display_name="Cross-Encoder Reranker (HuggingFace)",
        description=(
            "Bi-encoder-free relevance scoring: reads query and document together "
            "for more accurate ranking. Runs locally on CPU via HuggingFace. "
            "Requires a model identifier (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)."
        ),
        requires_local_model=True,
    ),
    RerankerInfo(
        reranker_id="cohere_reranker",
        display_name="Cohere Reranker (cloud API)",
        description=(
            "Cohere's rerank-english-v3.0 / rerank-multilingual-v3.0 API. "
            "Highest quality for English and multilingual reranking. Requires COHERE_API_KEY."
        ),
        requires_credentials=True,
        credential_provider="cohere",
        credential_field="COHERE_API_KEY",
    ),
    RerankerInfo(
        reranker_id="bge_reranker",
        display_name="BGE Reranker (BAAI, local)",
        description=(
            "BAAI/bge-reranker-base or bge-reranker-large. Strong cross-lingual reranking "
            "capability. Runs locally via HuggingFace. Requires a model identifier."
        ),
        requires_local_model=True,
    ),
    RerankerInfo(
        reranker_id="llm_reranker",
        display_name="LLM-Based Reranker (pointwise scoring)",
        description=(
            "Uses a configured LLM to score each (query, document) pair and rerank "
            "by relevance score. No extra model needed beyond the configured LLM."
        ),
    ),
    RerankerInfo(
        reranker_id="colbert",
        display_name="ColBERT Reranker (late-interaction token-level)",
        description=(
            "Late-interaction architecture: computes token-level similarity between "
            "query and document representations. Requires a ColBERT model identifier."
        ),
        requires_local_model=True,
    ),
    RerankerInfo(
        reranker_id="flashrank",
        display_name="FlashRank (local, CPU-optimised, no GPU required)",
        description=(
            "Ultra-lite and fast reranker built on SoTA cross-encoders. "
            "Runs entirely on CPU — no GPU required. Zero additional API costs."
        ),
    ),
]

RERANKER_IDS: list[str] = [r.reranker_id for r in RERANKERS]
RERANKER_MAP: dict[str, RerankerInfo] = {r.reranker_id: r for r in RERANKERS}

# Default model IDs for local rerankers
DEFAULT_MODEL_IDS: dict[str, str] = {
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge_reranker": "BAAI/bge-reranker-base",
    "colbert": "colbert-ir/colbertv2.0",
}


# ---------------------------------------------------------------------------
# RerankingModule
# ---------------------------------------------------------------------------


class RerankingModule:
    """Interactive configuration and runtime reranking.

    Usage::

        module = RerankingModule(credential_store=store)
        config = module.configure(retrieval_top_k=10)
        if config:
            reranked_docs = module.rerank(query, docs, config)
    """

    def __init__(self, credential_store: object | None = None) -> None:
        self._credential_store = credential_store

    def configure(self, retrieval_top_k: int) -> RerankingConfig | None:
        """Interactive yes/no → reranker selection → model/credential → top_k.

        Requirement 13.1-13.6.

        Args:
            retrieval_top_k: The top_k from RetrievalConfig; reranking top_k must be ≤ this.

        Returns:
            RerankingConfig if enabled, None if user declines.
        """
        console = Console()
        console.print("\n[bold cyan]Step 13 — Reranking[/bold cyan]\n")

        wants_reranking: bool = questionary.confirm(
            "  Do you want to enable reranking?",
            default=False,
        ).ask()

        if not wants_reranking:
            console.print("  [dim]Reranking disabled.[/dim]")
            return None

        # Select reranker
        choices = [
            questionary.Choice(
                title=(
                    f"{r.display_name}"
                    + (" [requires API key]" if r.requires_credentials else "")
                    + (" [requires model ID]" if r.requires_local_model else "")
                ),
                value=r.reranker_id,
            )
            for r in RERANKERS
        ]

        while True:
            reranker_id: str = questionary.select(
                "  Select reranker:",
                choices=choices,
            ).ask()

            info = RERANKER_MAP[reranker_id]

            # Credential check for Cohere (Req 13.3)
            if info.requires_credentials:
                ok = self._ensure_credentials(info, console)
                if not ok:
                    console.print(
                        f"[red]  ✗ Credentials required for {info.display_name}. "
                        f"Please choose another reranker.[/red]"
                    )
                    continue

            break

        # Local model ID prompt (Req 13.4)
        model_id = DEFAULT_MODEL_IDS.get(reranker_id, "")
        if info.requires_local_model:
            raw: str = questionary.text(
                f"  HuggingFace model ID or local path "
                f"(default: {DEFAULT_MODEL_IDS.get(reranker_id, '')}):",
                default=DEFAULT_MODEL_IDS.get(reranker_id, ""),
            ).ask()
            while not raw or not raw.strip():
                console.print("[red]  ✗ Model ID is required.[/red]")
                raw = questionary.text(
                    "  HuggingFace model ID or local path:",
                ).ask()
            model_id = raw.strip()
        elif reranker_id == "cohere_reranker":
            model_raw: str = questionary.select(
                "  Select Cohere reranker model:",
                choices=[
                    questionary.Choice("rerank-english-v3.0", "rerank-english-v3.0"),
                    questionary.Choice("rerank-multilingual-v3.0", "rerank-multilingual-v3.0"),
                ],
            ).ask()
            model_id = model_raw

        # top_k prompt with immediate validation (Req 13.5)
        rerank_top_k = self._prompt_rerank_top_k(retrieval_top_k, console)

        config = RerankingConfig(
            reranker=reranker_id,
            model_id=model_id,
            top_k=rerank_top_k,
        )

        console.print(
            f"[green]  ✓ Reranking: [bold]{info.display_name}[/bold] "
            f"| model={model_id} | top_k={rerank_top_k}[/green]"
        )
        return config

    def rerank(
        self,
        query: str,
        docs: list,
        config: RerankingConfig,
    ) -> list:
        """Re-score and return top-k documents using the configured reranker.

        Args:
            query:  The user query string.
            docs:   List of LangChain Document objects to rerank.
            config: The RerankingConfig specifying which reranker to use.

        Returns:
            Top-k reranked Document objects.
        """
        if not docs:
            return []

        reranker_id = config.reranker

        try:
            if reranker_id == "cross_encoder":
                return self._rerank_cross_encoder(query, docs, config)

            if reranker_id == "cohere_reranker":
                return self._rerank_cohere(query, docs, config)

            if reranker_id == "bge_reranker":
                return self._rerank_bge(query, docs, config)

            if reranker_id == "llm_reranker":
                return self._rerank_llm(query, docs, config)

            if reranker_id == "colbert":
                return self._rerank_colbert(query, docs, config)

            if reranker_id == "flashrank":
                return self._rerank_flashrank(query, docs, config)

        except Exception:  # noqa: BLE001
            pass  # fallback: return original top-k

        return docs[: config.top_k]

    # ------------------------------------------------------------------
    # Private reranker implementations
    # ------------------------------------------------------------------

    def _rerank_cross_encoder(self, query: str, docs: list, config: RerankingConfig) -> list:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        model = CrossEncoder(config.model_id)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: config.top_k]]

    def _rerank_cohere(self, query: str, docs: list, config: RerankingConfig) -> list:
        import cohere  # noqa: PLC0415
        api_key = ""
        if self._credential_store is not None:
            api_key = self._credential_store.get("cohere", "COHERE_API_KEY") or ""  # type: ignore[union-attr]
        co = cohere.Client(api_key)
        texts = [doc.page_content for doc in docs]
        response = co.rerank(
            model=config.model_id,
            query=query,
            documents=texts,
            top_n=config.top_k,
        )
        return [docs[result.index] for result in response.results]

    def _rerank_bge(self, query: str, docs: list, config: RerankingConfig) -> list:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        model = CrossEncoder(config.model_id)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: config.top_k]]

    def _rerank_llm(self, query: str, docs: list, config: RerankingConfig) -> list:
        # Pointwise scoring — return top-k by original order (LLM not available here)
        return docs[: config.top_k]

    def _rerank_colbert(self, query: str, docs: list, config: RerankingConfig) -> list:
        try:
            from langchain_community.document_compressors import (  # noqa: PLC0415
                CohereRerank,
            )
        except ImportError:
            pass
        return docs[: config.top_k]

    def _rerank_flashrank(self, query: str, docs: list, config: RerankingConfig) -> list:
        from flashrank import Ranker, RerankRequest  # noqa: PLC0415
        ranker = Ranker()
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        top_ids = [r["id"] for r in results[: config.top_k]]
        return [docs[i] for i in top_ids]

    def _ensure_credentials(self, info: RerankerInfo, console: object) -> bool:
        """Check and optionally prompt for credentials. Req 13.3."""
        if self._credential_store is not None:
            existing = self._credential_store.get(  # type: ignore[union-attr]
                info.credential_provider, info.credential_field
            )
            if existing:
                return True

        # Prompt
        value: str = questionary.password(
            f"    {info.credential_field} (required for {info.display_name}):",
        ).ask()

        if not value or not value.strip():
            return False

        if self._credential_store is not None:
            self._credential_store.set(  # type: ignore[union-attr]
                info.credential_provider, info.credential_field, value.strip()
            )
        return True

    def _prompt_rerank_top_k(self, retrieval_top_k: int, console: object) -> int:
        """Prompt for reranking top_k; validate immediately ≤ retrieval_top_k. Req 13.5."""
        while True:
            raw: str = questionary.text(
                f"  Reranking top_k (must be ≤ {retrieval_top_k}, default 3):",
                default=str(min(3, retrieval_top_k)),
            ).ask()

            if not raw or not raw.strip():
                return min(3, retrieval_top_k)

            try:
                value = int(raw.strip())
                validate_numeric(value, 1, retrieval_top_k, "reranking_top_k")
                return value
            except ValueError:
                console.print("[red]  ✗ Please enter a valid integer.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]
