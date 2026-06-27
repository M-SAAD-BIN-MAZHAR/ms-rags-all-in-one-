"""Retrieval Strategy Module for MS_RAG.

Interactive configuration and LangChain retriever factory for all 10
supported retrieval strategies.

Requirement 12:
- Display numbered list of all 10 strategies (12.1)
- Prompt top_k (1-1000, default 5) (12.2)
- Prompt hybrid alpha (0.0-1.0) for Hybrid Search (12.3)
- Prompt lambda_diversity (0.0-1.0) for MMR (12.4)
- Prompt metadata fields (name, data_type, description) for Self-Query (12.5)
- Prompt ensemble weights (must sum to 1.0 ±0.01) for Ensemble (12.6)
- Store all parameters in PipelineConfig (12.7)
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

from ms_rag.models import MetadataField, RetrievalConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_ensemble_weights, validate_numeric


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyInfo:
    """Metadata for a single retrieval strategy."""
    strategy_id: str
    display_name: str
    description: str


STRATEGIES: list[StrategyInfo] = [
    StrategyInfo(
        strategy_id="dense_vector",
        display_name="Dense Vector Search (cosine similarity)",
        description="Embeds the query and retrieves top-k chunks by cosine similarity",
    ),
    StrategyInfo(
        strategy_id="keyword_bm25",
        display_name="Keyword Search — BM25",
        description="Classic BM25 ranking — fast, no embeddings needed, strong for exact terms",
    ),
    StrategyInfo(
        strategy_id="tfidf",
        display_name="Keyword Search — TF-IDF",
        description="TF-IDF term-frequency ranking — lightweight and interpretable",
    ),
    StrategyInfo(
        strategy_id="hybrid",
        display_name="Hybrid Search (vector + keyword, configurable alpha)",
        description="Combines dense vector and keyword scores; alpha controls the balance",
    ),
    StrategyInfo(
        strategy_id="mmr",
        display_name="Maximum Marginal Relevance (MMR)",
        description="Retrieves diverse results by penalising redundancy; lambda controls diversity",
    ),
    StrategyInfo(
        strategy_id="ensemble",
        display_name="Ensemble Retrieval (multiple retrievers, configurable weights)",
        description="Combines 2+ sub-retrievers with user-defined weights (must sum to 1.0)",
    ),
    StrategyInfo(
        strategy_id="parent_child",
        display_name="Parent-Child Retrieval",
        description="Retrieves small child chunks but returns their larger parent context",
    ),
    StrategyInfo(
        strategy_id="multi_vector",
        display_name="Multi-Vector Retrieval (summary + full chunk)",
        description="Stores summary embeddings for retrieval but returns full chunk content",
    ),
    StrategyInfo(
        strategy_id="self_query",
        display_name="Self-Query Retrieval (LLM metadata filtering)",
        description="LLM generates structured metadata filter + vector query from natural language",
    ),
    StrategyInfo(
        strategy_id="time_weighted",
        display_name="Time-Weighted Retrieval (recency-boosted similarity)",
        description="Boosts recently added documents in ranking alongside cosine similarity",
    ),
]

STRATEGY_IDS: list[str] = [s.strategy_id for s in STRATEGIES]
STRATEGY_MAP: dict[str, StrategyInfo] = {s.strategy_id: s for s in STRATEGIES}

DATA_TYPES: list[str] = ["string", "integer", "float", "date"]


def _extract_corpus_texts(vector_store: object) -> list[str]:
    """Extract indexed document texts from a vector store for keyword retrievers."""
    texts: list[str] = []

    get_fn = getattr(vector_store, "get", None)
    if callable(get_fn):
        try:
            result = get_fn()
            if isinstance(result, dict):
                documents = result.get("documents") or []
            else:
                documents = getattr(result, "documents", []) or []
            for doc in documents:
                if isinstance(doc, str) and doc.strip():
                    texts.append(doc)
                elif hasattr(doc, "page_content") and doc.page_content.strip():
                    texts.append(doc.page_content)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not extract documents from vector_store.get(); keyword retrieval may degrade: {exc}",
                stacklevel=2,
            )

    if not texts:
        collection = getattr(vector_store, "_collection", None)
        if collection is not None and hasattr(collection, "get"):
            try:
                raw = collection.get(include=["documents"])
                for doc in raw.get("documents", []) or []:
                    if isinstance(doc, str) and doc.strip():
                        texts.append(doc)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not extract documents from vector store collection; keyword retrieval may degrade: {exc}",
                    stacklevel=2,
                )

    return texts


# ---------------------------------------------------------------------------
# RetrievalStrategyModule
# ---------------------------------------------------------------------------


class RetrievalStrategyModule:
    """Interactive configuration and LangChain retriever factory.

    Usage::

        module = RetrievalStrategyModule()
        config = module.configure()
        retriever = module.get_retriever(config, vector_store)
    """

    def configure(self) -> RetrievalConfig:
        """Interactive flow: select strategy → prompt parameters → return config.

        Requirement 12.1-12.7.
        """
        console = Console()
        console.print("\n[bold cyan]Step 11 — Retrieval Strategy[/bold cyan]\n")

        # Step 1: strategy selection
        choices = [
            questionary.Choice(
                title=f"{i + 1:2}. {s.display_name}  —  {s.description}",
                value=s.strategy_id,
            )
            for i, s in enumerate(STRATEGIES)
        ]
        from ms_rag.ui.prompts import prompt_select  # noqa: PLC0415

        strategy_id = prompt_select(
            "Select retrieval strategy:",
            choices,
            console=console,
        )

        # Step 2: top_k (Req 12.2)
        top_k = self._prompt_int(
            prompt="  Number of chunks to retrieve (top_k, default 5):",
            default=5, min_val=1, max_val=1000,
            field_name="retrieval_top_k", console=console,
        )

        alpha: float | None = None
        lambda_diversity: float | None = None
        metadata_fields: list[MetadataField] | None = None
        ensemble_weights: list[float] | None = None
        ensemble_sub_retrievers: list[str] | None = None

        # Strategy-specific params
        if strategy_id == "hybrid":
            alpha = self._prompt_float(
                prompt="  Alpha weight (vector vs keyword, 0.0=all keyword, 1.0=all vector, default 0.5):",
                default=0.5, min_val=0.0, max_val=1.0,
                field_name="hybrid_alpha", console=console,
            )

        elif strategy_id == "mmr":
            lambda_diversity = self._prompt_float(
                prompt="  Lambda diversity (0.0=max diversity, 1.0=max relevance, default 0.5):",
                default=0.5, min_val=0.0, max_val=1.0,
                field_name="mmr_lambda", console=console,
            )

        elif strategy_id == "self_query":
            metadata_fields = self._prompt_metadata_fields(console)

        elif strategy_id == "ensemble":
            ensemble_sub_retrievers, ensemble_weights = self._prompt_ensemble(console)

        config = RetrievalConfig(
            strategy=strategy_id,
            top_k=top_k,
            alpha=alpha,
            lambda_diversity=lambda_diversity,
            metadata_fields=metadata_fields,
            ensemble_weights=ensemble_weights,
            ensemble_sub_retrievers=ensemble_sub_retrievers,
        )

        console.print(
            f"[green]  ✓ Retrieval: [bold]{STRATEGY_MAP[strategy_id].display_name}[/bold] "
            f"| top_k={top_k}[/green]"
        )
        return config

    def get_retriever(
        self,
        config: RetrievalConfig,
        vector_store: object,
        llm: object | None = None,
    ) -> object:
        """Return the appropriate LangChain BaseRetriever for *config*.

        Args:
            config:       The RetrievalConfig from configure().
            vector_store: Initialised LangChain VectorStore.
            llm:          Optional LLM for Self-Query retrieval.

        Returns:
            A LangChain BaseRetriever instance.

        Raises:
            ValueError:  If the strategy is not recognised.
            ImportError: If required packages are missing.
        """
        strategy = config.strategy

        if strategy == "dense_vector":
            return vector_store.as_retriever(  # type: ignore[union-attr]
                search_type="similarity",
                search_kwargs={"k": config.top_k},
            )

        if strategy == "keyword_bm25":
            from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415
            texts = _extract_corpus_texts(vector_store)
            if not texts:
                warnings.warn(
                    "BM25 retrieval selected but no corpus texts were available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return vector_store.as_retriever(  # type: ignore[union-attr]
                    search_kwargs={"k": config.top_k},
                )
            return BM25Retriever.from_texts(texts, k=config.top_k)

        if strategy == "tfidf":
            from langchain_community.retrievers import TFIDFRetriever  # noqa: PLC0415
            texts = _extract_corpus_texts(vector_store)
            if not texts:
                warnings.warn(
                    "TF-IDF retrieval selected but no corpus texts were available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return vector_store.as_retriever(  # type: ignore[union-attr]
                    search_kwargs={"k": config.top_k},
                )
            return TFIDFRetriever.from_texts(texts, k=config.top_k)

        if strategy == "hybrid":
            from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415
            from langchain_classic.retrievers import EnsembleRetriever  # noqa: PLC0415
            alpha = config.alpha if config.alpha is not None else 0.5
            dense = vector_store.as_retriever(  # type: ignore[union-attr]
                search_kwargs={"k": config.top_k}
            )
            texts = _extract_corpus_texts(vector_store)
            if not texts:
                warnings.warn(
                    "Hybrid retrieval selected but no keyword corpus texts were available; using dense vector retrieval only.",
                    stacklevel=2,
                )
                return dense
            bm25 = BM25Retriever.from_texts(texts, k=config.top_k)
            return EnsembleRetriever(
                retrievers=[bm25, dense],
                weights=[1 - alpha, alpha],
            )

        if strategy == "mmr":
            lam = config.lambda_diversity if config.lambda_diversity is not None else 0.5
            return vector_store.as_retriever(  # type: ignore[union-attr]
                search_type="mmr",
                search_kwargs={"k": config.top_k, "lambda_mult": lam},
            )

        if strategy == "ensemble":
            from langchain_classic.retrievers import EnsembleRetriever  # noqa: PLC0415
            sub_ids = config.ensemble_sub_retrievers or ["dense_vector", "keyword_bm25"]
            weights = config.ensemble_weights or [1.0 / len(sub_ids)] * len(sub_ids)
            sub_retrievers = []
            for sub_id in sub_ids:
                sub_cfg = RetrievalConfig(strategy=sub_id, top_k=config.top_k)
                sub_retrievers.append(self.get_retriever(sub_cfg, vector_store, llm))
            return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)

        if strategy == "parent_child":
            from langchain_classic.retrievers import ParentDocumentRetriever  # noqa: PLC0415
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
            from langchain_classic.storage import InMemoryStore  # noqa: PLC0415
            return ParentDocumentRetriever(
                vectorstore=vector_store,  # type: ignore[arg-type]
                docstore=InMemoryStore(),
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
                search_kwargs={"k": config.top_k},
            )

        if strategy == "multi_vector":
            from langchain_classic.retrievers.multi_vector import MultiVectorRetriever  # noqa: PLC0415
            from langchain_classic.storage import InMemoryStore  # noqa: PLC0415
            return MultiVectorRetriever(
                vectorstore=vector_store,  # type: ignore[arg-type]
                docstore=InMemoryStore(),
                id_key="doc_id",
                search_kwargs={"k": config.top_k},
            )

        if strategy == "self_query":
            if llm is None:
                warnings.warn(
                    "Self-Query retrieval selected but no LLM is available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return vector_store.as_retriever(  # type: ignore[union-attr]
                    search_kwargs={"k": config.top_k}
                )
            from langchain_classic.retrievers.self_query.base import SelfQueryRetriever  # noqa: PLC0415
            from langchain_classic.chains.query_constructor.base import AttributeInfo  # noqa: PLC0415
            attr_infos = [
                AttributeInfo(
                    name=f.name,
                    description=f.description,
                    type=f.data_type,
                )
                for f in (config.metadata_fields or [])
            ]
            return SelfQueryRetriever.from_llm(
                llm=llm,  # type: ignore[arg-type]
                vectorstore=vector_store,  # type: ignore[arg-type]
                document_contents="Document chunks",
                metadata_field_info=attr_infos,
                search_kwargs={"k": config.top_k},
            )

        if strategy == "time_weighted":
            from langchain_classic.retrievers.time_weighted_retriever import (  # noqa: PLC0415
                TimeWeightedVectorStoreRetriever,
            )
            return TimeWeightedVectorStoreRetriever(
                vectorstore=vector_store,  # type: ignore[arg-type]
                decay_rate=0.01,
                k=config.top_k,
            )

        raise ValueError(
            f"Unsupported retrieval strategy: {strategy!r}. "
            f"Supported: {STRATEGY_IDS}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prompt_int(
        self,
        prompt: str,
        default: int,
        min_val: int,
        max_val: int,
        field_name: str,
        console: object,
    ) -> int:
        while True:
            raw: str = questionary.text(prompt, default=str(default)).ask()
            if not raw or not raw.strip():
                return default
            try:
                value = int(raw.strip())
                validate_numeric(value, min_val, max_val, field_name)
                return value
            except ValueError:
                console.print("[red]  ✗ Please enter a valid integer.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

    def _prompt_float(
        self,
        prompt: str,
        default: float,
        min_val: float,
        max_val: float,
        field_name: str,
        console: object,
    ) -> float:
        while True:
            raw: str = questionary.text(prompt, default=str(default)).ask()
            if not raw or not raw.strip():
                return default
            try:
                value = float(raw.strip())
                validate_numeric(value, min_val, max_val, field_name)
                return value
            except ValueError:
                console.print("[red]  ✗ Please enter a valid decimal number.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

    def _prompt_metadata_fields(self, console: object) -> list[MetadataField]:
        """Prompt for Self-Query metadata field definitions (Req 12.5)."""
        console.print(  # type: ignore[union-attr]
            "  [bold white]Define metadata fields for Self-Query filtering.[/bold white]\n"
            "  At least one field is required.\n"
        )
        fields: list[MetadataField] = []
        while True:
            name: str = questionary.text("    Field name (e.g. 'source', 'year'):").ask()
            if not name or not name.strip():
                if fields:
                    break
                console.print("[red]  ✗ At least one metadata field is required.[/red]")  # type: ignore[union-attr]
                continue

            while True:
                data_type: str | None = questionary.select(
                    "    Data type:",
                    choices=[questionary.Choice(dt, dt) for dt in DATA_TYPES],
                ).ask()
                if data_type:
                    break
                console.print("[yellow]  Selection cancelled — please choose a metadata data type.[/yellow]")  # type: ignore[union-attr]

            description: str | None = questionary.text("    Description:").ask()
            fields.append(MetadataField(
                name=name.strip(),
                data_type=data_type,
                description=(description or "").strip(),
            ))

            more_raw = questionary.confirm("    Add another field?", default=False).ask()
            more = bool(more_raw) if more_raw is not None else False
            if not more:
                break

        return fields

    def _prompt_ensemble(self, console: object) -> tuple[list[str], list[float]]:
        """Prompt for ensemble sub-retrievers and weights (Req 12.6)."""
        console.print(  # type: ignore[union-attr]
            "  [bold white]Ensemble Retrieval — select 2+ sub-retrievers.[/bold white]\n"
        )
        # Allow selecting from strategies that work as sub-retrievers
        sub_choices = [
            questionary.Choice(
                title=s.display_name,
                value=s.strategy_id,
            )
            for s in STRATEGIES
            if s.strategy_id not in ("ensemble",)  # can't nest ensemble
        ]

        selected_subs: list[str] = []
        while True:
            selected_subs = questionary.checkbox(
                "  Select sub-retrievers (minimum 2):",
                choices=sub_choices,
            ).ask()
            if selected_subs is None:
                console.print("[yellow]  Please select at least 2 sub-retrievers.[/yellow]")  # type: ignore[union-attr]
                continue
            if len(selected_subs) < 2:
                console.print(  # type: ignore[union-attr]
                    "[red]  ✗ Ensemble retrieval requires at least 2 sub-retrievers.[/red]"
                )
                continue
            break

        # Prompt weights
        weights: list[float] = []
        while True:
            console.print(  # type: ignore[union-attr]
                f"  Enter weight for each sub-retriever (must sum to 1.0 ±0.01):\n"
                f"  Sub-retrievers: {', '.join(selected_subs)}"
            )
            weight_strs: str = questionary.text(
                f"  Enter {len(selected_subs)} weights (comma-separated, e.g. 0.5,0.5):",
                default=",".join([f"{1.0 / len(selected_subs):.3f}"] * len(selected_subs)),
            ).ask()

            try:
                parsed = [float(w.strip()) for w in weight_strs.split(",")]
                if len(parsed) != len(selected_subs):
                    raise ValueError(
                        f"Expected {len(selected_subs)} weights, got {len(parsed)}"
                    )
                validate_ensemble_weights(parsed)
                weights = parsed
                break
            except (ValueError, ValidationError) as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

        return selected_subs, weights
