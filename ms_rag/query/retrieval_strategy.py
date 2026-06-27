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
from datetime import datetime
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
ADVANCED_STATE_STRATEGIES = {"parent_child", "multi_vector", "time_weighted"}
ADVANCED_REQUIREMENTS: dict[str, str] = {
    "parent_child": (
        "Needs parent documents plus child chunk IDs. MS_RAG creates this during ingestion "
        "and rebuilds it from original sources when loading a saved session."
    ),
    "multi_vector": (
        "Needs chunk documents, the selected embedding model, and a local FAISS representation "
        "index built at runtime. This does not write synthetic vectors into your production DB."
    ),
    "time_weighted": (
        "Needs ingestion timestamps on chunks. MS_RAG adds ms_rag_ingested_at metadata during ingestion."
    ),
}


def _extract_corpus_texts(
    vector_store: object,
    corpus_texts: list[str] | None = None,
) -> list[str]:
    """Extract indexed document texts from a vector store for keyword retrievers."""
    if corpus_texts:
        return [text for text in corpus_texts if isinstance(text, str) and text.strip()]

    texts: list[str] = []
    cached = getattr(vector_store, "_ms_rag_keyword_corpus", None)
    if isinstance(cached, list):
        texts.extend(text for text in cached if isinstance(text, str) and text.strip())

    if texts:
        return texts

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
        docstore = getattr(vector_store, "docstore", None)
        raw_docs = getattr(docstore, "_dict", None)
        if isinstance(raw_docs, dict):
            for doc in raw_docs.values():
                if isinstance(doc, str) and doc.strip():
                    texts.append(doc)
                elif hasattr(doc, "page_content") and isinstance(doc.page_content, str) and doc.page_content.strip():
                    texts.append(doc.page_content)

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


def _compact_multivector_text(content: str, metadata: dict) -> str:
    """Create a short representation text for multi-vector retrieval."""
    source = metadata.get("source", "")
    first_lines = " ".join(line.strip() for line in content.splitlines()[:3] if line.strip())
    snippet = first_lines or content.strip()
    snippet = snippet[:700]
    return f"Source: {source}\nSummary representation: {snippet}".strip()


def _recency_score(raw_timestamp: object) -> float:
    """Convert an ISO timestamp into a 0..1 recency score."""
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return 0.0
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_seconds = max((datetime.now(parsed.tzinfo) - parsed).total_seconds(), 0.0)
    one_day = 24 * 60 * 60
    return 1.0 / (1.0 + (age_seconds / one_day))


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
        self._display_strategy_guidance(console)

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

        self._confirm_advanced_requirements(config, console)

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
        corpus_texts: list[str] | None = None,
        embeddings: object | None = None,
        strict_advanced: bool = False,
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
            return self._dense_fallback(vector_store, config.top_k)

        if strategy == "keyword_bm25":
            from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415
            texts = _extract_corpus_texts(vector_store, corpus_texts)
            if not texts:
                warnings.warn(
                    "BM25 retrieval selected but no corpus texts were available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return self._dense_fallback(vector_store, config.top_k)
            return BM25Retriever.from_texts(texts, k=config.top_k)

        if strategy == "tfidf":
            from langchain_community.retrievers import TFIDFRetriever  # noqa: PLC0415
            texts = _extract_corpus_texts(vector_store, corpus_texts)
            if not texts:
                warnings.warn(
                    "TF-IDF retrieval selected but no corpus texts were available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return self._dense_fallback(vector_store, config.top_k)
            return TFIDFRetriever.from_texts(texts, k=config.top_k)

        if strategy == "hybrid":
            from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415
            from langchain_classic.retrievers import EnsembleRetriever  # noqa: PLC0415
            alpha = config.alpha if config.alpha is not None else 0.5
            dense = vector_store.as_retriever(  # type: ignore[union-attr]
                search_kwargs={"k": config.top_k}
            )
            texts = _extract_corpus_texts(vector_store, corpus_texts)
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
                sub_retrievers.append(
                    self.get_retriever(
                        sub_cfg,
                        vector_store,
                        llm,
                        corpus_texts=corpus_texts,
                        embeddings=embeddings,
                        strict_advanced=strict_advanced,
                    )
                )
            return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)

        if strategy == "parent_child":
            return self._parent_child_retriever(vector_store, config.top_k, strict=strict_advanced)

        if strategy == "multi_vector":
            return self._multi_vector_retriever(vector_store, config.top_k, embeddings, strict=strict_advanced)

        if strategy == "self_query":
            if llm is None:
                warnings.warn(
                    "Self-Query retrieval selected but no LLM is available; falling back to dense vector retrieval.",
                    stacklevel=2,
                )
                return self._dense_fallback(vector_store, config.top_k)
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
            return self._time_weighted_retriever(vector_store, config.top_k, strict=strict_advanced)

        raise ValueError(
            f"Unsupported retrieval strategy: {strategy!r}. "
            f"Supported: {STRATEGY_IDS}"
        )

    @staticmethod
    def _dense_fallback(vector_store: object, top_k: int) -> object:
        """Return a standard dense retriever as the universal safe fallback."""
        return vector_store.as_retriever(  # type: ignore[union-attr]
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def _parent_child_retriever(self, vector_store: object, top_k: int, *, strict: bool = False) -> object:
        """Retrieve child chunks by vector search, then return parent documents."""
        parent_documents = getattr(vector_store, "_ms_rag_parent_documents", None)
        if not isinstance(parent_documents, dict) or not parent_documents:
            message = (
                "Parent-Child retrieval selected but parent document state is unavailable. "
                "Re-ingest documents or load the session with original document sources available."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(
                f"{message} Falling back to dense vector retrieval.",
                stacklevel=2,
            )
            return self._dense_fallback(vector_store, top_k)

        dense = vector_store.as_retriever(search_kwargs={"k": top_k})  # type: ignore[union-attr]

        def retrieve(query: str) -> list:
            child_docs = dense.invoke(query)
            results: list = []
            seen_parent_ids: set[str] = set()
            for child in child_docs:
                parent_id = getattr(child, "metadata", {}).get("ms_rag_parent_id")
                parent_doc = parent_documents.get(parent_id)
                if parent_id and parent_doc is not None and parent_id not in seen_parent_ids:
                    results.append(parent_doc)
                    seen_parent_ids.add(parent_id)
                elif parent_id not in seen_parent_ids:
                    results.append(child)
                    if parent_id:
                        seen_parent_ids.add(parent_id)
            return results[:top_k]

        from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
        return RunnableLambda(retrieve)

    def _multi_vector_retriever(
        self,
        vector_store: object,
        top_k: int,
        embeddings: object | None,
        *,
        strict: bool = False,
    ) -> object:
        """Search synthetic summary/title vectors and return original chunks."""
        chunk_documents = getattr(vector_store, "_ms_rag_chunk_documents", None)
        if not isinstance(chunk_documents, list) or not chunk_documents:
            message = (
                "Multi-Vector retrieval selected but source chunk state is unavailable. "
                "Re-ingest documents or load the session with original document sources available."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(
                f"{message} Falling back to dense vector retrieval.",
                stacklevel=2,
            )
            return self._dense_fallback(vector_store, top_k)
        if embeddings is None:
            message = (
                "Multi-Vector retrieval selected but embeddings are unavailable for the representation index."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(
                f"{message} Falling back to dense vector retrieval.",
                stacklevel=2,
            )
            return self._dense_fallback(vector_store, top_k)

        try:
            from langchain_core.documents import Document  # noqa: PLC0415
            from langchain_community.vectorstores import FAISS  # noqa: PLC0415
        except ImportError as exc:
            message = (
                "Multi-Vector retrieval needs langchain-community and FAISS for its local representation index."
            )
            if strict:
                raise RuntimeError(f"{message} Missing dependency: {exc}") from exc
            warnings.warn(
                f"{message} Falling back to dense vector retrieval: {exc}",
                stacklevel=2,
            )
            return self._dense_fallback(vector_store, top_k)

        source_documents: dict[str, object] = {}
        representation_docs: list = []
        for doc in chunk_documents:
            content = getattr(doc, "page_content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            metadata = dict(getattr(doc, "metadata", {}) or {})
            source_id = metadata.get("ms_rag_multi_vector_source_id") or metadata.get("ms_rag_child_id")
            if not source_id:
                continue
            source_documents[source_id] = doc
            compact_text = _compact_multivector_text(content, metadata)
            representation_docs.append(
                Document(
                    page_content=compact_text,
                    metadata={"ms_rag_multi_vector_source_id": source_id},
                )
            )

        if not representation_docs:
            message = (
                "Multi-Vector retrieval could not build representation documents from the ingested chunks."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(
                f"{message} Falling back to dense vector retrieval.",
                stacklevel=2,
            )
            return self._dense_fallback(vector_store, top_k)

        representation_store = FAISS.from_documents(representation_docs, embeddings)
        representation_retriever = representation_store.as_retriever(search_kwargs={"k": max(top_k * 2, top_k)})

        def retrieve(query: str) -> list:
            hits = representation_retriever.invoke(query)
            results: list = []
            seen_source_ids: set[str] = set()
            for hit in hits:
                source_id = getattr(hit, "metadata", {}).get("ms_rag_multi_vector_source_id")
                source_doc = source_documents.get(source_id)
                if source_id and source_doc is not None and source_id not in seen_source_ids:
                    results.append(source_doc)
                    seen_source_ids.add(source_id)
                if len(results) >= top_k:
                    break
            return results

        from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
        return RunnableLambda(retrieve)

    def _time_weighted_retriever(self, vector_store: object, top_k: int, *, strict: bool = False) -> object:
        """Blend dense rank with ingestion recency metadata."""
        dense = vector_store.as_retriever(search_kwargs={"k": max(top_k * 4, top_k)})  # type: ignore[union-attr]

        def retrieve(query: str) -> list:
            docs = dense.invoke(query)
            if not docs:
                return []
            if strict and not any(getattr(doc, "metadata", {}).get("ms_rag_ingested_at") for doc in docs):
                raise RuntimeError(
                    "Time-Weighted retrieval selected but retrieved documents do not contain "
                    "ms_rag_ingested_at metadata. Re-ingest documents before using this strategy."
                )
            scored = []
            for rank, doc in enumerate(docs):
                dense_score = 1.0 / (rank + 1)
                recency_score = _recency_score(getattr(doc, "metadata", {}).get("ms_rag_ingested_at"))
                scored.append((0.4 * dense_score + 0.6 * recency_score, rank, doc))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [doc for _, _, doc in scored[:top_k]]

        from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
        return RunnableLambda(retrieve)

    def _display_strategy_guidance(self, console: object) -> None:
        """Show model/state requirements before strategy selection."""
        try:
            from rich.table import Table  # noqa: PLC0415
        except ImportError:
            return
        table = Table(title="Retrieval Model and State Requirements", border_style="cyan")
        table.add_column("Strategy", style="bold white")
        table.add_column("Needs", style="green")
        table.add_column("Best use", style="cyan")
        table.add_row("Dense / MMR", "Selected embedding model + vector DB", "General semantic retrieval baseline")
        table.add_row("BM25 / TF-IDF / Hybrid", "Chunk text corpus", "Exact terms, IDs, names, and mixed semantic/keyword search")
        table.add_row("Parent-Child", "Parent docs + child IDs", "Long PDFs, reports, legal/policy docs where larger context matters")
        table.add_row("Multi-Vector", "Selected embedding model + local FAISS representation index", "When summaries/titles retrieve better than raw chunks")
        table.add_row("Time-Weighted", "Chunk timestamps", "Freshness-sensitive docs, changelogs, support tickets, recent policies")
        table.add_row("Self-Query", "Generation LLM + metadata fields", "Questions that need metadata filters such as date, author, source, department")
        console.print(table)  # type: ignore[union-attr]

    def _confirm_advanced_requirements(self, config: RetrievalConfig, console: object) -> None:
        """Explain and confirm advanced retrieval requirements."""
        selected = {config.strategy}
        if config.strategy == "ensemble":
            selected.update(config.ensemble_sub_retrievers or [])
        advanced_selected = sorted(selected & ADVANCED_STATE_STRATEGIES)
        if not advanced_selected:
            return
        from ms_rag.ui.prompts import prompt_required_confirm  # noqa: PLC0415
        console.print("\n[bold yellow]Advanced retrieval requirements[/bold yellow]")  # type: ignore[union-attr]
        for strategy_id in advanced_selected:
            console.print(  # type: ignore[union-attr]
                f"  [cyan]{STRATEGY_MAP[strategy_id].display_name}[/cyan]: "
                f"{ADVANCED_REQUIREMENTS[strategy_id]}"
            )
        prompt_required_confirm(
            "Use these advanced retrieval requirements for this pipeline?",
            console=console,
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
