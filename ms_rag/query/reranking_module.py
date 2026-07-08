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
from functools import lru_cache
import warnings

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

from ms_rag.models import RerankingConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric


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
        from ms_rag.ui.prompts import prompt_confirm, prompt_select, prompt_text  # noqa: PLC0415

        console = Console()
        console.print("\n[bold cyan]Step 12 — Reranking[/bold cyan]\n")

        wants_reranking = prompt_confirm(
            "  Do you want to enable reranking?",
            default=False,
            console=console,
        )

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
            reranker_id = prompt_select(
                "  Select reranker:",
                choices=choices,
                console=console,
            )

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
            raw = prompt_text(
                f"  HuggingFace model ID or local path "
                f"(default: {DEFAULT_MODEL_IDS.get(reranker_id, '')}):",
                default=DEFAULT_MODEL_IDS.get(reranker_id, ""),
                required=True,
                console=console,
            )
            while not raw or not raw.strip():
                console.print("[red]  ✗ Model ID is required.[/red]")
                raw = prompt_text(
                    "  HuggingFace model ID or local path:",
                    required=True,
                    console=console,
                )
            model_id = raw.strip()
        elif reranker_id == "cohere_reranker":
            model_raw = prompt_select(
                "  Select Cohere reranker model:",
                choices=[
                    questionary.Choice("rerank-english-v3.0", "rerank-english-v3.0"),
                    questionary.Choice("rerank-multilingual-v3.0", "rerank-multilingual-v3.0"),
                ],
                console=console,
            )
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
        llm: object | None = None,
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
                return self._rerank_llm(query, docs, config, llm)

            if reranker_id == "colbert":
                return self._rerank_colbert(query, docs, config)

            if reranker_id == "flashrank":
                return self._rerank_flashrank(query, docs, config)

        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Reranker {reranker_id!r} failed; returning original top-k documents: {exc}",
                stacklevel=2,
            )

        return docs[: config.top_k]

    # ------------------------------------------------------------------
    # Private reranker implementations
    # ------------------------------------------------------------------

    def _rerank_cross_encoder(self, query: str, docs: list, config: RerankingConfig) -> list:
        model = _get_cross_encoder(config.model_id)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: config.top_k]]

    def _rerank_cohere(self, query: str, docs: list, config: RerankingConfig) -> list:
        import cohere  # noqa: PLC0415
        from ms_rag.utils.credentials import resolve_credential  # noqa: PLC0415

        api_key = resolve_credential("COHERE_API_KEY", self._credential_store, "cohere") or ""
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
        model = _get_cross_encoder(config.model_id)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: config.top_k]]

    def _rerank_llm(
        self,
        query: str,
        docs: list,
        config: RerankingConfig,
        llm: object | None = None,
    ) -> list:
        if llm is None:
            return docs[: config.top_k]

        from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
        from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Score relevance from 0 to 10. Reply with only the number.",
            ),
            (
                "human",
                "Query: {query}\n\nDocument:\n{document}",
            ),
        ])
        chain = prompt | llm | StrOutputParser()  # type: ignore[operator]

        scored: list[tuple[float, object]] = []
        for doc in docs:
            raw = chain.invoke({
                "query": query,
                "document": doc.page_content,
            })
            try:
                score = float(str(raw).strip().split()[0])
            except (ValueError, IndexError):
                score = 0.0
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[: config.top_k]]

    def _rerank_colbert(self, query: str, docs: list, config: RerankingConfig) -> list:
        """Rerank with genuine ColBERT-style late interaction (token-level MaxSim).

        Unlike a cross-encoder, this encodes the query and each document into
        per-token contextual embeddings and scores by summing, over every query
        token, its maximum cosine similarity to any document token (MaxSim).
        """
        scorer = _get_colbert_scorer(config.model_id)
        scores = scorer.score(query, [doc.page_content for doc in docs])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: config.top_k]]

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
        from ms_rag.ui.prompts import prompt_text  # noqa: PLC0415

        while True:
            raw = prompt_text(
                f"  Reranking top_k (must be ≤ {retrieval_top_k}, default 3):",
                default=str(min(3, retrieval_top_k)),
                console=console,  # type: ignore[arg-type]
            )

            if not raw or not str(raw).strip():
                return min(3, retrieval_top_k)

            try:
                value = int(str(raw).strip())
                validate_numeric(value, 1, retrieval_top_k, "reranking_top_k")
                return value
            except ValueError:
                console.print("[red]  ✗ Please enter a valid integer.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]


@lru_cache(maxsize=1)
def _get_cross_encoder(model_id: str) -> object:
    """Load and cache local CrossEncoder rerankers for the process lifetime.

    Capped at maxsize=1 to prevent memory exhaustion from holding multiple
    large CrossEncoder models (each ~500MB–2GB) simultaneously.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(model_id)


class _ColBERTLateInteractionScorer:
    """Genuine ColBERT-style late-interaction (MaxSim) scorer.

    Encodes query and documents into per-token contextual embeddings with a
    transformer encoder, L2-normalises them, and scores each document as the
    sum over query tokens of the maximum cosine similarity to any document
    token. This is the ColBERT MaxSim operator — fundamentally different from a
    cross-encoder's single joint relevance score.
    """

    def __init__(self, model_id: str, max_length: int = 512) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    def _encode(self, texts: list[str]) -> list:
        torch = self._torch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state  # (B, T, H)
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        mask = encoded["attention_mask"].bool()
        return [hidden[i][mask[i]] for i in range(hidden.shape[0])]

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_tokens = self._encode([query])[0]  # (Tq, H)
        scores: list[float] = []
        batch_size = 16
        for start in range(0, len(documents), batch_size):
            for doc_tokens in self._encode(documents[start : start + batch_size]):
                if doc_tokens.shape[0] == 0 or query_tokens.shape[0] == 0:
                    scores.append(0.0)
                    continue
                similarity = query_tokens @ doc_tokens.T  # (Tq, Td)
                scores.append(float(similarity.max(dim=1).values.sum().item()))
        return scores


@lru_cache(maxsize=1)
def _get_colbert_scorer(model_id: str) -> _ColBERTLateInteractionScorer:
    """Load and cache the late-interaction ColBERT scorer for the process."""
    return _ColBERTLateInteractionScorer(model_id or "colbert-ir/colbertv2.0")


def clear_reranker_model_cache() -> None:
    """Release cached local reranker models.

    This is primarily used by tests and long-running CLI sessions on memory
    constrained machines after reranking settings change.
    """
    _get_cross_encoder.cache_clear()
    _get_colbert_scorer.cache_clear()
