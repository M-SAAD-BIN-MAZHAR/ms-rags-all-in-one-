"""RAG Architecture Selector for MS_RAG.

Presents all 15 supported RAG architecture variants to the user,
shows descriptions, flags LangGraph-requiring types, and returns
the selected RAGTypeConfig.

- Display numbered list of all 15 RAG types (3.1)
- Show 2-4 sentence description on selection (3.2)
- Accept exactly one selection per Session (3.3)
- Show LangGraph note only for agentic types (3.4)
"""

from __future__ import annotations

from ms_rag.models import RAGTypeConfig

# ---------------------------------------------------------------------------
# RAG type definitions
# ---------------------------------------------------------------------------

LANGGRAPH_TYPES: frozenset[str] = frozenset(
    {"agentic_rag", "self_rag", "corrective_rag", "adaptive_rag"}
)

RAG_TYPES: list[RAGTypeConfig] = [
    RAGTypeConfig(
        rag_type="naive_rag",
        display_name="Naive RAG",
        description=(
            "Naive RAG follows a straightforward retrieve-then-generate pipeline: "
            "it embeds the query, retrieves the top-k chunks from the vector store, "
            "and passes them directly to the LLM with the user question. "
            "Best for simple Q&A over a well-structured, homogeneous document corpus."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="advanced_rag",
        display_name="Advanced RAG",
        description=(
            "Advanced RAG improves on Naive RAG by adding pre-retrieval steps "
            "(query rewriting, HyDE) and post-retrieval steps (reranking, context "
            "compression) to boost precision and recall. "
            "Best for production pipelines where answer quality matters more than "
            "raw latency."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="modular_rag",
        display_name="Modular RAG",
        description=(
            "Modular RAG decomposes the pipeline into interchangeable modules—routing, "
            "retrieval, fusion, generation—that can be swapped or combined independently. "
            "It supports search, memory, predict, and task-decomposition modules. "
            "Best for teams that need full control over each pipeline stage."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="agentic_rag",
        display_name="Agentic RAG",
        description=(
            "Agentic RAG wraps the RAG pipeline in a LangGraph agent loop that can "
            "plan whether to retrieve, rewrite the query, call approved tools, or "
            "answer directly based on intermediate results. External tools remain "
            "strictly permission-gated and allowlisted. "
            "Best for complex, multi-step reasoning tasks where a single retrieval pass "
            "is insufficient. Requires LangGraph."
        ),
        requires_langgraph=True,
    ),
    RAGTypeConfig(
        rag_type="self_rag",
        display_name="Self-RAG",
        description=(
            "Self-RAG reflects on retrieval need, grades retrieved evidence for "
            "relevance, generates from grounded context, and checks whether the answer "
            "is supported before returning it. "
            "Best for high-accuracy tasks where hallucination must be minimised. "
            "Requires LangGraph."
        ),
        requires_langgraph=True,
    ),
    RAGTypeConfig(
        rag_type="corrective_rag",
        display_name="Corrective RAG (CRAG)",
        description=(
            "Corrective RAG grades retrieved chunks, removes irrelevant chunks from the "
            "context, rewrites weak queries, and can fall back to approved Web Search "
            "before generating when the corpus is insufficient. "
            "Best for open-domain QA where the document corpus may not cover the query. "
            "Requires LangGraph."
        ),
        requires_langgraph=True,
    ),
    RAGTypeConfig(
        rag_type="speculative_rag",
        display_name="Speculative RAG",
        description=(
            "Speculative RAG generates a draft answer first, then retrieves evidence "
            "to verify or refine the draft, reversing the traditional retrieve-then-generate "
            "order. It is faster than iterative self-correction approaches. "
            "Best for tasks where a good initial hypothesis can guide more targeted retrieval."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="graphrag",
        display_name="GraphRAG",
        description=(
            "GraphRAG in MS-RAGS(ALL-IN-ONE) extracts entities and relationships during ingestion, "
            "stores a persistent graph, builds community summaries, and combines "
            "local/global graph context with hybrid evidence retrieval. "
            "Best for analytical questions requiring cross-document entity and "
            "relationship context."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="hyde_rag",
        display_name="HyDE RAG",
        description=(
            "HyDE (Hypothetical Document Embeddings) RAG generates a hypothetical "
            "ideal answer document for the query, embeds that document, and uses its "
            "embedding to retrieve real chunks. This bridges the lexical gap between "
            "short queries and long documents. "
            "Best for queries phrased very differently from the indexed document style."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="multi_query_rag",
        display_name="Multi-Query RAG",
        description=(
            "Multi-Query RAG uses an LLM to generate multiple rephrased variants of "
            "the original query, retrieves documents for each variant, and takes the "
            "union of results before deduplication. "
            "Best for queries with multiple interpretations or when a single phrasing "
            "misses relevant chunks."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="rag_fusion",
        display_name="RAG-Fusion",
        description=(
            "RAG-Fusion generates multiple search queries and merges the retrieved "
            "document lists using Reciprocal Rank Fusion (RRF) to produce a single "
            "high-quality ranked list. "
            "Best for search-heavy workloads where ranking accuracy across diverse "
            "query phrasings is critical."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="step_back_rag",
        display_name="Step-Back RAG",
        description=(
            "Step-Back RAG prompts the LLM to generate a higher-level 'step-back' "
            "question before retrieval, retrieving background principles rather than "
            "specific facts. The original and step-back results are combined. "
            "Best for science, law, and reasoning tasks that require foundational "
            "concepts alongside specific facts."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="parent_child_rag",
        display_name="Parent-Child RAG",
        description=(
            "Parent-Child RAG stores small child chunks for precise retrieval but "
            "returns larger parent chunks to the LLM for full context. "
            "This balances retrieval precision (small chunks) with generation quality "
            "(rich context). "
            "Best for long documents like legal contracts, books, or research papers."
        ),
        requires_langgraph=False,
    ),
    RAGTypeConfig(
        rag_type="adaptive_rag",
        display_name="Adaptive RAG",
        description=(
            "Adaptive RAG routes queries to the most appropriate path based on query "
            "complexity: simple prompts go directly to generation, standard questions "
            "use retrieval, and complex analytical questions use rewrite-plus-retrieval. "
            "Best for mixed workloads where different query types need different "
            "pipeline depths. Requires LangGraph."
        ),
        requires_langgraph=True,
    ),
    RAGTypeConfig(
        rag_type="contextual_compression_rag",
        display_name="Contextual Compression RAG",
        description=(
            "Contextual Compression RAG applies LangChain's ContextualCompressionRetriever "
            "to extract only the relevant portions of retrieved documents before passing "
            "them to the LLM, reducing token usage and noise. "
            "Best for verbose document corpora where retrieved chunks contain significant "
            "off-topic content."
        ),
        requires_langgraph=False,
    ),
]

# Build a lookup dict for fast access
RAG_TYPE_MAP: dict[str, RAGTypeConfig] = {r.rag_type: r for r in RAG_TYPES}

LANGGRAPH_NOTE = (
    "[bold yellow]  ⚡ Note:[/bold yellow] The generated code will use "
    "[bold cyan]LangGraph[/bold cyan] to implement the agentic workflow for this RAG type."
)


# ---------------------------------------------------------------------------
# RAGTypeSelector
# ---------------------------------------------------------------------------


class RAGTypeSelector:
    """Interactive selector for RAG architecture variants.

    Usage::

        selector = RAGTypeSelector()
        config = selector.display_and_select()
        print(config.rag_type, config.requires_langgraph)
    """

    def display_and_select(self) -> RAGTypeConfig:
        """Display numbered list, show description on selection, return config.

        show all 15 types
        show 2-4 sentence description
        accept exactly one selection
        LangGraph note only for agentic types

        Returns:
            The selected RAGTypeConfig.
        """
        import questionary  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        from ms_rag.ui.prompts import get_console, print_step, prompt_select  # noqa: PLC0415

        console = get_console()
        print_step(console, 3, "Select RAG Architecture")

        choices = [
            questionary.Choice(
                title=f"{i + 1:2}. {r.display_name}"
                      + (" [LangGraph]" if r.requires_langgraph else ""),
                value=r.rag_type,
            )
            for i, r in enumerate(RAG_TYPES)
        ]

        selected_id = prompt_select(
            "Which RAG architecture do you want to build?",
            choices,
            console=console,
        )

        config = RAG_TYPE_MAP[selected_id]

        # Show description panel
        desc_text = Text(config.description, style="white")
        panel = Panel(
            desc_text,
            title=f"[bold cyan]{config.display_name}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)

        # Show LangGraph note only for agentic types
        if config.requires_langgraph:
            console.print(LANGGRAPH_NOTE)

        return config
