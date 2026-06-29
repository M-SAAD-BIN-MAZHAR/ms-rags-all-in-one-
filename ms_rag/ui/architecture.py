"""Architecture visibility helpers for MS-RAGS(ALL-IN-ONE).

These helpers render the user's selected RAG architecture from PipelineConfig.
They intentionally avoid hard-coded demo diagrams so the terminal always shows
the actual configured workflow.
"""

from __future__ import annotations

from typing import Iterable

from ms_rag.models import PipelineConfig


def build_visibility_rows(config: PipelineConfig) -> list[tuple[str, str]]:
    """Return human-readable rows for every major configured component."""
    rows: list[tuple[str, str]] = []

    rows.append(("RAG architecture", config.rag_type.display_name if config.rag_type else "Not selected"))
    rows.append(
        (
            "Generation model",
            f"{config.llm_model.provider} / {config.llm_model.model_id}" if config.llm_model else "Not selected",
        )
    )
    rows.append(("LLM providers", ", ".join(config.configured_providers) or "None"))
    rows.append(("Document types", ", ".join(config.document_types) or "None"))
    rows.append(
        (
            "Loaders",
            ", ".join(f"{doc_type}:{loader}" for doc_type, loader in config.loader_map.items()) or "None",
        )
    )
    rows.append(
        (
            "Chunking",
            (
                f"{config.chunking.strategy} | size={config.chunking.chunk_size} | "
                f"overlap={config.chunking.chunk_overlap}"
            )
            if config.chunking
            else "Not selected",
        )
    )
    rows.append(
        (
            "Embedding model",
            (
                f"{config.embedding_model.provider} / {config.embedding_model.model_id}"
                + (
                    f" ({config.vector_db.dimension} dimensions)"
                    if config.vector_db and config.vector_db.dimension
                    else ""
                )
            )
            if config.embedding_model
            else "Not selected",
        )
    )
    rows.append(
        (
            "Vector database",
            f"{config.vector_db.db_type} / {config.vector_db.collection_name}" if config.vector_db else "Not selected",
        )
    )
    rows.append(
        (
            "Keyword store",
            (
                f"{config.keyword_store.store_type} / {config.keyword_store.collection_name}"
                if config.keyword_store
                else "Not configured"
            ),
        )
    )
    rows.append(
        (
            "Graph store",
            (
                f"{config.graph_store.store_type} / {config.graph_store.graph_name} / mode={config.graph_store.query_mode}"
                if config.graph_store
                else "Not configured"
            ),
        )
    )
    rows.append(
        (
            "Query enhancement",
            ", ".join(config.query_enhancement) if config.query_enhancement else "Disabled",
        )
    )
    rows.append(
        (
            "Retrieval",
            f"{config.retrieval.strategy} | top_k={config.retrieval.top_k}" if config.retrieval else "Not selected",
        )
    )
    rows.append(
        (
            "Reranking",
            (
                f"Enabled / {config.reranking.reranker} / top_k={config.reranking.top_k}"
                if config.reranking_enabled and config.reranking
                else "Disabled"
            ),
        )
    )
    rows.append(
        (
            "Compression",
            (
                "Enabled / " + ", ".join(config.compression.techniques)
                if config.compression_enabled and config.compression
                else "Disabled"
            ),
        )
    )
    rows.append(
        (
            "Evaluation",
            (
                "Enabled / " + ", ".join(config.evaluation.evaluators)
                if config.evaluation_enabled and config.evaluation
                else "Disabled"
            ),
        )
    )
    rows.append(
        (
            "Agent tools",
            (
                ", ".join(config.agent_tools.enabled_tools)
                if config.agent_tools and config.agent_tools.enabled_tools
                else "Not configured"
            ),
        )
    )
    rows.append(("Sources", ", ".join(config.document_sources) or "Not selected"))
    return rows


def build_architecture_flow_steps(config: PipelineConfig) -> list[str]:
    """Return ordered architecture stages selected by the user."""
    rag_name = config.rag_type.display_name if config.rag_type else "Selected RAG"
    steps = [
        f"User documents ({', '.join(config.document_types) or 'selected files'})",
        f"Extractors/loaders ({_compact_loader_names(config.loader_map.values())})",
        (
            f"Chunk splitter ({config.chunking.strategy}, size={config.chunking.chunk_size}, "
            f"overlap={config.chunking.chunk_overlap})"
            if config.chunking
            else "Chunk splitter (not selected)"
        ),
        (
            "Embeddings ("
            + config.embedding_model.model_id
            + (
                f", {config.vector_db.dimension}d"
                if config.vector_db and config.vector_db.dimension
                else ""
            )
            + ")"
            if config.embedding_model
            else "Embeddings (not selected)"
        ),
        (
            f"Vector database ({config.vector_db.db_type}:{config.vector_db.collection_name})"
            if config.vector_db
            else "Vector database (not selected)"
        ),
    ]

    if config.keyword_store:
        steps.append(f"Keyword store ({config.keyword_store.store_type}:{config.keyword_store.collection_name})")
    if config.graph_store:
        steps.append(
            f"Knowledge graph ({config.graph_store.store_type}:{config.graph_store.graph_name}, {config.graph_store.query_mode})"
        )
    if config.query_enhancement:
        steps.append(f"Query enhancement ({', '.join(config.query_enhancement)})")
    if config.retrieval:
        steps.append(f"Retriever ({config.retrieval.strategy}, top_k={config.retrieval.top_k})")
    if config.reranking_enabled and config.reranking:
        steps.append(f"Reranker ({config.reranking.reranker}, top_k={config.reranking.top_k})")
    if config.compression_enabled and config.compression:
        steps.append(f"Context compression ({', '.join(config.compression.techniques)})")
    if config.agent_tools and config.agent_tools.enabled_tools:
        steps.append(f"Approved agent tools ({', '.join(config.agent_tools.enabled_tools)})")

    steps.append(f"{rag_name} orchestration")
    steps.append(
        f"Answer generator ({config.llm_model.provider}:{config.llm_model.model_id})"
        if config.llm_model
        else "Answer generator (not selected)"
    )
    if config.evaluation_enabled and config.evaluation:
        steps.append(f"Evaluation ({', '.join(config.evaluation.evaluators)})")
    steps.append("Live query loop + visible traces + standalone code generation")
    return steps


def build_architecture_flow_text(config: PipelineConfig, *, width: int = 96) -> str:
    """Render a compact ASCII flowchart from selected architecture stages."""
    steps = build_architecture_flow_steps(config)
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        label = f"{index:02d}. {step}"
        lines.extend(_box(label, width=width))
        if index != len(steps):
            lines.append(" " * (width // 2 - 1) + "|")
            lines.append(" " * (width // 2 - 1) + "v")
    return "\n".join(lines)


def display_architecture_report(config: PipelineConfig, console: object) -> None:
    """Print selection visibility and the selected technical architecture."""
    from rich.panel import Panel  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415
    from rich.text import Text  # noqa: PLC0415

    table = Table(title="Selection Visibility Summary", border_style="cyan", show_header=True)
    table.add_column("Component", style="bold white", min_width=22)
    table.add_column("Selected value", style="cyan")
    for component, value in build_visibility_rows(config):
        table.add_row(component, value)
    console.print(table)  # type: ignore[union-attr]

    flow = build_architecture_flow_text(config)
    console.print(  # type: ignore[union-attr]
        Panel(
            Text(flow, style="white", overflow="fold", no_wrap=False),
            title="[bold yellow]Selected Technical Architecture[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def _compact_loader_names(loaders: Iterable[str]) -> str:
    unique = list(dict.fromkeys(str(loader) for loader in loaders if str(loader).strip()))
    if not unique:
        return "selected loaders"
    if len(unique) <= 3:
        return ", ".join(unique)
    return ", ".join(unique[:3]) + f", +{len(unique) - 3} more"


def _box(text: str, *, width: int) -> list[str]:
    inner_width = max(24, width - 4)
    wrapped = _wrap_text(text, inner_width)
    top = "+" + "-" * (inner_width + 2) + "+"
    bottom = "+" + "-" * (inner_width + 2) + "+"
    body = ["| " + line.ljust(inner_width) + " |" for line in wrapped]
    return [top, *body, bottom]


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
