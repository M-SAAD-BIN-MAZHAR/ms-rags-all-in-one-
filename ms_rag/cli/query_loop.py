"""Query Input Loop for MS_RAG.

The main interactive query interface after ingestion is complete.

Requirement 10:
- Present query input prompt (10.1)
- Accept queries 1-4096 characters; reject > 4096 (10.2)
- Re-prompt on empty/whitespace-only query (10.3)
- /exit or /quit always requires Y/n confirmation (10.4)
- /config displays structured Pipeline_Config summary (10.5)
- Unrecognised slash commands display valid command list (10.6)
"""

from __future__ import annotations

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]

from ms_rag.models import SessionState
from ms_rag.ui.prompts import get_console, print_error, print_hint, print_success
from ms_rag.utils.error_formatting import format_provider_error
from ms_rag.utils.logging import get_logger, log_error, log_event
from ms_rag.utils.telemetry import TelemetryReporter

# Valid slash commands
VALID_COMMANDS: list[str] = ["/exit", "/quit", "/config", "/save", "/help"]
MAX_QUERY_LENGTH: int = 4096


class QueryLoop:
    """Interactive query loop for the MS_RAG pipeline.

    Usage::

        loop = QueryLoop(query_pipeline=pipeline)
        loop.run(session_state)
    """

    def __init__(
        self,
        query_pipeline: object | None = None,
        session_manager: object | None = None,
    ) -> None:
        """
        Args:
            query_pipeline:  A callable or object with a `process(query, session)` method.
                             If None, live queries fail loudly until runtime is built.
            session_manager: SessionManager instance for /save handling.
        """
        self._pipeline = query_pipeline
        self._session_manager = session_manager

    def run(self, session_state: SessionState) -> None:
        """Run the interactive query loop until the user exits."""
        console = get_console()
        logger = get_logger()
        telemetry = TelemetryReporter()

        console.print(
            Panel(
                "\n".join(
                    [
                        "[bold white]Live Query Mode[/bold white]",
                        "",
                        "Type a natural-language question, or use a command:",
                        f"[cyan]{', '.join(VALID_COMMANDS)}[/cyan]",
                        "",
                        "[dim]Empty Enter re-prompts · /exit requires confirmation[/dim]",
                    ]
                ),
                border_style="green",
                padding=(1, 2),
            )
        )

        while True:
            raw: str | None = questionary.text(
                "  Query >",
                instruction="(question or /help)",
            ).ask()

            if raw is None:
                print_hint(console, "Input cancelled — type your query or /exit to leave.")
                continue

            action = self._classify_input(raw)

            if action == "empty":
                print_hint(console, "Please enter a question or command.")
                continue

            if action == "too_long":
                print_error(
                    console,
                    f"Query too long ({len(raw.strip())} chars). "
                    f"Maximum is {MAX_QUERY_LENGTH} characters.",
                )
                continue

            if action in ("exit", "quit"):
                if self._confirm_exit(console):
                    console.print("[dim]  Goodbye.[/dim]")
                    return
                continue

            if action == "help":
                self._display_help(console)
                continue

            if action == "config":
                self._display_config(session_state, console)
                continue

            if action == "save":
                self._handle_save(session_state, console)
                continue

            if action == "unknown_command":
                print_error(
                    console,
                    f"Unknown command. Valid commands: {', '.join(VALID_COMMANDS)}",
                )
                continue

            query = raw.strip()
            session_state.query_history.append((query, ""))

            try:
                with telemetry.span("query.process", query_length=len(query)):
                    with console.status(
                        "[bold cyan]Retrieving context and generating answer...[/bold cyan]",
                        spinner="dots",
                    ):
                        answer = self._process_query(query, session_state)
                log_event(logger, "query.completed", "Query answered", query_length=len(query))
                if session_state.query_history:
                    last = session_state.query_history[-1]
                    session_state.query_history[-1] = (last[0], answer)

                console.print(
                    Panel(
                        Text(answer, style="white", overflow="fold", no_wrap=False),
                        title="[bold cyan]Answer[/bold cyan]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print_error(console, f"Query error: {format_provider_error(exc)}")
                log_error(logger, "query.failed", "Query processing failed", query_length=len(query))
                telemetry.record_error(
                    "query.failed",
                    f"{type(exc).__name__}: {exc}",
                    query_length=len(query),
                )

    def _classify_input(self, raw: str) -> str:
        """Classify raw user input."""
        stripped = raw.strip()

        if not stripped:
            return "empty"

        if len(stripped) > MAX_QUERY_LENGTH:
            return "too_long"

        if stripped.startswith("/"):
            cmd = stripped.lower().split()[0]
            if cmd == "/exit":
                return "exit"
            if cmd == "/quit":
                return "quit"
            if cmd == "/config":
                return "config"
            if cmd == "/save":
                return "save"
            if cmd == "/help":
                return "help"
            return "unknown_command"

        return "query"

    def _confirm_exit(self, console: Console) -> bool:
        """Ask Y/n confirmation before exiting. Req 10.4."""
        while True:
            result = questionary.confirm(
                "  Are you sure you want to exit?",
                default=False,
            ).ask()
            if result is None:
                print_hint(console, "Please confirm Yes to exit or No to continue.")
                continue
            return bool(result)

    def _display_help(self, console: Console) -> None:
        """Show available slash commands."""
        table = Table(title="Query Loop Commands", border_style="cyan", show_header=True)
        table.add_column("Command", style="bold cyan", min_width=12)
        table.add_column("Description", style="white")
        table.add_row("/config", "Show full pipeline configuration summary")
        table.add_row("/save", "Save session config to JSON")
        table.add_row("/help", "Show this help message")
        table.add_row("/exit, /quit", "Exit the query loop (confirmation required)")
        table.add_row("(text)", "Ask a natural-language question")
        console.print(table)

    def _display_config(self, session_state: SessionState, console: Console) -> None:
        """Display structured Pipeline_Config summary. Req 10.5."""
        cfg = session_state.config

        table = Table(title="Pipeline Configuration", border_style="cyan", show_header=True)
        table.add_column("Component", style="bold white", min_width=24)
        table.add_column("Value", style="cyan")

        table.add_row("Providers", ", ".join(cfg.configured_providers) or "—")
        table.add_row(
            "RAG Type",
            cfg.rag_type.display_name if cfg.rag_type else "—",
        )
        table.add_row("Document Types", ", ".join(cfg.document_types) or "—")
        table.add_row(
            "Loaders",
            ", ".join(f"{k}:{v}" for k, v in cfg.loader_map.items()) or "—",
        )
        table.add_row(
            "Chunking",
            (
                f"{cfg.chunking.strategy} | size={cfg.chunking.chunk_size} "
                f"| overlap={cfg.chunking.chunk_overlap}"
            )
            if cfg.chunking
            else "—",
        )
        table.add_row(
            "Embedding Model",
            f"{cfg.embedding_model.model_id} ({cfg.embedding_model.provider})"
            if cfg.embedding_model
            else "—",
        )
        table.add_row(
            "Vector DB",
            f"{cfg.vector_db.db_type} / {cfg.vector_db.collection_name}"
            if cfg.vector_db
            else "—",
        )
        table.add_row(
            "Keyword Store",
            f"{cfg.keyword_store.store_type} / {cfg.keyword_store.collection_name}"
            if cfg.keyword_store
            else "—",
        )
        table.add_row(
            "Graph Store",
            f"{cfg.graph_store.store_type} / {cfg.graph_store.graph_name} / mode={cfg.graph_store.query_mode}"
            if cfg.graph_store
            else "—",
        )
        table.add_row(
            "Query Enhancement",
            ", ".join(cfg.query_enhancement) or "None",
        )
        table.add_row(
            "Retrieval",
            (
                f"{cfg.retrieval.strategy} | top_k={cfg.retrieval.top_k}"
            )
            if cfg.retrieval
            else "—",
        )
        table.add_row(
            "Reranking",
            f"{'Enabled' if cfg.reranking_enabled else 'Disabled'}"
            + (f" ({cfg.reranking.reranker})" if cfg.reranking_enabled and cfg.reranking else ""),
        )
        table.add_row(
            "Compression",
            f"{'Enabled' if cfg.compression_enabled else 'Disabled'}"
            + (
                f" ({', '.join(cfg.compression.techniques)})"
                if cfg.compression_enabled and cfg.compression
                else ""
            ),
        )
        table.add_row(
            "Evaluation",
            f"{'Enabled' if cfg.evaluation_enabled else 'Disabled'}"
            + (
                f" ({', '.join(cfg.evaluation.evaluators)})"
                if cfg.evaluation_enabled and cfg.evaluation
                else ""
            ),
        )

        console.print(table)

    def _handle_save(self, session_state: SessionState, console: Console) -> None:
        """Handle /save command. Req 18.1."""
        if self._session_manager is None:
            print_error(console, "Session manager not available. Cannot save.")
            return

        from ms_rag.ui.prompts import prompt_save_path  # noqa: PLC0415

        file_path = prompt_save_path(console=console)
        if not file_path:
            print_hint(console, "Save cancelled.")
            return

        try:
            from pathlib import Path  # noqa: PLC0415

            self._session_manager.save(session_state.config, Path(file_path))  # type: ignore[union-attr]
            print_success(console, f"Session saved to {file_path}")
        except Exception as exc:  # noqa: BLE001
            print_error(console, f"Save failed: {exc}")

    def _process_query(self, query: str, session_state: SessionState) -> str:
        """Route query to the initialized runtime pipeline."""
        if self._pipeline is not None:
            if hasattr(self._pipeline, "process"):
                return self._pipeline.process(query, session_state)  # type: ignore[union-attr]
            if callable(self._pipeline):
                return self._pipeline(query, session_state)  # type: ignore[operator]

        raise RuntimeError(
            "Live query runtime is not initialized. Complete setup and build the "
            "retriever, LLM, and RAG chain before entering query mode."
        )
