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

from typing import Callable

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

# Valid slash commands
VALID_COMMANDS: list[str] = ["/exit", "/quit", "/config", "/save"]
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
                             If None, a placeholder response is returned.
            session_manager: SessionManager instance for /save handling.
        """
        self._pipeline = query_pipeline
        self._session_manager = session_manager

    def run(self, session_state: SessionState) -> None:
        """Run the interactive query loop until the user exits.

        Handles all slash commands, validation, and query dispatch.

        Args:
            session_state: The current SessionState holding PipelineConfig and runtime objects.
        """
        console = Console()

        console.print(
            "\n[bold green]  ✓ Ready to query![/bold green]  "
            "Type your question, or use a command:\n"
            f"  [dim]{', '.join(VALID_COMMANDS)}[/dim]\n"
        )

        while True:
            raw: str | None = questionary.text(
                "  Query >",
                instruction="(Enter query or /help for commands)",
            ).ask()

            if raw is None:
                # User closed input (Ctrl+D in terminal) — treat as exit
                raw = ""

            if raw is None or (raw == "" and not raw):
                # Safety: if ask() returns None (mocked/Ctrl+D), exit loop
                return

            # Dispatch based on input type
            action = self._classify_input(raw)

            if action == "empty":
                # Req 10.3: re-prompt without processing
                continue

            if action == "too_long":
                # Req 10.2: reject and re-prompt
                console.print(
                    f"[red]  ✗ Query too long ({len(raw)} chars). "
                    f"Maximum is {MAX_QUERY_LENGTH} characters.[/red]"
                )
                continue

            if action in ("exit", "quit"):
                # Req 10.4: always confirm before terminating
                if self._confirm_exit(console):
                    console.print("[dim]  Goodbye.[/dim]")
                    return
                continue

            if action == "config":
                # Req 10.5: structured config summary
                self._display_config(session_state, console)
                continue

            if action == "save":
                # Req 18.1: /save command
                self._handle_save(session_state, console)
                continue

            if action == "unknown_command":
                # Req 10.6: unknown slash command
                console.print(
                    f"[yellow]  Unknown command. Valid commands: "
                    f"{', '.join(VALID_COMMANDS)}[/yellow]"
                )
                continue

            # action == "query": valid natural language query
            query = raw.strip()
            session_state.query_history.append((query, ""))  # placeholder

            try:
                answer = self._process_query(query, session_state)
                # Update history with actual answer
                if session_state.query_history:
                    last = session_state.query_history[-1]
                    session_state.query_history[-1] = (last[0], answer)

                console.print(
                    Panel(
                        Text(answer, style="white"),
                        title="[bold cyan]Answer[/bold cyan]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Req 19.3: query errors return to prompt without terminating
                console.print(
                    f"[red]  ✗ Query error: {type(exc).__name__}: {exc}[/red]"
                )

    # ------------------------------------------------------------------
    # Input classification
    # ------------------------------------------------------------------

    def _classify_input(self, raw: str) -> str:
        """Return a classification string for the raw input.

        Returns one of:
            "empty"           — zero length or whitespace only
            "too_long"        — exceeds MAX_QUERY_LENGTH
            "exit"            — /exit command
            "quit"            — /quit command
            "config"          — /config command
            "save"            — /save command
            "unknown_command" — unrecognised slash command
            "query"           — valid natural language query
        """
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
            return "unknown_command"

        return "query"

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _confirm_exit(self, console: object) -> bool:
        """Ask Y/n confirmation before exiting. Req 10.4."""
        confirmed: bool = questionary.confirm(
            "  Are you sure you want to exit?",
            default=False,
        ).ask()
        return bool(confirmed)

    def _display_config(self, session_state: SessionState, console: object) -> None:
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

        console.print(table)  # type: ignore[union-attr]

    def _handle_save(self, session_state: SessionState, console: object) -> None:
        """Handle /save command. Req 18.1."""
        if self._session_manager is None:
            console.print(  # type: ignore[union-attr]
                "[yellow]  Session manager not available. Cannot save.[/yellow]"
            )
            return

        file_path: str = questionary.text(
            "  Save config to file path:",
            default="ms_rag_session.json",
        ).ask()

        if not file_path or not file_path.strip():
            console.print("[yellow]  Save cancelled.[/yellow]")  # type: ignore[union-attr]
            return

        try:
            from pathlib import Path  # noqa: PLC0415
            self._session_manager.save(session_state.config, Path(file_path.strip()))  # type: ignore[union-attr]
            console.print(  # type: ignore[union-attr]
                f"[green]  ✓ Session saved to {file_path.strip()}[/green]"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]  ✗ Save failed: {exc}[/red]")  # type: ignore[union-attr]

    def _process_query(self, query: str, session_state: SessionState) -> str:
        """Route query to the pipeline or return placeholder."""
        if self._pipeline is not None:
            if hasattr(self._pipeline, "process"):
                return self._pipeline.process(query, session_state)  # type: ignore[union-attr]
            if callable(self._pipeline):
                return self._pipeline(query, session_state)  # type: ignore[operator]

        return (
            "[dim italic]Pipeline not yet initialised. "
            "Complete Steps 11-16 to enable live querying.[/dim italic]"
        )
