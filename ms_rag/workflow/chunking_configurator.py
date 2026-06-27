"""Chunking Parameter Configuration UI for MS_RAG.

Presents the user with a numbered strategy selector, then prompts for
all relevant parameters (chunk size, overlap, separators, tokenizer,
language) with strategy-specific defaults pre-filled.

Requirement 7:
- Prompt chunk size with pre-filled default (7.1)
- Prompt chunk overlap with pre-filled default (7.2)
- Prompt separators for recursive_character strategy (7.3)
- Prompt tokenizer for token_based / sentence strategies (7.4)
- Validate overlap < chunk_size; re-prompt on violation (7.5)
- Store all parameters in ChunkingConfig (7.6)
"""

from __future__ import annotations

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]

from ms_rag.ingestion.chunking_engine import (
    STRATEGY_DESCRIPTIONS,
    STRATEGY_IDS,
    SUPPORTED_LANGUAGES,
    ChunkingStrategyInfo,
)
from ms_rag.models import ChunkingConfig
from ms_rag.utils.validation import validate_chunk_overlap, validate_numeric
from ms_rag.utils.exceptions import ValidationError


class ChunkingConfigurator:
    """Interactive configurator for chunking strategy and parameters.

    Usage::

        configurator = ChunkingConfigurator()
        config = configurator.configure()
        # config: ChunkingConfig with strategy, chunk_size, chunk_overlap, etc.
    """

    def configure(self) -> ChunkingConfig:
        """Run the full chunking configuration flow.

        Steps:
        1. Display strategy selector with descriptions.
        2. Prompt chunk size (with default).
        3. Prompt chunk overlap (with default); validate < chunk_size.
        4. Conditionally prompt separators (recursive_character).
        5. Conditionally prompt tokenizer (token_based, sentence).
        6. Conditionally prompt language (code_aware).

        Returns:
            A fully-populated ChunkingConfig.
        """
        console = Console()

        console.print("\n[bold cyan]Step 6 — Select Chunking Strategy[/bold cyan]\n")

        # Step 1: strategy selection
        strategy_id = self._select_strategy(console)
        info = STRATEGY_DESCRIPTIONS[strategy_id]

        # Show description panel
        console.print(
            Panel(
                Text(info.description, style="white"),
                title=f"[bold cyan]{info.display_name}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        console.print("\n[bold cyan]Step 7 — Configure Chunking Parameters[/bold cyan]\n")

        # Step 2: chunk size (skip for semantic — threshold-based)
        chunk_size = 0
        if strategy_id != "semantic":
            chunk_size = self._prompt_int(
                prompt=f"  Chunk size (default: {info.default_chunk_size}):",
                default=info.default_chunk_size,
                min_val=1,
                max_val=32_000,
                field_name="chunk_size",
                console=console,
            )

        # Step 3: chunk overlap (validate < chunk_size)
        chunk_overlap = 0
        if strategy_id not in ("semantic", "agentic"):
            chunk_overlap = self._prompt_overlap(
                chunk_size=chunk_size,
                default_overlap=info.default_overlap,
                console=console,
            )

        # Step 4: separators (recursive_character only)
        separators: list[str] | None = None
        if info.supports_separators:
            separators = self._prompt_separators(console)

        # Step 5: tokenizer (token_based, sentence)
        tokenizer: str | None = None
        if info.requires_tokenizer:
            tokenizer = self._prompt_tokenizer(strategy_id, console)

        # Step 6: language (code_aware only)
        language: str | None = None
        if info.requires_language:
            language = self._prompt_language(console)

        config = ChunkingConfig(
            strategy=strategy_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            tokenizer=tokenizer,
            language=language,
        )

        console.print(
            f"[green]  ✓ Chunking configured: [bold]{info.display_name}[/bold] "
            f"| size={chunk_size} | overlap={chunk_overlap}[/green]"
        )

        return config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_strategy(self, console: object) -> str:
        choices = [
            questionary.Choice(
                title=f"{i + 1:2}. {info.display_name}",
                value=sid,
            )
            for i, (sid, info) in enumerate(STRATEGY_DESCRIPTIONS.items())
        ]
        while True:
            selected: str | None = questionary.select(
                "Select chunking strategy:",
                choices=choices,
            ).ask()
            if selected:
                return selected
            console.print("[yellow]  Selection cancelled — please choose a chunking strategy.[/yellow]")  # type: ignore[union-attr]

    def _prompt_int(
        self,
        prompt: str,
        default: int,
        min_val: int,
        max_val: int,
        field_name: str,
        console: object,
    ) -> int:
        """Prompt for an integer with range validation and re-prompt on error."""
        while True:
            raw: str = questionary.text(
                prompt,
                default=str(default),
            ).ask()

            if raw is None or not raw.strip():
                return default

            try:
                value = int(raw.strip())
                validate_numeric(value, min_val, max_val, field_name)
                return value
            except ValueError:
                console.print(f"[red]  ✗ Please enter a valid integer.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

    def _prompt_overlap(
        self,
        chunk_size: int,
        default_overlap: int,
        console: object,
    ) -> int:
        """Prompt for chunk overlap, enforcing overlap < chunk_size (Req 7.5)."""
        while True:
            raw: str = questionary.text(
                f"  Chunk overlap (default: {default_overlap}, must be < {chunk_size}):",
                default=str(min(default_overlap, max(0, chunk_size - 1))),
            ).ask()

            if raw is None or not raw.strip():
                overlap = min(default_overlap, chunk_size - 1)
                return max(0, overlap)

            try:
                overlap = int(raw.strip())
                validate_chunk_overlap(chunk_size, overlap)
                return overlap
            except ValueError:
                console.print("[red]  ✗ Please enter a valid integer.[/red]")  # type: ignore[union-attr]
            except ValidationError as exc:
                console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

    def _prompt_separators(self, console: object) -> list[str] | None:
        """Prompt for custom separators or accept defaults (Req 7.3)."""
        console.print(  # type: ignore[union-attr]
            "  [dim]Custom separators (comma-separated, e.g. '\\n\\n,\\n, ').[/dim]"
        )
        raw: str | None = questionary.text(
            "  Separators (leave blank for defaults ['\\n\\n','\\n',' ','']):",
            default="",
        ).ask()

        if not raw or not raw.strip():
            return None  # use LangChain defaults

        # Parse comma-separated separators, interpreting escape sequences
        parts = [s.strip() for s in raw.split(",")]
        separators = []
        for part in parts:
            # Convert literal \n, \t to actual characters
            sep = part.replace("\\n", "\n").replace("\\t", "\t")
            if sep:
                separators.append(sep)
        return separators if separators else None

    def _prompt_tokenizer(self, strategy_id: str, console: object) -> str | None:
        """Prompt for tokenizer selection (Req 7.4)."""
        if strategy_id == "token_based":
            choices = [
                questionary.Choice("cl100k_base (GPT-4, GPT-3.5-turbo)", "cl100k_base"),
                questionary.Choice("p50k_base (text-davinci-003)", "p50k_base"),
                questionary.Choice("r50k_base (GPT-3)", "r50k_base"),
                questionary.Choice("HuggingFace tokenizer (custom)", "__hf__"),
            ]
            while True:
                selected: str | None = questionary.select(
                    "  Select tokenizer:",
                    choices=choices,
                ).ask()
                if selected:
                    break
                console.print("[yellow]  Selection cancelled — please choose a tokenizer.[/yellow]")  # type: ignore[union-attr]
            if selected == "__hf__":
                while True:
                    hf_id: str | None = questionary.text(
                        "  HuggingFace tokenizer identifier (e.g. 'bert-base-uncased'):",
                        default="bert-base-uncased",
                    ).ask()
                    if hf_id and hf_id.strip():
                        return hf_id.strip()
                    console.print("[red]  ✗ Tokenizer identifier is required.[/red]")  # type: ignore[union-attr]
            return selected
        else:
            # sentence strategy
            raw: str | None = questionary.text(
                "  Sentence transformer model name "
                "(default: sentence-transformers/all-MiniLM-L6-v2):",
                default="sentence-transformers/all-MiniLM-L6-v2",
            ).ask()
            return raw.strip() if raw else "sentence-transformers/all-MiniLM-L6-v2"

    def _prompt_language(self, console: object) -> str:
        """Prompt for programming language (code_aware strategy)."""
        choices = [
            questionary.Choice(lang.capitalize(), lang)
            for lang in SUPPORTED_LANGUAGES
        ]
        while True:
            selected: str | None = questionary.select(
                "  Select programming language for code-aware splitting:",
                choices=choices,
            ).ask()
            if selected:
                return selected
            console.print("[yellow]  Selection cancelled — please choose a language.[/yellow]")  # type: ignore[union-attr]
