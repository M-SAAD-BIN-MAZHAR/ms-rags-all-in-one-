"""Shared terminal prompt helpers for MS_RAG.

Provides consistent re-prompt behaviour: required fields loop until valid,
cancel (Ctrl+C) is treated as "try again" for mandatory steps, and Rich
messages use a uniform style.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment,misc]

T = TypeVar("T")

_DEFAULT_CONSOLE: Console | None = None


def get_console() -> Console:
    """Return a shared Rich console instance."""
    global _DEFAULT_CONSOLE  # noqa: PLW0603
    if Console is None:
        raise ImportError("rich is required for MS_RAG terminal UI")
    if _DEFAULT_CONSOLE is None:
        _DEFAULT_CONSOLE = Console(highlight=False, soft_wrap=True)
    return _DEFAULT_CONSOLE


def print_step(console: Console, step: int | str, title: str) -> None:
    """Print a workflow step header."""
    label = f"Step {step}" if isinstance(step, int) else step
    console.print(f"\n[bold cyan]  {label} — {title}[/bold cyan]\n")


def print_error(console: Console, message: str) -> None:
    console.print(f"[red]  ✗ {message}[/red]")


def print_warning(console: Console, message: str) -> None:
    console.print(f"[yellow]  ⚠ {message}[/yellow]")


def print_success(console: Console, message: str) -> None:
    console.print(f"[green]  ✓ {message}[/green]")


def print_hint(console: Console, message: str) -> None:
    console.print(f"[dim]  {message}[/dim]")


def _cancelled(console: Console) -> None:
    print_warning(console, "Selection cancelled — please try again.")


def prompt_text(
    message: str,
    *,
    default: str | None = None,
    required: bool = False,
    secret: bool = False,
    console: Console | None = None,
    validator: Callable[[str], T] | None = None,
    error_message: str | None = None,
) -> str | T:
    """Prompt for text; re-prompt until valid when required=True."""
    if questionary is None:
        raise ImportError("questionary is required for MS_RAG interactive prompts")

    con = console or get_console()
    prompt_fn = questionary.password if secret else questionary.text

    while True:
        result: str | None = prompt_fn(
            message,
            default=default or "",
        ).ask()

        if result is None:
            _cancelled(con)
            continue

        value = result.strip()
        if required and not value:
            print_error(con, "This field is required — please enter a value.")
            continue

        if not value and default is not None:
            value = default

        if validator is not None:
            try:
                return validator(value)
            except Exception as exc:  # noqa: BLE001
                print_error(con, error_message or str(exc))
                continue

        return value


def prompt_select(
    message: str,
    choices: list[Any],
    *,
    console: Console | None = None,
) -> str:
    """Single-select prompt; re-prompt on cancel or empty result."""
    if questionary is None:
        raise ImportError("questionary is required for MS_RAG interactive prompts")

    con = console or get_console()
    while True:
        selected: str | None = questionary.select(message, choices=choices).ask()
        if selected is None:
            _cancelled(con)
            continue
        if not str(selected).strip():
            print_error(con, "Please select an option.")
            continue
        return str(selected)


def prompt_checkbox(
    message: str,
    choices: list[Any],
    *,
    min_selections: int = 1,
    console: Console | None = None,
) -> list[str]:
    """Multi-select prompt; re-prompt when fewer than min_selections chosen."""
    if questionary is None:
        raise ImportError("questionary is required for MS_RAG interactive prompts")

    con = console or get_console()
    while True:
        selected: list[str] | None = questionary.checkbox(message, choices=choices).ask()
        if selected is None:
            _cancelled(con)
            continue
        if len(selected) < min_selections:
            print_error(
                con,
                f"Please select at least {min_selections} option"
                f"{'' if min_selections == 1 else 's'}.",
            )
            continue
        return selected


def prompt_confirm(
    message: str,
    *,
    default: bool = False,
    console: Console | None = None,
) -> bool:
    """Yes/no confirm; re-prompt on cancel."""
    if questionary is None:
        raise ImportError("questionary is required for MS_RAG interactive prompts")

    con = console or get_console()
    while True:
        result = questionary.confirm(message, default=default).ask()
        if result is None:
            _cancelled(con)
            continue
        return bool(result)


def prompt_required_confirm(
    message: str,
    *,
    console: Console | None = None,
) -> None:
    """Loop until the user explicitly confirms Yes."""
    con = console or get_console()
    while not prompt_confirm(message, default=False, console=con):
        print_warning(con, "Please confirm to continue, or update your selection above.")


def prompt_document_sources(
    *,
    console: Console | None = None,
    default: str = "./docs",
) -> list[str]:
    """Prompt for comma-separated document paths/URLs until at least one is given."""
    con = console or get_console()
    print_hint(
        con,
        "Enter local paths, directories, or URLs separated by commas.",
    )

    while True:
        raw = prompt_text(
            "  Document paths/directories/URLs (comma-separated):",
            default=default,
            required=True,
            console=con,
        )
        sources = [s.strip() for s in str(raw).split(",") if s.strip()]
        if sources:
            return sources
        print_error(con, "At least one document source is required.")


def prompt_save_path(
    *,
    default: str = "ms_rag_session.json",
    console: Console | None = None,
) -> str | None:
    """Prompt for a save path; re-prompt until non-empty or user declines."""
    con = console or get_console()
    while True:
        path = prompt_text(
            "  Save config to file path:",
            default=default,
            required=False,
            console=con,
        )
        if path and str(path).strip():
            return str(path).strip()
        if prompt_confirm("  Save cancelled. Try again?", default=True, console=con):
            continue
        return None


def prompt_telemetry_configuration(
    *,
    console: Console | None = None,
) -> "TelemetryConfig | None":
    """Prompt the user to enable optional OpenTelemetry tracing."""
    from ms_rag.utils.telemetry import TelemetryConfig  # noqa: PLC0415

    con = console or get_console()
    print_hint(
        con,
        "Optional: enable tracing so you can inspect setup and query timings in a monitoring backend.",
    )

    if not prompt_confirm("  Enable OpenTelemetry tracing for this session?", default=False, console=con):
        return TelemetryConfig(enabled=False)

    service_name = prompt_text(
        "  Service name:",
        default="ms-rag",
        required=True,
        console=con,
    )
    environment = prompt_text(
        "  Environment (development/staging/production):",
        default="development",
        required=True,
        console=con,
    )
    endpoint = prompt_text(
        "  OTLP endpoint (leave blank for console spans only):",
        default="",
        required=False,
        console=con,
    )
    headers_raw = prompt_text(
        "  OTLP headers (optional, comma-separated k=v pairs):",
        default="",
        required=False,
        console=con,
    )

    headers: dict[str, str] = {}
    for pair in str(headers_raw).split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()

    return TelemetryConfig(
        enabled=True,
        service_name=str(service_name).strip() or "ms-rag",
        environment=str(environment).strip() or "development",
        otlp_endpoint=str(endpoint).strip(),
        otlp_headers=headers,
        console_exporter=True,
    )
