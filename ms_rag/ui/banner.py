"""MS_RAG ASCII banner display module.

Renders the branded ASCII art banner and tagline on CLI startup.

Behaviour (Requirement 1):
- Displays full-width ASCII art containing "MS_RAG" before any other output.
- Prints a one-line tagline immediately below the banner.
- Uses ANSI colour on colour-capable terminals via Rich.
- Falls back to plain print() if Rich raises any exception.
- Continues silently if both display methods fail — startup is never blocked.
"""

from __future__ import annotations

MS_RAG_BANNER = r"""
███╗   ███╗███████╗      ██████╗  █████╗  ██████╗
████╗ ████║██╔════╝      ██╔══██╗██╔══██╗██╔════╝
██╔████╔██║███████╗      ██████╔╝███████║██║  ███╗
██║╚██╔╝██║╚════██║      ██╔══██╗██╔══██║██║   ██║
██║ ╚═╝ ██║███████║      ██║  ██║██║  ██║╚██████╔╝
╚═╝     ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
                      MS_RAG
"""

TAGLINE = "Production-Grade RAG Framework Builder"
VERSION_LINE = "v0.1.0  |  Powered by LangChain & LangGraph"
SEPARATOR = "─" * 60


def display_banner(console: object | None = None) -> None:  # type: ignore[type-arg]
    """Render the MS_RAG ASCII banner and tagline to the terminal.

    Args:
        console: A ``rich.console.Console`` instance.  If *None*, one is
                 created internally.  Pass an explicit console in tests to
                 control output capture.

    Fallback chain (Requirement 1.3):
        1. Try Rich coloured rendering.
        2. If Rich raises, fall back to ``print()``.
        3. If ``print()`` also raises, continue silently — startup must
           never be blocked by a banner failure.
    """
    try:
        _display_rich(console)
    except Exception:  # noqa: BLE001
        try:
            _display_plain()
        except Exception:  # noqa: BLE001
            # Both methods failed — continue silently
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _display_rich(console: object | None = None) -> None:
    """Render banner with Rich colour formatting.

    Raises:
        Any exception raised by Rich — caller catches and falls back.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.text import Text         # noqa: PLC0415
    from rich.panel import Panel       # noqa: PLC0415
    from rich.align import Align       # noqa: PLC0415

    if console is None:
        console = Console()

    # Build the coloured banner text
    banner_text = Text(MS_RAG_BANNER, style="bold cyan", justify="center")
    tagline_text = Text(f"\n  {TAGLINE}", style="bold white")
    version_text = Text(f"  {VERSION_LINE}", style="dim white")

    combined = Text.assemble(banner_text, tagline_text, "\n", version_text, "\n")

    panel = Panel(
        Align.center(combined),
        border_style="cyan",
        padding=(0, 2),
    )

    console.print(panel)  # type: ignore[union-attr]


def _display_plain() -> None:
    """Plain-text fallback banner — no Rich dependency."""
    print(MS_RAG_BANNER)
    print(f"  {TAGLINE}")
    print(f"  {VERSION_LINE}")
    print(SEPARATOR)
