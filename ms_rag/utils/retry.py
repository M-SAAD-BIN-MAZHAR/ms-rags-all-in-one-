"""Retry utility for MS_RAG.

Provides retry_with_backoff() for wrapping external API calls (embedding,
LLM inference, vector DB writes) with exponential backoff.

Requirement 19.1: on failure, offer Retry / Skip / Abort choices.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    delays: tuple[float, ...] = (1.0, 2.0),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call *fn()* with exponential backoff.

    Args:
        fn:           Zero-argument callable to invoke.
        max_attempts: Maximum number of attempts before raising (default 3).
        delays:       Seconds to wait between attempts (default: 1s, 2s).
        on_retry:     Optional callback(attempt_number, exception) called before
                      each retry.

    Returns:
        Return value of *fn()* on success.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                if on_retry:
                    on_retry(attempt + 1, exc)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def retry_with_user_prompt(
    fn: Callable[[], T],
    operation_name: str = "operation",
    max_attempts: int = 3,
    delays: tuple[float, ...] = (1.0, 2.0),
) -> T | None:
    """Call *fn()* with backoff; after all attempts fail, prompt user for action.

    Requirement 19.1: present Retry / Skip / Abort on final failure.

    Args:
        fn:             Zero-argument callable.
        operation_name: Human-readable name for error messages.
        max_attempts:   Number of auto-retry attempts before prompting user.
        delays:         Seconds between auto-retries.

    Returns:
        Return value of *fn()* on success, or None if user skips.

    Raises:
        SystemExit: If user chooses Abort.
        Exception:  Re-raised if questionary is not available.
    """
    try:
        return retry_with_backoff(fn, max_attempts=max_attempts, delays=delays)
    except Exception as final_exc:  # noqa: BLE001
        try:
            console = Console()
            console.print(
                f"\n[red]  ✗ {operation_name} failed after {max_attempts} attempts:[/red]\n"
                f"  [dim]{type(final_exc).__name__}: {final_exc}[/dim]\n"
            )
            choice: str = questionary.select(
                "  What would you like to do?",
                choices=[
                    questionary.Choice("Retry", value="retry"),
                    questionary.Choice("Skip this operation", value="skip"),
                    questionary.Choice("Abort session", value="abort"),
                ],
            ).ask()

            if choice == "retry":
                return retry_with_backoff(fn, max_attempts=max_attempts, delays=delays)
            elif choice == "skip":
                return None
            else:
                console.print("[red]  Session aborted.[/red]")
                raise SystemExit(1)

        except ImportError:
            raise final_exc
