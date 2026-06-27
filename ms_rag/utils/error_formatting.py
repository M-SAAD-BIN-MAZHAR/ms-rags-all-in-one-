"""User-facing formatting for provider/runtime errors."""

from __future__ import annotations

import re


def format_provider_error(exc: BaseException) -> str:
    """Return a concise, actionable message for common provider failures."""
    text = str(exc)
    lowered = text.lower()

    removed_match = re.search(
        r"model ['\"](?P<model>[^'\"]+)['\"] was removed(?: on (?P<date>[^.]+))?",
        text,
        flags=re.IGNORECASE,
    )
    if removed_match:
        model = removed_match.group("model")
        date = removed_match.group("date")
        suffix = f" on {date}" if date else ""
        return (
            f"Selected model '{model}' was removed{suffix}. "
            "Choose a current model in setup, then rebuild the runtime."
        )

    if "not supported for task" in lowered and "supported task" in lowered:
        return (
            f"{type(exc).__name__}: the selected hosted model does not support this task. "
            "Choose a model whose task matches the feature being used."
        )

    if "status_code: 404" in lowered and "model" in lowered:
        return (
            f"{type(exc).__name__}: the selected model was not found by the provider. "
            "Check the model ID or choose a current model in setup."
        )

    return f"{type(exc).__name__}: {text}"
