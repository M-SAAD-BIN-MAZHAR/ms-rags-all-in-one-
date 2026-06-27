"""Shared logging helpers for MS_RAG."""

from __future__ import annotations

import json
import logging
import warnings
from datetime import UTC, datetime
from typing import Any


_LOGGER_NAME = "ms_rag"


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in (
            "event",
            "step",
            "component",
            "provider",
            "db_type",
            "collection_name",
            "strategy",
            "fallback",
            "reason",
            "action",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return the project logger with a conservative default configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, message: str, **fields: Any) -> None:
    """Emit a structured event."""
    logger.info(message, extra={"event": event, **fields})


def log_warning(logger: logging.Logger, event: str, message: str, **fields: Any) -> None:
    """Emit a structured warning event."""
    logger.warning(message, extra={"event": event, **fields})


def log_error(logger: logging.Logger, event: str, message: str, **fields: Any) -> None:
    """Emit a structured error event."""
    logger.error(message, extra={"event": event, **fields})


def install_warning_renderer(console: object | None = None) -> None:
    """Render Python warnings as visible terminal notices and structured logs.

    MS_RAG intentionally uses ``warnings.warn`` for non-fatal degradation paths.
    This hook keeps those warnings visible in the UI and searchable in logs.
    """
    if getattr(install_warning_renderer, "_installed", False):
        return

    logger = get_logger()
    original_showwarning = warnings.showwarning

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object | None = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        log_warning(
            logger,
            "runtime.warning",
            text,
            component="warnings",
            reason=category.__name__,
            action="review_terminal_notice",
        )
        if console is not None:
            try:
                from rich.panel import Panel  # noqa: PLC0415
                from rich.text import Text  # noqa: PLC0415

                body = Text(text, style="yellow", overflow="fold", no_wrap=False)
                body.append(f"\n\n{filename}:{lineno}", style="dim")
                console.print(  # type: ignore[union-attr]
                    Panel(
                        body,
                        title="[bold yellow]Runtime notice[/bold yellow]",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
                return
            except Exception:  # noqa: BLE001
                pass
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    setattr(install_warning_renderer, "_installed", True)
