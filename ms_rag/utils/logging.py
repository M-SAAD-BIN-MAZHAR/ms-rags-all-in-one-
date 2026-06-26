"""Shared logging helpers for MS_RAG."""

from __future__ import annotations

import json
import logging
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
        for key in ("event", "step", "component", "provider", "db_type", "collection_name"):
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


def log_error(logger: logging.Logger, event: str, message: str, **fields: Any) -> None:
    """Emit a structured error event."""
    logger.error(message, extra={"event": event, **fields})
