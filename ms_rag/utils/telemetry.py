"""Optional OpenTelemetry hooks for MS_RAG.

Tracing is disabled by default unless a caller explicitly enables it through
an in-memory TelemetryConfig or environment variables. When tracing is not
available, the app falls back to structured logging without changing behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from typing import Any, Iterator

from ms_rag.utils.logging import get_logger, log_error, log_event

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:  # pragma: no cover - optional dependency
        OTLPSpanExporter = None  # type: ignore[assignment]

    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:  # pragma: no cover - older OTEL variants
        Status = None  # type: ignore[assignment]
        StatusCode = None  # type: ignore[assignment]

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    ConsoleSpanExporter = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


@dataclass(slots=True)
class TelemetryConfig:
    """Runtime tracing settings for one MS_RAG session."""

    enabled: bool = False
    service_name: str = "ms-rag"
    environment: str = "development"
    otlp_endpoint: str = ""
    otlp_headers: dict[str, str] = field(default_factory=dict)
    console_exporter: bool = True


_TRACING_CONFIGURED = False
_ACTIVE_CONFIG: TelemetryConfig | None = None


def _truthy_env(name: str, default: str = "") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _config_from_env() -> TelemetryConfig:
    enabled = _truthy_env("MS_RAG_OTEL_ENABLED") or bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")) or bool(os.getenv("OTEL_TRACES_EXPORTER"))
    headers: dict[str, str] = {}
    raw_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    for pair in raw_headers.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()

    return TelemetryConfig(
        enabled=enabled,
        service_name=os.getenv("OTEL_SERVICE_NAME", "ms-rag"),
        environment=os.getenv("OTEL_ENVIRONMENT", "development"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        otlp_headers=headers,
        console_exporter=_truthy_env("MS_RAG_OTEL_CONSOLE", "1"),
    )


def _configure_tracing(config: TelemetryConfig) -> None:
    global _TRACING_CONFIGURED  # noqa: PLW0603
    if _TRACING_CONFIGURED or not config.enabled or not _OTEL_AVAILABLE:
        return

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "deployment.environment": config.environment,
        }
    )
    provider = TracerProvider(resource=resource)

    exporters: list[object] = []
    if config.otlp_endpoint and OTLPSpanExporter is not None:
        exporters.append(
            OTLPSpanExporter(
                endpoint=config.otlp_endpoint,
                headers=config.otlp_headers or None,
            )
        )

    if config.console_exporter or not exporters:
        exporters.append(ConsoleSpanExporter())

    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


class TelemetryReporter:
    """Local telemetry sink that can be swapped for a real backend later."""

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self._logger = get_logger("ms_rag.telemetry")
        self._config = config or _ACTIVE_CONFIG or _config_from_env()
        self._set_active_config(self._config)
        _configure_tracing(self._config)
        self._tracer = trace.get_tracer("ms_rag") if (_OTEL_AVAILABLE and self._config.enabled) else None

    @staticmethod
    def _set_active_config(config: TelemetryConfig) -> None:
        global _ACTIVE_CONFIG  # noqa: PLW0603
        _ACTIVE_CONFIG = config

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[object | None]:
        """Create an optional tracing span."""
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:  # type: ignore[union-attr]
            if span is not None:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def record_event(self, event: str, message: str, **fields: Any) -> None:
        log_event(self._logger, event, message, **fields)
        if self._tracer is not None and trace is not None:
            span = trace.get_current_span()
            if span is not None:
                span.add_event(message, attributes={k: str(v) for k, v in fields.items()})

    def record_error(self, event: str, message: str, **fields: Any) -> None:
        log_error(self._logger, event, message, **fields)
        if self._tracer is not None and trace is not None:
            span = trace.get_current_span()
            if span is not None:
                span.record_exception(RuntimeError(message))
                if Status is not None and StatusCode is not None:
                    span.set_status(Status(StatusCode.ERROR, message))
