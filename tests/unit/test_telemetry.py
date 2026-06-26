"""Unit tests for optional OpenTelemetry hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ms_rag.utils.telemetry import TelemetryReporter
from ms_rag.ui.prompts import prompt_telemetry_configuration


def test_telemetry_reporter_is_safe_without_configuration() -> None:
    reporter = TelemetryReporter()

    assert reporter.enabled in (True, False)

    with reporter.span("test.span", step="unit") as span:
        assert span is None or hasattr(span, "set_attribute")

    reporter.record_event("test.event", "Telemetry event", step="unit")
    reporter.record_error("test.error", "Telemetry error", step="unit")


def test_prompt_telemetry_configuration_builds_config() -> None:
    with patch("ms_rag.ui.prompts.prompt_confirm", return_value=True), \
         patch("ms_rag.ui.prompts.prompt_text") as mock_text:
        mock_text.side_effect = [
            "ms-rag-prod",
            "production",
            "http://localhost:4318",
            "Authorization=Bearer token",
        ]

        config = prompt_telemetry_configuration(console=MagicMock())

    assert config is not None
    assert config.enabled is True
    assert config.service_name == "ms-rag-prod"
    assert config.environment == "production"
    assert config.otlp_endpoint == "http://localhost:4318"
    assert config.otlp_headers["Authorization"] == "Bearer token"
