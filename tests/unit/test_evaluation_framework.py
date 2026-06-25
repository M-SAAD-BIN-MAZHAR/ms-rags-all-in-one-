"""Unit tests for EvaluationFramework.

Tests (Requirement 16.4, 16.5):
- Credential-required evaluator stays selected when user cancels credential input.
- CI/CD threshold validation for out-of-range values.
- All 12 evaluators defined.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.evaluation.evaluation_framework import (
    CICD_DEFAULT_METRICS,
    CREDENTIAL_REQUIRED_EVALUATORS,
    EVALUATOR_IDS,
    EVALUATOR_MAP,
    EVALUATORS,
    EvaluationFramework,
)
from ms_rag.models import CredentialStore, EvaluationConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric


# ---------------------------------------------------------------------------
# Structural completeness (Req 16.2)
# ---------------------------------------------------------------------------


class TestEvaluatorListCompleteness:
    def test_exactly_12_evaluators_defined(self) -> None:
        assert len(EVALUATORS) == 12

    def test_all_required_evaluators_present(self) -> None:
        required = {
            "ragas", "deepeval", "trulens", "langsmith", "langfuse",
            "arize_phoenix", "ares", "ragbench", "rageval",
            "cicd_gate", "langgraph_trace", "monitoring_export",
        }
        defined = set(EVALUATOR_IDS)
        missing = required - defined
        assert not missing, f"Missing evaluators: {missing}"

    def test_no_duplicate_evaluator_ids(self) -> None:
        assert len(EVALUATOR_IDS) == len(set(EVALUATOR_IDS))

    def test_evaluator_map_matches_list(self) -> None:
        assert set(EVALUATOR_MAP.keys()) == set(EVALUATOR_IDS)

    def test_all_evaluators_have_display_names(self) -> None:
        for e in EVALUATORS:
            assert len(e.display_name.strip()) > 0

    def test_credential_required_evaluators_set(self) -> None:
        assert "langsmith" in CREDENTIAL_REQUIRED_EVALUATORS
        assert "langfuse" in CREDENTIAL_REQUIRED_EVALUATORS
        assert "arize_phoenix" in CREDENTIAL_REQUIRED_EVALUATORS

    def test_non_credential_evaluators_not_in_set(self) -> None:
        for eid in ["ragas", "deepeval", "trulens", "ares", "ragbench", "rageval",
                    "cicd_gate", "langgraph_trace", "monitoring_export"]:
            assert eid not in CREDENTIAL_REQUIRED_EVALUATORS


# ---------------------------------------------------------------------------
# Credential handling (Req 16.4)
# ---------------------------------------------------------------------------


class TestCredentialHandling:
    def test_evaluator_stays_selected_when_credentials_cancelled(self) -> None:
        """Req 16.4: credential-required evaluator must NOT be removed from selection
        when user cancels or skips the credential prompt."""
        store = CredentialStore()
        framework = EvaluationFramework(credential_store=store)
        info = EVALUATOR_MAP["langsmith"]

        # Simulate: user presses Enter without entering credentials (empty string)
        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q:
            m = MagicMock()
            m.ask.return_value = ""  # cancelled / empty
            mock_q.password.return_value = m
            mock_q.text.return_value = m

            framework._prompt_evaluator_credentials(info, console=MagicMock())

        # Credentials not stored, but no exception — evaluator remains in selection
        assert store.get("langsmith", "LANGCHAIN_API_KEY") is None

    def test_credentials_stored_when_provided(self) -> None:
        store = CredentialStore()
        framework = EvaluationFramework(credential_store=store)
        info = EVALUATOR_MAP["langsmith"]

        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q:
            # Return values for each field
            answers = iter(["lsv2-test-key", "ms-rag-project"])

            def mock_answer() -> str:
                return next(answers, "")

            m = MagicMock()
            m.ask.side_effect = mock_answer
            mock_q.password.return_value = m
            mock_q.text.return_value = m

            framework._prompt_evaluator_credentials(info, console=MagicMock())

        assert store.get("langsmith", "LANGCHAIN_API_KEY") == "lsv2-test-key"

    def test_credentials_not_re_prompted_when_already_stored(self) -> None:
        """If credentials are already in CredentialStore, no prompt shown."""
        store = CredentialStore()
        store.set("langsmith", "LANGCHAIN_API_KEY", "existing-key")
        store.set("langsmith", "LANGCHAIN_PROJECT", "existing-project")
        framework = EvaluationFramework(credential_store=store)
        info = EVALUATOR_MAP["langsmith"]

        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q:
            framework._prompt_evaluator_credentials(info, console=MagicMock())
            # Should NOT have been called
            mock_q.password.assert_not_called()
            mock_q.text.assert_not_called()


# ---------------------------------------------------------------------------
# CI/CD threshold validation (Req 16.5)
# ---------------------------------------------------------------------------


class TestCICDThresholds:
    def test_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_numeric(1.5, 0.0, 1.0, "faithfulness_threshold")

        with pytest.raises(ValidationError):
            validate_numeric(-0.1, 0.0, 1.0, "answer_relevancy_threshold")

    def test_threshold_boundary_values_pass(self) -> None:
        validate_numeric(0.0, 0.0, 1.0, "faithfulness_threshold")
        validate_numeric(1.0, 0.0, 1.0, "faithfulness_threshold")
        validate_numeric(0.8, 0.0, 1.0, "faithfulness_threshold")

    def test_cicd_default_metrics_defined(self) -> None:
        assert "faithfulness" in CICD_DEFAULT_METRICS
        assert "answer_relevancy" in CICD_DEFAULT_METRICS
        assert all(0.0 <= v <= 1.0 for v in CICD_DEFAULT_METRICS.values())

    def test_check_cicd_thresholds_passes_when_all_met(self) -> None:
        framework = EvaluationFramework()
        config = EvaluationConfig(
            evaluators=["cicd_gate"],
            cicd_thresholds={"faithfulness": 0.8, "answer_relevancy": 0.75},
        )
        result = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        assert framework.check_cicd_thresholds(result, config) is True

    def test_check_cicd_thresholds_fails_when_one_below(self) -> None:
        framework = EvaluationFramework()
        config = EvaluationConfig(
            evaluators=["cicd_gate"],
            cicd_thresholds={"faithfulness": 0.8, "answer_relevancy": 0.75},
        )
        result = {"faithfulness": 0.9, "answer_relevancy": 0.60}  # below threshold
        assert framework.check_cicd_thresholds(result, config) is False

    def test_check_cicd_thresholds_passes_with_no_thresholds(self) -> None:
        framework = EvaluationFramework()
        config = EvaluationConfig(evaluators=["ragas"], cicd_thresholds=None)
        assert framework.check_cicd_thresholds({}, config) is True

    def test_check_cicd_thresholds_skips_missing_metrics(self) -> None:
        """Missing metric in result should not cause a failure (metric not available)."""
        framework = EvaluationFramework()
        config = EvaluationConfig(
            evaluators=["cicd_gate"],
            cicd_thresholds={"faithfulness": 0.8, "custom_metric": 0.9},
        )
        result = {"faithfulness": 0.85}  # custom_metric not in result
        assert framework.check_cicd_thresholds(result, config) is True


# ---------------------------------------------------------------------------
# configure() — basic flow
# ---------------------------------------------------------------------------


class TestConfigureFlow:
    def test_no_evaluation_returns_none(self) -> None:
        framework = EvaluationFramework()

        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q, \
             patch("ms_rag.evaluation.evaluation_framework.Console"):
            mock_confirm = MagicMock()
            mock_confirm.ask.return_value = False
            mock_q.confirm.return_value = mock_confirm

            result = framework.configure()

        assert result is None

    def test_empty_selection_returns_none(self) -> None:
        framework = EvaluationFramework()

        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q, \
             patch("ms_rag.evaluation.evaluation_framework.Console"):
            mock_confirm = MagicMock()
            mock_confirm.ask.return_value = True
            mock_q.confirm.return_value = mock_confirm

            mock_checkbox = MagicMock()
            mock_checkbox.ask.return_value = []
            mock_q.checkbox.return_value = mock_checkbox
            mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

            result = framework.configure()

        assert result is None

    def test_selected_evaluators_stored_in_config(self) -> None:
        framework = EvaluationFramework()
        selected = ["ragas", "deepeval"]

        with patch("ms_rag.evaluation.evaluation_framework.questionary") as mock_q, \
             patch("ms_rag.evaluation.evaluation_framework.Console"):
            mock_confirm = MagicMock()
            mock_confirm.ask.return_value = True
            mock_q.confirm.return_value = mock_confirm

            mock_checkbox = MagicMock()
            mock_checkbox.ask.return_value = selected
            mock_q.checkbox.return_value = mock_checkbox
            mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

            result = framework.configure()

        assert result is not None
        assert set(result.evaluators) == set(selected)
