"""Evaluation Framework for MS_RAG.

Interactive configuration and runtime evaluation using all 11 supported
evaluation frameworks.

Requirement 16:
- Ask yes/no for evaluation (16.1)
- Display all 11 evaluators as a checkbox (16.2)
- Allow multi-select (16.3)
- Prompt credentials for LangSmith/Langfuse/Arize Phoenix; keep selected on cancel (16.4)
- Prompt CI/CD thresholds for cicd_gate (16.5)
- Store all selections and thresholds in EvaluationConfig (16.6)
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

try:
    import questionary
    from rich.console import Console
    from rich.table import Table
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from ms_rag.evaluation.evaluator_runners import (
    EVALUATOR_RUNNERS,
    run_monitoring_export,
)
from ms_rag.models import EvaluationConfig
from ms_rag.utils.exceptions import ValidationError
from ms_rag.utils.validation import validate_numeric


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatorInfo:
    """Metadata for a single evaluation framework."""
    evaluator_id: str
    display_name: str
    description: str
    requires_credentials: bool = False
    credential_fields: list[str] = None  # type: ignore[assignment]
    credential_provider: str = ""

    def __post_init__(self) -> None:
        if self.credential_fields is None:
            object.__setattr__(self, "credential_fields", [])


EVALUATORS: list[EvaluatorInfo] = [
    EvaluatorInfo(
        evaluator_id="ragas",
        display_name="RAGAS",
        description="Reference-free RAG evaluation: faithfulness, answer relevancy, context precision",
    ),
    EvaluatorInfo(
        evaluator_id="deepeval",
        display_name="DeepEval",
        description="LLM evaluation with G-Eval, RAG metrics, and custom test cases",
    ),
    EvaluatorInfo(
        evaluator_id="trulens",
        display_name="TruLens",
        description="Modern TruLens package validation plus TruLens-prefixed groundedness scoring",
    ),
    EvaluatorInfo(
        evaluator_id="langsmith",
        display_name="LangSmith",
        description="LangChain-native tracing, evaluation datasets, and prompt versioning",
        requires_credentials=True,
        credential_fields=["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"],
        credential_provider="langsmith",
    ),
    EvaluatorInfo(
        evaluator_id="langfuse",
        display_name="Langfuse",
        description="Open-source LLM observability — self-hostable, GDPR-compliant",
        requires_credentials=True,
        credential_fields=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
        credential_provider="langfuse",
    ),
    EvaluatorInfo(
        evaluator_id="arize_phoenix",
        display_name="Arize Phoenix",
        description="OpenInference/Phoenix trace export when endpoint is configured, plus Phoenix-prefixed scores",
        requires_credentials=True,
        credential_fields=["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT"],
        credential_provider="arize_phoenix",
    ),
    EvaluatorInfo(
        evaluator_id="ares",
        display_name="ARES",
        description="ARES package-backed availability path plus ARES-compatible retrieval/generation scores",
    ),
    EvaluatorInfo(
        evaluator_id="ragbench",
        display_name="RAGBench",
        description="RAGBench-compatible single-query scoring with optional Hugging Face datasets tooling",
    ),
    EvaluatorInfo(
        evaluator_id="cicd_gate",
        display_name="CI/CD Pipeline Gate (pass/fail thresholds)",
        description="Blocks deployment if any RAG quality metric falls below configured thresholds",
    ),
    EvaluatorInfo(
        evaluator_id="langgraph_trace",
        display_name="Agentic System Tracing (LangGraph trace)",
        description="Full step-by-step tracing of LangGraph agentic workflow execution",
    ),
    EvaluatorInfo(
        evaluator_id="monitoring_export",
        display_name="Production Monitoring Dashboard Export",
        description="Exports evaluation metrics and traces to a monitoring dashboard (JSON/CSV)",
    ),
]

EVALUATOR_IDS: list[str] = [e.evaluator_id for e in EVALUATORS]
EVALUATOR_MAP: dict[str, EvaluatorInfo] = {e.evaluator_id: e for e in EVALUATORS}

CREDENTIAL_REQUIRED_EVALUATORS: frozenset[str] = frozenset(
    e.evaluator_id for e in EVALUATORS if e.requires_credentials
)

# Default CI/CD metric names with recommended thresholds
CICD_DEFAULT_METRICS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


# ---------------------------------------------------------------------------
# EvaluationFramework
# ---------------------------------------------------------------------------


class EvaluationFramework:
    """Interactive configuration and runtime evaluation.

    Usage::

        framework = EvaluationFramework(credential_store=store)
        config = framework.configure()
        if config:
            result = framework.evaluate(query, context, answer)
            passed = framework.check_cicd_thresholds(result)
    """

    def __init__(self, credential_store: object | None = None) -> None:
        self._credential_store = credential_store
        self._config: EvaluationConfig | None = None

    def configure(self) -> EvaluationConfig | None:
        """Interactive yes/no → evaluator checkbox → credentials → thresholds.

        Requirement 16.1-16.6.

        Returns:
            EvaluationConfig if enabled, None if user declines.
        """
        console = Console()
        console.print("\n[bold cyan]Step 15 — Evaluation Framework[/bold cyan]\n")

        while True:
            result = questionary.confirm(
                "  Do you want to configure evaluation?",
                default=False,
            ).ask()
            if result is not None:
                wants_evaluation = bool(result)
                break
            console.print("[yellow]  Selection cancelled — please answer yes or no.[/yellow]")

        if not wants_evaluation:
            console.print("  [dim]Evaluation disabled.[/dim]")
            return None

        choices = [
            questionary.Choice(
                title=f"{e.display_name}  —  {e.description}"
                      + (" [API key required]" if e.requires_credentials else ""),
                value=e.evaluator_id,
            )
            for e in EVALUATORS
        ]

        while True:
            selected: list[str] | None = questionary.checkbox(
                "  Select evaluation frameworks:",
                choices=choices,
            ).ask()
            if selected is None:
                console.print("[yellow]  Selection cancelled — please try again.[/yellow]")
                continue
            if not selected:
                console.print(
                    "[red]  ✗ Please select at least one evaluation framework.[/red]"
                )
                continue
            break

        for evaluator_id in selected:
            info = EVALUATOR_MAP[evaluator_id]
            if info.requires_credentials:
                self._prompt_evaluator_credentials(info, console)

        # CI/CD gate thresholds (Req 16.5)
        cicd_thresholds: dict[str, float] | None = None
        if "cicd_gate" in selected:
            cicd_thresholds = self._prompt_cicd_thresholds(console)

        config = EvaluationConfig(
            evaluators=selected,
            cicd_thresholds=cicd_thresholds,
        )

        console.print(
            f"[green]  ✓ Evaluation configured: "
            f"[bold]{', '.join(selected)}[/bold][/green]"
        )
        self._config = config
        return config

    def evaluate(
        self,
        query: str,
        context: list,
        answer: str,
        ground_truth: str | None = None,
        config: EvaluationConfig | None = None,
    ) -> dict[str, float]:
        """Run enabled evaluators and return metric scores.

        Args:
            query:        The user query.
            context:      List of LangChain Document objects used as context.
            answer:       The generated answer.
            ground_truth: Optional reference answer (used by some evaluators).
            config:       EvaluationConfig override; uses configure() result if omitted.

        Returns:
            Dict mapping metric_name -> score (0.0-1.0).
        """
        active = config or self._config
        if active is None or not active.evaluators:
            return {}

        scores: dict[str, float] = {}
        deferred_export = "monitoring_export" in active.evaluators

        for evaluator_id in active.evaluators:
            if evaluator_id in ("cicd_gate", "monitoring_export"):
                continue

            runner = EVALUATOR_RUNNERS.get(evaluator_id)
            if runner is None:
                warnings.warn(
                    f"Evaluator {evaluator_id!r} has no runtime runner; skipping it.",
                    stacklevel=2,
                )
                continue

            try:
                if evaluator_id in ("langsmith", "langfuse", "arize_phoenix", "ragas"):
                    result = runner(
                        query,
                        context,
                        answer,
                        credential_store=self._credential_store,
                    )
                elif evaluator_id == "langgraph_trace":
                    result = runner(query, context, answer)
                else:
                    result = runner(query, context, answer)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Evaluator {evaluator_id!r} failed and returned no scores: {exc}",
                    stacklevel=2,
                )
                result = {}

            for metric, value in result.items():
                if isinstance(value, (int, float)):
                    scores[metric] = float(value)

        if deferred_export:
            export_scores = run_monitoring_export(
                query,
                context,
                answer,
                scores,
            )
            scores.update(export_scores)

        if ground_truth:
            gt_tokens = set(ground_truth.lower().split())
            ans_tokens = set(answer.lower().split())
            if ans_tokens:
                overlap = len(gt_tokens & ans_tokens) / max(len(gt_tokens), 1)
                scores["ground_truth_overlap"] = round(min(overlap, 1.0), 4)

        return scores

    def check_cicd_thresholds(
        self,
        result: dict[str, float],
        config: EvaluationConfig | None = None,
    ) -> bool:
        """Return True if all metric scores meet configured thresholds.

        Args:
            result: Dict of metric_name -> score from evaluate().
            config: EvaluationConfig with cicd_thresholds; uses instance config if None.

        Returns:
            True if all metrics pass (or no thresholds configured).
        """
        if config is None or config.cicd_thresholds is None:
            return True

        for metric, threshold in config.cicd_thresholds.items():
            score = result.get(metric)
            if score is None:
                continue  # metric not available — don't block
            if score < threshold:
                return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prompt_evaluator_credentials(
        self,
        info: EvaluatorInfo,
        console: object,
    ) -> None:
        """Prompt for evaluator credentials; keep evaluator selected on cancel. Req 16.4."""
        if not info.credential_fields:
            return

        # Check if credentials already stored
        if self._credential_store is not None:
            all_present = all(
                self._credential_store.get(info.credential_provider, field)  # type: ignore[union-attr]
                for field in info.credential_fields
            )
            if all_present:
                return

        console.print(  # type: ignore[union-attr]
            f"\n  [bold white]Credentials for {info.display_name}[/bold white] "
            f"[dim](press Enter to skip and configure later via /config)[/dim]"
        )

        for field in info.credential_fields:
            is_secret = any(
                field.upper().endswith(s)
                for s in ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")
            )
            if is_secret:
                value: str | None = questionary.password(
                    f"    {field} (optional, skip to configure later):",
                ).ask()
            else:
                value = questionary.text(
                    f"    {field} (optional, skip to configure later):",
                    default="",
                ).ask()

            if value and value.strip() and self._credential_store is not None:
                self._credential_store.set(  # type: ignore[union-attr]
                    info.credential_provider, field, value.strip()
                )

    def _prompt_cicd_thresholds(self, console: object) -> dict[str, float]:
        """Prompt for CI/CD metric thresholds. Req 16.5."""
        console.print(  # type: ignore[union-attr]
            "\n  [bold white]CI/CD Gate Thresholds[/bold white]\n"
            "  Enter minimum acceptable score (0.0-1.0) for each metric.\n"
            "  Press Enter to use defaults.\n"
        )

        thresholds: dict[str, float] = {}

        for metric, default in CICD_DEFAULT_METRICS.items():
            while True:
                raw: str | None = questionary.text(
                    f"    {metric} threshold (default {default}):",
                    default=str(default),
                ).ask()

                if not raw or not raw.strip():
                    thresholds[metric] = default
                    break

                try:
                    value = float(raw.strip())
                    validate_numeric(value, 0.0, 1.0, f"{metric}_threshold")
                    thresholds[metric] = value
                    break
                except (ValueError, ValidationError) as exc:
                    console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

        # Allow adding custom metrics
        while True:
            while True:
                add_more_raw = questionary.confirm(
                    "  Add a custom metric threshold?",
                    default=False,
                ).ask()
                if add_more_raw is not None:
                    add_more = bool(add_more_raw)
                    break
                console.print("[yellow]  Selection cancelled — please answer yes or no.[/yellow]")  # type: ignore[union-attr]
            if not add_more:
                break

            metric_name: str | None = questionary.text("    Custom metric name:").ask()
            if not metric_name or not metric_name.strip():
                break

            metric_name = metric_name.strip().lower().replace(" ", "_")

            while True:
                raw = questionary.text(
                    f"    Threshold for {metric_name} (0.0-1.0):",
                    default="0.75",
                ).ask()
                try:
                    value = float((raw or "0.75").strip())
                    validate_numeric(value, 0.0, 1.0, metric_name)
                    thresholds[metric_name] = value
                    break
                except (ValueError, ValidationError) as exc:
                    console.print(f"[red]  ✗ {exc}[/red]")  # type: ignore[union-attr]

        return thresholds
