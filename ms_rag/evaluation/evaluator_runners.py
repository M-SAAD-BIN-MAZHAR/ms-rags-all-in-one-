"""Runtime evaluator implementations for EvaluationFramework.

Each runner returns metric_name -> score (0.0-1.0). Failures are non-fatal but
must emit warnings before falling back or returning an empty result.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ms_rag.utils.credentials import resolve_credential


def context_texts(context: list) -> list[str]:
    """Extract plain text from LangChain Document objects or strings."""
    texts: list[str] = []
    for item in context:
        if isinstance(item, str):
            texts.append(item)
        else:
            content = getattr(item, "page_content", None)
            if isinstance(content, str) and content.strip():
                texts.append(content)
    return texts


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def lexical_grounding_scores(
    query: str,
    answer: str,
    context: list,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Reference-free lexical overlap metrics usable without external APIs."""
    ctx_texts = context_texts(context)
    if not ctx_texts or not answer.strip():
        return {}

    ctx_tokens = _token_set(" ".join(ctx_texts))
    answer_tokens = _token_set(answer)
    query_tokens = _token_set(query)

    if not answer_tokens:
        return {}

    context_recall = len(answer_tokens & ctx_tokens) / len(answer_tokens)
    context_precision = (
        len(answer_tokens & ctx_tokens) / len(ctx_tokens) if ctx_tokens else 0.0
    )
    answer_relevancy = (
        len(answer_tokens & query_tokens) / len(answer_tokens) if query_tokens else 0.0
    )

    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}context_recall": round(context_recall, 4),
        f"{key}context_precision": round(context_precision, 4),
        f"{key}answer_relevancy": round(answer_relevancy, 4),
        f"{key}faithfulness": round(context_recall, 4),
    }


def _prefixed(scores: dict[str, float], prefix: str) -> dict[str, float]:
    return {key if key.startswith(f"{prefix}_") else f"{prefix}_{key}": value for key, value in scores.items()}


def run_ragas(
    query: str,
    context: list,
    answer: str,
    *,
    credential_store: object | None = None,
) -> dict[str, float]:
    """Run RAGAS metrics when ragas and its dependencies are available."""
    try:
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
        )

        sample = {
            "user_input": query,
            "response": answer,
            "retrieved_contexts": context_texts(context) or [""],
        }
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
        result = ragas_evaluate([sample], metrics=metrics)

        scores: dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            if not df.empty:
                row = df.iloc[0]
                for col in df.columns:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        scores[str(col).lower()] = float(value)
        return scores
    except Exception as exc:
        warnings.warn(
            f"RAGAS evaluation failed; using lexical fallback metrics: {exc}",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="ragas")


def run_deepeval(
    query: str,
    context: list,
    answer: str,
) -> dict[str, float]:
    """Run DeepEval answer relevancy when deepeval is installed."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric  # noqa: PLC0415
        from deepeval.test_case import LLMTestCase  # noqa: PLC0415

        metric = AnswerRelevancyMetric(threshold=0.5)
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context_texts(context) or [""],
        )
        metric.measure(test_case)
        score = float(metric.score or 0.0)
        return {"answer_relevancy": score, "deepeval_answer_relevancy": score}
    except Exception as exc:
        warnings.warn(
            f"DeepEval evaluation failed; using lexical fallback metrics: {exc}",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="deepeval")


def run_trulens(
    query: str,
    context: list,
    answer: str,
) -> dict[str, float]:
    """Run TruLens package-backed local feedback compatibility checks."""
    try:
        from trulens.core import Feedback  # noqa: PLC0415
        from trulens.core import Select  # noqa: PLC0415
        from trulens.apps.langchain import TruChain  # noqa: PLC0415

        _ = (Feedback, Select, TruChain)  # validate supported modern packages
        scores = lexical_grounding_scores(query, answer, context)
        scores["package_available"] = 1.0
        return _prefixed(scores, "trulens")
    except Exception as exc:
        warnings.warn(
            f"TruLens import/check failed; using lexical fallback metrics: {exc}",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="trulens")


def run_langsmith(
    query: str,
    context: list,
    answer: str,
    *,
    credential_store: object | None = None,
) -> dict[str, float]:
    """Log a run to LangSmith when credentials are configured."""
    api_key = resolve_credential("LANGCHAIN_API_KEY", credential_store, "langsmith")
    project = resolve_credential("LANGCHAIN_PROJECT", credential_store, "langsmith")
    if not api_key:
        api_key = os.getenv("LANGCHAIN_API_KEY")
    if not project:
        project = os.getenv("LANGCHAIN_PROJECT", "ms_rag_pipeline")

    if not api_key:
        warnings.warn(
            "LangSmith evaluator selected but LANGCHAIN_API_KEY is not configured; skipping LangSmith logging.",
            stacklevel=2,
        )
        return {}

    try:
        from langsmith import Client  # noqa: PLC0415

        client = Client(api_key=api_key)
        run = client.create_run(
            name="ms_rag_query",
            run_type="chain",
            inputs={"query": query},
            outputs={"answer": answer},
            project_name=project,
            extra={"context_count": len(context)},
        )
        return {"langsmith_logged": 1.0 if run else 0.0}
    except Exception as exc:
        warnings.warn(f"LangSmith logging failed: {exc}", stacklevel=2)
        return {}


def run_langfuse(
    query: str,
    context: list,
    answer: str,
    *,
    credential_store: object | None = None,
) -> dict[str, float]:
    """Log a trace span to Langfuse when credentials are configured."""
    public_key = resolve_credential("LANGFUSE_PUBLIC_KEY", credential_store, "langfuse")
    secret_key = resolve_credential("LANGFUSE_SECRET_KEY", credential_store, "langfuse")
    host = resolve_credential("LANGFUSE_HOST", credential_store, "langfuse") or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )

    if not public_key or not secret_key:
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        warnings.warn(
            "Langfuse evaluator selected but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not configured; skipping Langfuse logging.",
            stacklevel=2,
        )
        return {}

    try:
        from langfuse import Langfuse  # noqa: PLC0415

        langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        trace = langfuse.trace(name="ms_rag_query", input=query, output=answer)
        trace.span(name="retrieval", metadata={"context_count": len(context)})
        langfuse.flush()
        return {"langfuse_logged": 1.0}
    except Exception as exc:
        warnings.warn(f"Langfuse logging failed: {exc}", stacklevel=2)
        return {}


def run_arize_phoenix(
    query: str,
    context: list,
    answer: str,
    *,
    credential_store: object | None = None,
) -> dict[str, float]:
    """Record an OpenInference-compatible Phoenix trace when configured."""
    api_key = resolve_credential("PHOENIX_API_KEY", credential_store, "arize_phoenix")
    endpoint = resolve_credential(
        "PHOENIX_COLLECTOR_ENDPOINT", credential_store, "arize_phoenix"
    )
    if not api_key:
        api_key = os.getenv("PHOENIX_API_KEY")
    if not endpoint:
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    if not endpoint:
        warnings.warn(
            "Phoenix evaluator selected but PHOENIX_COLLECTOR_ENDPOINT is not configured; "
            "using local Phoenix-prefixed lexical metrics.",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="phoenix")

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

        provider = TracerProvider(resource=Resource.create({"service.name": "ms-rags-all-in-one-evaluation"}))
        headers = {"api_key": api_key} if api_key else None
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
        tracer = provider.get_tracer("ms_rag.evaluation.phoenix")
        with tracer.start_as_current_span("ms_rag.query_eval") as span:
            span.set_attribute("input.value", query)
            span.set_attribute("output.value", answer)
            span.set_attribute("retrieval.context_count", len(context))
        scores = lexical_grounding_scores(query, answer, context, prefix="phoenix")
        scores["phoenix_trace_exported"] = 1.0
        return scores
    except Exception as exc:
        warnings.warn(
            f"Phoenix import/check failed; using lexical fallback metrics: {exc}",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="phoenix")


def run_ares(query: str, context: list, answer: str) -> dict[str, float]:
    """Run ARES package-backed availability path plus local RAG scores.

    ARES (`ares-ai`) is primarily designed around dataset/config-driven evaluation.
    For a single live query, MS-RAGS(ALL-IN-ONE) validates the official package when installed
    and returns ARES-prefixed local scores for the same retrieval/generation fields.
    """
    try:
        import ares  # type: ignore[import-not-found]  # noqa: PLC0415

        _ = ares
        scores = lexical_grounding_scores(query, answer, context)
        scores["package_available"] = 1.0
        return _prefixed(scores, "ares")
    except Exception as exc:
        warnings.warn(
            f"ARES package-backed check failed; using ARES-compatible lexical metrics: {exc}",
            stacklevel=2,
        )
        return lexical_grounding_scores(query, answer, context, prefix="ares")


def run_ragbench(query: str, context: list, answer: str) -> dict[str, float]:
    """Run RAGBench-compatible single-query metrics with optional dataset package."""
    scores = lexical_grounding_scores(query, answer, context)
    try:
        import datasets  # noqa: PLC0415

        _ = datasets
        scores["datasets_package_available"] = 1.0
    except Exception as exc:
        warnings.warn(
            f"RAGBench dataset tooling is unavailable; using RAGBench-compatible lexical metrics: {exc}",
            stacklevel=2,
        )
    return _prefixed(scores, "ragbench")


def run_langgraph_trace(
    query: str,
    context: list,
    answer: str,
    *,
    export_path: str | None = None,
) -> dict[str, float]:
    """Append a lightweight trace record for agentic workflows."""
    path = Path(export_path or os.getenv("MS_RAG_TRACE_LOG", "./ms_rag_traces.jsonl"))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "context_count": len(context),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"langgraph_trace_logged": 1.0}
    except Exception as exc:
        warnings.warn(f"LangGraph trace export failed: {exc}", stacklevel=2)
        return {}


def run_monitoring_export(
    query: str,
    context: list,
    answer: str,
    metrics: dict[str, float],
    *,
    export_path: str | None = None,
) -> dict[str, float]:
    """Export aggregated metrics to JSONL for dashboards."""
    path = Path(
        export_path or os.getenv("MS_RAG_METRICS_EXPORT", "./ms_rag_metrics.jsonl")
    )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "context_count": len(context),
        "metrics": metrics,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"monitoring_export_logged": 1.0}
    except Exception as exc:
        warnings.warn(f"Monitoring metrics export failed: {exc}", stacklevel=2)
        return {}


EVALUATOR_RUNNERS: dict[str, Any] = {
    "ragas": run_ragas,
    "deepeval": run_deepeval,
    "trulens": run_trulens,
    "langsmith": run_langsmith,
    "langfuse": run_langfuse,
    "arize_phoenix": run_arize_phoenix,
    "ares": run_ares,
    "ragbench": run_ragbench,
    "langgraph_trace": run_langgraph_trace,
}
