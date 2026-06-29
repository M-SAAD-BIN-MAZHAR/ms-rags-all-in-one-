"""RAG type presets that keep later setup steps aligned with Step 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from ms_rag.models import CompressionConfig, RetrievalConfig


@dataclass(frozen=True)
class RAGTypePreset:
    """Required and allowed downstream settings for a selected RAG type."""

    rag_type: str
    summary: str
    query_enhancement: list[str] | None = None
    retrieval: RetrievalConfig | None = None
    compression: CompressionConfig | None = None
    allow_query_enhancement_prompt: bool = False
    allow_retrieval_prompt: bool = False
    allow_reranking_prompt: bool = False
    allow_compression_prompt: bool = False
    allow_evaluation_prompt: bool = True
    notes: list[str] = field(default_factory=list)


RAG_TYPE_PRESETS: dict[str, RAGTypePreset] = {
    "naive_rag": RAGTypePreset(
        rag_type="naive_rag",
        summary="Simple retrieve-then-generate baseline.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_evaluation_prompt=True,
    ),
    "advanced_rag": RAGTypePreset(
        rag_type="advanced_rag",
        summary="Quality-focused pipeline with user-selected enhancement, retrieval, reranking, and compression.",
        allow_query_enhancement_prompt=True,
        allow_retrieval_prompt=True,
        allow_reranking_prompt=True,
        allow_compression_prompt=True,
    ),
    "modular_rag": RAGTypePreset(
        rag_type="modular_rag",
        summary="Composable workbench mode: all modules remain available.",
        allow_query_enhancement_prompt=True,
        allow_retrieval_prompt=True,
        allow_reranking_prompt=True,
        allow_compression_prompt=True,
    ),
    "agentic_rag": RAGTypePreset(
        rag_type="agentic_rag",
        summary="LangGraph agent loop with retrieval and generation.",
        query_enhancement=["query_rewriting"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
        allow_compression_prompt=True,
    ),
    "self_rag": RAGTypePreset(
        rag_type="self_rag",
        summary="LangGraph relevance grading and groundedness checking.",
        query_enhancement=["query_rewriting"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
        allow_compression_prompt=True,
    ),
    "corrective_rag": RAGTypePreset(
        rag_type="corrective_rag",
        summary="LangGraph retrieval grading with query correction and optional approved web fallback.",
        query_enhancement=["query_rewriting"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
        allow_compression_prompt=True,
        notes=[
            "For full CRAG behavior, enable the Web Search tool in Step 3b so missing or irrelevant corpus context can fall back to approved web search."
        ],
    ),
    "speculative_rag": RAGTypePreset(
        rag_type="speculative_rag",
        summary="Draft-first answer, retrieve evidence, then verify/refine.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
    ),
    "graphrag": RAGTypePreset(
        rag_type="graphrag",
        summary="Persistent knowledge graph retrieval for cross-document relationship questions.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="hybrid", top_k=8, alpha=0.5),
        allow_reranking_prompt=True,
        notes=[
            "Requires a graph store for entity/relation graph retrieval and a keyword store for hybrid evidence retrieval."
        ],
    ),
    "hyde_rag": RAGTypePreset(
        rag_type="hyde_rag",
        summary="HyDE query expansion using a hypothetical answer document.",
        query_enhancement=["hyde"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
    ),
    "multi_query_rag": RAGTypePreset(
        rag_type="multi_query_rag",
        summary="Multiple LLM-generated query variants are retrieved and merged.",
        query_enhancement=["multi_query"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
    ),
    "rag_fusion": RAGTypePreset(
        rag_type="rag_fusion",
        summary="Multiple generated queries are merged with reciprocal-rank style fusion.",
        query_enhancement=["rag_fusion"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=7),
        allow_reranking_prompt=True,
    ),
    "step_back_rag": RAGTypePreset(
        rag_type="step_back_rag",
        summary="Retrieves with a broader conceptual step-back query.",
        query_enhancement=["step_back_prompting"],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
    ),
    "parent_child_rag": RAGTypePreset(
        rag_type="parent_child_rag",
        summary="Retrieves precise child chunks and returns larger parent context.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="parent_child", top_k=5),
    ),
    "adaptive_rag": RAGTypePreset(
        rag_type="adaptive_rag",
        summary="LangGraph routing for simple vs retrieval-heavy queries.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        allow_reranking_prompt=True,
    ),
    "contextual_compression_rag": RAGTypePreset(
        rag_type="contextual_compression_rag",
        summary="Dense retrieval followed by contextual compression.",
        query_enhancement=[],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=8),
        compression=CompressionConfig(techniques=["contextual_compression"], similarity_threshold=0.75),
    ),
}


def get_rag_preset(rag_type: str | None) -> RAGTypePreset:
    """Return the preset for a RAG type, defaulting to Naive RAG."""
    return RAG_TYPE_PRESETS.get(rag_type or "naive_rag", RAG_TYPE_PRESETS["naive_rag"])
