"""Tests for RAG type preset wiring."""

from __future__ import annotations

from ms_rag.workflow.rag_presets import RAG_TYPE_PRESETS, get_rag_preset
from ms_rag.workflow.rag_type_selector import RAG_TYPES


REQUIRED_RUNTIME_FEATURES = {
    "naive_rag": ("retrieval",),
    "advanced_rag": ("user_prompts",),
    "modular_rag": ("user_prompts",),
    "agentic_rag": ("langgraph", "retrieval"),
    "self_rag": ("langgraph", "grading"),
    "corrective_rag": ("langgraph", "grading", "web_fallback_option"),
    "speculative_rag": ("draft_verify_chain",),
    "graphrag": ("graph_store", "hybrid_keyword_store"),
    "hyde_rag": ("hyde",),
    "multi_query_rag": ("multi_query",),
    "rag_fusion": ("rag_fusion",),
    "step_back_rag": ("step_back_prompting",),
    "parent_child_rag": ("parent_child_retrieval",),
    "adaptive_rag": ("langgraph", "routing"),
    "contextual_compression_rag": ("contextual_compression",),
}


def test_every_rag_type_has_a_preset() -> None:
    defined = {rag.rag_type for rag in RAG_TYPES}
    assert set(RAG_TYPE_PRESETS) == defined
    assert set(REQUIRED_RUNTIME_FEATURES) == defined


def test_hyde_rag_locks_hyde_query_enhancement() -> None:
    preset = get_rag_preset("hyde_rag")
    assert preset.query_enhancement == ["hyde"]
    assert preset.retrieval is not None
    assert preset.retrieval.strategy == "dense_vector"
    assert not preset.allow_query_enhancement_prompt


def test_parent_child_rag_locks_parent_child_retrieval() -> None:
    preset = get_rag_preset("parent_child_rag")
    assert preset.retrieval is not None
    assert preset.retrieval.strategy == "parent_child"
    assert not preset.allow_retrieval_prompt


def test_advanced_rag_keeps_module_prompts_available() -> None:
    preset = get_rag_preset("advanced_rag")
    assert preset.allow_query_enhancement_prompt
    assert preset.allow_retrieval_prompt
    assert preset.allow_reranking_prompt
    assert preset.allow_compression_prompt


def test_contextual_compression_rag_locks_compression() -> None:
    preset = get_rag_preset("contextual_compression_rag")
    assert preset.compression is not None
    assert preset.compression.techniques == ["contextual_compression"]


def test_every_locked_rag_type_has_required_behavior_not_only_name() -> None:
    for rag_type, features in REQUIRED_RUNTIME_FEATURES.items():
        preset = get_rag_preset(rag_type)
        if "user_prompts" in features:
            assert preset.allow_query_enhancement_prompt
            assert preset.allow_retrieval_prompt
            continue
        if "langgraph" in features:
            assert rag_type in {"agentic_rag", "self_rag", "corrective_rag", "adaptive_rag"}
        if "hyde" in features:
            assert preset.query_enhancement == ["hyde"]
        if "multi_query" in features:
            assert preset.query_enhancement == ["multi_query"]
        if "rag_fusion" in features:
            assert preset.query_enhancement == ["rag_fusion"]
        if "step_back_prompting" in features:
            assert preset.query_enhancement == ["step_back_prompting"]
        if "parent_child_retrieval" in features:
            assert preset.retrieval and preset.retrieval.strategy == "parent_child"
        if "contextual_compression" in features:
            assert preset.compression and "contextual_compression" in preset.compression.techniques
        if "hybrid_keyword_store" in features:
            assert preset.retrieval and preset.retrieval.strategy == "hybrid"
        if "graph_store" in features:
            assert preset.query_enhancement == []
            assert any("graph store" in note.lower() for note in preset.notes)
        if "web_fallback_option" in features:
            assert any("Web Search" in note for note in preset.notes)
