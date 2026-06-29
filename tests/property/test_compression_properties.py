"""Property-based tests for ContextCompressor.

Properties covered:
    Property 16: Context Compressor Multi-Select Round-Trip (Req 14.2, 14.3)

Validates: Requirements 14.1-14.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.models import CompressionConfig
from ms_rag.query.context_compressor import (
    COMPRESSION_TECHNIQUES,
    LLM_REQUIRED_TECHNIQUES,
    TECHNIQUE_INFO,
    ContextCompressor,
)


# ---------------------------------------------------------------------------
# Property 16: Context Compressor Multi-Select Round-Trip
# ---------------------------------------------------------------------------


@given(
    techniques=st.frozensets(
        st.sampled_from(COMPRESSION_TECHNIQUES),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=50)
def test_context_compressor_multi_select_round_trip(
    techniques: frozenset[str],
) -> None:
    """Feature: ms-rag, Property 16: Context Compressor Multi-Select Round-Trip.

    Selected techniques must be stored exactly as selected in the returned
    CompressionConfig, in the order presented in the checklist.
    """
    techniques_in_order = [t for t in COMPRESSION_TECHNIQUES if t in techniques]
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = techniques_in_order
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        # Any embedding-based compression choice asks for the threshold.
        if set(techniques) & {
            "embeddings_filter",
            "contextual_compression",
            "document_compressor_pipeline",
        }:
            mock_text = MagicMock()
            mock_text.ask.return_value = "0.75"
            mock_q.text.return_value = mock_text

        result = compressor.configure(configured_providers=["openai"])

    assert result is not None
    assert set(result.techniques) == techniques, (
        f"Expected {techniques}, got {set(result.techniques)}"
    )


def test_techniques_stored_in_checklist_order() -> None:
    """Techniques must be stored in the order they appear in the checklist."""
    compressor = ContextCompressor()
    # Select techniques in reverse order — they should be stored in checklist order
    selected_ids = ["summary_compression", "embeddings_filter", "llm_chain_extraction"]
    expected_order = [t for t in COMPRESSION_TECHNIQUES if t in selected_ids]

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = expected_order
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        mock_text = MagicMock()
        mock_text.ask.return_value = "0.8"
        mock_q.text.return_value = mock_text

        result = compressor.configure(configured_providers=["openai"])

    assert result.techniques == expected_order


# ---------------------------------------------------------------------------
# No compression (Req 14.1)
# ---------------------------------------------------------------------------


def test_no_compression_returns_none() -> None:
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_q.confirm.return_value = mock_confirm

        result = compressor.configure()

    assert result is None


def test_zero_technique_selection_reprompts() -> None:
    """Req 14.3: zero-technique selection must show error and re-present."""
    compressor = ContextCompressor()
    call_count = {"n": 0}

    def checkbox_side_effect(*args, **kwargs) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        m.ask.return_value = [] if call_count["n"] == 1 else ["redundancy_removal"]
        return m

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_q.confirm.return_value = mock_confirm

        mock_q.checkbox.side_effect = checkbox_side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = compressor.configure(configured_providers=["openai"])

    assert result is not None
    assert "redundancy_removal" in result.techniques
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# LLM-dependency check (Req 14.5)
# ---------------------------------------------------------------------------


def test_llm_required_techniques_are_flagged() -> None:
    """llm_chain_extraction and summary_compression must require LLM."""
    assert "llm_chain_extraction" in LLM_REQUIRED_TECHNIQUES
    assert "summary_compression" in LLM_REQUIRED_TECHNIQUES


def test_non_llm_techniques_not_in_llm_required() -> None:
    non_llm = {"embeddings_filter", "redundancy_removal", "contextual_compression",
               "document_compressor_pipeline"}
    for t in non_llm:
        assert t not in LLM_REQUIRED_TECHNIQUES


def test_contextual_compression_builds_real_wrapper_with_embeddings() -> None:
    compressor = ContextCompressor()
    config = CompressionConfig(
        techniques=["contextual_compression"],
        similarity_threshold=0.5,
    )
    embeddings = MagicMock()
    base_retriever = MagicMock()

    with patch("langchain_classic.retrievers.document_compressors.EmbeddingsFilter") as filter_cls:
        filter_cls.return_value = MagicMock(name="filter")
        result = compressor.get_compressor(
            config,
            llm=None,
            embeddings=embeddings,
            base_retriever=base_retriever,
        )

    assert result.base_compressor is filter_cls.return_value
    assert result.base_retriever is base_retriever
    filter_cls.assert_called_once()


def test_document_compressor_pipeline_builds_real_wrapper_with_embeddings() -> None:
    compressor = ContextCompressor()
    config = CompressionConfig(
        techniques=["document_compressor_pipeline"],
        similarity_threshold=0.5,
    )
    embeddings = MagicMock()
    base_retriever = MagicMock()

    with patch("langchain_classic.retrievers.document_compressors.EmbeddingsFilter") as filter_cls, \
         patch("langchain_community.document_transformers.EmbeddingsRedundantFilter") as redundant_cls, \
         patch("langchain_classic.retrievers.document_compressors.DocumentCompressorPipeline") as pipeline_cls:
        filter_cls.return_value = MagicMock(name="filter")
        redundant_cls.return_value = MagicMock(name="redundant")
        pipeline_cls.return_value = MagicMock(name="pipeline")
        result = compressor.get_compressor(
            config,
            llm=None,
            embeddings=embeddings,
            base_retriever=base_retriever,
        )

    assert result.base_compressor is pipeline_cls.return_value
    assert result.base_retriever is base_retriever
    pipeline_cls.assert_called_once()


def test_document_compressor_pipeline_is_meta_when_combined() -> None:
    compressor = ContextCompressor()
    config = CompressionConfig(
        techniques=["document_compressor_pipeline", "embeddings_filter"],
        similarity_threshold=0.5,
    )
    embeddings = MagicMock()
    base_retriever = MagicMock()

    with patch("langchain_classic.retrievers.document_compressors.EmbeddingsFilter") as filter_cls, \
         patch("langchain_classic.retrievers.document_compressors.DocumentCompressorPipeline") as pipeline_cls:
        filter_cls.return_value = MagicMock(name="filter")
        pipeline_cls.return_value = MagicMock(name="pipeline")
        result = compressor.get_compressor(
            config,
            llm=None,
            embeddings=embeddings,
            base_retriever=base_retriever,
        )

    assert result.base_compressor is filter_cls.return_value
    assert result.base_retriever is base_retriever
    filter_cls.assert_called_once()
    pipeline_cls.assert_not_called()


def test_summary_compression_builds_summary_compressor() -> None:
    from ms_rag.query.compression_retriever import LLMSummaryCompressor

    compressor = ContextCompressor()
    config = CompressionConfig(techniques=["summary_compression"])
    llm = MagicMock()

    result = compressor.get_compressor(
        config,
        llm=llm,
        embeddings=None,
        base_retriever=None,
    )

    assert isinstance(result, LLMSummaryCompressor)
    assert result.llm is llm


def test_safe_compression_falls_back_when_compressor_removes_all_docs() -> None:
    from langchain_core.documents import Document
    from ms_rag.query.compression_retriever import SafeCompressionRetriever

    docs = [Document(page_content="Elephants are large mammals.")]
    base_retriever = MagicMock()
    base_retriever.invoke.return_value = docs
    compressor = MagicMock()
    compressor.compress_documents.return_value = []

    retriever = SafeCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    with pytest.warns(UserWarning, match="too little context"):
        result = retriever.invoke("tell me about elephants")

    assert result == docs


def test_safe_compression_accepts_transformer_only_components() -> None:
    from langchain_core.documents import Document
    from ms_rag.query.compression_retriever import SafeCompressionRetriever

    docs = [
        Document(page_content="Elephants are large mammals."),
        Document(page_content="Elephants are large mammals."),
    ]
    transformed = [docs[0]]
    base_retriever = MagicMock()
    base_retriever.invoke.return_value = docs
    class TransformerOnly:
        def __init__(self) -> None:
            self.seen_documents: list[Document] | None = None

        def transform_documents(self, documents: list[Document]) -> list[Document]:
            self.seen_documents = documents
            return transformed

    transformer = TransformerOnly()

    retriever = SafeCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=transformer,
    )

    result = retriever.invoke("tell me about elephants")

    assert result == transformed
    assert transformer.seen_documents == docs


def test_safe_compression_keeps_minimum_context_when_too_aggressive() -> None:
    from langchain_core.documents import Document
    from ms_rag.query.compression_retriever import SafeCompressionRetriever

    docs = [
        Document(page_content="Elephant overview."),
        Document(page_content="Elephant habitat."),
        Document(page_content="Elephant behavior."),
        Document(page_content="Elephant conservation."),
    ]
    base_retriever = MagicMock()
    base_retriever.invoke.return_value = docs
    compressor = MagicMock()
    compressor.compress_documents.return_value = [docs[0]]

    retriever = SafeCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
        min_retained_documents=3,
    )

    with pytest.warns(UserWarning, match="too little context"):
        result = retriever.invoke("tell me about elephants")

    assert result == docs[:3]
    assert retriever._ms_rag_pre_compression_count == 4
    assert retriever._ms_rag_post_compression_count == 3
    assert retriever._ms_rag_compression_fallback is True


# ---------------------------------------------------------------------------
# Threshold prompt (Req 14.4)
# ---------------------------------------------------------------------------


def test_threshold_default_is_0_75() -> None:
    compressor = ContextCompressor()

    with patch("ms_rag.query.context_compressor.questionary") as mock_q:
        m = MagicMock()
        m.ask.return_value = ""  # accept default
        mock_q.text.return_value = m
        result = compressor._prompt_threshold(console=MagicMock())

    assert result == 0.75


def test_threshold_out_of_range_reprompts() -> None:
    compressor = ContextCompressor()
    call_count = {"n": 0}

    def side_effect(*a, **kw) -> MagicMock:
        m = MagicMock()
        call_count["n"] += 1
        m.ask.return_value = "1.5" if call_count["n"] == 1 else "0.8"
        return m

    with patch("ms_rag.query.context_compressor.questionary") as mock_q, \
         patch("ms_rag.query.context_compressor.Console"):
        mock_q.text.side_effect = side_effect
        result = compressor._prompt_threshold(console=MagicMock())

    assert result == 0.8
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------


class TestCompressionTechniqueList:
    def test_exactly_6_techniques_defined(self) -> None:
        assert len(COMPRESSION_TECHNIQUES) == 6

    def test_all_required_techniques_present(self) -> None:
        required = {
            "llm_chain_extraction", "embeddings_filter",
            "document_compressor_pipeline", "redundancy_removal",
            "contextual_compression", "summary_compression",
        }
        defined = set(COMPRESSION_TECHNIQUES)
        missing = required - defined
        assert not missing, f"Missing techniques: {missing}"

    def test_no_duplicate_technique_ids(self) -> None:
        assert len(COMPRESSION_TECHNIQUES) == len(set(COMPRESSION_TECHNIQUES))

    def test_all_techniques_have_display_names(self) -> None:
        for tid in COMPRESSION_TECHNIQUES:
            assert len(TECHNIQUE_INFO[tid]["display"].strip()) > 0

    def test_all_techniques_have_descriptions(self) -> None:
        for tid in COMPRESSION_TECHNIQUES:
            assert len(TECHNIQUE_INFO[tid]["description"].strip()) > 0

    def test_compression_config_stores_all_params(self) -> None:
        config = CompressionConfig(
            techniques=["embeddings_filter", "redundancy_removal"],
            similarity_threshold=0.8,
        )
        assert config.techniques == ["embeddings_filter", "redundancy_removal"]
        assert config.similarity_threshold == 0.8
