"""Property-based tests for IngestionOrchestrator.

Properties covered:
    Property 26: Ingestion Failure Isolation (Req 19.2)
    Property 28: Recursive Document Discovery (Req 20.2)
    Property 29: Loader Assignment by File Type (Req 20.3)

Validates: Requirements 9.6, 9.7, 19.2, 20.1, 20.2, 20.3
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ms_rag.ingestion.document_type_selector import (
    DOCUMENT_TYPE_IDS,
    EXTENSION_TO_DOCTYPE,
)
from ms_rag.ingestion.ingestion_orchestrator import (
    IngestionOrchestrator,
    retry_with_backoff,
)
from ms_rag.models import (
    ChunkingConfig,
    IngestionResult,
    VectorDBConfig,
    EmbeddingModelConfig,
)


# ---------------------------------------------------------------------------
# Property 26: Ingestion Failure Isolation
# ---------------------------------------------------------------------------


@given(
    num_docs=st.integers(min_value=1, max_value=20),
    fail_indices=st.frozensets(
        st.integers(min_value=0, max_value=19),
        max_size=10,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_ingestion_failure_isolation(
    num_docs: int, fail_indices: frozenset[int]
) -> None:
    """Feature: ms-rag, Property 26: Ingestion Failure Isolation.

    Ingestion must complete for all non-failing documents, and
    IngestionResult.failed_documents must list exactly the failed ones.
    """
    orchestrator = IngestionOrchestrator()

    # Create fake source files that "exist"
    sources = [f"doc_{i}.txt" for i in range(num_docs)]
    actual_fails = {i for i in fail_indices if i < num_docs}
    expected_success = num_docs - len(actual_fails)

    loader_map = {"txt": "TextLoader"}
    chunking_config = ChunkingConfig(strategy="fixed_size", chunk_size=100, chunk_overlap=0)
    vector_db = VectorDBConfig(db_type="chroma", connection_params={}, collection_name="test")
    embedding_model = EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small")

    call_counter = {"n": 0}

    def mock_load_source(source, loader_map, youtube_language="en") -> list:
        idx = int(source.replace("doc_", "").replace(".txt", ""))
        call_counter["n"] += 1
        if idx in actual_fails:
            raise RuntimeError(f"Simulated failure for doc {idx}")
        # Return a fake document
        doc = MagicMock()
        doc.page_content = f"content of doc {idx}"
        return [doc]

    mock_splitter = MagicMock()
    mock_splitter.split_documents.side_effect = lambda docs: docs  # identity

    mock_store = MagicMock()

    with patch.object(orchestrator, "_load_source", side_effect=mock_load_source), \
         patch.object(orchestrator, "discover_documents", return_value=sources), \
         patch("ms_rag.ingestion.ingestion_orchestrator.ChunkingEngine") as mock_engine_cls, \
         patch("ms_rag.ingestion.ingestion_orchestrator.Console"):

        mock_engine = MagicMock()
        mock_engine.get_splitter.return_value = mock_splitter
        mock_engine_cls.return_value = mock_engine

        result = orchestrator.ingest(
            sources=sources,
            loader_map=loader_map,
            chunking_config=chunking_config,
            embedding_model=embedding_model,
            vector_db=vector_db,
            vector_store=mock_store,
        )

    # Verify failure isolation
    assert len(result.failed_documents) == len(actual_fails), (
        f"Expected {len(actual_fails)} failures, got {len(result.failed_documents)}"
    )

    # Verify successful documents were stored
    failed_paths = {fd[0] for fd in result.failed_documents}
    for i in range(num_docs):
        source = f"doc_{i}.txt"
        if i in actual_fails:
            assert source in failed_paths, f"doc_{i} should be in failed_documents"
        else:
            assert source not in failed_paths, f"doc_{i} should NOT be in failed_documents"


# ---------------------------------------------------------------------------
# Property 28: Recursive Document Discovery
# ---------------------------------------------------------------------------


@given(
    doc_types=st.frozensets(
        st.sampled_from(["pdf", "txt", "docx", "markdown", "code"]),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=30)
def test_recursive_document_discovery(doc_types: frozenset[str]) -> None:
    """Feature: ms-rag, Property 28: Recursive Document Discovery.

    discover_documents must return exactly the files whose extensions match
    the selected doc types — no non-matching files, no matching files omitted.
    """
    orchestrator = IngestionOrchestrator()

    # Build extension sets for selected and non-selected types
    selected_exts: set[str] = {
        ext for ext, dt in EXTENSION_TO_DOCTYPE.items() if dt in doc_types
    }
    other_exts = {".log", ".bak", ".tmp", ".xyz_unknown"}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create nested directory structure
        subdir = root / "sub" / "nested"
        subdir.mkdir(parents=True)

        created_matching: set[Path] = set()
        created_other: set[Path] = set()

        # Create matching files at various depths
        for i, ext in enumerate(list(selected_exts)[:5]):
            p1 = root / f"file{i}{ext}"
            p1.touch()
            created_matching.add(p1)

            p2 = subdir / f"nested_file{i}{ext}"
            p2.touch()
            created_matching.add(p2)

        # Create non-matching files
        for ext in other_exts:
            p = root / f"other{ext}"
            p.touch()
            created_other.add(p)

        result = orchestrator.discover_documents(
            sources=[str(root)],
            doc_types=list(doc_types),
        )

        result_paths = {Path(str(r)) for r in result}

        # All matching files must be discovered
        for expected in created_matching:
            assert expected in result_paths, (
                f"Matching file {expected} not found in discovery results"
            )

        # No non-matching files should appear
        for unexpected in created_other:
            assert unexpected not in result_paths, (
                f"Non-matching file {unexpected} should not appear in results"
            )


def test_discover_documents_handles_youtube_url() -> None:
    """YouTube URLs must be included when youtube doc_type is selected."""
    orchestrator = IngestionOrchestrator()
    sources = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    result = orchestrator.discover_documents(sources, ["youtube"])
    assert len(result) == 1
    assert "youtube.com" in str(result[0])


def test_discover_documents_excludes_youtube_without_type() -> None:
    """YouTube URLs must NOT appear when youtube is not in selected doc_types."""
    orchestrator = IngestionOrchestrator()
    sources = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    result = orchestrator.discover_documents(sources, ["pdf"])
    assert len(result) == 0


def test_discover_documents_handles_http_url() -> None:
    """HTTP URLs must be included when url doc_type is selected."""
    orchestrator = IngestionOrchestrator()
    sources = ["https://example.com/page"]
    result = orchestrator.discover_documents(sources, ["url"])
    assert len(result) == 1


def test_discover_documents_empty_sources() -> None:
    """Empty sources list returns empty result."""
    orchestrator = IngestionOrchestrator()
    result = orchestrator.discover_documents([], ["pdf", "txt"])
    assert result == []


# ---------------------------------------------------------------------------
# Property 29: Loader Assignment by File Type
# ---------------------------------------------------------------------------


@given(
    file_ext=st.sampled_from(list(EXTENSION_TO_DOCTYPE.keys()))
)
@settings(max_examples=50)
def test_loader_assignment_by_file_type(file_ext: str) -> None:
    """Feature: ms-rag, Property 29: Loader Assignment by File Type.

    The loader used for a file must be the one mapped to the file's
    detected document type — no file should use a loader for a different type.
    """
    orchestrator = IngestionOrchestrator()

    expected_doc_type = EXTENSION_TO_DOCTYPE[file_ext]
    expected_loader = f"TestLoader_{expected_doc_type}"

    loader_map = {expected_doc_type: expected_loader}

    # Create a fake file with the given extension
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
        fake_path = Path(f.name)
        f.write(b"test content")

    try:
        # Mock _invoke_loader to capture what loader was called
        invoked_loaders: list[str] = []

        def mock_invoke_loader(loader_class_name: str, source: str, language: str = "en") -> list:
            invoked_loaders.append(loader_class_name)
            doc = MagicMock()
            doc.page_content = "test"
            return [doc]

        with patch.object(orchestrator, "_invoke_loader", side_effect=mock_invoke_loader):
            docs = orchestrator._load_source(
                source=fake_path,
                loader_map=loader_map,
            )

        assert len(invoked_loaders) == 1, "Exactly one loader should be invoked"
        assert invoked_loaders[0] == expected_loader, (
            f"For ext {file_ext!r} → doc_type {expected_doc_type!r}: "
            f"expected loader {expected_loader!r}, got {invoked_loaders[0]!r}"
        )

    finally:
        fake_path.unlink(missing_ok=True)


def test_load_source_raises_ingestion_error_for_missing_loader() -> None:
    """If no loader is configured for a doc type, IngestionError must be raised."""
    from ms_rag.utils.exceptions import IngestionError

    orchestrator = IngestionOrchestrator()
    fake_path = Path("test_doc.pdf")

    # No pdf loader in the map
    with pytest.raises(IngestionError, match="No loader configured"):
        orchestrator._load_source(
            source=fake_path,
            loader_map={"txt": "TextLoader"},  # pdf not included
        )


def test_tabula_loader_falls_back_to_pypdf_for_general_pdf_text() -> None:
    orchestrator = IngestionOrchestrator()
    fallback_docs = [MagicMock(page_content="resume text")]

    with pytest.warns(UserWarning, match="falling back to PyPDFLoader"), \
         patch("langchain_community.document_loaders.UnstructuredPDFLoader", side_effect=RuntimeError("tabula unavailable")), \
         patch("langchain_community.document_loaders.PyPDFLoader") as mock_pypdf:
            mock_pypdf.return_value.load.return_value = fallback_docs
            result = orchestrator._invoke_loader("TabulaLoader", "Resume\\resume.pdf")

    assert result == fallback_docs
    mock_pypdf.assert_called_once_with("Resume\\resume.pdf")


def test_camelot_loader_falls_back_to_pypdf_for_general_pdf_text() -> None:
    orchestrator = IngestionOrchestrator()
    fallback_docs = [MagicMock(page_content="resume text")]

    with pytest.warns(UserWarning, match="falling back to PyPDFLoader"), \
         patch("langchain_community.document_loaders.CamelotPDFLoader", side_effect=RuntimeError("camelot unavailable"), create=True), \
         patch("langchain_community.document_loaders.PyPDFLoader") as mock_pypdf:
            mock_pypdf.return_value.load.return_value = fallback_docs
            result = orchestrator._invoke_loader("CamelotLoader", "Resume\\resume.pdf")

    assert result == fallback_docs
    mock_pypdf.assert_called_once_with("Resume\\resume.pdf")


def test_unstructured_pdf_loader_warns_before_falling_back_to_pypdf() -> None:
    orchestrator = IngestionOrchestrator()
    fallback_docs = [MagicMock(page_content="resume text")]

    with pytest.warns(UserWarning, match="UnstructuredPDFLoader could not parse"), \
         patch("langchain_unstructured.UnstructuredLoader", side_effect=RuntimeError("missing dependency")), \
         patch("langchain_community.document_loaders.PyPDFLoader") as mock_pypdf:
            mock_pypdf.return_value.load.return_value = fallback_docs
            result = orchestrator._invoke_loader("UnstructuredPDFLoader", "Resume\\resume.pdf")

    assert result == fallback_docs
    mock_pypdf.assert_called_once_with("Resume\\resume.pdf")


def test_unstructured_docx_loader_warns_before_falling_back_to_docx2txt() -> None:
    orchestrator = IngestionOrchestrator()
    fallback_docs = [MagicMock(page_content="resume text")]

    with pytest.warns(UserWarning, match="UnstructuredWordDocumentLoader could not parse"), \
         patch("langchain_unstructured.UnstructuredLoader", side_effect=RuntimeError("missing dependency")), \
         patch("langchain_community.document_loaders.Docx2txtLoader") as mock_docx:
            mock_docx.return_value.load.return_value = fallback_docs
            result = orchestrator._invoke_loader("UnstructuredWordDocumentLoader", "Resume\\resume.docx")

    assert result == fallback_docs
    mock_docx.assert_called_once_with("Resume\\resume.docx")


def test_ingest_attaches_keyword_corpus_to_vector_store() -> None:
    orchestrator = IngestionOrchestrator()
    mock_store = MagicMock()
    mock_splitter = MagicMock()
    chunk_a = MagicMock(page_content="first chunk")
    chunk_b = MagicMock(page_content="second chunk")
    mock_splitter.split_documents.return_value = [chunk_a, chunk_b]

    with patch.object(orchestrator, "discover_documents", return_value=[Path("doc.txt")]), \
         patch.object(orchestrator, "_load_source", return_value=[MagicMock(page_content="raw doc")]), \
         patch("ms_rag.ingestion.ingestion_orchestrator.ChunkingEngine") as mock_chunking:
        mock_chunking.return_value.get_splitter.return_value = mock_splitter
        result = orchestrator.ingest(
            sources=["doc.txt"],
            loader_map={"txt": "TextLoader"},
            chunking_config=ChunkingConfig(strategy="recursive_character", chunk_size=100, chunk_overlap=0),
            embedding_model=EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small"),
            vector_db=VectorDBConfig(db_type="faiss", connection_params={}, collection_name="test"),
            vector_store=mock_store,
        )

    assert result.chunk_count == 2
    assert getattr(mock_store, "_ms_rag_keyword_corpus") == ["first chunk", "second chunk"]


def test_ingest_attaches_advanced_retrieval_state_to_vector_store() -> None:
    orchestrator = IngestionOrchestrator()
    mock_store = MagicMock()
    parent_doc = MagicMock(page_content="full resume")
    parent_doc.metadata = {"source": "doc.txt"}
    chunk = MagicMock(page_content="resume chunk")
    chunk.metadata = {"source": "doc.txt", "ms_rag_parent_id": "doc.txt::parent::0"}
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [chunk]

    with patch.object(orchestrator, "discover_documents", return_value=[Path("doc.txt")]), \
         patch.object(orchestrator, "_load_source", return_value=[parent_doc]), \
         patch("ms_rag.ingestion.ingestion_orchestrator.ChunkingEngine") as mock_chunking:
        mock_chunking.return_value.get_splitter.return_value = mock_splitter
        orchestrator.ingest(
            sources=["doc.txt"],
            loader_map={"txt": "TextLoader"},
            chunking_config=ChunkingConfig(strategy="recursive_character", chunk_size=100, chunk_overlap=0),
            embedding_model=EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small"),
            vector_db=VectorDBConfig(db_type="faiss", connection_params={}, collection_name="test"),
            vector_store=mock_store,
        )

    parent_documents = getattr(mock_store, "_ms_rag_parent_documents")
    chunk_documents = getattr(mock_store, "_ms_rag_chunk_documents")
    assert parent_documents
    assert chunk_documents == [chunk]
    assert chunk.metadata["ms_rag_child_id"]
    assert chunk.metadata["ms_rag_multi_vector_source_id"]
    assert chunk.metadata["ms_rag_ingested_at"]


# ---------------------------------------------------------------------------
# retry_with_backoff tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    def test_succeeds_on_first_attempt(self) -> None:
        call_count = {"n": 0}

        def fn() -> str:
            call_count["n"] += 1
            return "success"

        result = retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001))
        assert result == "success"
        assert call_count["n"] == 1

    def test_retries_on_failure_and_eventually_succeeds(self) -> None:
        call_count = {"n": 0}

        def fn() -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        result = retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001))
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_after_max_attempts(self) -> None:
        def fn() -> None:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(fn, max_attempts=3, delays=(0.001, 0.001))

    def test_on_retry_callback_called(self) -> None:
        retries: list[int] = []

        def fn() -> None:
            raise RuntimeError("fail")

        def on_retry(attempt: int, exc: Exception) -> None:
            retries.append(attempt)

        with pytest.raises(RuntimeError):
            retry_with_backoff(
                fn, max_attempts=3, delays=(0.001, 0.001), on_retry=on_retry
            )

        assert retries == [1, 2]  # called before attempt 2 and 3

    def test_single_attempt_raises_immediately(self) -> None:
        def fn() -> None:
            raise IOError("io error")

        with pytest.raises(IOError):
            retry_with_backoff(fn, max_attempts=1, delays=(0.001,))
