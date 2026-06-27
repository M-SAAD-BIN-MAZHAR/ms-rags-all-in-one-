"""Property-based and unit tests for SessionManager.

Properties covered:
    Property 21 (extended): Pipeline Config Serialization Round-Trip via file (Req 18.1, 18.4)

Unit tests: Req 18.2, 18.3
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.models import (
    ChunkingConfig,
    EmbeddingModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    VectorDBConfig,
)
from ms_rag.session.session_manager import SessionManager
from ms_rag.utils.exceptions import SessionLoadError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_config() -> PipelineConfig:
    return PipelineConfig(
        configured_providers=["openai"],
        rag_type=RAGTypeConfig(
            rag_type="naive_rag", display_name="Naive RAG",
            description="test", requires_langgraph=False,
        ),
        document_types=["pdf"],
        loader_map={"pdf": "PyPDFLoader"},
        chunking=ChunkingConfig(strategy="recursive_character", chunk_size=500, chunk_overlap=50),
        embedding_model=EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small"),
        vector_db=VectorDBConfig(db_type="chroma", connection_params={}, collection_name="test"),
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        system_prompt="Be helpful.",
    )


# ---------------------------------------------------------------------------
# Property 21 (file round-trip extension): save → load must preserve all data
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_basic() -> None:
    """Feature: ms-rag, Property 21 (file): save/load round-trip preserves config."""
    manager = SessionManager()
    config = _make_minimal_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.json"
        manager.save(config, path)

        # File must exist and contain valid JSON
        assert path.exists()
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in raw  # Req 18.4
        assert raw["schema_version"] == "1.0"

        # Load and verify round-trip
        restored = manager.load(path)
        assert restored.schema_version == config.schema_version
        assert restored.configured_providers == config.configured_providers
        assert restored.document_types == config.document_types
        assert restored.loader_map == config.loader_map
        assert restored.system_prompt == config.system_prompt

        if config.rag_type:
            assert restored.rag_type is not None
            assert restored.rag_type.rag_type == config.rag_type.rag_type

        if config.chunking:
            assert restored.chunking is not None
            assert restored.chunking.strategy == config.chunking.strategy
            assert restored.chunking.chunk_size == config.chunking.chunk_size


@given(
    providers=st.lists(
        st.sampled_from(["openai", "anthropic", "cohere", "groq"]),
        min_size=0, max_size=3, unique=True,
    ),
    system_prompt=st.text(max_size=200),
)
@settings(max_examples=30)
def test_save_load_preserves_providers_and_prompt(
    providers: list[str], system_prompt: str
) -> None:
    """Arbitrary provider lists and system prompts survive file round-trip."""
    manager = SessionManager()
    config = PipelineConfig(
        configured_providers=providers,
        system_prompt=system_prompt,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.json"
        manager.save(config, path)
        restored = manager.load(path)

    assert restored.configured_providers == providers
    assert restored.system_prompt == system_prompt
    assert restored.schema_version == "1.0"


# ---------------------------------------------------------------------------
# Unit tests — error handling (Req 18.2, 18.3)
# ---------------------------------------------------------------------------


class TestSessionManagerErrors:
    def test_load_missing_file_raises_session_load_error(self) -> None:
        """Req 18.3: missing file raises SessionLoadError with descriptive message."""
        manager = SessionManager()
        with pytest.raises(SessionLoadError) as exc_info:
            manager.load(Path("/nonexistent/path/session.json"))

        error = exc_info.value
        assert "not found" in str(error).lower() or "session config" in str(error).lower()
        assert error.file_path != ""

    def test_load_malformed_json_raises_session_load_error(self) -> None:
        """Req 18.3: malformed JSON raises SessionLoadError."""
        manager = SessionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("{not valid json}", encoding="utf-8")

            with pytest.raises(SessionLoadError) as exc_info:
                manager.load(path)

            assert exc_info.value.file_path != ""

    def test_load_empty_file_raises_session_load_error(self) -> None:
        """Empty file must raise SessionLoadError."""
        manager = SessionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.write_text("", encoding="utf-8")

            with pytest.raises(SessionLoadError):
                manager.load(path)

    def test_save_creates_parent_directories(self) -> None:
        """save() must create parent directories that don't exist."""
        manager = SessionManager()
        config = PipelineConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "c" / "session.json"
            assert not path.parent.exists()
            manager.save(config, path)
            assert path.exists()

    def test_load_returns_pipeline_config_instance(self) -> None:
        manager = SessionManager()
        config = PipelineConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            manager.save(config, path)
            restored = manager.load(path)

        assert isinstance(restored, PipelineConfig)

    def test_schema_version_always_present_in_saved_file(self) -> None:
        """Req 18.4: schema_version must be in the saved JSON."""
        manager = SessionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            manager.save(PipelineConfig(), path)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "schema_version" in data
            assert data["schema_version"] == "1.0"

    def test_save_does_not_persist_vector_db_secrets(self) -> None:
        manager = SessionManager()
        config = PipelineConfig(
            vector_db=VectorDBConfig(
                db_type="pinecone",
                connection_params={
                    "PINECONE_API_KEY": "pc-secret-value",
                    "PINECONE_INDEX_NAME": "ms-rag-index",
                },
                collection_name="ms-rag-index",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            manager.save(config, path)
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)

        assert "pc-secret-value" not in raw
        assert data["vector_db"]["connection_params"]["PINECONE_API_KEY"] == "PINECONE_API_KEY"
        assert data["vector_db"]["connection_params"]["PINECONE_INDEX_NAME"] == "ms-rag-index"
