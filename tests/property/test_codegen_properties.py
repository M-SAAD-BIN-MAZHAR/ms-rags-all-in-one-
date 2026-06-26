"""Property-based tests for CodeGenerator.

Properties covered:
    Property 22: Generated Code Structural Completeness (Req 17.1, 17.4)
    Property 23: LangGraph Usage for Agentic RAG Types (Req 17.3)
    Property 24: LangChain Usage in All Generated Code (Req 17.2)

Validates: Requirements 17.1-17.8
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.codegen.code_generator import CodeGenerator
from ms_rag.models import (
    ChunkingConfig,
    EmbeddingModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    VectorDBConfig,
)
from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    rag_type_id: str = "naive_rag",
    providers: list[str] | None = None,
) -> PipelineConfig:
    requires_lg = rag_type_id in LANGGRAPH_TYPES
    config = PipelineConfig(
        configured_providers=providers or ["openai"],
        rag_type=RAGTypeConfig(
            rag_type=rag_type_id,
            display_name=rag_type_id.replace("_", " ").title(),
            description="test",
            requires_langgraph=requires_lg,
        ),
        document_types=["pdf"],
        loader_map={"pdf": "PyPDFLoader"},
        chunking=ChunkingConfig(strategy="recursive_character", chunk_size=1000, chunk_overlap=200),
        embedding_model=EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small"),
        vector_db=VectorDBConfig(db_type="chroma", connection_params={}, collection_name="test"),
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
        system_prompt="You are helpful.",
    )
    return config


# ---------------------------------------------------------------------------
# Property 22: Generated Code Structural Completeness
# ---------------------------------------------------------------------------


@given(rag_type=st.sampled_from([
    "naive_rag", "advanced_rag", "modular_rag",
    "self_rag", "corrective_rag", "agentic_rag", "adaptive_rag",
]))
@settings(max_examples=7)
def test_generated_code_structural_completeness(rag_type: str) -> None:
    """Feature: ms-rag, Property 22: Generated Code Structural Completeness.

    For any valid PipelineConfig, generated code must:
    (a) be syntactically valid Python
    (b) contain 'import os'
    (c) contain 'os.getenv'
    (d) contain 'def main():'
    (e) contain a requirements.txt comment block
    """
    config = _make_config(rag_type_id=rag_type)
    generator = CodeGenerator()
    result = generator.generate(config)

    code = result.python_code
    assert isinstance(code, str), "Generated code must be a string"
    assert len(code.strip()) > 0, "Generated code must not be empty"

    # (b) import os
    assert "import os" in code, "Generated code must contain 'import os'"

    # (c) os.getenv
    assert "os.getenv" in code, "Generated code must contain os.getenv for credentials"

    # (d) def main():
    assert "def main():" in code, "Generated code must contain a main() function"

    # (e) requirements.txt block
    assert "requirements.txt" in code, "Generated code must embed requirements.txt block"

    # (a) syntactically valid Python — parse with ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"Generated code for {rag_type!r} has syntax error: {exc}")


# ---------------------------------------------------------------------------
# Property 23: LangGraph Usage for Agentic RAG Types
# ---------------------------------------------------------------------------


@given(rag_type=st.sampled_from(sorted(LANGGRAPH_TYPES)))
@settings(max_examples=4)
def test_langgraph_usage_for_agentic_rag_types(rag_type: str) -> None:
    """Feature: ms-rag, Property 23: LangGraph Usage for Agentic RAG Types.

    For any agentic RAG type (requires_langgraph=True), generated code must
    contain LangGraph imports and StateGraph instantiation.
    """
    config = _make_config(rag_type_id=rag_type)
    generator = CodeGenerator()
    result = generator.generate(config)
    code = result.python_code

    assert "langgraph" in code.lower() or "StateGraph" in code, (
        f"Agentic RAG type {rag_type!r} must use LangGraph. "
        f"Code does not contain langgraph references."
    )
    assert "StateGraph" in code, (
        f"Agentic RAG type {rag_type!r} must instantiate StateGraph."
    )


# ---------------------------------------------------------------------------
# Property 24: LangChain Usage in All Generated Code
# ---------------------------------------------------------------------------


@given(rag_type=st.sampled_from([
    "naive_rag", "advanced_rag", "self_rag", "corrective_rag",
    "hyde_rag", "multi_query_rag", "contextual_compression_rag",
]))
@settings(max_examples=7)
def test_langchain_usage_in_all_generated_code(rag_type: str) -> None:
    """Feature: ms-rag, Property 24: LangChain Usage in All Generated Code.

    Regardless of RAG type, generated code must contain at least one
    langchain import statement.
    """
    config = _make_config(rag_type_id=rag_type)
    generator = CodeGenerator()
    result = generator.generate(config)
    code = result.python_code

    has_langchain_import = (
        "from langchain" in code
        or "import langchain" in code
        or "langchain_core" in code
        or "langchain_openai" in code
        or "langchain_community" in code
    )
    assert has_langchain_import, (
        f"Generated code for {rag_type!r} must contain at least one langchain import."
    )


# ---------------------------------------------------------------------------
# Save functionality (Req 17.7, 17.8)
# ---------------------------------------------------------------------------


def test_save_creates_directory_and_files() -> None:
    """Req 17.8: save() must create the output directory if it doesn't exist."""
    config = _make_config()
    generator = CodeGenerator()
    result = generator.generate(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "new_subdir" / "pipeline_output"
        # Directory does not exist yet
        assert not output_dir.exists()

        generator.save(result, output_dir)

        assert output_dir.exists()
        assert (output_dir / "pipeline.py").exists()
        assert (output_dir / "requirements.txt").exists()


def test_saved_pipeline_py_contains_main() -> None:
    config = _make_config()
    generator = CodeGenerator()
    result = generator.generate(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator.save(result, output_dir)
        content = (output_dir / "pipeline.py").read_text(encoding="utf-8")
        assert "def main():" in content


def test_saved_requirements_txt_non_empty() -> None:
    config = _make_config()
    generator = CodeGenerator()
    result = generator.generate(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator.save(result, output_dir)
        content = (output_dir / "requirements.txt").read_text(encoding="utf-8")
        assert len(content.strip()) > 0
        assert "langchain" in content


def test_generated_code_rag_type_field() -> None:
    """GeneratedCode.rag_type must match the configured RAG type."""
    config = _make_config(rag_type_id="naive_rag")
    generator = CodeGenerator()
    result = generator.generate(config)
    assert result.rag_type == "naive_rag"


def test_requirements_txt_deduplicated() -> None:
    """Requirements must be deduplicated."""
    config = _make_config(providers=["openai", "openai"])  # duplicate provider
    generator = CodeGenerator()
    result = generator.generate(config)
    lines = [l for l in result.requirements_txt.splitlines() if l.strip()]
    assert len(lines) == len(set(lines)), "Requirements must be deduplicated"


def test_generated_faiss_pipeline_loads_or_saves_persisted_index() -> None:
    """Generated FAISS pipelines must work in ingest and query-only modes."""
    config = _make_config()
    config.vector_db = VectorDBConfig(
        db_type="faiss",
        connection_params={},
        collection_name="faiss_test",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "FAISS_INDEX_PATH" in code
    assert "FAISS.from_documents(chunks, embeddings)" in code
    assert "vector_store.save_local(str(index_path))" in code
    assert "FAISS.load_local(" in code
    assert "allow_dangerous_deserialization=True" in code


def test_generated_qdrant_query_mode_connects_to_existing_collection() -> None:
    """Generated Qdrant pipelines must not require chunks for query-only mode."""
    config = _make_config()
    config.vector_db = VectorDBConfig(
        db_type="qdrant",
        connection_params={},
        collection_name="qdrant_test",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "from qdrant_client import QdrantClient" in code
    assert "if chunks:" in code
    assert "QdrantVectorStore.from_documents(" in code
    assert "client = QdrantClient(url=url, api_key=api_key)" in code
    assert 'collection_name="qdrant_test"' in code


def test_generated_huggingface_endpoint_embedding_uses_token_only_hosted_class() -> None:
    """Hosted HuggingFace embeddings must not generate local sentence-transformers wiring."""
    config = _make_config(providers=["huggingface"])
    config.embedding_model = EmbeddingModelConfig(
        provider="huggingface_endpoint",
        model_id="hf-endpoint:sentence-transformers/all-MiniLM-L6-v2",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "from langchain_huggingface import HuggingFaceEndpointEmbeddings" in code
    assert 'model="sentence-transformers/all-MiniLM-L6-v2"' in code
    assert 'huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")' in code
    assert "sentence-transformers>=3.0.0" not in result.requirements_txt


def test_generated_ollama_pipeline_supports_cloud_headers() -> None:
    """Generated Ollama pipelines must support bearer-token cloud auth and local fallback."""
    config = _make_config(providers=["ollama"])
    config.embedding_model = EmbeddingModelConfig(
        provider="ollama",
        model_id="gpt-oss:120b",
        local_path="gpt-oss:120b",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert 'OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")' in code
    assert 'return OLLAMA_BASE_URL or ("https://ollama.com" if OLLAMA_API_KEY else "http://localhost:11434")' in code
    assert 'return {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}} if OLLAMA_API_KEY else {}' in code
    assert 'client_kwargs=_ollama_client_kwargs()' in code


@pytest.mark.parametrize(
    ("db_type", "expected_snippets"),
    [
        ("chroma", ["from langchain_chroma import Chroma", "vector_store.add_documents(chunks)"]),
        ("pinecone", ["from langchain_pinecone import PineconeVectorStore", "vector_store.add_documents(chunks)"]),
        ("weaviate", ["from langchain_weaviate import WeaviateVectorStore", "vector_store.add_documents(chunks)"]),
        ("pgvector", ["from langchain_postgres import PGVector", "vector_store.add_documents(chunks)"]),
        ("milvus", ["from langchain_milvus import Milvus", "vector_store.add_documents(chunks)"]),
        ("elasticsearch", ["from langchain_elasticsearch import ElasticsearchStore", "vector_store.add_documents(chunks)"]),
        ("redis", ["from langchain_redis import RedisConfig, RedisVectorStore", "vector_store.add_documents(chunks)"]),
        ("opensearch", ["from langchain_community.vectorstores import OpenSearchVectorSearch", "vector_store.add_documents(chunks)"]),
        ("azure_ai_search", ["from langchain_community.vectorstores import AzureSearch", "vector_store.add_documents(chunks)"]),
        ("mongodb_atlas", ["from langchain_mongodb import MongoDBAtlasVectorSearch", "vector_store.add_documents(chunks)"]),
    ],
)
def test_generated_vector_db_pipelines_import_and_ingest_supported_backends(
    db_type: str,
    expected_snippets: list[str],
) -> None:
    """Generated pipelines must import and ingest into the selected vector DB."""
    config = _make_config()
    config.vector_db = VectorDBConfig(
        db_type=db_type,
        connection_params={},
        collection_name=f"{db_type}_test",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    for snippet in expected_snippets:
        assert snippet in code
