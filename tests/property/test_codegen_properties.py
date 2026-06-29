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
    CompressionConfig,
    EmbeddingModelConfig,
    GraphStoreConfig,
    LLMModelConfig,
    PipelineConfig,
    RAGTypeConfig,
    RerankingConfig,
    RetrievalConfig,
    VectorDBConfig,
    AgentToolConfig,
    EvaluationConfig,
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
    selected_providers = providers or ["openai"]
    config = PipelineConfig(
        configured_providers=selected_providers,
        llm_model=LLMModelConfig(
            provider=selected_providers[0],
            model_id="meta-llama/Meta-Llama-3-8B-Instruct"
            if selected_providers[0] == "huggingface"
            else "gpt-4o",
        ),
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


def test_generated_langgraph_preserves_distinct_advanced_rag_flows() -> None:
    """Generated LangGraph pipelines must preserve advanced RAG behavior."""
    for rag_type in ["self_rag", "corrective_rag", "agentic_rag", "adaptive_rag"]:
        config = _make_config(rag_type_id=rag_type)
        code = CodeGenerator().generate(config).python_code
        assert "StateGraph" in code

        if rag_type == "self_rag":
            assert "self_retrieval_need" in code
            assert "check_support" in code
            assert '"direct": "direct_answer"' in code
        elif rag_type == "corrective_rag":
            assert "grade_documents" in code
            assert "corrective_fallback" in code
            assert "rewrite_query" in code
        elif rag_type == "agentic_rag":
            assert "agent_plan" in code
            assert "route_agent_action" in code
            assert '"rewrite": "rewrite_query"' in code
            assert '"answer": "direct_answer"' in code
        elif rag_type == "adaptive_rag":
            assert "route_question" in code
            assert "deep_retrieve" in code
            assert '"deep": "deep_retrieve"' in code


def test_generated_agentic_tools_are_standalone_and_parseable() -> None:
    """Generated pipelines must include configured Agentic/CRAG tool helpers."""
    for rag_type in ["corrective_rag", "agentic_rag"]:
        config = _make_config(rag_type_id=rag_type)
        config.agent_tools = AgentToolConfig(
            enabled_tools=[
                "web_search",
                "url_fetch",
                "file_read",
                "api_request",
                "memory",
                "document_summarization",
            ],
            tool_settings={
                "web_search": {"provider": "tavily"},
                "url_fetch": {"allowed_domains": ["example.com"]},
                "file_read": {"allowed_paths": ["."]},
                "api_request": {
                    "allowed_base_urls": ["https://api.example.com"],
                    "allowed_methods": ["GET"],
                },
                "memory": {
                    "memory_types": ["short_term", "long_term", "semantic"],
                    "path": "./agent_memory/memory.json",
                },
            },
        )
        code = CodeGenerator().generate(config).python_code
        ast.parse(code)
        assert "AGENT_TOOLS" in code
        assert "def run_agent_tools" in code
        assert "TAVILY_API_KEY" in code
        assert "allowed_domains" in code
        assert "allowed_base_urls" in code
        if rag_type == "corrective_rag":
            assert "corrective_fallback" in code
            assert "run_agent_tools(state[\"question\"], [], llm)" in code


def test_generated_chunking_preserves_selected_strategy() -> None:
    """Generated pipeline.py must not collapse every chunker to recursive splitting."""
    strategies = [
        "recursive_character",
        "fixed_size",
        "semantic",
        "sentence",
        "paragraph",
        "token_based",
        "markdown_aware",
        "html_aware",
        "code_aware",
        "agentic",
        "document_aware",
    ]
    for strategy in strategies:
        config = _make_config()
        config.chunking = ChunkingConfig(
            strategy=strategy,
            chunk_size=512 if strategy != "semantic" else 0,
            chunk_overlap=64 if strategy != "semantic" else 0,
            tokenizer="cl100k_base" if strategy == "token_based" else None,
            language="python" if strategy == "code_aware" else None,
        )
        code = CodeGenerator().generate(config).python_code
        ast.parse(code)
        assert f'strategy = "{strategy}"' in code
        if strategy == "semantic":
            assert "SemanticChunker" in code
        if strategy == "agentic":
            assert "_AgenticGeneratedChunker" in code
            assert "<MS_RAG_CHUNK>" in code
        if strategy == "document_aware":
            assert "_DocumentAwareGeneratedSplitter" in code


def test_generated_evaluation_supports_all_selected_evaluators() -> None:
    config = _make_config()
    config.evaluation_enabled = True
    config.evaluation = EvaluationConfig(
        evaluators=[
            "ragas",
            "deepeval",
            "trulens",
            "langsmith",
            "langfuse",
            "arize_phoenix",
            "ares",
            "ragbench",
            "cicd_gate",
            "langgraph_trace",
            "monitoring_export",
        ],
        cicd_thresholds={"faithfulness": 0.8},
        evaluator_llm_provider="openai",
        evaluator_llm_model="gpt-4o-mini",
    )

    code = CodeGenerator().generate(config).python_code
    ast.parse(code)
    assert "def evaluate_response" in code
    assert "ENABLED_EVALUATORS" in code
    assert "EVALUATOR_LLM_MODEL = 'gpt-4o-mini'" in code
    assert "EvaluationDataset.from_list" in code
    assert "LLMContextPrecisionWithoutReference" in code
    assert "_finite_scores" in code
    assert "deepeval_answer_relevancy" in code
    assert "trulens_package_available" in code
    assert "phoenix_endpoint_configured" in code
    assert "ares_package_available" in code
    assert "ragbench_datasets_package_available" in code
    assert "langgraph_trace_logged" in code
    assert "monitoring_export_logged" in code


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
        assert (output_dir / ".env").exists()


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


def test_saved_env_file_contains_selected_runtime_variables_only() -> None:
    config = _make_config(rag_type_id="agentic_rag", providers=["groq"])
    config.embedding_model = EmbeddingModelConfig(
        provider="huggingface_endpoint",
        model_id="sentence-transformers/all-mpnet-base-v2",
    )
    config.loader_map = {"pdf": "LlamaParseLoader"}
    config.vector_db = VectorDBConfig(
        db_type="pinecone",
        connection_params={"PINECONE_API_KEY": "PINECONE_API_KEY", "PINECONE_INDEX_NAME": "saadi"},
        collection_name="saadi",
    )
    config.evaluation_enabled = True
    config.evaluation = EvaluationConfig(evaluators=["langsmith"])
    config.agent_tools = AgentToolConfig(
        enabled_tools=["web_search", "memory"],
        tool_settings={"web_search": {"provider": "brave"}, "memory": {"path": "./agent_memory/memory.json"}},
    )

    result = CodeGenerator().generate(config)

    assert "GROQ_API_KEY=" in result.env_txt
    assert "HUGGINGFACEHUB_API_TOKEN=" in result.env_txt
    assert "LLAMA_CLOUD_API_KEY=" in result.env_txt
    assert "PINECONE_API_KEY=" in result.env_txt
    assert "PINECONE_INDEX_NAME=" in result.env_txt
    assert "LANGCHAIN_API_KEY=" in result.env_txt
    assert "BRAVE_SEARCH_API_KEY=" in result.env_txt
    assert "MS_RAG_AGENT_MEMORY_PATH=" in result.env_txt
    assert "OPENAI_API_KEY=" not in result.env_txt
    assert "sk-" not in result.env_txt


def test_generated_code_and_env_preserve_selected_provider_and_database() -> None:
    config = _make_config(providers=["huggingface"])
    config.llm_model = LLMModelConfig(provider="huggingface", model_id="meta-llama/Meta-Llama-3-8B-Instruct")
    config.embedding_model = EmbeddingModelConfig(
        provider="huggingface_endpoint",
        model_id="sentence-transformers/all-mpnet-base-v2",
    )
    config.vector_db = VectorDBConfig(
        db_type="qdrant",
        connection_params={"QDRANT_URL": "QDRANT_URL", "QDRANT_API_KEY": "QDRANT_API_KEY"},
        collection_name="docs_qdrant",
        dimension=768,
    )

    result = CodeGenerator().generate(config)

    assert "HuggingFaceEndpoint" in result.python_code
    assert "HuggingFaceEndpointEmbeddings" in result.python_code
    assert "QdrantVectorStore" in result.python_code
    assert "docs_qdrant" in result.python_code
    assert "HUGGINGFACEHUB_API_TOKEN=" in result.env_txt
    assert "QDRANT_URL=" in result.env_txt
    assert "QDRANT_API_KEY=" in result.env_txt


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


def test_generated_local_huggingface_embedding_disables_xet_download_path() -> None:
    """Local HuggingFace generated pipelines should avoid hf-xet download incompatibilities."""
    config = _make_config(providers=["mistral"])
    config.llm_model = LLMModelConfig(provider="mistral", model_id="mistral-large-latest")
    config.embedding_model = EmbeddingModelConfig(
        provider="huggingface",
        model_id="sentence-transformers/all-mpnet-base-v2",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "def _local_huggingface_embeddings(model_name: str):" in code
    assert 'os.environ["HF_HUB_DISABLE_XET"] = "1"' in code
    assert 'os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)' in code
    assert 'HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")' in code
    assert 'embeddings = _local_huggingface_embeddings("sentence-transformers/all-mpnet-base-v2")' in code


def test_generated_speculative_rag_contains_draft_then_verify_flow() -> None:
    config = _make_config()
    config.rag_type = RAGTypeConfig(
        rag_type="speculative_rag",
        display_name="Speculative RAG",
        description="test",
        requires_langgraph=False,
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "def speculative(query: str) -> str:" in code
    assert "Draft answer:" in code
    assert "Verify the draft against the evidence" in code
    assert "RunnableLambda(speculative)" in code


def test_generated_graphrag_contains_full_graph_index_runtime() -> None:
    config = _make_config()
    config.rag_type = RAGTypeConfig(
        rag_type="graphrag",
        display_name="GraphRAG",
        description="test",
        requires_langgraph=False,
    )
    config.graph_store = GraphStoreConfig(
        store_type="local_json",
        connection_params={"GRAPH_STORE_PATH": "./graph_indexes/test_graph.json"},
        graph_name="test_graph",
        query_mode="hybrid",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "def build_graph_index(chunks: list, llm=None) -> dict:" in code
    assert "def persist_graph_index(graph: dict) -> None:" in code
    assert "def load_graph_index() -> dict:" in code
    assert "def retrieve_graph_context(query: str, llm=None) -> str:" in code
    assert "def graph_guided(query: str) -> str:" in code
    assert "Extract key entities, topics, and relationships" in code
    assert "graph_context = retrieve_graph_context(query, llm=llm)" in code
    assert "Building GraphRAG knowledge graph..." in code
    assert "persist_graph_index(graph)" in code
    assert "RunnableLambda(graph_guided)" in code


def test_generated_huggingface_llm_does_not_fall_back_to_openai() -> None:
    """A HuggingFace-only run must generate a HuggingFace answer model, not OpenAI."""
    config = _make_config(providers=["huggingface"])
    config.llm_model = LLMModelConfig(
        provider="huggingface",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    config.embedding_model = EmbeddingModelConfig(
        provider="huggingface_endpoint",
        model_id="hf-endpoint:sentence-transformers/all-MiniLM-L6-v2",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint" in code
    assert "ChatHuggingFace(llm=HuggingFaceEndpoint(" in code
    assert "repo_id='meta-llama/Meta-Llama-3-8B-Instruct'" in code
    assert "task='conversational'" in code
    assert "HUGGINGFACEHUB_API_TOKEN" in code
    assert "ChatOpenAI(model='gpt-4o'" not in code
    assert "OPENAI_API_KEY" not in code


def test_codegen_requires_generation_llm_when_no_provider_selected() -> None:
    """Generated code should fail loudly instead of silently defaulting to OpenAI."""
    config = _make_config()
    config.configured_providers = []
    config.llm_model = None

    with pytest.raises(ValueError, match="generation LLM"):
        CodeGenerator().generate(config)


@pytest.mark.parametrize(
    ("provider", "model_id", "expected_snippets"),
    [
        (
            "together_ai",
            "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            [
                "from langchain_openai import ChatOpenAI",
                "base_url='https://api.together.xyz/v1'",
                "openai_api_key=TOGETHER_API_KEY",
            ],
        ),
        (
            "replicate",
            "meta/meta-llama-3-8b-instruct",
            [
                "from langchain_community.llms import Replicate",
                "replicate_api_token=REPLICATE_API_TOKEN",
            ],
        ),
        (
            "aws_bedrock",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            [
                "from langchain_aws import ChatBedrock",
                "aws_access_key_id=AWS_ACCESS_KEY_ID",
                "aws_secret_access_key=AWS_SECRET_ACCESS_KEY",
            ],
        ),
    ],
)
def test_generated_llm_providers_match_runtime_support(
    provider: str,
    model_id: str,
    expected_snippets: list[str],
) -> None:
    config = _make_config(providers=[provider])
    config.llm_model = LLMModelConfig(provider=provider, model_id=model_id)

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    for snippet in expected_snippets:
        assert snippet in code


def test_generated_ollama_pipeline_supports_cloud_headers() -> None:
    """Generated Ollama pipelines must support cloud chat and local-only embeddings."""
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
    assert "def _normalize_ollama_base_url(base_url: str) -> str:" in code
    assert 'if normalized.endswith("/v1"):' in code
    assert "Ollama Cloud currently supports chat models only." in code
    assert 'base_url=_ollama_base_url(usage="embedding")' in code
    assert "base_url=_ollama_base_url(usage='chat')" in code
    assert 'client_kwargs=_ollama_client_kwargs(usage="embedding")' in code


def test_generated_hybrid_retriever_can_read_faiss_docstore_texts() -> None:
    config = _make_config()
    config.retrieval = RetrievalConfig(strategy="hybrid", top_k=5, alpha=0.5)

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert 'docstore = getattr(vector_store, "docstore", None)' in code
    assert 'raw_docs = getattr(docstore, "_dict", None)' in code
    assert 'elif hasattr(doc, "page_content")' in code


def test_generated_vector_store_ingest_caches_keyword_corpus_texts() -> None:
    config = _make_config()

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert '"_ms_rag_keyword_corpus"' in code
    assert '[chunk.page_content for chunk in chunks if getattr(chunk, "page_content", "").strip()]' in code


def test_generated_tfidf_retriever_fails_loudly_when_corpus_is_missing() -> None:
    config = _make_config()
    config.retrieval = RetrievalConfig(strategy="tfidf", top_k=7)

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "from langchain_community.retrievers import TFIDFRetriever" in code
    assert "_require_keyword_texts(_extract_corpus_texts(vector_store), \"TF-IDF\")" in code
    assert "or choose dense_vector retrieval" in code


def test_generated_ensemble_retriever_supports_keyword_and_dense_sub_retrievers() -> None:
    config = _make_config()
    config.retrieval = RetrievalConfig(
        strategy="ensemble",
        top_k=5,
        ensemble_sub_retrievers=["dense_vector", "keyword_bm25", "tfidf", "mmr"],
        ensemble_weights=[0.25, 0.25, 0.25, 0.25],
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "def _build_single_retriever(vector_store, strategy_id, *, top_k, alpha, lam):" in code
    assert 'if strategy_id == "keyword_bm25":' in code
    assert 'if strategy_id == "tfidf":' in code
    assert 'if strategy_id == "mmr":' in code
    assert "return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)" in code


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("parent_child", "_parent_child_retriever(vector_store, top_k=5)"),
        ("multi_vector", "_multi_vector_retriever(vector_store, top_k=5)"),
        ("time_weighted", "_time_weighted_retriever(vector_store, top_k=5)"),
    ],
)
def test_generated_advanced_retrievers_use_specialized_runtime(strategy: str, expected: str) -> None:
    config = _make_config()
    config.retrieval = RetrievalConfig(strategy=strategy, top_k=5)

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert expected in code
    assert "ADVANCED_PARENT_DOCUMENTS" in code
    assert "ADVANCED_CHUNK_DOCUMENTS" in code


def test_generated_self_query_documents_dense_extension_point() -> None:
    config = _make_config()
    config.retrieval = RetrievalConfig(strategy="self_query", top_k=5)

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "Self-Query needs a live LLM object" in code
    assert 'search_type="similarity"' in code


def test_generated_camelot_loader_uses_direct_camelot_package() -> None:
    config = _make_config()
    config.loader_map = {"pdf": "CamelotLoader"}

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert "import camelot" in code
    assert "camelot.read_pdf(source, pages=\"all\")" in code
    assert "CamelotPDFLoader" not in code


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


def test_generated_chroma_pipeline_uses_configured_persist_directory() -> None:
    config = _make_config()
    config.vector_db = VectorDBConfig(
        db_type="chroma",
        connection_params={"CHROMA_PERSIST_DIRECTORY": "./custom_chroma/saadi"},
        collection_name="saadi",
    )

    result = CodeGenerator().generate(config)
    code = result.python_code

    ast.parse(code)
    assert 'os.getenv("CHROMA_PERSIST_DIRECTORY")' in code
    assert 'os.getenv("CHROMA_PERSIST_DIR")' in code
    assert "'./custom_chroma/saadi'" in code


def test_generated_code_uses_updated_live_settings_config() -> None:
    config = _make_config()
    config.query_enhancement = ["hyde"]
    config.reranking_enabled = False
    config.reranking = None
    config.compression_enabled = True
    config.compression = CompressionConfig(techniques=["embeddings_filter"], similarity_threshold=0.65)

    result = CodeGenerator().generate(config)

    assert "rerank_documents" not in result.python_code
    assert "compress_context" in result.python_code
    assert "similarity_threshold=0.65" in result.python_code
    assert "retriever = compress_context(" in result.python_code


def test_generated_code_applies_selected_reranking() -> None:
    config = _make_config()
    config.reranking_enabled = True
    config.reranking = RerankingConfig(
        reranker="cohere_reranker",
        model_id="rerank-english-v3.0",
        top_k=3,
    )

    result = CodeGenerator().generate(config)

    ast.parse(result.python_code)
    assert "def rerank_documents" in result.python_code
    assert "reranker = 'cohere_reranker'" in result.python_code
    assert "retriever = apply_reranking(retriever)" in result.python_code
    assert "COHERE_API_KEY" in result.python_code


@pytest.mark.parametrize(
    "technique",
    [
        "query_rewriting",
        "query_expansion",
        "hyde",
        "multi_query",
        "step_back_prompting",
        "sub_question_decomposition",
        "rag_fusion",
    ],
)
def test_generated_code_applies_each_query_enhancement(technique: str) -> None:
    config = _make_config()
    config.query_enhancement = [technique]

    result = CodeGenerator().generate(config)

    ast.parse(result.python_code)
    assert "def enhance_queries" in result.python_code
    assert "Query enhancement trace" in result.python_code
    assert repr([technique]) in result.python_code
    assert "RunnableLambda(prepare_inputs)" in result.python_code
    assert 'setattr(rag_chain, "_ms_rag_retrieve_docs", retrieve_docs)' in result.python_code
    assert 'getattr(rag_chain, "_ms_rag_retrieve_docs", retriever.invoke)' in result.python_code


@pytest.mark.parametrize(
    ("technique", "expected_snippet"),
    [
        ("llm_chain_extraction", "LLMChainExtractor.from_llm"),
        ("embeddings_filter", "EmbeddingsFilter(embeddings=embeddings"),
        ("document_compressor_pipeline", "DocumentCompressorPipeline(transformers=["),
        ("redundancy_removal", "EmbeddingsRedundantFilter(embeddings=embeddings)"),
        ("contextual_compression", "technique == \"contextual_compression\""),
        ("summary_compression", "_LLMSummaryCompressor"),
    ],
)
def test_generated_code_supports_each_compression_technique(
    technique: str,
    expected_snippet: str,
) -> None:
    config = _make_config()
    config.compression_enabled = True
    config.compression = CompressionConfig(techniques=[technique], similarity_threshold=0.65)

    result = CodeGenerator().generate(config)

    ast.parse(result.python_code)
    assert "compress_context" in result.python_code
    assert "retriever = compress_context(" in result.python_code
    assert "_SafeCompressionRetriever" in result.python_code
    assert expected_snippet in result.python_code
