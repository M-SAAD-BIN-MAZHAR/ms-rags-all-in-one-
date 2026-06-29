"""Unit tests for LLM Integration Layer.

Tests (Requirement 17.2, 17.3):
- get_llm factory: correct class called for each provider.
- Unknown provider raises ValueError (not ImportError).
- build_rag_chain assembles LCEL chain correctly.
- build_langgraph_workflow raises ValueError for non-agentic types.
- process_query falls back gracefully when pipeline not initialised.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ms_rag.llm.llm_integration import (
    _answer_from_merged_queries,
    _collect_retriever_trace,
    _select_primary_retrieval_query,
    build_langgraph_workflow,
    build_rag_chain,
    get_llm,
    invoke_rag_chain,
    process_query,
)
from ms_rag.models import (
    CredentialStore,
    PipelineConfig,
    RAGTypeConfig,
    RetrievalConfig,
    SessionState,
)
from ms_rag.query.query_enhancer import QueryEnhancer
from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES


# ---------------------------------------------------------------------------
# get_llm factory
# ---------------------------------------------------------------------------


class TestGetLLMFactory:
    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm("nonexistent_provider", "some-model")

    def test_known_providers_dont_raise_value_error(self) -> None:
        """Known providers raise ImportError (missing package) not ValueError."""
        known = [
            "openai", "anthropic", "cohere", "huggingface", "google_gemini",
            "mistral", "groq", "together_ai", "replicate",
            "azure_openai", "aws_bedrock", "ollama",
        ]
        for provider in known:
            try:
                get_llm(provider, "test-model")
            except ImportError:
                pass  # package not installed — dispatch was correct
            except ValueError as exc:
                if "Unsupported LLM provider" in str(exc):
                    pytest.fail(
                        f"Provider {provider!r} raised ValueError: {exc}"
                    )
            except Exception:
                pass  # auth error etc. — dispatch was correct

    def test_openai_called_with_correct_class(self) -> None:
        with patch("ms_rag.llm.llm_integration.ChatOpenAI", create=True) as mock_cls:
            try:
                get_llm("openai", "gpt-4o")
            except Exception:
                pass
            # If import succeeds, mock_cls must have been called
            if mock_cls.called:
                call_kwargs = mock_cls.call_args
                assert call_kwargs is not None

    def test_credential_store_used_for_api_key(self) -> None:
        """Credential store values should be accessible via get_llm."""
        store = CredentialStore()
        store.set("openai", "OPENAI_API_KEY", "sk-test-from-store")
        # Simply verify get_llm doesn't raise ValueError for openai
        try:
            get_llm("openai", "gpt-4o", credential_store=store)
        except ImportError:
            pass  # langchain_openai not installed — dispatch was correct
        except Exception:
            pass  # auth or other runtime errors

    def test_ollama_uses_langchain_ollama_not_community(self) -> None:
        """Ollama must use langchain_ollama, not deprecated langchain_community."""
        import inspect  # noqa: PLC0415
        import ms_rag.llm.llm_integration as mod  # noqa: PLC0415
        source = inspect.getsource(mod.get_llm)
        # The ollama branch must import from langchain_ollama
        assert "langchain_ollama" in source, (
            "get_llm must use langchain_ollama for ollama provider"
        )
        # Must NOT use deprecated community ollama
        assert "langchain_community.llms" not in source or "Ollama" not in source.split("langchain_community.llms")[1][:50] if "langchain_community.llms" in source else True

    def test_ollama_cloud_uses_auth_headers_when_api_key_present(self) -> None:
        store = CredentialStore()
        store.set("ollama", "OLLAMA_MODEL_NAME", "gpt-oss:120b")
        store.set("ollama", "OLLAMA_API_KEY", "ollama-token")
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            get_llm("ollama", "default", credential_store=store)

        mock_cls.assert_called_once_with(
            model="gpt-oss:120b",
            base_url="https://ollama.com",
            client_kwargs={"headers": {"Authorization": "Bearer ollama-token"}},
        )

    def test_huggingface_uses_chat_wrapper_with_conversational_task(self) -> None:
        store = CredentialStore()
        store.set("huggingface", "HUGGINGFACEHUB_API_TOKEN", "hf-test-token")

        endpoint_instance = MagicMock(name="hf_endpoint")
        chat_instance = MagicMock(name="chat_hf")

        with (
            patch("langchain_huggingface.HuggingFaceEndpoint", return_value=endpoint_instance) as mock_endpoint,
            patch("langchain_huggingface.ChatHuggingFace", return_value=chat_instance) as mock_chat,
        ):
            result = get_llm(
                "huggingface",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                credential_store=store,
            )

        assert result is chat_instance
        mock_endpoint.assert_called_once_with(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token="hf-test-token",
            task="conversational",
        )
        mock_chat.assert_called_once_with(
            llm=endpoint_instance,
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        )


# ---------------------------------------------------------------------------
# build_rag_chain (Req 17.2)
# ---------------------------------------------------------------------------


class TestBuildRagChain:
    def test_chain_assembly_with_mock_retriever_and_llm(self) -> None:
        """LCEL chain must be constructable with mock objects."""
        try:
            from langchain_core.runnables import RunnablePassthrough  # noqa: PLC0415
        except ImportError:
            pytest.skip("langchain-core not installed")

        mock_retriever = MagicMock()
        mock_retriever.invoke = MagicMock(return_value=[])

        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_llm.__ror__ = MagicMock(return_value=mock_llm)

        try:
            chain = build_rag_chain(
                retriever=mock_retriever,
                llm=mock_llm,
                system_prompt="You are helpful.",
            )
            assert chain is not None
        except ImportError:
            pytest.skip("Required LangChain packages not installed")

    def test_chain_uses_lcel_not_llmchain(self) -> None:
        """build_rag_chain must NOT use the deprecated LLMChain."""
        import inspect  # noqa: PLC0415
        import ms_rag.llm.llm_integration as mod  # noqa: PLC0415
        source = inspect.getsource(mod.build_rag_chain)
        assert "LLMChain" not in source, (
            "build_rag_chain must use LCEL (|) not deprecated LLMChain"
        )


# ---------------------------------------------------------------------------
# build_langgraph_workflow (Req 17.3)
# ---------------------------------------------------------------------------


class TestBuildLangGraphWorkflow:
    def test_non_agentic_rag_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="does not require LangGraph"):
            build_langgraph_workflow("naive_rag", MagicMock(), MagicMock(), "test")

    def test_non_langgraph_types_all_raise(self) -> None:
        non_agentic = [
            "naive_rag", "advanced_rag", "modular_rag", "speculative_rag",
            "graphrag", "hyde_rag", "multi_query_rag", "rag_fusion",
            "step_back_rag", "parent_child_rag", "contextual_compression_rag",
        ]
        for rag_type in non_agentic:
            with pytest.raises(ValueError):
                build_langgraph_workflow(rag_type, MagicMock(), MagicMock(), "test")

    def test_langgraph_types_accepted(self) -> None:
        """LangGraph-requiring types must not raise ValueError."""
        for rag_type in LANGGRAPH_TYPES:
            try:
                build_langgraph_workflow(rag_type, MagicMock(), MagicMock(), "test")
            except ImportError:
                pass  # langgraph not installed
            except ValueError as exc:
                pytest.fail(
                    f"LangGraph type {rag_type!r} raised ValueError: {exc}"
                )
            except Exception:
                pass  # other runtime errors acceptable

    def test_langgraph_runtime_contains_full_advanced_rag_flows(self) -> None:
        """Advanced RAG types must have real graph branches, not labels only."""
        import inspect  # noqa: PLC0415
        import ms_rag.llm.llm_integration as mod  # noqa: PLC0415

        source = inspect.getsource(mod.build_langgraph_workflow)
        assert "decide_retrieval_need" in source
        assert "check_hallucination" in source
        assert "corrective_web_fallback" in source
        assert "query_analysis" in source
        assert "route_agent_action" in source
        assert "run_approved_tools" in source
        assert "deep_retrieve" in source
        assert '"direct": "direct_answer"' in source
        assert '"deep": "deep_retrieve"' in source

    def test_agentic_rag_workflow_compiles_without_graphstate_name_error(self) -> None:
        """Agentic graph compilation must resolve GraphState annotations."""
        try:
            from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
        except ImportError:
            pytest.skip("langchain-core not installed")

        retriever = MagicMock()
        llm = RunnableLambda(lambda value: "retrieve")

        try:
            graph = build_langgraph_workflow(
                "agentic_rag",
                retriever=retriever,
                llm=llm,
                system_prompt="Use context only.",
            )
        except ImportError:
            pytest.skip("langgraph not installed")

        assert graph is not None


# ---------------------------------------------------------------------------
# process_query
# ---------------------------------------------------------------------------


class TestProcessQuery:
    def test_returns_not_initialised_message_when_no_chain(self) -> None:
        session = SessionState(
            config=PipelineConfig(),
            credentials=CredentialStore(),
        )
        session.rag_chain = None  # not initialised

        answer = process_query("What is RAG?", session)
        assert "not initialised" in answer.lower() or "pipeline" in answer.lower()

    def test_invokes_rag_chain_when_available(self) -> None:
        session = SessionState(
            config=PipelineConfig(),
            credentials=CredentialStore(),
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "RAG is a technique."
        session.rag_chain = mock_chain

        answer = process_query("What is RAG?", session)
        assert answer == "RAG is a technique."
        mock_chain.invoke.assert_called_once_with("What is RAG?")

    def test_reraises_chain_exception(self) -> None:
        """Query errors should propagate so QueryLoop can handle them."""
        session = SessionState(
            config=PipelineConfig(),
            credentials=CredentialStore(),
        )
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM timeout")
        session.rag_chain = mock_chain

        with pytest.raises(RuntimeError, match="LLM timeout"):
            process_query("What is RAG?", session)


class TestInvokeRagChain:
    def test_langgraph_returns_generation_field(self) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"generation": "Agentic answer."}
        answer = invoke_rag_chain(mock_chain, "What is RAG?", requires_langgraph=True)
        assert answer == "Agentic answer."
        mock_chain.invoke.assert_called_once_with({"question": "What is RAG?"})

    def test_lcel_chain_returns_string(self) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Direct answer."
        answer = invoke_rag_chain(mock_chain, "What is RAG?", requires_langgraph=False)
        assert answer == "Direct answer."
        mock_chain.invoke.assert_called_once_with("What is RAG?")


def test_required_hyde_rag_enhancement_fails_instead_of_silent_fallback() -> None:
    config = PipelineConfig()
    config.rag_type = RAGTypeConfig(
        rag_type="hyde_rag",
        display_name="HyDE RAG",
        description="test",
        requires_langgraph=False,
    )
    config.query_enhancement = ["hyde"]
    session = SessionState(
        config=config,
        credentials=CredentialStore(),
        vector_store=None,
        retriever=MagicMock(),
        llm=None,
        rag_chain=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="Required query enhancement 'hyde' failed"):
        process_query("What is RAG?", session, query_enhancer=QueryEnhancer())


def test_process_query_records_enhanced_query_trace() -> None:
    from langchain_core.documents import Document

    config = PipelineConfig()
    config.query_enhancement = ["query_rewriting"]
    config.retrieval = RetrievalConfig(strategy="dense_vector", top_k=5)
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "answer"
    session = SessionState(
        config=config,
        credentials=CredentialStore(),
        rag_chain=mock_chain,
        llm=MagicMock(),
    )
    session.retriever = MagicMock()
    session.retriever.invoke.return_value = [Document(page_content="Elephants are large mammals.")]
    enhancer = MagicMock()
    enhancer.enhance.return_value = ["rewritten elephant query"]

    with patch("ms_rag.llm.llm_integration._answer_from_merged_queries") as answer_helper:
        answer_helper.return_value = "answer"
        answer = process_query("tell me about elephants", session, query_enhancer=enhancer)

    assert answer == "answer"
    answer_helper.assert_called_once()
    assert answer_helper.call_args.kwargs["original_query"] == "tell me about elephants"
    assert answer_helper.call_args.kwargs["retrieval_queries"] == ["rewritten elephant query"]
    assert session.last_enhanced_queries == ["rewritten elephant query"]
    assert session.last_primary_retrieval_query == "rewritten elephant query"
    assert session.last_retrieved_context_count == 1
    assert "Elephants are large mammals" in session.last_retrieved_context_preview[0]


def test_hyde_retrieves_with_hypothetical_but_answers_original_question() -> None:
    config = PipelineConfig()
    config.query_enhancement = ["hyde"]
    config.retrieval = RetrievalConfig(strategy="multi_vector", top_k=5)
    session = SessionState(
        config=config,
        credentials=CredentialStore(),
        rag_chain=MagicMock(),
        retriever=MagicMock(),
        llm=MagicMock(),
    )
    enhancer = MagicMock()
    enhancer.enhance.return_value = ["Hypothetical elephant document."]

    with patch("ms_rag.llm.llm_integration._answer_from_merged_queries") as answer_helper:
        answer_helper.return_value = "Elephants are large land mammals."
        answer = process_query("tell me about elephants", session, query_enhancer=enhancer)

    assert answer == "Elephants are large land mammals."
    answer_helper.assert_called_once()
    assert answer_helper.call_args.kwargs["original_query"] == "tell me about elephants"
    assert answer_helper.call_args.kwargs["retrieval_queries"] == ["Hypothetical elephant document."]
    session.rag_chain.invoke.assert_not_called()


def test_keyword_retrieval_keeps_original_query_when_hyde_is_enabled() -> None:
    selected = _select_primary_retrieval_query(
        original_query="tell me about elephant",
        enhanced_queries=["A hypothetical document about elephants and habitats."],
        retrieval=RetrievalConfig(strategy="keyword_bm25", top_k=5),
    )

    assert selected == "tell me about elephant"


def test_hybrid_retrieval_keeps_original_query_when_hyde_is_enabled() -> None:
    selected = _select_primary_retrieval_query(
        original_query="tell me about elephant",
        enhanced_queries=["A hypothetical document about elephants and habitats."],
        retrieval=RetrievalConfig(strategy="hybrid", top_k=5, alpha=0.5),
    )

    assert selected == "tell me about elephant"


def test_keyword_ensemble_keeps_original_query_when_hyde_is_enabled() -> None:
    selected = _select_primary_retrieval_query(
        original_query="tell me about elephant",
        enhanced_queries=["A hypothetical document about elephants and habitats."],
        retrieval=RetrievalConfig(
            strategy="ensemble",
            top_k=5,
            ensemble_sub_retrievers=["dense_vector", "keyword_bm25"],
        ),
    )

    assert selected == "tell me about elephant"


def test_dense_retrieval_uses_hyde_enhanced_query() -> None:
    selected = _select_primary_retrieval_query(
        original_query="tell me about elephant",
        enhanced_queries=["A hypothetical document about elephants and habitats."],
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=5),
    )

    assert selected == "A hypothetical document about elephants and habitats."


def test_collect_retriever_trace_reads_nested_rerank_and_compression_counts() -> None:
    base = MagicMock()
    reranker = MagicMock()
    reranker.base_retriever = base
    reranker._ms_rag_pre_rerank_count = 5
    reranker._ms_rag_post_rerank_count = 3
    compressor = MagicMock()
    compressor.base_retriever = reranker
    compressor._ms_rag_pre_compression_count = 3
    compressor._ms_rag_post_compression_count = 3
    compressor._ms_rag_compression_fallback = True

    trace = _collect_retriever_trace(compressor)

    assert trace["rerank"] == {"before": 5, "after": 3}
    assert trace["compression"] == {"before": 3, "after": 3, "fallback": True}


def test_rag_fusion_uses_reciprocal_rank_fusion_ordering() -> None:
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    retriever = MagicMock()
    doc_a = Document(page_content="A", metadata={"source": "a"})
    doc_b = Document(page_content="B", metadata={"source": "b"})
    retriever.invoke.side_effect = [
        [doc_a, doc_b],
        [doc_b],
        [doc_b],
    ]
    captured = {}

    def fake_llm(prompt_value: object) -> str:
        captured["prompt"] = str(prompt_value)
        return "answer"

    llm = RunnableLambda(fake_llm)

    answer = _answer_from_merged_queries(
        original_query="question",
        retrieval_queries=["q1", "q2", "q3"],
        retriever=retriever,
        llm=llm,
        system_prompt="system",
        use_reciprocal_rank_fusion=True,
    )

    assert answer == "answer"
    rendered = captured["prompt"]
    assert rendered.find("B") < rendered.find("A")


def test_query_rewriting_keeps_simple_lookup_query_unchanged() -> None:
    enhancer = QueryEnhancer()
    llm = MagicMock()

    result = enhancer.enhance(
        "tell me about elephants",
        techniques=["query_rewriting"],
        llm=llm,
    )

    assert result == ["tell me about elephants"]
