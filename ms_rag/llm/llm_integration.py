"""LLM Integration Layer for MS_RAG.

LLM factory, LCEL RAG chain assembly, and LangGraph agentic workflow
builder for all supported RAG architecture variants.

Requirement 17.2: generated code uses LangChain (LCEL chains)
Requirement 17.3: generated code uses LangGraph for agentic RAG types
"""

from __future__ import annotations

import re
from typing import Any, TypedDict
import warnings

from ms_rag.models import PipelineConfig, RetrievalConfig
from ms_rag.utils.credentials import (
    resolve_credential,
    resolve_model_id,
    resolve_ollama_connection,
    validate_llm_model,
)
from ms_rag.utils.telemetry import TelemetryReporter
from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES


class GraphState(TypedDict, total=False):
    """Shared LangGraph state for agentic RAG variants.

    This is intentionally module-scoped because LangGraph inspects node
    function annotations with global type-hint resolution while compiling.
    """

    question: str
    generation: str
    documents: list
    rewrite_count: int
    hallucination_count: int
    tool_results: list[str]
    action: str
    route: str
    trace: list[str]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(
    provider: str,
    model_id: str,
    credential_store: object | None = None,
    **kwargs: Any,
) -> object:
    """Return the appropriate LangChain LLM/ChatModel instance.

    Package routing (current as of 2025 — no deprecated community classes):
        openai        → langchain_openai.ChatOpenAI
        anthropic     → langchain_anthropic.ChatAnthropic
        cohere        → langchain_cohere.ChatCohere
        huggingface   → langchain_huggingface.ChatHuggingFace + HuggingFaceEndpoint
        google_gemini → langchain_google_genai.ChatGoogleGenerativeAI
        mistral       → langchain_mistralai.ChatMistralAI
        groq          → langchain_groq.ChatGroq
        together_ai   → langchain_together.ChatTogether  (or ChatOpenAI with base_url)
        replicate     → langchain_community.llms.Replicate
        azure_openai  → langchain_openai.AzureChatOpenAI
        aws_bedrock   → langchain_aws.ChatBedrock
        ollama        → langchain_ollama.ChatOllama  (NOT langchain-community)

    Args:
        provider:         Provider ID from PROVIDER_IDS.
        model_id:         Model name/ID to use.
        credential_store: CredentialStore instance for API key lookup.
        **kwargs:         Additional kwargs passed to the LLM constructor.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ImportError: If the required integration package is not installed.
        ValueError:  If the provider is not recognised.
    """

    def _env(field: str, provider_id: str | None = None) -> str | None:
        """Get credential from store or fall back to environment variable."""
        return resolve_credential(field, credential_store, provider_id or provider)

    resolved_model = resolve_model_id(provider, model_id, credential_store)
    validate_llm_model(provider, resolved_model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        openai_kwargs: dict[str, Any] = {"model": resolved_model}
        api_key = _env("OPENAI_API_KEY")
        if api_key:
            openai_kwargs["openai_api_key"] = api_key
        org_id = _env("OPENAI_ORG_ID")
        if org_id:
            openai_kwargs["openai_organization"] = org_id
        openai_kwargs.update(kwargs)
        return ChatOpenAI(**openai_kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        return ChatAnthropic(
            model=resolved_model,
            api_key=_env("ANTHROPIC_API_KEY"),  # type: ignore[arg-type]
            **kwargs,
        )

    if provider == "cohere":
        from langchain_cohere import ChatCohere  # noqa: PLC0415
        return ChatCohere(
            model=resolved_model,
            cohere_api_key=_env("COHERE_API_KEY"),  # type: ignore[arg-type]
            **kwargs,
        )

    if provider == "huggingface":
        from langchain_huggingface import (  # noqa: PLC0415
            ChatHuggingFace,
            HuggingFaceEndpoint,
        )
        hf_task = kwargs.pop("task", "conversational")
        endpoint = HuggingFaceEndpoint(
            repo_id=resolved_model,
            huggingfacehub_api_token=_env("HUGGINGFACEHUB_API_TOKEN"),
            task=hf_task,
            **kwargs,
        )
        return ChatHuggingFace(llm=endpoint, model_id=resolved_model)

    if provider == "google_gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=_env("GOOGLE_API_KEY"),  # type: ignore[arg-type]
            **kwargs,
        )

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI  # noqa: PLC0415
        return ChatMistralAI(
            model=resolved_model,
            api_key=_env("MISTRAL_API_KEY"),  # type: ignore[arg-type]
            **kwargs,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq  # noqa: PLC0415
        return ChatGroq(
            model=resolved_model,
            groq_api_key=_env("GROQ_API_KEY"),  # type: ignore[arg-type]
            **kwargs,
        )

    if provider == "together_ai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        together_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "base_url": "https://api.together.xyz/v1",
        }
        together_key = _env("TOGETHER_API_KEY")
        if together_key:
            together_kwargs["openai_api_key"] = together_key
        together_kwargs.update(kwargs)
        return ChatOpenAI(**together_kwargs)

    if provider == "replicate":
        from langchain_community.llms import Replicate  # noqa: PLC0415
        return Replicate(
            model=resolved_model,
            replicate_api_token=_env("REPLICATE_API_TOKEN"),
            **kwargs,
        )

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI  # noqa: PLC0415
        azure_kwargs: dict[str, Any] = {
            "azure_deployment": resolved_model,
            "azure_endpoint": _env("AZURE_OPENAI_ENDPOINT") or "",
            "api_version": _env("AZURE_OPENAI_API_VERSION") or "2024-02-01",
        }
        azure_key = _env("AZURE_OPENAI_API_KEY")
        if azure_key:
            azure_kwargs["openai_api_key"] = azure_key
        azure_kwargs.update(kwargs)
        return AzureChatOpenAI(**azure_kwargs)

    if provider == "aws_bedrock":
        from langchain_aws import ChatBedrock  # noqa: PLC0415
        bedrock_kwargs: dict[str, Any] = {
            "model_id": resolved_model,
            "region_name": _env("AWS_REGION") or "us-east-1",
        }
        access_key = _env("AWS_ACCESS_KEY_ID")
        secret_key = _env("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            bedrock_kwargs["aws_access_key_id"] = access_key
            bedrock_kwargs["aws_secret_access_key"] = secret_key
        bedrock_kwargs.update(kwargs)
        return ChatBedrock(**bedrock_kwargs)

    if provider == "ollama":
        # Use langchain-ollama (NOT deprecated langchain-community)
        from langchain_ollama import ChatOllama  # noqa: PLC0415
        base_url, client_kwargs = resolve_ollama_connection(credential_store)
        return ChatOllama(
            model=resolved_model,
            base_url=base_url,
            client_kwargs=client_kwargs,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. "
        f"Supported: openai, anthropic, cohere, huggingface, google_gemini, "
        f"mistral, groq, together_ai, replicate, azure_openai, aws_bedrock, ollama"
    )


# ---------------------------------------------------------------------------
# LCEL RAG chain (standard, non-agentic)
# ---------------------------------------------------------------------------


def build_rag_chain(
    retriever: object,
    llm: object,
    system_prompt: str,
    rag_type: str = "naive_rag",
    graph_store_config: object | None = None,
    credential_store: object | None = None,
) -> object:
    """Build a standard LCEL RAG chain.

    Architecture:
        user_query
          ├─ context branch: retriever → format_docs
          └─ question branch: RunnablePassthrough
          ↓
        ChatPromptTemplate (system_prompt + context + question)
          ↓
        llm
          ↓
        StrOutputParser → answer string

    Args:
        retriever:     A LangChain BaseRetriever.
        llm:           A LangChain BaseChatModel.
        system_prompt: System prompt string from SystemPromptConfigurator.

    Returns:
        A LangChain RunnableSequence (LCEL chain).
    """
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: PLC0415
    telemetry = TelemetryReporter()

    def format_docs(docs: list) -> str:
        return "\n\n".join(
            f"[Source: {getattr(doc, 'metadata', {}).get('source', f'chunk_{i}')}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
    ])

    with telemetry.span("rag.chain.build"):
        if rag_type == "speculative_rag":
            draft_prompt = ChatPromptTemplate.from_messages([
                ("system", "Draft a concise tentative answer. It may be incomplete; evidence will be retrieved next."),
                ("human", "{question}"),
            ])
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                (
                    "human",
                    "Draft answer:\n{draft}\n\nEvidence passages:\n{context}\n\n"
                    "Question: {question}\n\nVerify the draft against the evidence. Correct it if needed.",
                ),
            ])
            draft_chain = draft_prompt | llm | StrOutputParser()  # type: ignore[operator]
            final_chain = final_prompt | llm | StrOutputParser()  # type: ignore[operator]

            def speculative(query: str) -> str:
                draft = draft_chain.invoke({"question": query})
                docs = retriever.invoke(f"{query}\nDraft answer: {draft}")  # type: ignore[union-attr]
                return final_chain.invoke({
                    "question": query,
                    "draft": draft,
                    "context": format_docs(docs),
                })

            return RunnableLambda(speculative)

        if rag_type == "graphrag":
            from ms_rag.ingestion.graph_store import GraphStoreConnector  # noqa: PLC0415

            entity_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract key entities, topics, and relationships as a short comma-separated query expansion."),
                ("human", "{question}"),
            ])
            entity_chain = entity_prompt | llm | StrOutputParser()  # type: ignore[operator]
            final_chain = prompt | llm | StrOutputParser()  # type: ignore[operator]

            def graph_guided(query: str) -> str:
                entities = entity_chain.invoke({"question": query})
                docs = retriever.invoke(f"{query}\nEntities and relationships: {entities}")  # type: ignore[union-attr]
                graph_context = ""
                if graph_store_config is not None:
                    graph_context = GraphStoreConnector(credential_store).retrieve_graph_context(
                        graph_store_config,  # type: ignore[arg-type]
                        query,
                        llm=llm,
                    )
                context = "\n\n".join(
                    part for part in (graph_context, format_docs(docs)) if part.strip()
                )
                if not context.strip():
                    raise RuntimeError(
                        "GraphRAG could not retrieve graph or vector context. "
                        "Rebuild the graph index and verify the graph store/keyword store configuration."
                    )
                return final_chain.invoke({"question": query, "context": context})

            return RunnableLambda(graph_guided)

        rag_chain = (
            {
                "context": retriever | format_docs,  # type: ignore[operator]
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm  # type: ignore[operator]
            | StrOutputParser()
        )

    return rag_chain


def build_retriever_stack(
    base_retriever: object,
    config: PipelineConfig,
    *,
    context_compressor: object | None = None,
    reranking_module: object | None = None,
    llm: object | None = None,
    embeddings: object | None = None,
) -> object:
    """Apply configured reranking and compression layers to a base retriever."""
    retriever = base_retriever

    if config.reranking_enabled and config.reranking and reranking_module is not None:
        from ms_rag.query.reranking_retriever import RerankingRetriever  # noqa: PLC0415

        retriever = RerankingRetriever(
            base_retriever=retriever,
            reranking_module=reranking_module,
            config=config.reranking,
            llm=llm,
        )

    if config.compression_enabled and config.compression and context_compressor is not None:
        wrapped = context_compressor.get_compressor(  # type: ignore[union-attr]
            config.compression,
            llm=llm,
            embeddings=embeddings,
            base_retriever=retriever,
        )
        if wrapped is not None:
            retriever = wrapped

    return retriever


def rebuild_session_runtime(
    config: PipelineConfig,
    credential_store: object,
) -> dict[str, object]:
    """Rebuild vector store, retriever, LLM, and RAG chain from a saved config."""
    from ms_rag.ingestion.vectorization_module import VectorizationModule  # noqa: PLC0415
    from ms_rag.ingestion.vectordb_connector import VectorDBConnector  # noqa: PLC0415
    telemetry = TelemetryReporter()

    if config.embedding_model is None or config.vector_db is None or config.retrieval is None:
        raise ValueError(
            "Session config is incomplete — embedding model, vector DB, and retrieval "
            "strategy are required to rebuild the runtime pipeline."
        )

    with telemetry.span("session.runtime.rebuild"):
        vectorization = VectorizationModule()
        embeddings = vectorization.get_embeddings(config.embedding_model, credential_store)
        db_connector = VectorDBConnector(credential_store=credential_store)
        vector_store = db_connector.get_vector_store(config.vector_db, embeddings)

        runtime = build_session_runtime_from_vector_store(
            config,
            credential_store,
            vector_store=vector_store,
            embeddings=embeddings,
        )
    runtime["vector_store"] = vector_store
    return runtime


def build_session_runtime_from_vector_store(
    config: PipelineConfig,
    credential_store: object,
    *,
    vector_store: object,
    embeddings: object | None = None,
) -> dict[str, object]:
    """Build retriever, LLM, and RAG chain from an existing vector store.

    The interactive setup path has already populated ``vector_store`` during
    ingestion. Reusing it is important for in-memory stores such as FAISS,
    where creating a new wrapper would lose the just-ingested documents.
    """
    from ms_rag.ingestion.ingestion_orchestrator import IngestionOrchestrator  # noqa: PLC0415
    from ms_rag.query.context_compressor import ContextCompressor  # noqa: PLC0415
    from ms_rag.query.reranking_module import RerankingModule  # noqa: PLC0415
    from ms_rag.query.retrieval_strategy import RetrievalStrategyModule  # noqa: PLC0415

    if config.retrieval is None:
        raise ValueError(
            "Session config is incomplete — retrieval strategy is required "
            "to build the runtime pipeline."
        )
    if embeddings is not None:
        setattr(vector_store, "_ms_rag_embeddings", embeddings)

    if config.llm_model:
        provider = config.llm_model.provider
        model_id = config.llm_model.model_id
    elif config.configured_providers:
        provider = config.configured_providers[0]
        model_id = "default"
    else:
        raise ValueError(
            "No generation LLM model is selected. "
            "Please configure an LLM provider and model before building the runtime."
        )
    llm = get_llm(provider, model_id, credential_store=credential_store)

    keyword_corpus: list[str] | None = None
    if _retrieval_uses_keyword_corpus(config.retrieval):
        if config.keyword_store is not None:
            try:
                from ms_rag.ingestion.keyword_store import KeywordStoreConnector  # noqa: PLC0415

                keyword_corpus = KeywordStoreConnector(credential_store).load_texts(config.keyword_store)
                if keyword_corpus:
                    setattr(vector_store, "_ms_rag_keyword_corpus", keyword_corpus)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not load keyword corpus from persistent keyword store; will try runtime cache/source rebuild: {exc}",
                    stacklevel=2,
                )
        cached = getattr(vector_store, "_ms_rag_keyword_corpus", None)
        if isinstance(cached, list) and cached:
            keyword_corpus = cached
        elif config.document_sources and config.loader_map and config.chunking is not None:
            try:
                keyword_corpus = IngestionOrchestrator(credential_store=credential_store).build_keyword_corpus(
                    sources=config.document_sources,
                    loader_map=config.loader_map,
                    chunking_config=config.chunking,
                )
                setattr(vector_store, "_ms_rag_keyword_corpus", keyword_corpus)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not rebuild keyword corpus for {config.retrieval.strategy}; "
                    f"keyword retrieval may degrade: {exc}",
                    stacklevel=2,
                )
        if not keyword_corpus:
            raise RuntimeError(
                f"{config.retrieval.strategy} retrieval requires a keyword corpus, but no keyword texts "
                "were available from the persistent keyword store, runtime cache, or original sources. "
                "Re-ingest documents and configure a keyword store before using this retrieval mode."
            )

    if _retrieval_uses_advanced_state(config.retrieval):
        has_parent_state = bool(getattr(vector_store, "_ms_rag_parent_documents", None))
        has_chunk_state = bool(getattr(vector_store, "_ms_rag_chunk_documents", None))
        if not (has_parent_state and has_chunk_state) and config.document_sources and config.loader_map and config.chunking is not None:
            try:
                state = IngestionOrchestrator(credential_store=credential_store).build_retrieval_state(
                    sources=config.document_sources,
                    loader_map=config.loader_map,
                    chunking_config=config.chunking,
                )
                setattr(vector_store, "_ms_rag_parent_documents", state.get("parent_documents", {}))
                setattr(vector_store, "_ms_rag_chunk_documents", state.get("chunk_documents", []))
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not rebuild advanced retrieval state for {config.retrieval.strategy}; "
                    f"the selected retriever may degrade: {exc}",
                    stacklevel=2,
                )

    if config.rag_type and config.rag_type.rag_type == "graphrag":
        if config.graph_store is None:
            raise RuntimeError("GraphRAG requires a configured graph store.")
        try:
            from ms_rag.ingestion.graph_store import GraphStoreConnector  # noqa: PLC0415

            graph_connector = GraphStoreConnector(credential_store)
            graph = graph_connector.load_graph(config.graph_store)
            setattr(vector_store, "_ms_rag_graph_index", graph)
        except Exception as exc:  # noqa: BLE001
            if config.document_sources and config.loader_map and config.chunking is not None:
                state = IngestionOrchestrator(credential_store=credential_store).build_retrieval_state(
                    sources=config.document_sources,
                    loader_map=config.loader_map,
                    chunking_config=config.chunking,
                )
                chunks = state.get("chunk_documents", [])
                graph_connector = GraphStoreConnector(credential_store)
                graph = graph_connector.build_graph_index(chunks, llm=llm)
                graph_connector.persist_graph(config.graph_store, graph)
                setattr(vector_store, "_ms_rag_graph_index", graph)
            else:
                raise RuntimeError(
                    f"Could not load GraphRAG graph index and original sources are unavailable for rebuild: {exc}"
                ) from exc

    retrieval_module = RetrievalStrategyModule()
    base_retriever = retrieval_module.get_retriever(
        config.retrieval,
        vector_store,
        llm=llm,
        corpus_texts=keyword_corpus,
        embeddings=embeddings,
        strict_advanced=True,
    )

    reranking_module = RerankingModule(credential_store=credential_store)
    compressor = ContextCompressor()
    retriever = build_retriever_stack(
        base_retriever,
        config,
        context_compressor=compressor,
        reranking_module=reranking_module,
        llm=llm,
        embeddings=embeddings,
    )
    compression_active = bool(
        config.compression_enabled
        and config.compression
        and retriever is not base_retriever
    )

    # Build one shared agent tool runtime so short-term memory survives across
    # turns and semantic memory can reuse the pipeline's embedding model.
    agent_runtime: object | None = None
    if config.agent_tools and config.agent_tools.enabled_tools:
        from ms_rag.agent.tools import AgentToolRuntime  # noqa: PLC0415

        agent_runtime = AgentToolRuntime(
            config.agent_tools,
            credential_store,
            llm,
            embeddings=embeddings,
        )

    if config.rag_type and config.rag_type.requires_langgraph:
        rag_chain = build_langgraph_workflow(
            config.rag_type.rag_type,
            retriever,
            llm,
            config.system_prompt,
            agent_tools_config=config.agent_tools,
            credential_store=credential_store,
            tool_runtime=agent_runtime,
        )
    else:
        rag_chain = build_rag_chain(
            retriever,
            llm,
            config.system_prompt,
            config.rag_type.rag_type if config.rag_type else "naive_rag",
            graph_store_config=config.graph_store,
            credential_store=credential_store,
        )

    return {
        "vector_store": vector_store,
        "retriever": retriever,
        "llm": llm,
        "rag_chain": rag_chain,
        "compression_active": compression_active,
        "agent_runtime": agent_runtime,
    }


def _retrieval_uses_keyword_corpus(config: PipelineConfig | RetrievalConfig | None) -> bool:
    """Return True when the retrieval strategy depends on raw text corpus access."""
    retrieval = config.retrieval if isinstance(config, PipelineConfig) else config
    if retrieval is None:
        return False
    if retrieval.strategy in {"keyword_bm25", "tfidf", "hybrid"}:
        return True
    if retrieval.strategy == "ensemble":
        sub_ids = retrieval.ensemble_sub_retrievers or ["dense_vector", "keyword_bm25"]
        return any(sub_id in {"keyword_bm25", "tfidf", "hybrid"} for sub_id in sub_ids)
    return False


def _retrieval_uses_advanced_state(config: PipelineConfig | RetrievalConfig | None) -> bool:
    """Return True when retrieval needs parent/chunk/timestamp runtime state."""
    retrieval = config.retrieval if isinstance(config, PipelineConfig) else config
    if retrieval is None:
        return False
    advanced = {"parent_child", "multi_vector", "time_weighted"}
    if retrieval.strategy in advanced:
        return True
    if retrieval.strategy == "ensemble":
        sub_ids = retrieval.ensemble_sub_retrievers or ["dense_vector", "keyword_bm25"]
        return any(sub_id in advanced for sub_id in sub_ids)
    return False


def _select_primary_retrieval_query(
    *,
    original_query: str,
    enhanced_queries: list[str],
    retrieval: RetrievalConfig | None,
) -> str:
    """Choose the query string that should hit the configured retriever."""
    if _retrieval_uses_keyword_corpus(retrieval):
        return original_query
    return enhanced_queries[0] if enhanced_queries else original_query


def _append_trace(state: dict, message: str) -> list[str]:
    """Append one visible workflow trace message to LangGraph state."""
    return list(state.get("trace", []) or []) + [message]


def _lexically_relevant_docs(question: str, docs: list, limit: int = 3) -> list:
    """Keep obvious evidence when an LLM grader over-filters all documents."""
    stop_words = {
        "about",
        "tell",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
    }
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2 and term not in stop_words
    }
    scored: list[tuple[int, object]] = []
    for doc in docs:
        text = str(getattr(doc, "page_content", "") or "").lower()
        score = sum(1 for term in query_terms if term in text)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def _answer_is_unknown(answer: object) -> bool:
    """Return True for the framework's exact grounded-no-answer response."""
    normalized = re.sub(r"[^a-z]+", " ", str(answer or "").lower()).strip()
    return normalized in {"i don t know", "i do not know"}


def invoke_rag_chain(rag_chain: object, query: str, *, requires_langgraph: bool) -> str:
    """Invoke an LCEL chain or LangGraph workflow and return an answer string."""
    if requires_langgraph:
        result = rag_chain.invoke({"question": query})  # type: ignore[union-attr]
        if isinstance(result, dict):
            generation = result.get("generation")
            if generation is not None:
                return str(generation)
        return str(result)

    answer = rag_chain.invoke(query)  # type: ignore[union-attr]
    return answer if isinstance(answer, str) else str(answer)


# ---------------------------------------------------------------------------
# LangGraph agentic RAG workflows
# ---------------------------------------------------------------------------


def build_langgraph_workflow(
    rag_type: str,
    retriever: object,
    llm: object,
    system_prompt: str,
    agent_tools_config: object | None = None,
    credential_store: object | None = None,
    tool_runtime: object | None = None,
) -> object:
    """Build a LangGraph StateGraph for agentic RAG variants.

    Supports:
        self_rag       — Self-RAG with document grading and hallucination check
        corrective_rag — CRAG with web search fallback
        agentic_rag    — General agentic loop with query analysis
        adaptive_rag   — Routes to appropriate retrieval depth

    Args:
        rag_type:      One of LANGGRAPH_TYPES.
        retriever:     A LangChain BaseRetriever.
        llm:           A LangChain BaseChatModel.
        system_prompt: System prompt string.

    Returns:
        A compiled LangGraph app (CompiledGraph).

    Raises:
        ValueError: If rag_type is not in LANGGRAPH_TYPES.
    """
    if rag_type not in LANGGRAPH_TYPES:
        raise ValueError(
            f"{rag_type!r} does not require LangGraph. "
            f"Use build_rag_chain() instead."
        )

    from langgraph.graph import StateGraph, END  # noqa: PLC0415
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
    from ms_rag.agent.tools import AgentToolRuntime, ToolExecutionError  # noqa: PLC0415

    if tool_runtime is None:
        tool_runtime = AgentToolRuntime(agent_tools_config, credential_store, llm)

    # ── Shared nodes ──────────────────────────────────────────────────

    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])  # type: ignore[union-attr]
        return {
            "documents": docs,
            "trace": _append_trace(state, f"retrieve: fetched {len(docs)} candidate document(s)"),
        }

    def direct_answer(state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
        answer = chain.invoke({"question": state["question"]})
        return {
            "generation": answer,
            "documents": list(state.get("documents", [])),
            "trace": _append_trace(state, "direct_answer: answered without retrieval context"),
        }

    def generate(state: GraphState) -> dict:
        docs = list(state.get("documents", []) or [])
        context = "\n\n".join(d.page_content for d in docs)
        tool_context = "\n\n".join(state.get("tool_results", []))
        if tool_context:
            context = f"{context}\n\nTool results:\n{tool_context}".strip()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Context:\n\n{context}\n\nQuestion: {{question}}"),
        ])
        chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
        answer = chain.invoke({"question": state["question"]})
        trace = _append_trace(
            state,
            f"generate: generated answer from {len(docs)} document(s)"
            + (f" and {len(state.get('tool_results', []))} tool result(s)" if state.get("tool_results") else ""),
        )
        if docs and _answer_is_unknown(answer):
            retry_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    system_prompt
                    + "\n\nSELF-RAG RETRY: Relevant context was retrieved. "
                    "Before saying you do not know, extract the directly supported facts from the context. "
                    "Do not use outside knowledge.",
                ),
                ("human", f"Context:\n\n{context}\n\nQuestion: {{question}}"),
            ])
            retry_chain = retry_prompt | llm | StrOutputParser()  # type: ignore[operator]
            retry_answer = retry_chain.invoke({"question": state["question"]})
            trace = trace + [
                "generate: first answer was 'I don't know' despite retrieved evidence; ran one grounded retry"
            ]
            answer = retry_answer
        return {
            "generation": answer,
            "hallucination_count": state.get("hallucination_count", 0) + 1,
            "trace": trace,
        }

    def rewrite_query(state: GraphState) -> dict:
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question to be clearer and more specific."),
            ("human", "{question}"),
        ])
        chain = rewrite_prompt | llm | StrOutputParser()  # type: ignore[operator]
        new_q = chain.invoke({"question": state["question"]})
        return {
            "question": new_q.strip(),
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "trace": _append_trace(state, f"rewrite_query: {state['question']} -> {new_q.strip()}"),
        }

    def _contains_any(text: str, words: set[str]) -> bool:
        lower = text.lower()
        return any(word in lower for word in words)

    def _llm_choice(prompt_messages: list[tuple[str, str]], values: dict[str, object], allowed: set[str], default: str) -> str:
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
        try:
            raw = chain.invoke(values).strip().lower()
        except Exception as exc:
            raise RuntimeError(f"LangGraph routing LLM call failed: {exc}") from exc
        for item in allowed:
            if item in raw:
                return item
        return default

    def grade_documents(state: GraphState) -> dict:
        """Grade each document for relevance — keep only relevant ones."""
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "Is this document relevant to the question? Answer yes or no."),
            ("human", "Question: {question}\n\nDocument: {document}"),
        ])
        chain = grade_prompt | llm | StrOutputParser()  # type: ignore[operator]
        original_docs = list(state.get("documents", []))
        relevant = []
        for doc in state["documents"]:
            grade = chain.invoke({
                "question": state["question"],
                "document": doc.page_content,
            }).lower()
            if "yes" in grade:
                relevant.append(doc)
        trace = _append_trace(
            state,
            f"grade_documents: LLM relevance grader kept {len(relevant)}/{len(original_docs)} document(s)",
        )
        if not relevant and original_docs:
            lexical = _lexically_relevant_docs(state["question"], original_docs)
            if lexical:
                trace = trace + [
                    f"grade_documents: safety kept {len(lexical)} document(s) with lexical query overlap after grader removed all evidence"
                ]
                relevant = lexical
            else:
                trace = trace + ["grade_documents: no relevant evidence found; graph may rewrite query"]
        return {"documents": relevant, "trace": trace}

    # ── Routing functions ──────────────────────────────────────────────

    def decide_to_generate(state: GraphState) -> str:
        """Route to generate if enough docs; rewrite if none (max 2 rewrites)."""
        if state["documents"] and state.get("rewrite_count", 0) < 2:
            return "generate"
        if state.get("rewrite_count", 0) >= 2:
            return "generate"  # force generation after 2 rewrites
        return "rewrite_query"

    def check_hallucination(state: GraphState) -> str:
        """Check if generation is grounded in documents; retry up to 2 times."""
        count = state.get("hallucination_count", 0)
        if not state["documents"] or count >= 2:
            if count >= 2:
                warnings.warn(
                    "Self-RAG max hallucination retries (2) reached. Returning the last generated answer.",
                    stacklevel=2,
                )
            return "end"
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", "Is the answer fully supported by the provided context? Answer yes or no."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
        ])
        chain = check_prompt | llm | StrOutputParser()  # type: ignore[operator]
        context = "\n\n".join(d.page_content for d in state["documents"])
        result = chain.invoke({
            "context": context,
            "answer": state["generation"],
        }).lower()
        if "yes" not in result:
            warnings.warn(
                f"Self-RAG support check ({count + 1}/2) marked the answer as not fully grounded; retrying generation.",
                stacklevel=2,
            )
            return "generate"
        return "end"

    # ── Build graph ────────────────────────────────────────────────────

    workflow = StateGraph(GraphState)

    if rag_type in ("self_rag", "corrective_rag"):
        # Self-RAG / CRAG: retrieve → grade → generate → check hallucination
        def record_retrieval_decision(state: GraphState) -> dict:
            if rag_type == "self_rag":
                return {
                    "trace": _append_trace(
                        state,
                        "decide_retrieval_need: checking whether the private document corpus is needed",
                    )
                }
            return {
                "trace": _append_trace(
                    state,
                    "corrective_rag: retrieval is required before correction fallback",
                )
            }

        def decide_retrieval_need(state: GraphState) -> str:
            if rag_type != "self_rag":
                return "retrieve"
            question = state["question"]
            route = _llm_choice(
                [
                    (
                        "system",
                        "Decide if this question needs the private document corpus. "
                        "Answer exactly retrieve or direct.",
                    ),
                    ("human", "{question}"),
                ],
                {"question": question},
                {"retrieve", "direct"},
                "retrieve",
            )
            if route == "direct" and not _is_obvious_direct_chat(question):
                route = "retrieve"
            return route

        def corrective_web_fallback(state: GraphState) -> dict:
            if rag_type != "corrective_rag" or state.get("documents"):
                return {"tool_results": list(state.get("tool_results", []))}
            if not tool_runtime.enabled("web_search"):
                return {"tool_results": list(state.get("tool_results", []))}
            try:
                web_results = tool_runtime.web_search(state["question"])
                if tool_runtime.enabled("document_summarization") and len(web_results) > 4000:
                    web_results = tool_runtime.summarize(web_results)
            except ToolExecutionError as exc:
                raise RuntimeError(f"Configured CRAG Web Search fallback failed: {exc}") from exc
            return {"tool_results": list(state.get("tool_results", [])) + [f"CRAG web fallback:\n{web_results}"]}

        def decide_after_grade(state: GraphState) -> str:
            if state["documents"]:
                return "generate"
            if rag_type == "corrective_rag" and tool_runtime.enabled("web_search"):
                return "web_fallback"
            if state.get("rewrite_count", 0) >= 2:
                return "generate"
            return "rewrite_query"

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("direct_answer", direct_answer)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("web_fallback", corrective_web_fallback)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite_query", rewrite_query)

        workflow.add_node("decide_retrieval_need", record_retrieval_decision)
        workflow.set_entry_point("decide_retrieval_need")
        workflow.add_conditional_edges(
            "decide_retrieval_need",
            decide_retrieval_need,
            {"retrieve": "retrieve", "direct": "direct_answer"},
        )
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_after_grade,
            {"generate": "generate", "rewrite_query": "rewrite_query", "web_fallback": "web_fallback"},
        )
        workflow.add_edge("web_fallback", "generate")
        workflow.add_conditional_edges(
            "generate",
            check_hallucination,
            {"end": END, "generate": "generate"},
        )
        workflow.add_edge("rewrite_query", "retrieve")

    elif rag_type == "agentic_rag":
        # Agentic: LLM plan → retrieve/rewrite/tools/generate with hard allowlists.
        def query_analysis(state: GraphState) -> dict:
            question = state["question"]
            tool_names = set(getattr(tool_runtime.config, "enabled_tools", []) or [])
            trigger_notes: list[str] = []
            if "url_fetch" in tool_names and re.search(r"https?://", question):
                trigger_notes.append("url_fetch")
            if "file_read" in tool_names and re.search(r"file:", question, flags=re.IGNORECASE):
                trigger_notes.append("file_read")
            if "api_request" in tool_names and re.search(r"api\s+(GET|POST|PUT|PATCH)\s+https?://", question, flags=re.IGNORECASE):
                trigger_notes.append("api_request")
            if "memory" in tool_names:
                trigger_notes.append("memory")
            if "web_search" in tool_names:
                trigger_notes.append("web_search_when_retrieval_empty")
            planner_action = _llm_choice(
                [
                    (
                        "system",
                        "You are routing an approved RAG agent. Choose exactly one action: "
                        "retrieve, rewrite, tools, or answer. Use tools only if an approved "
                        "tool trigger exists. Use rewrite for unclear questions. Use retrieve "
                        "when private corpus evidence is needed. Use answer for simple general "
                        "conversation.",
                    ),
                    (
                        "human",
                        "Question: {question}\nApproved tool triggers: {triggers}\nAction:",
                    ),
                ],
                {"question": question, "triggers": ", ".join(trigger_notes) or "none"},
                {"retrieve", "rewrite", "tools", "answer"},
                "retrieve",
            )
            if planner_action == "tools" and not trigger_notes:
                planner_action = "retrieve"
            if planner_action == "answer" and not _is_obvious_direct_chat(question):
                planner_action = "retrieve"
            return {
                "tool_results": [],
                "action": planner_action,
                "trace": _append_trace(state, f"agentic_rag: planner selected action={planner_action}"),
            }

        def route_agent_action(state: GraphState) -> str:
            action = str(state.get("action") or "retrieve")
            if action == "rewrite" and state.get("rewrite_count", 0) >= 2:
                return "retrieve"
            return action if action in {"retrieve", "rewrite", "tools", "answer"} else "retrieve"

        def _maybe_summarize(text: str) -> str:
            if tool_runtime.enabled("document_summarization") and len(text) > 4000:
                return tool_runtime.summarize(text)
            return text

        def run_approved_tools(state: GraphState) -> dict:
            results = list(state.get("tool_results", []))
            question = state["question"]
            if tool_runtime.enabled("memory"):
                memory = tool_runtime.recall_memory(question)
                if memory:
                    results.append(f"Memory recall:\n{memory}")
            if tool_runtime.enabled("url_fetch"):
                urls = re.findall(r"https?://[^\s)>\"]+", question)
                for url in urls[:3]:
                    clean_url = url.rstrip(".,")
                    fetched = tool_runtime.fetch_url(clean_url)
                    results.append(f"URL fetch result for {clean_url}:\n{_maybe_summarize(fetched)}")
            if tool_runtime.enabled("file_read"):
                quoted_files = re.findall(r"file:\"([^\"]+)\"", question, flags=re.IGNORECASE)
                bare_files = re.findall(r"file:([^\s\"]+)", question, flags=re.IGNORECASE)
                file_matches = quoted_files + bare_files
                for file_path in file_matches[:3]:
                    clean_path = file_path.strip()
                    content = tool_runtime.read_file(clean_path)
                    results.append(f"File read result for {clean_path}:\n{_maybe_summarize(content)}")
            if tool_runtime.enabled("api_request"):
                api_matches = re.findall(
                    r"api\s+(GET|POST|PUT|PATCH)\s+(https?://\S+)",
                    question,
                    flags=re.IGNORECASE,
                )
                for method, url in api_matches[:2]:
                    clean_url = url.rstrip(".,")
                    response = tool_runtime.api_request(method.upper(), clean_url)
                    results.append(f"API response for {method.upper()} {clean_url}:\n{_maybe_summarize(response)}")
            if tool_runtime.enabled("web_search") and not state.get("documents"):
                try:
                    web_results = tool_runtime.web_search(question)
                    web_results = _maybe_summarize(web_results)
                    results.append(f"Web search results:\n{web_results}")
                except ToolExecutionError as exc:
                    raise RuntimeError(f"Configured Web Search Tool failed: {exc}") from exc
            return {
                "tool_results": results,
                "trace": _append_trace(state, f"agentic_rag: approved tools returned {len(results)} result block(s)"),
            }

        workflow.add_node("query_analysis", query_analysis)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("run_approved_tools", run_approved_tools)
        workflow.add_node("generate", generate)
        workflow.add_node("direct_answer", direct_answer)
        workflow.add_node("rewrite_query", rewrite_query)

        workflow.set_entry_point("query_analysis")
        # NOTE: the "tools" action routes through retrieve so approved tools
        # (memory, web search, etc.) AUGMENT corpus context instead of replacing
        # it. Retrieval always runs before generation; otherwise a query that
        # the planner sends to tools would generate from zero documents even
        # when the vector store holds the answer.
        workflow.add_conditional_edges(
            "query_analysis",
            route_agent_action,
            {
                "retrieve": "retrieve",
                "rewrite": "rewrite_query",
                "tools": "retrieve",
                "answer": "direct_answer",
            },
        )
        workflow.add_edge("rewrite_query", "query_analysis")
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("retrieve", "run_approved_tools")
        workflow.add_edge("run_approved_tools", "generate")
        workflow.add_edge("generate", END)

    elif rag_type == "adaptive_rag":
        # Adaptive: direct, standard retrieval, or deeper rewrite+retrieval route.
        def route_question(state: GraphState) -> str:
            question = state["question"]
            heuristic_deep = _contains_any(
                question,
                {"compare", "analyze", "relationship", "relationships", "across", "multi-hop", "why", "explain"},
            )
            route = _llm_choice(
                [
                    (
                        "system",
                        "Route this query. Answer exactly direct, retrieve, or deep. "
                        "direct means no private corpus needed. retrieve means one corpus "
                        "retrieval pass. deep means rewrite/decompose then retrieve for "
                        "multi-hop, comparison, analysis, or relationship questions.",
                    ),
                    ("human", "{question}"),
                ],
                {"question": question},
                {"direct", "retrieve", "deep"},
                "deep" if heuristic_deep else "retrieve",
            )
            if heuristic_deep and route == "retrieve":
                return "deep"
            return route

        def record_adaptive_route(state: GraphState) -> dict:
            return {
                "trace": _append_trace(
                    state,
                    "adaptive_rag: routing query as direct, standard retrieval, or deep retrieval",
                )
            }

        def deep_retrieve(state: GraphState) -> dict:
            original_question = state["question"]
            rewritten = rewrite_query(state)
            rewritten_question = str(rewritten.get("question") or original_question)
            original_docs = retriever.invoke(original_question)  # type: ignore[union-attr]
            rewritten_docs = retriever.invoke(rewritten_question)  # type: ignore[union-attr]
            seen: set[tuple[str, str]] = set()
            merged = []
            for doc in list(original_docs) + list(rewritten_docs):
                key = (
                    getattr(doc, "page_content", ""),
                    str(getattr(doc, "metadata", {}).get("source", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
            return {
                "documents": merged,
                "question": rewritten_question,
                "rewrite_count": state.get("rewrite_count", 0) + 1,
                "trace": _append_trace(
                    state,
                    f"adaptive_rag: deep retrieval merged {len(merged)} document(s) from original and rewritten queries",
                ),
            }

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("deep_retrieve", deep_retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("direct_answer", direct_answer)
        workflow.add_node("route_question", record_adaptive_route)

        workflow.set_entry_point("route_question")
        workflow.add_conditional_edges(
            "route_question",
            route_question,
            {"direct": "direct_answer", "retrieve": "retrieve", "deep": "deep_retrieve"},
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("deep_retrieve", "generate")
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Query pipeline wiring
# ---------------------------------------------------------------------------


def process_query(
    query: str,
    session_state: object,
    query_enhancer: object | None = None,
    evaluation_framework: object | None = None,
) -> str:
    """End-to-end query processing pipeline.

    Flow:
        query → enhance → retrieve → rerank → compress → LLM → evaluate → answer

    Args:
        query:               The user query string.
        session_state:       SessionState with rag_chain, retriever, etc.
        query_enhancer:      Optional QueryEnhancer instance.
        evaluation_framework: Optional EvaluationFramework instance.

    Returns:
        The generated answer string.
    """
    from ms_rag.models import SessionState  # noqa: PLC0415
    from ms_rag.workflow.rag_presets import get_rag_preset  # noqa: PLC0415
    telemetry = TelemetryReporter()

    ss: SessionState = session_state  # type: ignore[assignment]
    cfg = ss.config

    with telemetry.span("query.process", query_length=len(query)):
        # Step 1: Query Enhancement
        enhanced_queries = [query]
        if query_enhancer and cfg.query_enhancement:
            preset = get_rag_preset(cfg.rag_type.rag_type if cfg.rag_type else None)
            strict_enhancement = (
                bool(preset.query_enhancement)
                and list(preset.query_enhancement) == list(cfg.query_enhancement)
                and not preset.allow_query_enhancement_prompt
            )
            try:
                enhanced_queries = query_enhancer.enhance(  # type: ignore[union-attr]
                    query=query,
                    techniques=cfg.query_enhancement,
                    llm=ss.llm,
                    hyde_provider=cfg.hyde_llm_provider,
                    strict=strict_enhancement,
                )
            except Exception as exc:  # noqa: BLE001
                if strict_enhancement:
                    raise
                warnings.warn(
                    f"Query enhancement failed; using the original query: {exc}",
                    stacklevel=2,
                )
                enhanced_queries = [query]

        # HyDE is useful for dense embedding retrieval, but keyword-only
        # retrievers need the user's exact terms. Do not replace BM25/TF-IDF
        # queries with a hypothetical document.
        primary_query = _select_primary_retrieval_query(
            original_query=query,
            enhanced_queries=enhanced_queries,
            retrieval=cfg.retrieval,
        )
        ss.last_enhanced_queries = list(enhanced_queries)
        ss.last_primary_retrieval_query = primary_query

        # Step 2: invoke the RAG chain
        if ss.rag_chain is None:
            return "Pipeline not initialised. Please complete all setup steps first."

        try:
            requires_langgraph = bool(
                cfg.rag_type and cfg.rag_type.requires_langgraph
            )
            if (
                (len(enhanced_queries) > 1 or primary_query != query)
                and not requires_langgraph
                and ss.retriever is not None
                and ss.llm is not None
            ):
                use_reciprocal_rank_fusion = "rag_fusion" in set(cfg.query_enhancement or [])
                ss.last_rag_trace = [
                    "standard_rag: generated multiple retrieval queries",
                    (
                        "standard_rag: merged retrieved evidence with reciprocal-rank fusion"
                        if use_reciprocal_rank_fusion
                        else "standard_rag: merged retrieved evidence from enhanced queries"
                    ),
                    "standard_rag: generated the answer from merged context",
                ]
                answer = _answer_from_merged_queries(
                    original_query=query,
                    retrieval_queries=(
                        enhanced_queries if len(enhanced_queries) > 1 else [primary_query]
                    ),
                    retriever=ss.retriever,
                    llm=ss.llm,
                    system_prompt=cfg.system_prompt,
                    use_reciprocal_rank_fusion=use_reciprocal_rank_fusion,
                )
            else:
                if requires_langgraph:
                    result = ss.rag_chain.invoke({"question": primary_query, "trace": []})  # type: ignore[union-attr]
                    if isinstance(result, dict):
                        ss.last_rag_trace = list(result.get("trace", []) or [])
                        answer = str(result.get("generation", result))
                    else:
                        ss.last_rag_trace = []
                        answer = str(result)
                else:
                    ss.last_rag_trace = [
                        "standard_rag: retrieved context using the selected retriever",
                        "standard_rag: generated the answer with the selected LLM",
                    ]
                    answer = invoke_rag_chain(
                        ss.rag_chain,
                        primary_query,
                        requires_langgraph=requires_langgraph,
                    )
        except Exception as exc:  # noqa: BLE001
            raise exc  # re-raise so QueryLoop can handle it

        # Step 3: Evaluation (if enabled)
        context_docs = []
        if ss.retriever:
            try:
                context_docs = ss.retriever.invoke(primary_query)
                ss.last_retrieved_context_count = len(context_docs)
                ss.last_retrieved_context_preview = [
                    str(getattr(doc, "page_content", "") or "")[:240]
                    for doc in context_docs[:3]
                ]
                trace = _collect_retriever_trace(ss.retriever)
                ss.last_rerank_trace = trace.get("rerank", {})
                ss.last_compression_trace = trace.get("compression", {})
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not collect retrieval trace for this query: {exc}",
                    stacklevel=2,
                )
                ss.last_retrieved_context_count = 0
                ss.last_retrieved_context_preview = []
                ss.last_rerank_trace = {}
                ss.last_compression_trace = {}

        if evaluation_framework and cfg.evaluation_enabled and cfg.evaluation:
            try:
                scores = evaluation_framework.evaluate(  # type: ignore[union-attr]
                    query=query,
                    context=context_docs,
                    answer=answer,
                    config=cfg.evaluation,
                )
                ss.last_evaluation_scores = dict(scores or {})
                ss.last_evaluation_warning = "" if scores else "Evaluation ran but returned no scores."
            except Exception as exc:  # noqa: BLE001
                ss.last_evaluation_scores = {}
                ss.last_evaluation_warning = f"Evaluation failed for this query: {exc}"
                warnings.warn(
                    f"Evaluation failed for this query; answer returned without scores: {exc}",
                    stacklevel=2,
                )
        else:
            ss.last_evaluation_scores = {}
            ss.last_evaluation_warning = ""

    return answer


def _collect_retriever_trace(retriever: object) -> dict[str, dict[str, object]]:
    """Collect trace counters from nested retriever wrappers."""
    trace: dict[str, dict[str, object]] = {}
    current = retriever
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_ms_rag_pre_rerank_count") or hasattr(current, "_ms_rag_post_rerank_count"):
            trace["rerank"] = {
                "before": getattr(current, "_ms_rag_pre_rerank_count", None),
                "after": getattr(current, "_ms_rag_post_rerank_count", None),
            }
        if hasattr(current, "_ms_rag_pre_compression_count") or hasattr(current, "_ms_rag_post_compression_count"):
            trace["compression"] = {
                "before": getattr(current, "_ms_rag_pre_compression_count", None),
                "after": getattr(current, "_ms_rag_post_compression_count", None),
                "fallback": bool(getattr(current, "_ms_rag_compression_fallback", False)),
            }
        current = getattr(current, "base_retriever", None)
    return trace


def _is_obvious_direct_chat(question: str) -> bool:
    """Return True only for safe context-free conversational turns."""
    cleaned = " ".join(str(question or "").strip().lower().split())
    if not cleaned:
        return True
    direct_phrases = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "who are you",
    }
    return cleaned in direct_phrases


def _answer_from_merged_queries(
    *,
    original_query: str,
    retrieval_queries: list[str],
    retriever: object,
    llm: object,
    system_prompt: str,
    use_reciprocal_rank_fusion: bool = False,
) -> str:
    """Retrieve over multiple query variants, rank/dedupe docs, and answer once."""
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415

    docs_by_key: dict[tuple[str, str], object] = {}
    scores: dict[tuple[str, str], float] = {}
    first_seen: dict[tuple[str, str], int] = {}
    order = 0
    rrf_k = 60
    for retrieval_query in retrieval_queries:
        for rank, doc in enumerate(retriever.invoke(retrieval_query)):  # type: ignore[union-attr]
            metadata = getattr(doc, "metadata", {}) or {}
            key = (
                str(metadata.get("source", "")),
                str(getattr(doc, "page_content", ""))[:500],
            )
            docs_by_key.setdefault(key, doc)
            first_seen.setdefault(key, order)
            order += 1
            if use_reciprocal_rank_fusion:
                scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1))
            else:
                scores[key] = scores.get(key, 0.0) + (1.0 / (rank + 1))

    ranked_keys = sorted(
        docs_by_key,
        key=lambda key: (-scores.get(key, 0.0), first_seen.get(key, 0)),
    )
    docs = [docs_by_key[key] for key in ranked_keys]

    context = "\n\n".join(
        f"[Source: {getattr(doc, 'metadata', {}).get('source', f'chunk_{i}')}]"
        f"\n{getattr(doc, 'page_content', '')}"
        for i, doc in enumerate(docs)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            "The following context was retrieved from multiple query variants.\n\n"
            "{context}\n\nOriginal question: {question}",
        ),
    ])
    chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
    return chain.invoke({"context": context, "question": original_query})
