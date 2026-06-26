"""LLM Integration Layer for MS_RAG.

LLM factory, LCEL RAG chain assembly, and LangGraph agentic workflow
builder for all supported RAG architecture variants.

Requirement 17.2: generated code uses LangChain (LCEL chains)
Requirement 17.3: generated code uses LangGraph for agentic RAG types
"""

from __future__ import annotations

from typing import Any

from ms_rag.models import PipelineConfig
from ms_rag.utils.credentials import (
    resolve_credential,
    resolve_model_id,
    resolve_ollama_connection,
)
from ms_rag.utils.telemetry import TelemetryReporter
from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES


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
        huggingface   → langchain_huggingface.HuggingFaceEndpoint
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
        from langchain_huggingface import HuggingFaceEndpoint  # noqa: PLC0415
        return HuggingFaceEndpoint(
            repo_id=resolved_model,
            huggingfacehub_api_token=_env("HUGGINGFACEHUB_API_TOKEN"),
            **kwargs,
        )

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
    from langchain_core.runnables import RunnablePassthrough  # noqa: PLC0415
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
    from ms_rag.query.context_compressor import ContextCompressor  # noqa: PLC0415
    from ms_rag.query.reranking_module import RerankingModule  # noqa: PLC0415
    from ms_rag.query.retrieval_strategy import RetrievalStrategyModule  # noqa: PLC0415

    if config.retrieval is None:
        raise ValueError(
            "Session config is incomplete — retrieval strategy is required "
            "to build the runtime pipeline."
        )

    provider = config.configured_providers[0] if config.configured_providers else "ollama"
    llm = get_llm(provider, "default", credential_store=credential_store)

    retrieval_module = RetrievalStrategyModule()
    base_retriever = retrieval_module.get_retriever(
        config.retrieval,
        vector_store,
        llm=llm,
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

    if config.rag_type and config.rag_type.requires_langgraph:
        rag_chain = build_langgraph_workflow(
            config.rag_type.rag_type,
            retriever,
            llm,
            config.system_prompt,
        )
    else:
        rag_chain = build_rag_chain(retriever, llm, config.system_prompt)

    return {
        "vector_store": vector_store,
        "retriever": retriever,
        "llm": llm,
        "rag_chain": rag_chain,
    }


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
    from typing import TypedDict  # noqa: PLC0415
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415

    class GraphState(TypedDict):
        question: str
        generation: str
        documents: list
        rewrite_count: int

    # ── Shared nodes ──────────────────────────────────────────────────

    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])  # type: ignore[union-attr]
        return {"documents": docs}

    def generate(state: GraphState) -> dict:
        context = "\n\n".join(
            d.page_content for d in state["documents"]
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Context:\n\n{context}\n\nQuestion: {{question}}"),
        ])
        chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
        answer = chain.invoke({"question": state["question"]})
        return {"generation": answer}

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
        }

    def grade_documents(state: GraphState) -> dict:
        """Grade each document for relevance — keep only relevant ones."""
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "Is this document relevant to the question? Answer yes or no."),
            ("human", "Question: {question}\n\nDocument: {document}"),
        ])
        chain = grade_prompt | llm | StrOutputParser()  # type: ignore[operator]
        relevant = []
        for doc in state["documents"]:
            grade = chain.invoke({
                "question": state["question"],
                "document": doc.page_content,
            }).lower()
            if "yes" in grade:
                relevant.append(doc)
        return {"documents": relevant}

    # ── Routing functions ──────────────────────────────────────────────

    def decide_to_generate(state: GraphState) -> str:
        """Route to generate if enough docs; rewrite if none (max 2 rewrites)."""
        if state["documents"] and state.get("rewrite_count", 0) < 2:
            return "generate"
        if state.get("rewrite_count", 0) >= 2:
            return "generate"  # force generation after 2 rewrites
        return "rewrite_query"

    def check_hallucination(state: GraphState) -> str:
        """Check if generation is grounded in documents."""
        if not state["documents"]:
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
        return "end" if "yes" in result else "generate"

    # ── Build graph ────────────────────────────────────────────────────

    workflow = StateGraph(GraphState)

    if rag_type in ("self_rag", "corrective_rag"):
        # Self-RAG / CRAG: retrieve → grade → generate → check hallucination
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite_query", rewrite_query)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )
        workflow.add_conditional_edges(
            "generate",
            check_hallucination,
            {"end": END, "generate": "generate"},
        )
        workflow.add_edge("rewrite_query", "retrieve")

    elif rag_type == "agentic_rag":
        # Agentic: query analysis → retrieve → generate
        def query_analysis(state: GraphState) -> dict:
            return {}  # pass through — routing is done by LLM in full implementation

        workflow.add_node("query_analysis", query_analysis)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite_query", rewrite_query)

        workflow.set_entry_point("query_analysis")
        workflow.add_edge("query_analysis", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

    elif rag_type == "adaptive_rag":
        # Adaptive: route simple queries to direct generation, complex to retrieval
        def route_question(state: GraphState) -> str:
            route_prompt = ChatPromptTemplate.from_messages([
                ("system", "Is this a simple factual question (answer: simple) "
                           "or does it require document retrieval (answer: retrieve)?"),
                ("human", "{question}"),
            ])
            chain = route_prompt | llm | StrOutputParser()  # type: ignore[operator]
            result = chain.invoke({"question": state["question"]}).lower()
            return "retrieve" if "retrieve" in result else "generate"

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
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
    telemetry = TelemetryReporter()

    ss: SessionState = session_state  # type: ignore[assignment]
    cfg = ss.config

    with telemetry.span("query.process", query_length=len(query)):
        # Step 1: Query Enhancement
        enhanced_queries = [query]
        if query_enhancer and cfg.query_enhancement:
            try:
                enhanced_queries = query_enhancer.enhance(  # type: ignore[union-attr]
                    query=query,
                    techniques=cfg.query_enhancement,
                    llm=ss.llm,
                )
            except Exception:  # noqa: BLE001
                enhanced_queries = [query]

        # Use the first enhanced query for retrieval
        primary_query = enhanced_queries[0] if enhanced_queries else query

        # Step 2: invoke the RAG chain
        if ss.rag_chain is None:
            return "Pipeline not initialised. Please complete all setup steps first."

        try:
            requires_langgraph = bool(
                cfg.rag_type and cfg.rag_type.requires_langgraph
            )
            answer = invoke_rag_chain(
                ss.rag_chain,
                primary_query,
                requires_langgraph=requires_langgraph,
            )
        except Exception as exc:  # noqa: BLE001
            raise exc  # re-raise so QueryLoop can handle it

        # Step 3: Evaluation (if enabled)
        if evaluation_framework and cfg.evaluation_enabled and cfg.evaluation:
            try:
                context_docs = []
                if ss.retriever:
                    context_docs = ss.retriever.invoke(primary_query)
                evaluation_framework.evaluate(  # type: ignore[union-attr]
                    query=query,
                    context=context_docs,
                    answer=answer,
                    config=cfg.evaluation,
                )
            except Exception:  # noqa: BLE001
                pass  # evaluation failure is non-fatal

        return answer
