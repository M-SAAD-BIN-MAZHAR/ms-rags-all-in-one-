"""MS_RAG CLI main entry point.

Wires together all 16 workflow steps and the code generator.
Launched via `ms-rag` or `python -m ms_rag`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ms_rag.models import CredentialStore, PipelineConfig, SessionState
from ms_rag.ui.banner import display_banner


@click.command()
@click.option(
    "--load",
    "load_path",
    default=None,
    type=click.Path(exists=False),
    help="Load a previously saved session config JSON and skip to the query loop.",
)
def run(load_path: str | None = None) -> None:
    """MS_RAG — Production-Grade RAG Framework Builder.

    Run interactively to build a complete RAG pipeline step by step,
    or use --load to resume a saved session.
    """
    from rich.console import Console  # noqa: PLC0415

    console = Console()

    # Step 1: Banner
    display_banner(console)

    credential_store = CredentialStore()

    # --load: skip setup and jump to query loop
    if load_path:
        from ms_rag.session.session_manager import SessionManager  # noqa: PLC0415
        from ms_rag.utils.exceptions import SessionLoadError  # noqa: PLC0415
        manager = SessionManager()
        try:
            config = manager.load(Path(load_path))
            console.print(
                f"\n[green]  ✓ Session loaded from {load_path}[/green]\n"
                f"  RAG Type: {config.rag_type.display_name if config.rag_type else 'unknown'}\n"
                f"  Providers: {', '.join(config.configured_providers) or 'none'}\n"
            )
            session = SessionState(config=config, credentials=credential_store)
            _run_query_loop(session)
            return
        except SessionLoadError as exc:
            console.print(f"[yellow]  ⚠ {exc} Falling back to interactive setup.[/yellow]\n")

    # Interactive setup: Steps 2-16
    config = _run_interactive_setup(credential_store, console)

    # Generate code
    _generate_and_offer_save(config, console)


def _run_interactive_setup(credential_store: CredentialStore, console: object) -> PipelineConfig:
    """Run Steps 2-16 interactively and return the complete PipelineConfig."""
    from ms_rag.config.credential_manager import CredentialManager  # noqa: PLC0415
    from ms_rag.workflow.rag_type_selector import RAGTypeSelector  # noqa: PLC0415
    from ms_rag.ingestion.document_type_selector import DocumentTypeSelector  # noqa: PLC0415
    from ms_rag.ingestion.loader_selector import LoaderSelector  # noqa: PLC0415
    from ms_rag.workflow.chunking_configurator import ChunkingConfigurator  # noqa: PLC0415
    from ms_rag.ingestion.vectorization_module import VectorizationModule  # noqa: PLC0415
    from ms_rag.ingestion.vectordb_connector import VectorDBConnector  # noqa: PLC0415
    from ms_rag.ingestion.ingestion_orchestrator import IngestionOrchestrator  # noqa: PLC0415
    from ms_rag.query.query_enhancer import QueryEnhancer  # noqa: PLC0415
    from ms_rag.query.retrieval_strategy import RetrievalStrategyModule  # noqa: PLC0415
    from ms_rag.query.reranking_module import RerankingModule  # noqa: PLC0415
    from ms_rag.query.context_compressor import ContextCompressor  # noqa: PLC0415
    from ms_rag.workflow.system_prompt_configurator import SystemPromptConfigurator  # noqa: PLC0415
    from ms_rag.evaluation.evaluation_framework import EvaluationFramework  # noqa: PLC0415
    import questionary  # noqa: PLC0415

    config = PipelineConfig()

    # Step 2: Credentials
    cred_manager = CredentialManager(credential_store=credential_store)
    selected_providers = cred_manager.prompt_providers()
    for pid in selected_providers:
        creds = cred_manager.collect_credentials(pid)
        cred_manager.store(pid, creds)
    cred_manager.display_summary_and_confirm()
    config.configured_providers = selected_providers

    # Step 3: RAG Type
    rag_selector = RAGTypeSelector()
    config.rag_type = rag_selector.display_and_select()

    # Step 4: Document Types
    doc_selector = DocumentTypeSelector()
    config.document_types = doc_selector.display_checklist()

    # Step 5: Loaders
    loader_selector = LoaderSelector(credential_store=credential_store)
    config.loader_map = loader_selector.display_filtered_loaders(config.document_types)

    # Steps 6-7: Chunking
    chunking_configurator = ChunkingConfigurator()
    config.chunking = chunking_configurator.configure()

    # Step 8: Embedding Model
    vectorization = VectorizationModule()
    config.embedding_model = vectorization.display_and_select(config.configured_providers)

    # Step 9: Vector DB + Ingestion
    db_connector = VectorDBConnector(credential_store=credential_store)
    config.vector_db = db_connector.prompt_and_configure()

    # Connection test
    while True:
        result = db_connector.test_connection(config.vector_db)
        if result.success:
            break
        console.print(f"[red]  ✗ Connection failed: {result.error_message}[/red]")  # type: ignore[union-attr]
        retry = questionary.confirm("  Retry with different credentials?", default=True).ask()
        if not retry:
            break

    # Prompt for document sources
    sources_raw: str = questionary.text(
        "  Enter document paths/directories/URLs (comma-separated):",
        default="./docs",
    ).ask()
    config.document_sources = [s.strip() for s in sources_raw.split(",") if s.strip()]

    # Ingest
    embeddings = vectorization.get_embeddings(config.embedding_model, credential_store)
    vector_store = db_connector.get_vector_store(config.vector_db, embeddings)

    orchestrator = IngestionOrchestrator()
    ingestion_result = orchestrator.ingest(
        sources=config.document_sources,
        loader_map=config.loader_map,
        chunking_config=config.chunking,
        embedding_model=config.embedding_model,
        vector_db=config.vector_db,
        vector_store=vector_store,
    )
    config.ingestion_result = ingestion_result

    # Step 11: Query Enhancement (configured before query, applied at query time)
    query_enhancer = QueryEnhancer()
    config.query_enhancement = query_enhancer.configure(config.configured_providers)

    # Step 12: Retrieval Strategy
    retrieval_module = RetrievalStrategyModule()
    config.retrieval = retrieval_module.configure()

    # Step 13: Reranking
    reranking_module = RerankingModule(credential_store=credential_store)
    reranking_config = reranking_module.configure(config.retrieval.top_k)
    if reranking_config:
        config.reranking = reranking_config
        config.reranking_enabled = True

    # Step 14: Context Compression
    compressor = ContextCompressor()
    compression_config = compressor.configure(config.configured_providers)
    if compression_config:
        config.compression = compression_config
        config.compression_enabled = True

    # Step 15: System Prompt
    prompt_configurator = SystemPromptConfigurator()
    config.system_prompt = prompt_configurator.configure()

    # Step 16: Evaluation
    eval_framework = EvaluationFramework(credential_store=credential_store)
    eval_config = eval_framework.configure()
    if eval_config:
        config.evaluation = eval_config
        config.evaluation_enabled = True

    # Build retriever and RAG chain
    retriever = retrieval_module.get_retriever(config.retrieval, vector_store)

    from ms_rag.llm.llm_integration import build_rag_chain, build_langgraph_workflow, get_llm  # noqa: PLC0415
    from ms_rag.workflow.rag_type_selector import LANGGRAPH_TYPES  # noqa: PLC0415

    provider = config.configured_providers[0] if config.configured_providers else "ollama"
    llm = get_llm(provider, "default", credential_store=credential_store)

    if config.rag_type and config.rag_type.requires_langgraph:
        rag_chain = build_langgraph_workflow(
            config.rag_type.rag_type, retriever, llm, config.system_prompt
        )
    else:
        rag_chain = build_rag_chain(retriever, llm, config.system_prompt)

    session = SessionState(
        config=config,
        credentials=credential_store,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
        rag_chain=rag_chain,
    )

    _run_query_loop(session)
    return config


def _run_query_loop(session: SessionState) -> None:
    """Run the interactive query loop."""
    from ms_rag.cli.query_loop import QueryLoop  # noqa: PLC0415
    from ms_rag.llm.llm_integration import process_query  # noqa: PLC0415
    from ms_rag.session.session_manager import SessionManager  # noqa: PLC0415

    def pipeline(query: str, sess: SessionState) -> str:
        return process_query(query, sess)

    loop = QueryLoop(
        query_pipeline=pipeline,
        session_manager=SessionManager(),
    )
    loop.run(session)


def _generate_and_offer_save(config: PipelineConfig, console: object) -> None:
    """Generate code and offer to save."""
    from ms_rag.codegen.code_generator import CodeGenerator  # noqa: PLC0415
    generator = CodeGenerator()
    code = generator.generate(config)
    generator.display_and_offer_save(code, console)
