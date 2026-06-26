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
        from ms_rag.config.credential_manager import CredentialManager  # noqa: PLC0415
        from ms_rag.llm.llm_integration import rebuild_session_runtime  # noqa: PLC0415
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
            console.print(  # type: ignore[union-attr]
                "[bold cyan]  Re-enter credentials to rebuild the live pipeline.[/bold cyan]"
            )
            cred_manager = CredentialManager(credential_store=credential_store)
            while True:
                for pid in config.configured_providers:
                    creds = cred_manager.collect_credentials(pid)
                    cred_manager.store(pid, creds)
                if cred_manager.display_summary_and_confirm():
                    break
                credential_store.clear()
                console.print("[yellow]  Please re-enter credentials.[/yellow]")
            runtime = rebuild_session_runtime(config, credential_store)
            session = SessionState(
                config=config,
                credentials=credential_store,
                **runtime,
            )
            eval_framework = None
            if config.evaluation_enabled:
                from ms_rag.evaluation.evaluation_framework import EvaluationFramework  # noqa: PLC0415
                eval_framework = EvaluationFramework(credential_store=credential_store)
            _run_query_loop(session, eval_framework=eval_framework)
            return
        except SessionLoadError as exc:
            console.print(f"[yellow]  ⚠ {exc} Falling back to interactive setup.[/yellow]\n")
        except ValueError as exc:
            console.print(f"[yellow]  ⚠ {exc} Falling back to interactive setup.[/yellow]\n")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]  ⚠ Failed to rebuild session: {exc}[/yellow]\n")

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
    from ms_rag.ui.prompts import (  # noqa: PLC0415
        get_console,
        print_error,
        print_success,
        print_warning,
        prompt_document_sources,
        prompt_select,
    )

    con = get_console() if console is None else console  # type: ignore[assignment]
    config = PipelineConfig()

    # Step 2: Credentials
    cred_manager = CredentialManager(credential_store=credential_store)
    while True:
        selected_providers = cred_manager.prompt_providers()
        if not selected_providers:
            print_warning(con, "Please select at least one LLM provider.")
            continue
        for pid in selected_providers:
            creds = cred_manager.collect_credentials(pid)
            cred_manager.store(pid, creds)
        if cred_manager.display_summary_and_confirm():
            config.configured_providers = selected_providers
            break
        credential_store.clear()
        print_warning(con, "Let's re-configure your providers.")

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

    # Connection test — must succeed before ingestion (Req 9.4-9.5)
    config.vector_db = _ensure_vector_db_connection(db_connector, config.vector_db, con)

    config.document_sources = prompt_document_sources(console=con)

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
    from ms_rag.llm.llm_integration import rebuild_session_runtime  # noqa: PLC0415

    runtime = rebuild_session_runtime(config, credential_store)
    retriever = runtime["retriever"]
    llm = runtime["llm"]
    rag_chain = runtime["rag_chain"]
    vector_store = runtime["vector_store"]

    session = SessionState(
        config=config,
        credentials=credential_store,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
        rag_chain=rag_chain,
    )

    _run_query_loop(session, eval_framework=eval_framework if config.evaluation_enabled else None)
    return config


def _ensure_vector_db_connection(
    db_connector: object,
    vector_db_config: object,
    console: object,
) -> object:
    """Test vector DB connection; re-prompt until successful."""
    import questionary  # noqa: PLC0415

    from ms_rag.ui.prompts import print_error, print_success, prompt_select  # noqa: PLC0415

    config = vector_db_config
    while True:
        result = db_connector.test_connection(config)  # type: ignore[union-attr]
        if result.success:
            print_success(console, "Vector database connection successful.")  # type: ignore[arg-type]
            return config

        print_error(
            console,  # type: ignore[arg-type]
            f"Connection failed: {result.error_message}",
        )
        action = prompt_select(
            "  Connection failed. What would you like to do?",
            [
                questionary.Choice("Re-enter credentials", value="creds"),
                questionary.Choice("Re-select vector database", value="reselect"),
            ],
            console=console,  # type: ignore[arg-type]
        )
        if action == "creds":
            config = db_connector.reprompt_credentials(config)  # type: ignore[union-attr]
        else:
            config = db_connector.prompt_and_configure()  # type: ignore[union-attr]


def _run_query_loop(session: SessionState, eval_framework: object | None = None) -> None:
    """Run the interactive query loop."""
    from ms_rag.cli.query_loop import QueryLoop  # noqa: PLC0415
    from ms_rag.llm.llm_integration import process_query  # noqa: PLC0415
    from ms_rag.query.query_enhancer import QueryEnhancer  # noqa: PLC0415
    from ms_rag.session.session_manager import SessionManager  # noqa: PLC0415

    query_enhancer = QueryEnhancer()

    def pipeline(query: str, sess: SessionState) -> str:
        return process_query(
            query,
            sess,
            query_enhancer=query_enhancer,
            evaluation_framework=eval_framework,
        )

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
