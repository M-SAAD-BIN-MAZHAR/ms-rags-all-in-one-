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
from ms_rag.ui.prompts import prompt_telemetry_configuration
from ms_rag.utils.logging import get_logger
from ms_rag.utils.telemetry import TelemetryReporter


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
    logger = get_logger()

    # Step 1: Banner
    display_banner(console)
    telemetry_config = prompt_telemetry_configuration(console=console)
    telemetry = TelemetryReporter(telemetry_config)
    telemetry.record_event("cli.start", "MS_RAG CLI started")

    credential_store = CredentialStore()

    # --load: skip setup and jump to query loop
    if load_path:
        from ms_rag.config.credential_manager import CredentialManager  # noqa: PLC0415
        from ms_rag.llm.llm_integration import rebuild_session_runtime  # noqa: PLC0415
        from ms_rag.session.session_manager import SessionManager  # noqa: PLC0415
        from ms_rag.utils.exceptions import SessionLoadError  # noqa: PLC0415

        manager = SessionManager()
        try:
            with telemetry.span("session.load", load_path=load_path):
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
            telemetry.record_event("session.load", "Session loaded", load_path=load_path)
            with telemetry.span("session.rebuild", load_path=load_path):
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
            with telemetry.span("query_loop.run", load_path=load_path):
                _run_query_loop(session, eval_framework=eval_framework)
            return
        except SessionLoadError as exc:
            console.print(f"[yellow]  ⚠ {exc} Falling back to interactive setup.[/yellow]\n")
            telemetry.record_error("session.load_failed", str(exc), load_path=load_path)
        except ValueError as exc:
            console.print(f"[yellow]  ⚠ {exc} Falling back to interactive setup.[/yellow]\n")
            telemetry.record_error("session.load_failed", str(exc), load_path=load_path)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]  ⚠ Failed to rebuild session: {exc}[/yellow]\n")
            telemetry.record_error("session.rebuild_failed", str(exc), load_path=load_path)

    # Interactive setup: Steps 2-16
    with telemetry.span("cli.interactive_setup"):
        config = _run_interactive_setup(credential_store, console)

    # Generate code
    _generate_and_offer_save(config, console)
    logger.info("Interactive setup completed", extra={"event": "cli.setup_complete"})


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
        print_hint,
        print_success,
        print_warning,
        prompt_document_sources,
        prompt_confirm,
        prompt_required_confirm,
        prompt_select,
    )

    con = get_console() if console is None else console  # type: ignore[assignment]
    telemetry = TelemetryReporter()
    config = PipelineConfig()

    # Step 2: Credentials
    with telemetry.span("setup.credentials"):
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
                telemetry.record_event("credentials.confirmed", "Providers confirmed", providers=selected_providers)
                break
            credential_store.clear()
            print_warning(con, "Let's re-configure your providers.")

    # Step 3: RAG Type
    with telemetry.span("setup.rag_type"):
        rag_selector = RAGTypeSelector()
        config.rag_type = rag_selector.display_and_select()
        telemetry.record_event("rag_type.selected", "RAG type selected", rag_type=config.rag_type.rag_type if config.rag_type else "")

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
    with telemetry.span("setup.embedding"):
        vectorization = VectorizationModule()
        config.embedding_model = vectorization.display_and_select(config.configured_providers)
        telemetry.record_event(
            "embedding.selected",
            "Embedding model selected",
            provider=config.embedding_model.provider if config.embedding_model else "",
            model_id=config.embedding_model.model_id if config.embedding_model else "",
        )

    # Step 9: Vector DB + Ingestion
    with telemetry.span("setup.vector_db"):
        db_connector = VectorDBConnector(credential_store=credential_store)
        config.vector_db = db_connector.prompt_and_configure(config.embedding_model)
        telemetry.record_event(
            "vector_db.selected",
            "Vector database selected",
            db_type=config.vector_db.db_type if config.vector_db else "",
            collection_name=config.vector_db.collection_name if config.vector_db else "",
        )

    # Connection test — must succeed before ingestion (Req 9.4-9.5)
    with telemetry.span("setup.vector_db_connection"):
        config.vector_db = _ensure_vector_db_connection(db_connector, config.vector_db, con)

    with telemetry.span("setup.sources"):
        config.document_sources = prompt_document_sources(console=con)

    _display_ingestion_review(config, con)
    prompt_required_confirm("Start ingestion with the settings above?", console=con)
    telemetry.record_event("ingestion.confirmed", "User approved ingestion start")

    with telemetry.span("setup.ingestion"):
        embeddings, vector_store, ingestion_result = _run_ingestion_with_recovery(
            config,
            credential_store,
            vectorization,
            db_connector,
            IngestionOrchestrator(),
            con,
        )
    config.ingestion_result = ingestion_result
    if ingestion_result.chunk_count == 0:
        print_warning(con, "No chunks were stored. Query quality will be poor until documents are ingested.")
        if not prompt_confirm("Continue setup anyway?", default=False, console=con):
            raise click.ClickException("Setup stopped before query loop because ingestion stored no chunks.")
    telemetry.record_event(
        "ingestion.completed",
        "Ingestion completed",
        chunk_count=ingestion_result.chunk_count,
        failed_documents=len(ingestion_result.failed_documents),
    )

    # Step 10: Query Enhancement (configured before query, applied at query time)
    with telemetry.span("setup.query_enhancement"):
        query_enhancer = QueryEnhancer()
        config.query_enhancement = query_enhancer.configure(config.configured_providers)

    # Step 11: Retrieval Strategy
    with telemetry.span("setup.retrieval"):
        retrieval_module = RetrievalStrategyModule()
        config.retrieval = retrieval_module.configure()

    # Step 12: Reranking
    with telemetry.span("setup.reranking"):
        reranking_module = RerankingModule(credential_store=credential_store)
        reranking_config = reranking_module.configure(config.retrieval.top_k)
        if reranking_config:
            config.reranking = reranking_config
            config.reranking_enabled = True

    # Step 13: Context Compression
    with telemetry.span("setup.compression"):
        compressor = ContextCompressor()
        compression_config = compressor.configure(config.configured_providers)
        if compression_config:
            config.compression = compression_config
            config.compression_enabled = True

    # Step 14: System Prompt
    with telemetry.span("setup.system_prompt"):
        prompt_configurator = SystemPromptConfigurator()
        config.system_prompt = prompt_configurator.configure()

    # Step 15: Evaluation
    with telemetry.span("setup.evaluation"):
        eval_framework = EvaluationFramework(credential_store=credential_store)
        eval_config = eval_framework.configure()
        if eval_config:
            config.evaluation = eval_config
            config.evaluation_enabled = True

    # Step 16: Build retriever/RAG chain, then enter the live query loop.
    from ms_rag.llm.llm_integration import build_session_runtime_from_vector_store  # noqa: PLC0415

    with telemetry.span("setup.runtime"):
        runtime = _build_runtime_with_recovery(
            config,
            credential_store,
            vector_store,
            embeddings,
            build_session_runtime_from_vector_store,
            con,
        )
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


def _display_ingestion_review(config: PipelineConfig, console: object) -> None:
    """Show a final human-readable review before writing vectors."""
    from rich.table import Table  # noqa: PLC0415

    table = Table(title="Ready to Ingest", border_style="cyan")
    table.add_column("Setting", style="bold white")
    table.add_column("Value", style="green")
    table.add_row("Embedding", config.embedding_model.model_id if config.embedding_model else "not selected")
    table.add_row(
        "Embedding Dimension",
        str(config.vector_db.dimension) if config.vector_db and config.vector_db.dimension else "custom/unknown",
    )
    table.add_row(
        "Vector DB",
        f"{config.vector_db.db_type} / {config.vector_db.collection_name}" if config.vector_db else "not selected",
    )
    table.add_row("Sources", "\n".join(config.document_sources) or "none")
    table.add_row(
        "Note",
        "Existing indexes must use the same embedding dimension. Use a new collection when changing models.",
    )
    console.print(table)  # type: ignore[union-attr]


def _run_ingestion_with_recovery(
    config: PipelineConfig,
    credential_store: CredentialStore,
    vectorization: object,
    db_connector: object,
    orchestrator: object,
    console: object,
) -> tuple[object, object, object]:
    """Build embeddings/store and ingest, giving the user recovery choices."""
    import questionary  # noqa: PLC0415
    from ms_rag.ui.prompts import print_error, print_hint, prompt_select  # noqa: PLC0415

    while True:
        try:
            console.print("[bold cyan]  Preparing embeddings and vector store...[/bold cyan]")  # type: ignore[union-attr]
            embeddings = vectorization.get_embeddings(config.embedding_model, credential_store)  # type: ignore[union-attr,arg-type]
            vector_store = db_connector.get_vector_store(config.vector_db, embeddings)  # type: ignore[union-attr,arg-type]
            console.print("[bold cyan]  Loading, chunking, embedding, and storing documents...[/bold cyan]")  # type: ignore[union-attr]
            ingestion_result = orchestrator.ingest(  # type: ignore[union-attr]
                sources=config.document_sources,
                loader_map=config.loader_map,
                chunking_config=config.chunking,
                embedding_model=config.embedding_model,
                vector_db=config.vector_db,
                vector_store=vector_store,
            )
            return embeddings, vector_store, ingestion_result
        except Exception as exc:  # noqa: BLE001
            print_error(console, f"Ingestion setup failed: {type(exc).__name__}: {exc}")  # type: ignore[arg-type]
            print_hint(
                console,  # type: ignore[arg-type]
                "Common causes: wrong embedding model for an existing index, missing DB credentials, "
                "unreachable local service, or incompatible collection dimension.",
            )
            action = prompt_select(
                "  What would you like to do?",
                [
                    questionary.Choice("Retry with the same settings", value="retry"),
                    questionary.Choice("Change embedding model", value="embedding"),
                    questionary.Choice("Change vector database settings", value="vectordb"),
                    questionary.Choice("Abort setup", value="abort"),
                ],
                console=console,  # type: ignore[arg-type]
            )
            if action == "retry":
                continue
            if action == "embedding":
                config.embedding_model = vectorization.display_and_select(config.configured_providers)  # type: ignore[union-attr]
                config.vector_db = db_connector.prompt_and_configure(config.embedding_model)  # type: ignore[union-attr]
                config.vector_db = _ensure_vector_db_connection(db_connector, config.vector_db, console)
                _display_ingestion_review(config, console)
                continue
            if action == "vectordb":
                config.vector_db = db_connector.prompt_and_configure(config.embedding_model)  # type: ignore[union-attr]
                config.vector_db = _ensure_vector_db_connection(db_connector, config.vector_db, console)
                _display_ingestion_review(config, console)
                continue
            raise click.ClickException("Setup aborted during ingestion.")


def _build_runtime_with_recovery(
    config: PipelineConfig,
    credential_store: CredentialStore,
    vector_store: object,
    embeddings: object,
    builder: object,
    console: object,
) -> dict[str, object]:
    """Build query runtime with a clear error if final wiring fails."""
    try:
        console.print("[bold cyan]  Building retriever, LLM, and RAG chain...[/bold cyan]")  # type: ignore[union-attr]
        runtime = builder(  # type: ignore[operator]
            config,
            credential_store,
            vector_store=vector_store,
            embeddings=embeddings,
        )
        console.print("[green]  ✓ Runtime pipeline ready.[/green]")  # type: ignore[union-attr]
        return runtime
    except Exception as exc:  # noqa: BLE001
        from ms_rag.ui.prompts import print_error  # noqa: PLC0415

        print_error(console, f"Failed to build query runtime: {type(exc).__name__}: {exc}")  # type: ignore[arg-type]
        raise click.ClickException(
            "Runtime build failed. Check provider credentials, retrieval settings, and vector DB compatibility."
        ) from exc


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
