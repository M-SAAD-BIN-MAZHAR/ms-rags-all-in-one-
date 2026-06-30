"""MS-RAGS(ALL-IN-ONE) CLI main entry point.

Wires together all 16 workflow steps and the code generator.
Launched via `ms-rags`, `ms-rag`, or `python -m ms_rag`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ms_rag.models import CredentialStore, PipelineConfig, SessionState
from ms_rag.ui.banner import display_banner
from ms_rag.ui.prompts import prompt_telemetry_configuration
from ms_rag.utils.logging import get_logger, install_warning_renderer
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
    """MS-RAGS(ALL-IN-ONE) — Production-Grade RAG Framework Builder.

    Run interactively to build a complete RAG pipeline step by step,
    or use --load to resume a saved session.
    """
    from rich.console import Console  # noqa: PLC0415

    console = Console()
    logger = get_logger()
    install_warning_renderer(console)

    # Step 1: Banner
    display_banner(console)
    telemetry_config = prompt_telemetry_configuration(console=console)
    telemetry = TelemetryReporter(telemetry_config)
    telemetry.record_event("cli.start", "MS-RAGS(ALL-IN-ONE) CLI started")

    credential_store = CredentialStore()

    # --load: skip setup and jump to query loop
    if load_path:
        from ms_rag.config.credential_manager import CredentialManager  # noqa: PLC0415
        from ms_rag.ingestion.vectordb_connector import VECTOR_DB_MAP, VectorDBConnector  # noqa: PLC0415
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
                credential_providers = list(dict.fromkeys(
                    list(config.configured_providers)
                    + ([config.llm_model.provider] if config.llm_model else [])
                ))
                for pid in credential_providers:
                    creds = cred_manager.collect_credentials(pid)
                    cred_manager.store(pid, creds)
                if cred_manager.display_summary_and_confirm():
                    break
                credential_store.clear()
                console.print("[yellow]  Please re-enter credentials.[/yellow]")
            if config.vector_db:
                db_info = VECTOR_DB_MAP.get(config.vector_db.db_type)
                if db_info and db_info.credential_fields:
                    console.print(
                        "[bold cyan]  Re-enter vector database credentials for this saved session.[/bold cyan]"
                    )
                    db_connector = VectorDBConnector(credential_store=credential_store)
                    config.vector_db = db_connector.reprompt_credentials(config.vector_db)
            if config.keyword_store:
                from ms_rag.ingestion.keyword_store import KEYWORD_STORE_MAP, KeywordStoreConnector  # noqa: PLC0415
                from ms_rag.ui.prompts import prompt_text  # noqa: PLC0415

                store_info = KEYWORD_STORE_MAP.get(config.keyword_store.store_type)
                if store_info and store_info.credential_fields:
                    console.print(
                        "[bold cyan]  Re-enter keyword store credentials for this saved session.[/bold cyan]"
                    )
                    params = dict(config.keyword_store.connection_params)
                    for field_name in store_info.credential_fields:
                        value = prompt_text(
                            f"  {field_name}:",
                            required=True,
                            secret=True,
                            console=console,
                        )
                        params[field_name] = str(value)
                        credential_store.set(config.keyword_store.store_type, field_name, str(value))
                    for field_name in store_info.optional_fields:
                        value = prompt_text(
                            f"  {field_name} (optional):",
                            required=False,
                            secret=any(token in field_name for token in ("KEY", "PASSWORD", "TOKEN", "SECRET")),
                            console=console,
                        )
                        if value:
                            params[field_name] = str(value)
                            credential_store.set(config.keyword_store.store_type, field_name, str(value))
                    config.keyword_store.connection_params = params
                    KeywordStoreConnector(credential_store).test_connection(config.keyword_store)
            if config.graph_store:
                from ms_rag.ingestion.graph_store import GRAPH_STORE_MAP, GraphStoreConnector  # noqa: PLC0415

                graph_info = GRAPH_STORE_MAP.get(config.graph_store.store_type)
                if graph_info and graph_info.credential_fields:
                    console.print(
                        "[bold cyan]  Re-enter graph database credentials for this saved session.[/bold cyan]"
                    )
                    config.graph_store = GraphStoreConnector(credential_store).reprompt_credentials(config.graph_store)
            if config.agent_tools:
                from ms_rag.ui.prompts import prompt_text  # noqa: PLC0415

                if "web_search" in config.agent_tools.enabled_tools:
                    web_settings = config.agent_tools.tool_settings.get("web_search", {})
                    provider = web_settings.get("provider", "tavily")
                    key_name = "TAVILY_API_KEY" if provider == "tavily" else "BRAVE_SEARCH_API_KEY"
                    console.print("[bold cyan]  Re-enter web search credentials for agent tools.[/bold cyan]")
                    value = prompt_text(f"  {key_name}:", required=True, secret=True, console=console)
                    credential_store.set("web_search", key_name, str(value))
                if "api_request" in config.agent_tools.enabled_tools:
                    api_settings = config.agent_tools.tool_settings.get("api_request", {})
                    auth_env = str(api_settings.get("auth_env_var") or "")
                    if auth_env:
                        console.print("[bold cyan]  Re-enter API Request Tool credential.[/bold cyan]")
                        value = prompt_text(f"  {auth_env}:", required=True, secret=True, console=console)
                        credential_store.set("api_request", auth_env, str(value))
            telemetry.record_event("session.load", "Session loaded", load_path=load_path)
            with telemetry.span("session.rebuild", load_path=load_path):
                runtime = rebuild_session_runtime(config, credential_store)
            session = SessionState(
                config=config,
                credentials=credential_store,
                **runtime,
            )
            _display_selected_architecture(config, console)
            eval_framework = None
            if config.evaluation_enabled:
                from ms_rag.evaluation.evaluation_framework import EvaluationFramework  # noqa: PLC0415
                eval_framework = EvaluationFramework(credential_store=credential_store)
            try:
                with telemetry.span("query_loop.run", load_path=load_path):
                    _run_query_loop(session, eval_framework=eval_framework)
            finally:
                _close_session_runtime(session)
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
    from ms_rag.workflow.rag_presets import get_rag_preset  # noqa: PLC0415
    from ms_rag.ingestion.document_type_selector import DocumentTypeSelector  # noqa: PLC0415
    from ms_rag.ingestion.loader_selector import LoaderSelector  # noqa: PLC0415
    from ms_rag.workflow.chunking_configurator import ChunkingConfigurator  # noqa: PLC0415
    from ms_rag.ingestion.vectorization_module import VectorizationModule  # noqa: PLC0415
    from ms_rag.ingestion.vectordb_connector import VectorDBConnector  # noqa: PLC0415
    from ms_rag.ingestion.keyword_store import KeywordStoreConnector, retrieval_needs_keyword_store  # noqa: PLC0415
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
                config.llm_model = _prompt_llm_model_selection(
                    selected_providers,
                    credential_store,
                    cred_manager,
                    con,
                )
                telemetry.record_event("credentials.confirmed", "Providers confirmed", providers=selected_providers)
                telemetry.record_event(
                    "llm.selected",
                    "Generation LLM selected",
                    provider=config.llm_model.provider if config.llm_model else "",
                    model_id=config.llm_model.model_id if config.llm_model else "",
                )
                break
            credential_store.clear()
            print_warning(con, "Let's re-configure your providers.")

    # Step 3: RAG Type
    with telemetry.span("setup.rag_type"):
        rag_selector = RAGTypeSelector()
        config.rag_type = rag_selector.display_and_select()
        rag_preset = get_rag_preset(config.rag_type.rag_type if config.rag_type else None)
        con.print(f"[cyan]  Preset applied:[/cyan] {rag_preset.summary}")
        for note in rag_preset.notes:
            con.print(f"[yellow]  Note:[/yellow] {note}")
        telemetry.record_event("rag_type.selected", "RAG type selected", rag_type=config.rag_type.rag_type if config.rag_type else "")

    if config.rag_type and config.rag_type.rag_type in {"agentic_rag", "corrective_rag"}:
        from ms_rag.agent.tool_configurator import AgentToolConfigurator  # noqa: PLC0415

        with telemetry.span("setup.agent_tools"):
            config.agent_tools = AgentToolConfigurator(credential_store).configure()
            telemetry.record_event(
                "agent_tools.configured",
                "Agentic tools configured",
                tools=config.agent_tools.enabled_tools if config.agent_tools else [],
            )

    if config.rag_type and config.rag_type.rag_type == "graphrag":
        from ms_rag.ingestion.graph_store import GraphStoreConnector  # noqa: PLC0415

        with telemetry.span("setup.graph_store"):
            config.graph_store = GraphStoreConnector(credential_store).prompt_and_configure()
            telemetry.record_event(
                "graph_store.configured",
                "GraphRAG graph store configured",
                store_type=config.graph_store.store_type,
                query_mode=config.graph_store.query_mode,
            )

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
        config.embedding_model = vectorization.display_and_select(
            config.configured_providers,
            credential_store=credential_store,
        )
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
        config.vector_db = _ensure_vector_db_connection(
            db_connector,
            config.vector_db,
            con,
            embedding_model=config.embedding_model,
        )

    with telemetry.span("setup.sources"):
        config.document_sources = prompt_document_sources(console=con)

    _display_runtime_dependency_report(config, con)
    _display_ingestion_review(config, con)
    prompt_required_confirm("Start ingestion with the settings above?", console=con)
    telemetry.record_event("ingestion.confirmed", "User approved ingestion start")

    with telemetry.span("setup.ingestion"):
        embeddings, vector_store, ingestion_result = _run_ingestion_with_recovery(
            config,
            credential_store,
            vectorization,
            db_connector,
            IngestionOrchestrator(credential_store=credential_store),
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

    if config.rag_type and config.rag_type.rag_type == "graphrag":
        from ms_rag.ingestion.graph_store import GraphStoreConnector  # noqa: PLC0415
        from ms_rag.llm.llm_integration import get_llm  # noqa: PLC0415

        with telemetry.span("setup.graphrag_index"):
            if config.graph_store is None:
                raise click.ClickException("GraphRAG requires a configured graph store.")
            if config.llm_model is None:
                raise click.ClickException("GraphRAG requires a selected generation LLM for graph extraction.")
            con.print("[cyan]  Building persistent GraphRAG knowledge graph...[/cyan]")
            graph_llm = get_llm(
                config.llm_model.provider,
                config.llm_model.model_id,
                credential_store=credential_store,
            )
            graph_connector = GraphStoreConnector(credential_store)
            graph = graph_connector.build_graph_index(
                getattr(vector_store, "_ms_rag_chunk_documents", []) or [],
                llm=graph_llm,
            )
            graph_connector.persist_graph(config.graph_store, graph)
            setattr(vector_store, "_ms_rag_graph_index", graph)
            print_success(
                con,
                f"GraphRAG graph built: {len(graph.get('nodes', []))} entities, "
                f"{len(graph.get('edges', []))} relationships, {len(graph.get('communities', []))} communities",
            )

    # Step 10: Query Enhancement (configured before query, applied at query time)
    with telemetry.span("setup.query_enhancement"):
        query_enhancer = QueryEnhancer()
        if rag_preset.allow_query_enhancement_prompt:
            config.query_enhancement = query_enhancer.configure(config.configured_providers)
            config.hyde_llm_provider = query_enhancer.hyde_llm_provider
        else:
            config.query_enhancement = _confirm_query_enhancement_preset(
                rag_preset,
                query_enhancer,
                config.configured_providers,
                con,
            )
            config.hyde_llm_provider = query_enhancer.hyde_llm_provider
            if "hyde" in config.query_enhancement and not config.hyde_llm_provider and config.llm_model:
                config.hyde_llm_provider = config.llm_model.provider

    # Step 11: Retrieval Strategy
    with telemetry.span("setup.retrieval"):
        retrieval_module = RetrievalStrategyModule()
        if rag_preset.allow_retrieval_prompt:
            config.retrieval = retrieval_module.configure()
        else:
            config.retrieval = rag_preset.retrieval
            if config.retrieval:
                print_success(con, f"Retrieval preset: {config.retrieval.strategy} | top_k={config.retrieval.top_k}")
        if retrieval_needs_keyword_store(config.retrieval):
            keyword_connector = KeywordStoreConnector(credential_store=credential_store)
            production_recommended = bool(config.vector_db and config.vector_db.db_type not in {"chroma", "faiss"})
            config.keyword_store = keyword_connector.prompt_and_configure(
                production_recommended=production_recommended,
            )
            keyword_texts = keyword_connector.persist_documents(
                config.keyword_store,
                getattr(vector_store, "_ms_rag_chunk_documents", []) or [],
            )
            setattr(vector_store, "_ms_rag_keyword_corpus", keyword_texts)
            print_success(
                con,
                f"Keyword store ready: {config.keyword_store.store_type} / {config.keyword_store.collection_name} ({len(keyword_texts)} chunks)",
            )

    # Step 12: Reranking
    with telemetry.span("setup.reranking"):
        if rag_preset.allow_reranking_prompt:
            reranking_module = RerankingModule(credential_store=credential_store)
            reranking_config = reranking_module.configure(config.retrieval.top_k)
            if reranking_config:
                config.reranking = reranking_config
                config.reranking_enabled = True
        else:
            con.print("  [dim]Reranking skipped by selected RAG preset.[/dim]")

    # Step 13: Context Compression
    with telemetry.span("setup.compression"):
        compressor = ContextCompressor()
        if rag_preset.allow_compression_prompt:
            compression_config = compressor.configure(config.configured_providers)
        else:
            compression_config = _confirm_compression_preset(
                rag_preset,
                compressor,
                config.configured_providers,
                con,
            )
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

    try:
        _run_query_loop(session, eval_framework=eval_framework if config.evaluation_enabled else None)
    finally:
        _close_session_runtime(session)
    return config


def _prompt_llm_model_selection(
    configured_providers: list[str],
    credential_store: CredentialStore,
    cred_manager: object,
    console: object,
) -> object:
    """Prompt for the generation LLM, with a recovery path for accidental empty selection."""
    import questionary  # noqa: PLC0415
    from ms_rag.models import LLMModelConfig  # noqa: PLC0415
    from ms_rag.ui.prompts import (  # noqa: PLC0415
        print_step,
        print_success,
        print_warning,
        prompt_confirm,
        prompt_select,
        prompt_text,
    )
    from ms_rag.utils.credentials import DEFAULT_LLM_MODELS  # noqa: PLC0415

    def provider_label(provider_id: str) -> str:
        from ms_rag.config.credential_manager import PROVIDER_DISPLAY_NAMES  # noqa: PLC0415

        return PROVIDER_DISPLAY_NAMES.get(provider_id, provider_id)

    while True:
        providers = [pid for pid in configured_providers if pid in DEFAULT_LLM_MODELS]
        if not providers:
            print_warning(
                console,
                "No generation model is available from the selected providers.",
            )
            reconfigure = prompt_confirm(
                "  Configure LLM providers again?",
                default=True,
                console=console,
            )
            if reconfigure:
                credential_store.clear()
                new_providers = cred_manager.prompt_providers()  # type: ignore[union-attr]
                if not new_providers:
                    print_warning(console, "Please select at least one LLM provider.")
                    continue
                for pid in new_providers:
                    creds = cred_manager.collect_credentials(pid)  # type: ignore[union-attr]
                    cred_manager.store(pid, creds)  # type: ignore[union-attr]
                if cred_manager.display_summary_and_confirm():  # type: ignore[union-attr]
                    configured_providers[:] = new_providers
                    continue
                credential_store.clear()
                continue
            raise click.ClickException("A generation LLM model is required to continue.")

        print_step(console, "2b", "Select Generation Model")
        provider = prompt_select(
            "  Which provider should answer user questions?",
            [
                questionary.Choice(
                    f"{provider_label(pid)} — default: {DEFAULT_LLM_MODELS[pid]}",
                    value=pid,
                )
                for pid in providers
            ]
            + [questionary.Choice("Configure providers again", value="__reconfigure__")],
            console=console,
        )
        if provider == "__reconfigure__":
            credential_store.clear()
            new_providers = cred_manager.prompt_providers()  # type: ignore[union-attr]
            if not new_providers:
                print_warning(console, "Please select at least one LLM provider.")
                continue
            for pid in new_providers:
                creds = cred_manager.collect_credentials(pid)  # type: ignore[union-attr]
                cred_manager.store(pid, creds)  # type: ignore[union-attr]
            if cred_manager.display_summary_and_confirm():  # type: ignore[union-attr]
                configured_providers[:] = new_providers
            else:
                credential_store.clear()
            continue

        default_model = DEFAULT_LLM_MODELS[provider]
        model_id = prompt_text(
            "  Generation model ID:",
            default=default_model,
            required=True,
            console=console,
        )
        selected = LLMModelConfig(provider=provider, model_id=str(model_id))
        print_success(
            console,
            f"Generation model: {selected.provider} / {selected.model_id}",
        )
        return selected


def _confirm_query_enhancement_preset(
    rag_preset: object,
    query_enhancer: object,
    configured_providers: list[str],
    console: object,
) -> list[str]:
    """Ask permission before applying a RAG-type query enhancement preset."""
    import questionary  # noqa: PLC0415
    from ms_rag.ui.prompts import print_success, prompt_select  # noqa: PLC0415

    preset = list(getattr(rag_preset, "query_enhancement", []) or [])
    if not preset:
        console.print("  [dim]Query enhancement skipped by selected RAG preset.[/dim]")  # type: ignore[union-attr]
        return []

    console.print("\n[bold cyan]Step 10 — Query Enhancement[/bold cyan]\n")  # type: ignore[union-attr]
    console.print(  # type: ignore[union-attr]
        "[yellow]  The selected RAG type recommends a query enhancement preset.[/yellow]\n"
        f"  Preset: [bold]{', '.join(preset)}[/bold]\n"
        "  You can keep it, edit it, or disable query enhancement for this session."
    )
    action = prompt_select(
        "  What do you want to do with this preset?",
        [
            questionary.Choice("Keep recommended preset", "keep"),
            questionary.Choice("Edit query enhancement choices", "edit"),
            questionary.Choice("Disable query enhancement", "disable"),
        ],
        console=console,  # type: ignore[arg-type]
    )
    if action == "edit":
        selected = query_enhancer.configure(configured_providers)  # type: ignore[union-attr]
        return list(selected or [])
    if action == "disable":
        console.print("  [dim]Query enhancement disabled by user.[/dim]")  # type: ignore[union-attr]
        return []

    print_success(console, f"Query enhancement preset kept: {', '.join(preset)}")  # type: ignore[arg-type]
    return preset


def _confirm_compression_preset(
    rag_preset: object,
    compressor: object,
    configured_providers: list[str],
    console: object,
) -> object | None:
    """Ask permission before applying a RAG-type compression preset."""
    import questionary  # noqa: PLC0415
    from ms_rag.ui.prompts import print_success, prompt_select  # noqa: PLC0415

    preset = getattr(rag_preset, "compression", None)
    if preset is None:
        console.print("  [dim]Context compression skipped by selected RAG preset.[/dim]")  # type: ignore[union-attr]
        return None

    techniques = list(getattr(preset, "techniques", []) or [])
    console.print("\n[bold cyan]Step 13 — Context Compression[/bold cyan]\n")  # type: ignore[union-attr]
    console.print(  # type: ignore[union-attr]
        "[yellow]  The selected RAG type recommends a compression preset.[/yellow]\n"
        f"  Preset: [bold]{', '.join(techniques)}[/bold]\n"
        "  You can keep it, edit it, or disable compression for this session."
    )
    action = prompt_select(
        "  What do you want to do with this preset?",
        [
            questionary.Choice("Keep recommended preset", "keep"),
            questionary.Choice("Edit compression choices", "edit"),
            questionary.Choice("Disable compression", "disable"),
        ],
        console=console,  # type: ignore[arg-type]
    )
    if action == "edit":
        return compressor.configure(configured_providers)  # type: ignore[union-attr]
    if action == "disable":
        console.print("  [dim]Context compression disabled by user.[/dim]")  # type: ignore[union-attr]
        return None

    print_success(console, f"Compression preset kept: {', '.join(techniques)}")  # type: ignore[arg-type]
    return preset


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


def _display_runtime_dependency_report(config: PipelineConfig, console: object) -> None:
    """Tell users which non-Python runtime tools their selected loaders may need."""
    from rich.table import Table  # noqa: PLC0415
    from ms_rag.ingestion.dependency_preflight import (  # noqa: PLC0415
        build_dependency_checks,
        missing_required_dependencies,
    )
    from ms_rag.ui.prompts import print_warning, prompt_confirm  # noqa: PLC0415

    checks = build_dependency_checks(config.loader_map, config.document_sources)
    if not checks:
        return

    table = Table(title="External Tools Needed For Selected Loaders", border_style="yellow")
    table.add_column("Tool", style="bold white")
    table.add_column("Status", style="cyan")
    table.add_column("Needed For", style="white")
    table.add_column("Install Hint", style="green")
    for check in checks:
        status = "installed" if check.installed else ("missing required" if check.required else "missing optional")
        table.add_row(check.tool, status, check.needed_for, check.install_hint)
    console.print(table)  # type: ignore[union-attr]

    missing = missing_required_dependencies(checks)
    if missing:
        names = ", ".join(check.tool for check in missing)
        print_warning(
            console,  # type: ignore[arg-type]
            f"Missing required external tool(s): {names}. Scanned/image PDFs may fail or be skipped.",
        )
        if not prompt_confirm(
            "Continue ingestion anyway? Choose No to install the missing tool(s) first.",
            default=False,
            console=console,  # type: ignore[arg-type]
        ):
            raise click.ClickException(
                f"Ingestion stopped because required external tool(s) are missing: {names}."
            )


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
    from ms_rag.ui.prompts import print_error, print_hint, prompt_confirm, prompt_select  # noqa: PLC0415

    while True:
        try:
            console.print("[bold cyan]  Preparing embeddings and vector store...[/bold cyan]")  # type: ignore[union-attr]
            embeddings = vectorization.get_embeddings(config.embedding_model, credential_store)  # type: ignore[union-attr,arg-type]
            vector_store = db_connector.get_vector_store(config.vector_db, embeddings)  # type: ignore[union-attr,arg-type]
            chunking_llm = None
            if config.chunking and config.chunking.strategy == "agentic":
                from ms_rag.llm.llm_integration import get_llm  # noqa: PLC0415

                if not config.llm_model:
                    raise RuntimeError("Agentic chunking requires a selected generation model.")
                chunking_llm = get_llm(
                    config.llm_model.provider,
                    config.llm_model.model_id,
                    credential_store=credential_store,
                )
            console.print("[bold cyan]  Loading, chunking, embedding, and storing documents...[/bold cyan]")  # type: ignore[union-attr]
            ingestion_result = orchestrator.ingest(  # type: ignore[union-attr]
                sources=config.document_sources,
                loader_map=config.loader_map,
                chunking_config=config.chunking,
                embedding_model=config.embedding_model,
                vector_db=config.vector_db,
                vector_store=vector_store,
                embeddings=embeddings,
                llm=chunking_llm,
            )
            return embeddings, vector_store, ingestion_result
        except Exception as exc:  # noqa: BLE001
            print_error(console, f"Ingestion setup failed: {type(exc).__name__}: {exc}")  # type: ignore[arg-type]
            print_hint(
                console,  # type: ignore[arg-type]
                "Common causes: wrong embedding model for an existing index, missing DB credentials, "
                "unreachable local service, or incompatible collection dimension.",
            )
            hf_cache_path = None
            if (
                config.embedding_model
                and config.embedding_model.provider in {"huggingface", "local"}
                and "Can't load the model" in str(exc)
            ):
                from ms_rag.ingestion.vectorization_module import (  # noqa: PLC0415
                    local_huggingface_cache_path,
                    remove_local_huggingface_cache,
                )

                hf_cache_path = local_huggingface_cache_path(config.embedding_model.model_id)
                print_hint(
                    console,  # type: ignore[arg-type]
                    "Local HuggingFace download failed before the weights were fully cached. "
                    "MS-RAGS can clean only this model's local cache after your approval, "
                    "or you can choose the Hosted HuggingFace Inference API option to avoid local downloads.",
                )
                if hf_cache_path and hf_cache_path.exists():
                    print_hint(console, f"Detected partial/failed cache: {hf_cache_path}")  # type: ignore[arg-type]

            recovery_choices = [
                questionary.Choice("Retry with the same settings", value="retry"),
                questionary.Choice("Change embedding model", value="embedding"),
                questionary.Choice("Change vector database settings", value="vectordb"),
                questionary.Choice("Abort setup", value="abort"),
            ]
            if hf_cache_path and hf_cache_path.exists():
                recovery_choices.insert(
                    0,
                    questionary.Choice(
                        "Clean this HuggingFace model cache and retry",
                        value="clean_hf_cache",
                    ),
                )
            action = prompt_select(
                "  What would you like to do?",
                recovery_choices,
                console=console,  # type: ignore[arg-type]
            )
            if action == "clean_hf_cache":
                if not hf_cache_path:
                    print_error(console, "No HuggingFace cache folder was detected for this model.")  # type: ignore[arg-type]
                    continue
                if prompt_confirm(
                    f"  Delete only this model cache and retry? {hf_cache_path}",
                    default=False,
                    console=console,  # type: ignore[arg-type]
                ):
                    deleted_path = remove_local_huggingface_cache(config.embedding_model.model_id)  # type: ignore[union-attr]
                    if deleted_path:
                        print_hint(console, f"Deleted HuggingFace cache: {deleted_path}")  # type: ignore[arg-type]
                    else:
                        print_hint(console, "Cache folder was already absent; retrying download.")  # type: ignore[arg-type]
                    continue
                print_hint(console, "Cache cleanup skipped; choose another recovery option.")  # type: ignore[arg-type]
                continue
            if action == "retry":
                continue
            if action == "embedding":
                config.embedding_model = vectorization.display_and_select(  # type: ignore[union-attr]
                    config.configured_providers,
                    credential_store=credential_store,
                )
                config.vector_db = db_connector.prompt_and_configure(config.embedding_model)  # type: ignore[union-attr]
                config.vector_db = _ensure_vector_db_connection(
                    db_connector,
                    config.vector_db,
                    console,
                    embedding_model=config.embedding_model,
                )
                _display_ingestion_review(config, console)
                continue
            if action == "vectordb":
                config.vector_db = db_connector.prompt_and_configure(config.embedding_model)  # type: ignore[union-attr]
                config.vector_db = _ensure_vector_db_connection(
                    db_connector,
                    config.vector_db,
                    console,
                    embedding_model=config.embedding_model,
                )
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
        _display_runtime_readiness(config, runtime, console)
        _display_selected_architecture(config, console)
        return runtime
    except Exception as exc:  # noqa: BLE001
        from ms_rag.ui.prompts import print_error  # noqa: PLC0415

        print_error(console, f"Failed to build query runtime: {type(exc).__name__}: {exc}")  # type: ignore[arg-type]
        raise click.ClickException(
            "Runtime build failed. Check provider credentials, retrieval settings, and vector DB compatibility."
        ) from exc


def _display_runtime_readiness(
    config: PipelineConfig,
    runtime: dict[str, object],
    console: object,
) -> None:
    """Show final runtime wiring and retrieval-state status."""
    from rich.table import Table  # noqa: PLC0415

    vector_store = runtime.get("vector_store")
    retrieval = config.retrieval
    table = Table(title="Runtime Wiring Check", border_style="green")
    table.add_column("Component", style="bold white")
    table.add_column("Status", style="green")
    table.add_column("Details", style="cyan")

    table.add_row(
        "Generation model",
        "ready" if config.llm_model else "missing",
        f"{config.llm_model.provider} / {config.llm_model.model_id}" if config.llm_model else "No model selected",
    )
    table.add_row(
        "Vector store",
        "ready" if vector_store is not None else "missing",
        config.vector_db.db_type if config.vector_db else "No vector DB selected",
    )
    table.add_row(
        "Retriever",
        "ready" if runtime.get("retriever") is not None else "missing",
        retrieval.strategy if retrieval else "No retrieval strategy selected",
    )
    if config.compression_enabled and config.compression:
        table.add_row(
            "Compression",
            "ready" if runtime.get("compression_active") else "not applied",
            ", ".join(config.compression.techniques),
        )
    else:
        table.add_row("Compression", "disabled", "No compression configured")

    if config.agent_tools and config.agent_tools.enabled_tools:
        table.add_row("Agent tools", "ready", ", ".join(config.agent_tools.enabled_tools))
        if "memory" in config.agent_tools.enabled_tools:
            memory_settings = dict((config.agent_tools.tool_settings or {}).get("memory") or {})
            backend = str(memory_settings.get("backend") or "json")
            detail = backend
            if backend in {"json", "sqlite"}:
                detail += f" / {memory_settings.get('path', 'default path')}"
            elif backend == "postgres":
                detail += f" / table={memory_settings.get('table', 'ms_rag_agent_memory')}"
            elif backend == "mongodb_atlas":
                detail += (
                    f" / db={memory_settings.get('database', 'ms_rag_memory')}"
                    f" / collection={memory_settings.get('collection', 'agent_memory')}"
                )
            table.add_row("Memory store", "ready", detail)

    selected = {retrieval.strategy} if retrieval else set()
    if retrieval and retrieval.strategy == "ensemble":
        selected.update(retrieval.ensemble_sub_retrievers or [])
    if selected & {"parent_child", "multi_vector", "time_weighted"}:
        parent_count = len(getattr(vector_store, "_ms_rag_parent_documents", {}) or {})
        chunk_count = len(getattr(vector_store, "_ms_rag_chunk_documents", []) or [])
        has_embeddings = getattr(vector_store, "_ms_rag_embeddings", None) is not None
        table.add_row(
            "Advanced state",
            "ready" if parent_count or chunk_count else "missing",
            f"parents={parent_count}, chunks={chunk_count}, representation_embeddings={'yes' if has_embeddings else 'no'}",
        )
    console.print(table)  # type: ignore[union-attr]


def _display_selected_architecture(config: PipelineConfig, console: object) -> None:
    """Show the final selected architecture and all selected components."""
    from ms_rag.ui.architecture import display_architecture_report  # noqa: PLC0415

    display_architecture_report(config, console)


def _close_session_runtime(session: SessionState) -> None:
    """Close vector DB clients and other live resources after query mode exits."""
    from ms_rag.utils.runtime_cleanup import close_session_runtime  # noqa: PLC0415

    close_session_runtime(session)


def _ensure_vector_db_connection(
    db_connector: object,
    vector_db_config: object,
    console: object,
    *,
    embedding_model: object | None = None,
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
            config = db_connector.prompt_and_configure(embedding_model)  # type: ignore[union-attr]


def _run_query_loop(session: SessionState, eval_framework: object | None = None) -> None:
    """Run the interactive query loop."""
    from ms_rag.cli.query_loop import QueryLoop  # noqa: PLC0415
    from ms_rag.llm.llm_integration import build_session_runtime_from_vector_store  # noqa: PLC0415
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

    def edit_settings(sess: SessionState, console: object) -> bool:
        changed = _edit_live_query_settings(sess, console)
        if not changed:
            return False
        runtime = build_session_runtime_from_vector_store(
            sess.config,
            sess.credentials,
            vector_store=sess.vector_store,
            embeddings=getattr(sess.vector_store, "_ms_rag_embeddings", None),
        )
        sess.vector_store = runtime["vector_store"]
        sess.retriever = runtime["retriever"]
        sess.llm = runtime["llm"]
        sess.rag_chain = runtime["rag_chain"]
        _display_runtime_readiness(sess.config, runtime, console)
        return True

    loop = QueryLoop(
        query_pipeline=pipeline,
        session_manager=SessionManager(),
        settings_editor=edit_settings,
    )
    loop.run(session)


def _edit_live_query_settings(session: SessionState, console: object) -> bool:
    """Edit query-time settings that can be safely rebuilt without re-ingestion."""
    from ms_rag.query.query_enhancer import QueryEnhancer  # noqa: PLC0415
    from ms_rag.query.reranking_module import RerankingModule  # noqa: PLC0415
    from ms_rag.query.context_compressor import ContextCompressor  # noqa: PLC0415
    from ms_rag.ui.prompts import prompt_select, print_hint, print_success  # noqa: PLC0415
    import questionary  # noqa: PLC0415

    cfg = session.config
    action = prompt_select(
        "  What do you want to edit?",
        choices=[
            questionary.Choice("Query enhancement", "query_enhancement"),
            questionary.Choice("Reranking", "reranking"),
            questionary.Choice("Context compression", "compression"),
            questionary.Choice("Retrieval/vector DB path (requires re-ingestion)", "requires_reingest"),
            questionary.Choice("Cancel", "cancel"),
        ],
        console=console,  # type: ignore[arg-type]
    )

    if action == "cancel":
        return False
    if action == "requires_reingest":
        print_hint(
            console,  # type: ignore[arg-type]
            "Retrieval strategy, vector DB, embedding model, loader, chunking, and source path changes require a fresh ingestion run so indexes stay consistent.",
        )
        return False

    if action == "query_enhancement":
        enhancer = QueryEnhancer()
        cfg.query_enhancement = enhancer.configure(cfg.configured_providers)
        cfg.hyde_llm_provider = enhancer.hyde_llm_provider
        print_success(console, "Query enhancement updated.")  # type: ignore[arg-type]
        return True

    if action == "reranking":
        reranker = RerankingModule(credential_store=session.credentials)
        reranking_config = reranker.configure(cfg.retrieval.top_k if cfg.retrieval else 5)
        cfg.reranking = reranking_config
        cfg.reranking_enabled = reranking_config is not None
        print_success(console, "Reranking settings updated.")  # type: ignore[arg-type]
        return True

    if action == "compression":
        compressor = ContextCompressor()
        compression_config = compressor.configure(cfg.configured_providers)
        cfg.compression = compression_config
        cfg.compression_enabled = compression_config is not None
        print_success(console, "Compression settings updated.")  # type: ignore[arg-type]
        return True

    return False


def _generate_and_offer_save(config: PipelineConfig, console: object) -> None:
    """Generate code and offer to save."""
    from ms_rag.codegen.code_generator import CodeGenerator  # noqa: PLC0415
    generator = CodeGenerator()
    code = generator.generate(config)
    generator.display_and_offer_save(code, console)
