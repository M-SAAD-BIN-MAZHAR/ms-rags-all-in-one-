"""Interactive configuration for Agentic RAG tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ms_rag.models import AgentToolConfig, CredentialStore
from ms_rag.ui.prompts import (
    get_console,
    print_hint,
    print_step,
    print_success,
    print_warning,
    prompt_checkbox,
    prompt_confirm,
    prompt_select,
    prompt_text,
)


TOOL_CHOICES = [
    {
        "name": "Web Search",
        "value": "web_search",
        "description": "Search the public web through an approved provider API.",
    },
    {
        "name": "Memory Systems",
        "value": "memory",
        "description": "Short-term, long-term, semantic, episodic, and user profile memory.",
    },
    {
        "name": "URL Fetch / Web Page Reader",
        "value": "url_fetch",
        "description": "Read approved URLs from allowlisted domains.",
    },
    {
        "name": "File System Read",
        "value": "file_read",
        "description": "Read approved local files from allowlisted paths only.",
    },
    {
        "name": "Document Summarization",
        "value": "document_summarization",
        "description": "Summarize long tool results before generation.",
    },
    {
        "name": "API Request",
        "value": "api_request",
        "description": "Call safe external REST APIs with method and URL allowlists.",
    },
]


class AgentToolConfigurator:
    """Collect permission-gated agentic tool configuration."""

    def __init__(self, credential_store: CredentialStore) -> None:
        self.credential_store = credential_store

    def configure(self) -> AgentToolConfig | None:
        con = get_console()
        print_step(con, "3b", "Agentic Tools")
        print_hint(
            con,
            "Tools are optional and permission-gated. If enabled, MS-RAGS(ALL-IN-ONE) asks for allowlists and credentials now.",
        )
        if not prompt_confirm("  Enable external/local tools for Agentic RAG?", default=False, console=con):
            print_warning(con, "Agentic tools skipped. The agent will use retrieval and the selected LLM only.")
            return None

        selected = prompt_checkbox(
            "  Select tools to enable:",
            choices=TOOL_CHOICES,
            min_selections=1,
            console=con,
        )
        settings: dict[str, dict[str, Any]] = {}
        if "web_search" in selected:
            settings["web_search"] = self._configure_web_search()
        if "memory" in selected:
            settings["memory"] = self._configure_memory()
        if "url_fetch" in selected:
            settings["url_fetch"] = self._configure_url_fetch()
        if "file_read" in selected:
            settings["file_read"] = self._configure_file_read()
        if "document_summarization" in selected:
            settings["document_summarization"] = self._configure_summarization()
        if "api_request" in selected:
            settings["api_request"] = self._configure_api_request()

        print_success(con, f"Agentic tools enabled: {', '.join(selected)}")
        return AgentToolConfig(enabled_tools=selected, tool_settings=settings)

    def _configure_web_search(self) -> dict[str, Any]:
        con = get_console()
        provider = prompt_select(
            "  Web search provider:",
            choices=[
                {"name": "Tavily - search API for AI agents", "value": "tavily"},
                {"name": "Brave Search API", "value": "brave"},
            ],
            console=con,
        )
        key_name = "TAVILY_API_KEY" if provider == "tavily" else "BRAVE_SEARCH_API_KEY"
        value = prompt_text(f"  {key_name}:", required=True, secret=True, console=con)
        self.credential_store.set("web_search", key_name, str(value))
        max_results = prompt_text(
            "  Max web results per query:",
            default="5",
            required=True,
            console=con,
            validator=lambda raw: max(1, min(20, int(raw))),
        )
        return {"provider": provider, "max_results": int(max_results), "timeout_seconds": 20}

    def _configure_memory(self) -> dict[str, Any]:
        con = get_console()
        memory_types = prompt_checkbox(
            "  Select memory types:",
            choices=[
                {"name": "Short-Term Memory - only this live session", "value": "short_term"},
                {"name": "Long-Term Memory - persistent facts and decisions", "value": "long_term"},
                {"name": "Semantic Memory - reusable knowledge snippets", "value": "semantic"},
                {"name": "Episodic Memory - past task/query events", "value": "episodic"},
                {"name": "User Profile Memory - approved user preferences", "value": "user_profile"},
            ],
            min_selections=1,
            console=con,
        )
        backend = prompt_select(
            "  Memory backend:",
            choices=[
                {"name": "Local JSON file - simplest local/container memory", "value": "json"},
                {"name": "Local SQLite - durable local memory database", "value": "sqlite"},
                {"name": "Postgres / cloud Postgres - production shared memory", "value": "postgres"},
                {"name": "MongoDB Atlas - managed cloud document memory", "value": "mongodb_atlas"},
            ],
            console=con,
        )
        recall_limit = prompt_text(
            "  How many memories to recall per query (default 5):",
            default="5",
            required=True,
            console=con,
            validator=lambda raw: max(1, min(50, int(raw))),
        )
        settings: dict[str, Any] = {
            "memory_types": memory_types,
            "backend": backend,
            "max_records": 1000,
            "recall_limit": int(recall_limit),
        }
        if backend == "json":
            path = prompt_text(
                "  Memory JSON path for local/container deploys:",
                default="./agent_memory/memory.json",
                required=True,
                console=con,
            )
            settings["path"] = str(path)
        elif backend == "sqlite":
            path = prompt_text(
                "  Memory SQLite path for local/container deploys:",
                default="./agent_memory/memory.sqlite3",
                required=True,
                console=con,
            )
            settings["path"] = str(path)
        elif backend == "postgres":
            conn = prompt_text(
                "  MEMORY_POSTGRES_CONNECTION_STRING:",
                required=True,
                secret=True,
                console=con,
            )
            self.credential_store.set("memory", "MEMORY_POSTGRES_CONNECTION_STRING", str(conn))
            settings["connection_env"] = "MEMORY_POSTGRES_CONNECTION_STRING"
            settings["table"] = str(prompt_text("  Memory table name:", default="ms_rag_agent_memory", required=True, console=con))
        elif backend == "mongodb_atlas":
            conn = prompt_text(
                "  MEMORY_MONGODB_CONNECTION_STRING:",
                required=True,
                secret=True,
                console=con,
            )
            self.credential_store.set("memory", "MEMORY_MONGODB_CONNECTION_STRING", str(conn))
            settings["connection_env"] = "MEMORY_MONGODB_CONNECTION_STRING"
            settings["database"] = str(prompt_text("  Memory database name:", default="ms_rag_memory", required=True, console=con))
            settings["collection"] = str(prompt_text("  Memory collection name:", default="agent_memory", required=True, console=con))
        print_success(con, f"Memory backend configured: {backend}")
        return settings

    def _configure_url_fetch(self) -> dict[str, Any]:
        con = get_console()
        raw = prompt_text(
            "  Allowed URL domains (comma-separated, e.g. docs.python.org,github.com):",
            required=True,
            console=con,
        )
        domains = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
        return {"allowed_domains": domains, "timeout_seconds": 20, "max_bytes": 250_000}

    def _configure_file_read(self) -> dict[str, Any]:
        con = get_console()
        raw = prompt_text(
            "  Allowed local files/folders (comma-separated absolute or project-relative paths):",
            required=True,
            console=con,
        )
        paths = []
        for item in str(raw).split(","):
            cleaned = item.strip()
            if cleaned:
                paths.append(str(Path(cleaned).expanduser().resolve()))
        return {"allowed_paths": paths, "max_bytes": 250_000}

    def _configure_summarization(self) -> dict[str, Any]:
        return {"max_input_chars": 12000, "fallback_chars": 1500}

    def _configure_api_request(self) -> dict[str, Any]:
        con = get_console()
        raw_bases = prompt_text(
            "  Allowed API base URLs (comma-separated, e.g. https://api.example.com):",
            required=True,
            console=con,
        )
        methods = prompt_checkbox(
            "  Allowed HTTP methods:",
            choices=[
                {"name": "GET", "value": "GET"},
                {"name": "POST", "value": "POST"},
                {"name": "PUT", "value": "PUT"},
                {"name": "PATCH", "value": "PATCH"},
            ],
            min_selections=1,
            console=con,
        )
        auth_env = prompt_text(
            "  Auth env var name (optional, e.g. CUSTOMER_API_TOKEN):",
            default="",
            required=False,
            console=con,
        )
        if auth_env:
            secret = prompt_text(f"  {auth_env}:", required=True, secret=True, console=con)
            self.credential_store.set("api_request", str(auth_env), str(secret))
        bases = [item.strip().rstrip("/") for item in str(raw_bases).split(",") if item.strip()]
        return {
            "allowed_base_urls": bases,
            "allowed_methods": methods,
            "auth_env_var": str(auth_env or ""),
            "timeout_seconds": 20,
        }
