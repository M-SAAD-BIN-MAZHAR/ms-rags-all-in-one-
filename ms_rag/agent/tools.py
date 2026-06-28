"""Permission-gated Agentic RAG tools.

Every tool in this module is deny-by-default. Network, file, and API tools
only run when the saved AgentToolConfig contains explicit allowlists.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ms_rag.models import AgentToolConfig, CredentialStore


class ToolExecutionError(RuntimeError):
    """Raised when an agent tool cannot run safely."""


def _now() -> float:
    return time.time()


def _normalize_domain(hostname: str | None) -> str:
    return (hostname or "").lower().strip()


def _domain_allowed(hostname: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return False
    host = _normalize_domain(hostname)
    for allowed in allowed_domains:
        item = allowed.lower().strip()
        if host == item or host.endswith(f".{item}"):
            return True
    return False


def _safe_text(value: Any, max_chars: int = 4000) -> str:
    text = str(value or "")
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


@dataclass
class MemoryRecord:
    memory_type: str
    text: str
    metadata: dict[str, Any]
    created_at: float


class AgentMemoryStore:
    """Small JSON-backed memory store that works locally and in containers."""

    VALID_TYPES = {"short_term", "long_term", "semantic", "episodic", "user_profile"}

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        self.enabled_types = set(self.settings.get("memory_types") or [])
        self.short_term: list[MemoryRecord] = []
        default_path = Path(os.getenv("MS_RAG_AGENT_MEMORY_PATH", "./agent_memory/memory.json"))
        self.path = Path(str(self.settings.get("path") or default_path)).expanduser()
        self.max_records = int(self.settings.get("max_records", 1000))

    def remember(self, memory_type: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        if memory_type not in self.VALID_TYPES:
            raise ToolExecutionError(f"Unsupported memory type: {memory_type}")
        if memory_type not in self.enabled_types:
            raise ToolExecutionError(f"Memory type is not enabled: {memory_type}")
        record = MemoryRecord(memory_type, text, metadata or {}, _now())
        if memory_type == "short_term":
            self.short_term.append(record)
            self.short_term = self.short_term[-self.max_records :]
            return
        records = self._load()
        records.append(record)
        self._save(records[-self.max_records :])

    def recall(self, query: str, memory_type: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        records = list(self.short_term) + self._load()
        if memory_type:
            records = [record for record in records if record.memory_type == memory_type]
        filtered = [record for record in records if record.memory_type in self.enabled_types]
        ranked = sorted(
            filtered,
            key=lambda record: (
                len(query_terms.intersection(record.text.lower().split())),
                record.created_at,
            ),
            reverse=True,
        )
        return [
            {
                "memory_type": record.memory_type,
                "text": record.text,
                "metadata": record.metadata,
                "created_at": record.created_at,
            }
            for record in ranked[:limit]
        ]

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolExecutionError(f"Could not load agent memory store: {exc}") from exc
        return [
            MemoryRecord(
                memory_type=str(item.get("memory_type", "")),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata") or {}),
                created_at=float(item.get("created_at") or 0),
            )
            for item in raw
            if item.get("memory_type") in self.VALID_TYPES
        ]

    def _save(self, records: list[MemoryRecord]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "memory_type": record.memory_type,
                    "text": record.text,
                    "metadata": record.metadata,
                    "created_at": record.created_at,
                }
                for record in records
            ]
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Could not write agent memory store: {exc}") from exc


class AgentToolRuntime:
    """Runtime facade for configured Agentic RAG tools."""

    def __init__(
        self,
        config: AgentToolConfig | None,
        credential_store: CredentialStore | None = None,
        llm: object | None = None,
    ) -> None:
        self.config = config or AgentToolConfig()
        self.credential_store = credential_store or CredentialStore()
        self.llm = llm
        self.memory = AgentMemoryStore(self.settings("memory"))

    def enabled(self, tool_name: str) -> bool:
        return tool_name in set(self.config.enabled_tools)

    def settings(self, tool_name: str) -> dict[str, Any]:
        return dict((self.config.tool_settings or {}).get(tool_name) or {})

    def web_search(self, query: str) -> str:
        self._require_enabled("web_search")
        settings = self.settings("web_search")
        provider = str(settings.get("provider") or "tavily")
        max_results = int(settings.get("max_results") or 5)
        if provider == "tavily":
            key = self._credential("web_search", "TAVILY_API_KEY")
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": max_results},
                timeout=float(settings.get("timeout_seconds") or 20),
            )
            return self._format_response(response, "Tavily search")
        if provider == "brave":
            key = self._credential("web_search", "BRAVE_SEARCH_API_KEY")
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": key},
                timeout=float(settings.get("timeout_seconds") or 20),
            )
            return self._format_response(response, "Brave search")
        raise ToolExecutionError(f"Unsupported web search provider: {provider}")

    def fetch_url(self, url: str) -> str:
        self._require_enabled("url_fetch")
        settings = self.settings("url_fetch")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolExecutionError("URL Fetch only allows http/https URLs.")
        allowed_domains = list(settings.get("allowed_domains") or [])
        if not _domain_allowed(parsed.hostname or "", allowed_domains):
            raise ToolExecutionError(f"URL domain is not allowlisted: {parsed.hostname}")
        timeout = float(settings.get("timeout_seconds") or 20)
        max_bytes = int(settings.get("max_bytes") or 250_000)
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        content = response.raw.read(max_bytes + 1, decode_content=True)
        if len(content) > max_bytes:
            raise ToolExecutionError(f"URL response exceeded max page size of {max_bytes} bytes.")
        return content.decode(response.encoding or "utf-8", errors="replace")

    def read_file(self, path: str) -> str:
        self._require_enabled("file_read")
        settings = self.settings("file_read")
        requested = Path(path).expanduser().resolve()
        allowed_paths = [Path(p).expanduser().resolve() for p in settings.get("allowed_paths") or []]
        if not any(requested == allowed or requested.is_relative_to(allowed) for allowed in allowed_paths):
            raise ToolExecutionError(f"File path is not allowlisted: {requested}")
        if requested.is_dir():
            raise ToolExecutionError("File System Read Tool reads files only. Select specific files.")
        max_bytes = int(settings.get("max_bytes") or 250_000)
        data = requested.read_bytes()
        if len(data) > max_bytes:
            raise ToolExecutionError(f"File exceeded max read size of {max_bytes} bytes.")
        return data.decode("utf-8", errors="replace")

    def api_request(self, method: str, url: str, body: dict[str, Any] | None = None) -> str:
        self._require_enabled("api_request")
        settings = self.settings("api_request")
        method = method.upper()
        allowed_methods = {str(item).upper() for item in settings.get("allowed_methods") or ["GET"]}
        if method not in allowed_methods:
            raise ToolExecutionError(f"HTTP method is not allowlisted: {method}")
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        allowed_bases = [str(item).rstrip("/") for item in settings.get("allowed_base_urls") or []]
        if base_url not in allowed_bases:
            raise ToolExecutionError(f"API base URL is not allowlisted: {base_url}")
        headers: dict[str, str] = {}
        auth_env = str(settings.get("auth_env_var") or "")
        if auth_env:
            token = self._credential("api_request", auth_env) or os.getenv(auth_env)
            if not token:
                raise ToolExecutionError(f"Missing API auth secret: {auth_env}")
            headers["Authorization"] = f"Bearer {token}"
        response = requests.request(
            method,
            url,
            json=body if method in {"POST", "PUT", "PATCH"} else None,
            headers=headers,
            timeout=float(settings.get("timeout_seconds") or 20),
        )
        return self._format_response(response, "API request")

    def summarize(self, text: str, max_chars: int | None = None) -> str:
        self._require_enabled("document_summarization")
        settings = self.settings("document_summarization")
        limit = int(max_chars or settings.get("max_input_chars") or 12_000)
        clipped = _safe_text(text, limit)
        if self.llm is None:
            return clipped[: int(settings.get("fallback_chars") or 1500)]
        prompt = (
            "Summarize the following document for a RAG agent. Keep factual details, "
            "entities, dates, URLs, and decisions.\n\n"
            f"{clipped}"
        )
        result = self.llm.invoke(prompt)  # type: ignore[attr-defined]
        return getattr(result, "content", str(result))

    def recall_memory(self, query: str, limit: int = 5) -> str:
        self._require_enabled("memory")
        records = self.memory.recall(query, limit=limit)
        if not records:
            return ""
        return "\n\n".join(f"[{item['memory_type']}] {item['text']}" for item in records)

    def remember(self, memory_type: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._require_enabled("memory")
        self.memory.remember(memory_type, text, metadata)

    def _require_enabled(self, tool_name: str) -> None:
        if not self.enabled(tool_name):
            raise ToolExecutionError(f"Agent tool is not enabled: {tool_name}")

    def _credential(self, provider_id: str, field: str) -> str:
        value = self.credential_store.get(provider_id, field) or os.getenv(field)
        if not value:
            raise ToolExecutionError(f"Missing credential for {field}.")
        return value

    @staticmethod
    def _format_response(response: requests.Response, label: str) -> str:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ToolExecutionError(f"{label} failed with HTTP {response.status_code}: {response.text[:500]}") from exc
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return json.dumps(response.json(), indent=2)[:12000]
        return response.text[:12000]
