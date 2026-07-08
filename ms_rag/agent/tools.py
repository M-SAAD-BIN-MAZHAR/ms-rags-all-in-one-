"""Permission-gated Agentic RAG tools.

Every tool in this module is deny-by-default. Network, file, and API tools
only run when the saved AgentToolConfig contains explicit allowlists.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
import warnings
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
    key: str | None = None                 # user_profile upsert key
    session_id: str | None = None          # short_term / episodic scoping
    embedding: list[float] | None = None   # semantic / long_term similarity vector


# Memory types whose recall is driven by embedding similarity (meaning), not
# literal keyword overlap. These store durable knowledge/facts.
SEMANTIC_TYPES: frozenset[str] = frozenset({"semantic", "long_term"})


def _slug_key(text: str) -> str:
    """Normalise a profile attribute name into a stable key.

    Uses Unicode word characters (``\\w``), so non-Latin keys (Arabic, Chinese,
    Cyrillic, etc.) are preserved instead of collapsing to a single fallback key
    that would make international profile attributes overwrite each other.
    """
    cleaned = re.sub(r"\W+", "_", str(text or "").strip().lower(), flags=re.UNICODE).strip("_")
    return cleaned[:64] or "attribute"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 if degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_overlap(query_terms: set[str], text: str) -> float:
    """Fraction of query terms present in ``text`` — a bounded 0..1 fallback score."""
    if not query_terms:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & tokens) / len(query_terms)


class AgentMemoryStore:
    """Permission-gated memory store where each memory type behaves per its purpose.

    - ``semantic`` / ``long_term``: durable facts recalled by embedding similarity
      (meaning), falling back to lexical overlap only when no embedding model is wired.
    - ``episodic``: time/session-scoped events, recalled by recency blended with relevance.
    - ``user_profile``: keyed attributes that are upserted (never duplicated) and
      always injected regardless of query similarity.
    - ``short_term``: this-session working memory kept as a shared rolling window.

    Records are appended incrementally (not full-store rewrites) and evicted
    per memory type so a burst of episodes can never drop stable facts or profile.
    """

    VALID_TYPES = {"short_term", "long_term", "semantic", "episodic", "user_profile"}

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        credential_store: CredentialStore | None = None,
        embeddings: object | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings or {}
        self.credential_store = credential_store
        self.embeddings = embeddings
        self.session_id = session_id or uuid.uuid4().hex
        self.enabled_types = set(self.settings.get("memory_types") or [])
        self.short_term: list[MemoryRecord] = []
        self.backend = str(self.settings.get("backend") or "json")
        default_path = Path(os.getenv("MS_RAG_AGENT_MEMORY_PATH", "./agent_memory/memory.json"))
        self.path = Path(str(self.settings.get("path") or default_path)).expanduser()
        # Per-type cap for persistent types so no single type starves the others.
        self.max_records = int(self.settings.get("max_records", 1000))
        self.short_term_limit = int(self.settings.get("short_term_limit", 30))
        self.recall_limit = max(1, int(self.settings.get("recall_limit", 5)))
        self._warned_no_embeddings = False

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def remember(
        self,
        memory_type: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        key: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if memory_type not in self.VALID_TYPES:
            raise ToolExecutionError(f"Unsupported memory type: {memory_type}")
        if memory_type not in self.enabled_types:
            raise ToolExecutionError(f"Memory type is not enabled: {memory_type}")
        text = str(text or "").strip()
        if not text:
            raise ToolExecutionError("Cannot store an empty memory entry.")
        meta = dict(metadata or {})
        sid = session_id or self.session_id

        if memory_type == "user_profile":
            record = MemoryRecord(
                memory_type="user_profile",
                text=text,
                metadata=meta,
                created_at=_now(),
                key=key or meta.get("key") or _slug_key(text.split(":", 1)[0]),
            )
            self._upsert_profile(record)
            return

        if memory_type == "short_term":
            record = MemoryRecord("short_term", text, meta, _now(), session_id=sid)
            self.short_term.append(record)
            self.short_term = self.short_term[-self.short_term_limit :]
            return

        embedding = self._embed(text) if memory_type in SEMANTIC_TYPES else None
        record = MemoryRecord(
            memory_type=memory_type,
            text=text,
            metadata=meta,
            created_at=_now(),
            session_id=sid if memory_type == "episodic" else None,
            embedding=embedding,
        )
        self._append(record)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        memory_type: str | None = None,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories using the retrieval behaviour appropriate to each type."""
        if memory_type is not None:
            if memory_type not in self.enabled_types:
                return []
            return [self._as_dict(r) for r in self._recall_type(query, memory_type, limit, session_id)]

        results: list[MemoryRecord] = []
        seen: set[tuple[str, str]] = set()
        # User profile is always injected first, unfiltered by similarity.
        ordered_types = [t for t in ("user_profile", "semantic", "long_term", "episodic", "short_term")
                         if t in self.enabled_types]
        for mtype in ordered_types:
            for record in self._recall_type(query, mtype, limit, session_id):
                dedupe_key = (record.memory_type, record.text)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                results.append(record)
        return [self._as_dict(r) for r in results]

    def _recall_type(
        self,
        query: str,
        memory_type: str,
        limit: int,
        session_id: str | None,
    ) -> list[MemoryRecord]:
        if memory_type == "short_term":
            sid = session_id or self.session_id
            scoped = [r for r in self.short_term if r.session_id in (None, sid)]
            return list(reversed(scoped))[:limit]

        if memory_type == "user_profile":
            # Always return the latest value per key, unfiltered by the query.
            profile = [r for r in self._load() if r.memory_type == "user_profile"]
            latest: dict[str, MemoryRecord] = {}
            for record in sorted(profile, key=lambda r: r.created_at):
                latest[record.key or _slug_key(record.text)] = record
            return sorted(latest.values(), key=lambda r: r.key or "")

        candidates = [r for r in self._load() if r.memory_type == memory_type]
        if not candidates:
            return []

        if memory_type in SEMANTIC_TYPES:
            return self._rank_semantic(query, candidates, limit)

        if memory_type == "episodic":
            return self._rank_episodic(query, candidates, limit)

        # Defensive default — recency ordering.
        return sorted(candidates, key=lambda r: r.created_at, reverse=True)[:limit]

    def _rank_semantic(self, query: str, candidates: list[MemoryRecord], limit: int) -> list[MemoryRecord]:
        query_vec = self._embed(query)
        if query_vec is not None:
            scored: list[tuple[float, MemoryRecord]] = []
            for record in candidates:
                vec = record.embedding if record.embedding else self._embed(record.text)
                scored.append((_cosine(query_vec, vec) if vec else 0.0, record))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [record for score, record in scored if score > 0.0][:limit] or [
                record for _, record in scored[:limit]
            ]

        if not self._warned_no_embeddings:
            warnings.warn(
                "Semantic/long-term memory is running in lexical fallback mode because no "
                "embedding model is wired into the memory store; recall matches keywords, not meaning.",
                stacklevel=2,
            )
            self._warned_no_embeddings = True
        query_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        ranked = sorted(
            candidates,
            key=lambda r: (_lexical_overlap(query_terms, r.text), r.created_at),
            reverse=True,
        )
        return ranked[:limit]

    def _rank_episodic(self, query: str, candidates: list[MemoryRecord], limit: int) -> list[MemoryRecord]:
        query_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        newest = max((r.created_at for r in candidates), default=0.0)
        oldest = min((r.created_at for r in candidates), default=0.0)
        span = (newest - oldest) or 1.0
        scored = []
        for record in candidates:
            recency = (record.created_at - oldest) / span
            relevance = _lexical_overlap(query_terms, record.text)
            scored.append((0.5 * recency + 0.5 * relevance, record))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [record for _, record in scored[:limit]]

    def _embed(self, text: str) -> list[float] | None:
        if self.embeddings is None:
            return None
        embed_query = getattr(self.embeddings, "embed_query", None)
        if not callable(embed_query):
            return None
        try:
            vector = embed_query(text)
            return [float(x) for x in vector]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Memory embedding failed; falling back to lexical recall: {exc}", stacklevel=2)
            return None

    @staticmethod
    def _as_dict(record: MemoryRecord) -> dict[str, Any]:
        return {
            "memory_type": record.memory_type,
            "text": record.text,
            "metadata": record.metadata,
            "created_at": record.created_at,
            "key": record.key,
        }

    # ------------------------------------------------------------------
    # Persistence — incremental append / keyed upsert with per-type eviction
    # ------------------------------------------------------------------

    def _append(self, record: MemoryRecord) -> None:
        if self.backend == "sqlite":
            self._sqlite_append(record)
        elif self.backend == "postgres":
            self._postgres_append(record)
        elif self.backend == "mongodb_atlas":
            self._mongo_append(record)
        else:
            records = self._load()
            records.append(record)
            self._save_json(self._evict_per_type(records))

    def _upsert_profile(self, record: MemoryRecord) -> None:
        if self.backend == "sqlite":
            self._sqlite_upsert_profile(record)
        elif self.backend == "postgres":
            self._postgres_upsert_profile(record)
        elif self.backend == "mongodb_atlas":
            self._mongo_upsert_profile(record)
        else:
            records = [
                r for r in self._load()
                if not (r.memory_type == "user_profile" and (r.key or "") == (record.key or ""))
            ]
            records.append(record)
            self._save_json(self._evict_per_type(records))

    def _evict_per_type(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Keep newest ``max_records`` per type; user_profile is keyed, never volume-evicted."""
        kept: list[MemoryRecord] = []
        by_type: dict[str, list[MemoryRecord]] = {}
        for record in records:
            by_type.setdefault(record.memory_type, []).append(record)
        for mtype, items in by_type.items():
            if mtype == "user_profile":
                kept.extend(items)
                continue
            kept.extend(sorted(items, key=lambda r: r.created_at)[-self.max_records :])
        return sorted(kept, key=lambda r: r.created_at)

    def _load(self) -> list[MemoryRecord]:
        if self.backend == "sqlite":
            return self._load_sqlite()
        if self.backend == "postgres":
            return self._load_postgres()
        if self.backend == "mongodb_atlas":
            return self._load_mongodb()
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolExecutionError(f"Could not load agent memory store: {exc}") from exc
        return [self._record_from_item(item) for item in raw if item.get("memory_type") in self.VALID_TYPES]

    def _record_from_item(self, item: dict[str, Any]) -> MemoryRecord:
        embedding = item.get("embedding")
        return MemoryRecord(
            memory_type=str(item.get("memory_type", "")),
            text=str(item.get("text", "")),
            metadata=dict(item.get("metadata") or {}),
            created_at=float(item.get("created_at") or 0),
            key=item.get("key"),
            session_id=item.get("session_id"),
            embedding=[float(x) for x in embedding] if isinstance(embedding, list) else None,
        )

    def _to_item(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "memory_type": record.memory_type,
            "text": record.text,
            "metadata": record.metadata,
            "created_at": record.created_at,
            "key": record.key,
            "session_id": record.session_id,
            "embedding": record.embedding,
        }

    def _save_json(self, records: list[MemoryRecord]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = [self._to_item(record) for record in records]
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Could not write agent memory store: {exc}") from exc

    def _record_from_row(
        self,
        memory_type: str,
        text: str,
        metadata: Any,
        created_at: Any,
        key: Any = None,
        session_id: Any = None,
        embedding: Any = None,
    ) -> MemoryRecord:
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
                embedding = None
        return MemoryRecord(
            memory_type=str(memory_type),
            text=str(text),
            metadata=dict(metadata or {}),
            created_at=float(created_at or 0),
            key=str(key) if key else None,
            session_id=str(session_id) if session_id else None,
            embedding=[float(x) for x in embedding] if isinstance(embedding, list) else None,
        )

    # ── SQLite ────────────────────────────────────────────────────────
    _SQLITE_COLUMNS = "memory_type, text, metadata, created_at, mem_key, session_id, embedding"

    def _load_sqlite(self) -> list[MemoryRecord]:
        import sqlite3  # noqa: PLC0415

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            self._ensure_sqlite(conn)
            rows = conn.execute(
                f"SELECT {self._SQLITE_COLUMNS} FROM memory ORDER BY created_at DESC"
            ).fetchall()
        return [self._record_from_row(*row) for row in rows]

    def _sqlite_append(self, record: MemoryRecord) -> None:
        import sqlite3  # noqa: PLC0415

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            self._ensure_sqlite(conn)
            self._sqlite_insert(conn, record)
            self._sqlite_trim(conn, record.memory_type)

    def _sqlite_upsert_profile(self, record: MemoryRecord) -> None:
        import sqlite3  # noqa: PLC0415

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            self._ensure_sqlite(conn)
            conn.execute(
                "DELETE FROM memory WHERE memory_type = 'user_profile' AND mem_key = ?",
                (record.key,),
            )
            self._sqlite_insert(conn, record)

    def _sqlite_insert(self, conn: Any, record: MemoryRecord) -> None:
        conn.execute(
            f"INSERT INTO memory({self._SQLITE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_type,
                record.text,
                json.dumps(record.metadata),
                record.created_at,
                record.key,
                record.session_id,
                json.dumps(record.embedding) if record.embedding else None,
            ),
        )

    def _sqlite_trim(self, conn: Any, memory_type: str) -> None:
        if memory_type == "user_profile":
            return
        conn.execute(
            "DELETE FROM memory WHERE memory_type = ? AND id NOT IN ("
            "SELECT id FROM memory WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?)",
            (memory_type, memory_type, self.max_records),
        )

    def _ensure_sqlite(self, conn: Any) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, memory_type TEXT, text TEXT, metadata TEXT, "
            "created_at REAL, mem_key TEXT, session_id TEXT, embedding TEXT)"
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
        for column in ("mem_key", "session_id", "embedding"):
            if column not in existing:
                conn.execute(f"ALTER TABLE memory ADD COLUMN {column} TEXT")

    # ── Postgres ──────────────────────────────────────────────────────
    def _postgres_conn_string(self) -> str:
        env_name = str(self.settings.get("connection_env") or "MEMORY_POSTGRES_CONNECTION_STRING")
        value = (
            self.credential_store.get("memory", env_name)
            if self.credential_store is not None
            else None
        ) or os.getenv(env_name)
        if not value:
            raise ToolExecutionError(f"Missing memory Postgres connection string: {env_name}")
        return value

    def _postgres_table(self) -> str:
        table = str(self.settings.get("table") or "ms_rag_agent_memory")
        if not table.replace("_", "").isalnum():
            raise ToolExecutionError(f"Unsafe memory table name: {table}")
        return table

    def _load_postgres(self) -> list[MemoryRecord]:
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ToolExecutionError("Postgres memory requires psycopg. Install psycopg[binary].") from exc
        table = self._postgres_table()
        with psycopg.connect(self._postgres_conn_string()) as conn:
            self._ensure_postgres(conn, table)
            rows = conn.execute(
                f"SELECT memory_type, text, metadata, created_at, mem_key, session_id, embedding "
                f"FROM {table} ORDER BY created_at DESC"
            ).fetchall()
        return [self._record_from_row(*row) for row in rows]

    def _postgres_append(self, record: MemoryRecord) -> None:
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ToolExecutionError("Postgres memory requires psycopg. Install psycopg[binary].") from exc
        table = self._postgres_table()
        with psycopg.connect(self._postgres_conn_string()) as conn:
            self._ensure_postgres(conn, table)
            self._postgres_insert(conn, table, record)
            if record.memory_type != "user_profile":
                conn.execute(
                    f"DELETE FROM {table} WHERE memory_type = %s AND id NOT IN ("
                    f"SELECT id FROM {table} WHERE memory_type = %s ORDER BY created_at DESC LIMIT %s)",
                    (record.memory_type, record.memory_type, self.max_records),
                )

    def _postgres_upsert_profile(self, record: MemoryRecord) -> None:
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ToolExecutionError("Postgres memory requires psycopg. Install psycopg[binary].") from exc
        table = self._postgres_table()
        with psycopg.connect(self._postgres_conn_string()) as conn:
            self._ensure_postgres(conn, table)
            conn.execute(
                f"DELETE FROM {table} WHERE memory_type = 'user_profile' AND mem_key = %s",
                (record.key,),
            )
            self._postgres_insert(conn, table, record)

    def _postgres_insert(self, conn: Any, table: str, record: MemoryRecord) -> None:
        conn.execute(
            f"INSERT INTO {table}(memory_type, text, metadata, created_at, mem_key, session_id, embedding) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                record.memory_type,
                record.text,
                json.dumps(record.metadata),
                record.created_at,
                record.key,
                record.session_id,
                json.dumps(record.embedding) if record.embedding else None,
            ),
        )

    def _ensure_postgres(self, conn: Any, table: str) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "id SERIAL PRIMARY KEY, memory_type TEXT, text TEXT, metadata JSONB, "
            "created_at DOUBLE PRECISION, mem_key TEXT, session_id TEXT, embedding JSONB)"
        )
        for column, col_type in (("mem_key", "TEXT"), ("session_id", "TEXT"), ("embedding", "JSONB")):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}")

    # ── MongoDB Atlas ─────────────────────────────────────────────────
    def _mongodb_collection(self) -> Any:
        try:
            from pymongo import MongoClient  # noqa: PLC0415
        except ImportError as exc:
            raise ToolExecutionError("MongoDB memory requires pymongo.") from exc
        env_name = str(self.settings.get("connection_env") or "MEMORY_MONGODB_CONNECTION_STRING")
        uri = (
            self.credential_store.get("memory", env_name)
            if self.credential_store is not None
            else None
        ) or os.getenv(env_name)
        if not uri:
            raise ToolExecutionError(f"Missing MongoDB memory connection string: {env_name}")
        client = MongoClient(uri)
        db = client[str(self.settings.get("database") or "ms_rag_memory")]
        return db[str(self.settings.get("collection") or "agent_memory")]

    def _load_mongodb(self) -> list[MemoryRecord]:
        coll = self._mongodb_collection()
        rows = coll.find({}, {"_id": 0}).sort("created_at", -1)
        return [self._record_from_item(row) for row in rows]

    def _mongo_append(self, record: MemoryRecord) -> None:
        coll = self._mongodb_collection()
        coll.insert_one(self._to_item(record))
        if record.memory_type != "user_profile":
            keep_ids = [
                doc["_id"]
                for doc in coll.find({"memory_type": record.memory_type}, {"_id": 1})
                .sort("created_at", -1)
                .limit(self.max_records)
            ]
            coll.delete_many({"memory_type": record.memory_type, "_id": {"$nin": keep_ids}})

    def _mongo_upsert_profile(self, record: MemoryRecord) -> None:
        coll = self._mongodb_collection()
        coll.delete_many({"memory_type": "user_profile", "key": record.key})
        coll.insert_one(self._to_item(record))


class AgentToolRuntime:
    """Runtime facade for configured Agentic RAG tools."""

    def __init__(
        self,
        config: AgentToolConfig | None,
        credential_store: CredentialStore | None = None,
        llm: object | None = None,
        embeddings: object | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or AgentToolConfig()
        self.credential_store = credential_store or CredentialStore()
        self.llm = llm
        self.memory = AgentMemoryStore(
            self.settings("memory"),
            credential_store=self.credential_store,
            embeddings=embeddings,
            session_id=session_id,
        )

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

    def recall_memory(self, query: str, limit: int | None = None) -> str:
        self._require_enabled("memory")
        records = self.memory.recall(query, limit=limit if limit is not None else self.memory.recall_limit)
        if not records:
            return ""
        return "\n\n".join(f"[{item['memory_type']}] {item['text']}" for item in records)

    def remember(self, memory_type: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._require_enabled("memory")
        self.memory.remember(memory_type, text, metadata)

    def capture_interaction(
        self,
        memory_type: str,
        query: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Store a query/answer turn as the content shape that memory type expects.

        Instead of dumping a raw "User query / Assistant answer" log into every
        type, this extracts type-appropriate content: durable facts for
        semantic/long-term, keyed attributes for user_profile, and a timestamped
        event for episodic. Returns the text fragments actually stored.
        """
        self._require_enabled("memory")
        meta = {"source": "live_query_loop", **(metadata or {})}

        if memory_type == "short_term":
            text = f"User asked: {query}\nAssistant answered: {answer}"
            self.memory.remember("short_term", text, meta)
            return [text]

        if memory_type == "episodic":
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            text = f"[{timestamp}] User asked '{query.strip()}'. Outcome: {_safe_text(answer, 400)}"
            self.memory.remember("episodic", text, meta)
            return [text]

        if memory_type == "user_profile":
            attributes = self._extract_profile_attributes(query, answer)
            if not attributes:
                raise ToolExecutionError(
                    "No personal profile attributes could be extracted from this answer. "
                    "User Profile memory stores durable facts about a person (name, role, skills, "
                    "education, etc.). To keep this as reusable knowledge instead, save it to "
                    "Semantic or Long-Term memory."
                )
            stored: list[str] = []
            for key, value in attributes.items():
                text = f"{key.replace('_', ' ')}: {value}"
                self.memory.remember("user_profile", text, meta, key=key)
                stored.append(text)
            return stored

        if memory_type in SEMANTIC_TYPES:
            facts = self._extract_facts(query, answer)
            if not facts:
                # Never fall back to a raw Q/A log — store the answer as a knowledge snippet.
                facts = [_safe_text(answer, 600)]
            for fact in facts:
                self.memory.remember(memory_type, fact, meta)
            return facts

        self.memory.remember(memory_type, f"{query}\n{answer}", meta)
        return []

    def _extract_facts(self, query: str, answer: str) -> list[str]:
        """Ask the LLM for standalone durable facts worth remembering."""
        if self.llm is None:
            return []
        prompt = (
            "Extract up to 5 standalone, durable factual statements worth remembering from the "
            "assistant's answer below. Write each fact on its own line as a complete sentence, "
            "with no numbering, bullets, or preamble. Skip greetings, opinions, and anything "
            "specific only to this one conversation. If nothing is worth storing, output exactly NONE.\n\n"
            f"User question: {query}\nAssistant answer: {answer}"
        )
        try:
            result = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            content = getattr(result, "content", str(result))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Fact extraction for memory failed: {exc}", stacklevel=2)
            return []
        facts: list[str] = []
        for line in str(content).splitlines():
            cleaned = line.strip().lstrip("-•*0123456789. ").strip()
            if cleaned and cleaned.upper() != "NONE":
                facts.append(cleaned)
        return facts[:5]

    def _extract_profile_attributes(self, query: str, answer: str) -> dict[str, str]:
        """Ask the LLM for stable person/profile attributes as key/value pairs.

        Works whether the answer is phrased in first person ("my name is…") or as
        a third-person bio ("Muhammad is an AI/ML Engineer…") — a profile save is
        an explicit user action, so attributes of the person described are stored.
        """
        if self.llm is None:
            return {}
        prompt = (
            "Extract durable profile attributes about the main person described below as "
            "'key: value' lines using snake_case keys — for example name, role, title, "
            "employer, location, education, skills, achievements, or contact. Include only "
            "concrete, stable facts stated in the text. No opinions, no general knowledge, and "
            "no conversation summary. If the text describes no specific person's attributes, "
            "output exactly NONE.\n\n"
            f"Question: {query}\nAnswer: {answer}"
        )
        try:
            result = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            content = getattr(result, "content", str(result))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"User-profile extraction for memory failed: {exc}", stacklevel=2)
            return {}
        attributes: dict[str, str] = {}
        for line in str(content).splitlines():
            if ":" not in line or line.strip().upper() == "NONE":
                continue
            raw_key, _, raw_value = line.partition(":")
            key = _slug_key(raw_key)
            value = raw_value.strip().strip("-•* ").strip()
            if key and value:
                attributes[key] = value
        return attributes

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
