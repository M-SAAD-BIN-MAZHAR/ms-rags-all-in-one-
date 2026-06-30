"""Persistent text/keyword store for hybrid and keyword-based retrieval.

The keyword store persists raw chunk text outside the vector DB so that
keyword-based retrievers (BM25, TF-IDF, hybrid, and ensemble strategies
with keyword sub-retrievers) can access searchable text even when the
vector DB (e.g. Pinecone) does not expose stored text.

Supported backends:
    sqlite          — Local SQLite database (default, zero config)
    postgres        — PostgreSQL with full-text search
    elasticsearch   — Elasticsearch
    opensearch      — OpenSearch
    memory          — In-memory list (development/evaluation only)

NOTE: Secrets rendered as env-var markers are re-resolved at runtime
against the active CredentialStore.  The KEYWORD_SQLITE_PATH parameter
is deliberately unmasked in saved-session JSON because it is a file path,
not a secret.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_rag.models import CredentialStore, KeywordStoreConfig, RetrievalConfig
from ms_rag.ui.prompts import (
    get_console,
    print_error,
    print_hint,
    print_step,
    print_success,
    prompt_confirm,
    prompt_select,
    prompt_text,
)


# ---------------------------------------------------------------------------
# Keyword store catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeywordStoreInfo:
    """Metadata for a single keyword store backend."""

    store_type: str
    display_name: str
    description: str
    credential_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    default_collection: str = "ms_rag_keywords"


KEYWORD_STORES: list[KeywordStoreInfo] = [
    KeywordStoreInfo(
        store_type="sqlite",
        display_name="SQLite (local file)",
        description="Local SQLite database — zero config, ideal for development and single-machine deployments.",
        optional_fields=("KEYWORD_SQLITE_PATH",),
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="postgres",
        display_name="PostgreSQL (full-text search)",
        description="PostgreSQL with full-text search indexes for production deployments.",
        credential_fields=("KEYWORD_POSTGRES_CONNECTION_STRING",),
        optional_fields=("KEYWORD_POSTGRES_TABLE",),
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="elasticsearch",
        display_name="Elasticsearch",
        description="Elasticsearch — full-text search engine suitable for large-scale production.",
        credential_fields=("ELASTICSEARCH_URL",),
        optional_fields=("ELASTICSEARCH_USERNAME", "ELASTICSEARCH_PASSWORD", "ELASTICSEARCH_API_KEY"),
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="opensearch",
        display_name="OpenSearch",
        description="OpenSearch — AWS-compatible full-text search with k-NN plugin.",
        credential_fields=("OPENSEARCH_URL",),
        optional_fields=("OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"),
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="memory",
        display_name="In-Memory (development only)",
        description="Plain Python list in memory — data is lost when the process exits. "
        "Use only for evaluation and quick experiments.",
        default_collection="ms_rag_keywords",
    ),
]

KEYWORD_STORE_MAP: dict[str, KeywordStoreInfo] = {s.store_type: s for s in KEYWORD_STORES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_secret(field_name: str) -> bool:
    """Return True when a field value should be masked in saved-session JSON.

    File paths (e.g. KEYWORD_SQLITE_PATH) are deliberately unmasked because
    they are local filesystem references, not secrets.
    """
    upper = field_name.upper()
    if upper == "KEYWORD_SQLITE_PATH":
        return False
    for token in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CONNECTION_STRING", "URI", "URL"):
        if token in upper:
            return True
    return False


def _safe_collection_name(name: str) -> str:
    """Return a filesystem-safe name for local keyword store files."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name.strip())
    return cleaned or "ms_rag_keywords"


def _safe_sql_identifier(name: str, *, default: str = "ms_rag_keywords") -> str:
    """Return a safe SQL identifier for user-configured table names."""
    candidate = (name or default).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", candidate):
        raise ValueError(
            f"Unsafe SQL table name {name!r}. Use letters, numbers, and underscores, "
            "starting with a letter or underscore."
        )
    return candidate


def _default_sqlite_path(config: KeywordStoreConfig) -> Path:
    """Return the default SQLite database file path."""
    return Path(f"./keyword_store/{_safe_collection_name(config.collection_name)}.sqlite")


def retrieval_needs_keyword_store(config: RetrievalConfig | None) -> bool:
    """Return True when the retrieval strategy depends on a keyword store.

    Keyword-based retrieval (BM25, TF-IDF, hybrid) and ensemble strategies
    with keyword sub-retrievers need raw text to be available outside the
    vector DB.
    """
    if config is None:
        return False
    if config.strategy in {"keyword_bm25", "tfidf", "hybrid"}:
        return True
    if config.strategy == "ensemble":
        sub_ids = config.ensemble_sub_retrievers or ["dense_vector", "keyword_bm25"]
        return any(sid in {"keyword_bm25", "tfidf", "hybrid"} for sid in sub_ids)
    return False


# ---------------------------------------------------------------------------
# KeywordStoreConnector
# ---------------------------------------------------------------------------


class KeywordStoreConnector:
    """Persistent text store for keyword-based retrieval.

    Usage::

        connector = KeywordStoreConnector(credential_store)
        config = connector.prompt_and_configure(production_recommended=True)
        connector.test_connection(config)
        texts = connector.persist_documents(config, docs)
        loaded = connector.load_texts(config)
    """

    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self.credential_store = credential_store or CredentialStore()

    # ------------------------------------------------------------------
    # Interactive configuration
    # ------------------------------------------------------------------

    def prompt_and_configure(
        self,
        production_recommended: bool = False,
    ) -> KeywordStoreConfig:
        """Interactive prompt with Rich UI for keyword store selection.

        Args:
            production_recommended: When True, local/volatile backends show a
                warning so the user prefers a persistent option.

        Returns:
            A fully configured KeywordStoreConfig.
        """
        con = get_console()
        print_step(con, "11k", "Keyword Store Configuration")

        print_hint(
            con,
            "Hybrid, BM25, and TF-IDF retrieval need chunk text stored outside "
            "the vector DB so keyword search is possible even when the vector "
            "DB (e.g. Pinecone) does not store raw text.",
        )
        if production_recommended:
            print_hint(
                con,
                "Production deployment detected (cloud vector DB). "
                "Memory-only keyword storage will be lost on restart. "
                "Use SQLite, PostgreSQL, Elasticsearch, or OpenSearch for persistence.",
            )

        choices = [
            {
                "name": f"{info.display_name}  --  {info.description}",
                "value": info.store_type,
            }
            for info in KEYWORD_STORES
        ]

        store_type = prompt_select(
            "  Select keyword store backend:",
            choices=choices,
            console=con,
        )

        info = KEYWORD_STORE_MAP[store_type]
        connection_params: dict[str, str] = {}

        for field in info.credential_fields:
            value = prompt_text(
                f"  {field}:",
                required=True,
                secret=_is_secret(field),
                console=con,
            )
            connection_params[field] = str(value)
            self.credential_store.set(store_type, field, str(value))

        for field in info.optional_fields:
            default = ""
            if field == "KEYWORD_SQLITE_PATH":
                default = str(_default_sqlite_path(KeywordStoreConfig(store_type="sqlite", connection_params={})))
            elif field == "KEYWORD_POSTGRES_TABLE":
                default = "ms_rag_keywords"
            value = prompt_text(
                f"  {field} (optional):",
                default=default,
                required=False,
                secret=_is_secret(field),
                console=con,
            )
            if value:
                connection_params[field] = str(value)
                self.credential_store.set(store_type, field, str(value))

        collection_name = prompt_text(
            "  Collection / index name:",
            default=info.default_collection,
            required=True,
            console=con,
        )

        config = KeywordStoreConfig(
            store_type=store_type,
            connection_params=connection_params,
            collection_name=str(collection_name),
        )

        self.test_connection(config)
        print_success(con, f"Keyword store ready: {info.display_name} / {config.collection_name}")
        return config

    # ------------------------------------------------------------------
    # Connection testing
    # ------------------------------------------------------------------

    def test_connection(self, config: KeywordStoreConfig) -> None:
        """Test that the keyword store backend is reachable.

        Raises:
            RuntimeError: If the backend is unavailable or required
                dependencies are missing.
        """
        store_type = config.store_type

        if store_type == "memory":
            return

        if store_type == "sqlite":
            db_path = self._resolved_sqlite_path(config)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("SELECT 1")
            finally:
                conn.close()
            return

        if store_type == "postgres":
            try:
                import psycopg2  # type: ignore[import-not-found]  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL keyword store requires psycopg2. "
                    "Install it with: pip install psycopg2-binary"
                ) from exc
            conn_string = self._resolve_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")
            conn = psycopg2.connect(conn_string)  # type: ignore[arg-type]
            conn.close()
            return

        if store_type in ("elasticsearch", "opensearch"):
            try:
                import requests  # type: ignore[import-not-found]  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    f"{store_type} keyword store requires the requests package."
                ) from exc
            url = self._search_url(config, via_opensearch=store_type == "opensearch")
            auth, headers = self._search_auth(config, via_opensearch=store_type == "opensearch")
            try:
                resp = requests.get(url, auth=auth, headers=headers, timeout=5)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot reach {store_type} at {url}: {exc}"
                ) from exc
            return

        raise ValueError(f"Unsupported keyword store type: {store_type!r}")

    # ------------------------------------------------------------------
    # Persist documents
    # ------------------------------------------------------------------

    def persist_documents(
        self,
        config: KeywordStoreConfig,
        docs: list,
    ) -> list[str]:
        """Store chunk text in the keyword store backend.

        Args:
            config: KeywordStoreConfig with backend connection settings.
            docs:   List of LangChain Document objects.

        Returns:
            List of extracted text strings that were persisted.
        """
        texts = [
            str(getattr(doc, "page_content", "") or "")
            for doc in docs
            if str(getattr(doc, "page_content", "") or "").strip()
        ]
        if not texts:
            return []

        store_type = config.store_type

        if store_type == "memory":
            self._memory_store[config.collection_name] = list(texts)
            return texts

        if store_type == "sqlite":
            self._persist_sqlite(config, texts)
            return texts

        if store_type == "postgres":
            self._persist_postgres(config, texts)
            return texts

        if store_type == "elasticsearch":
            self._persist_elasticsearch(config, texts, via_opensearch=False)
            return texts

        if store_type == "opensearch":
            self._persist_elasticsearch(config, texts, via_opensearch=True)
            return texts

        raise ValueError(f"Unsupported keyword store type: {store_type!r}")

    # ------------------------------------------------------------------
    # Load texts
    # ------------------------------------------------------------------

    def load_texts(self, config: KeywordStoreConfig) -> list[str]:
        """Load stored chunk texts from the keyword store backend.

        Args:
            config: KeywordStoreConfig with backend connection settings.

        Returns:
            List of text strings.
        """
        store_type = config.store_type

        if store_type == "memory":
            return list(self._memory_store.get(config.collection_name, []))

        if store_type == "sqlite":
            return self._load_sqlite(config)

        if store_type == "postgres":
            return self._load_postgres(config)

        if store_type == "elasticsearch":
            return self._load_elasticsearch(config, via_opensearch=False)

        if store_type == "opensearch":
            return self._load_elasticsearch(config, via_opensearch=True)

        raise ValueError(f"Unsupported keyword store type: {store_type!r}")

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    _memory_store: dict[str, list[str]] = {}

    def _resolve_param(self, config: KeywordStoreConfig, field_name: str, default: str = "") -> str:
        """Resolve config/env-marker/store/env values for a keyword-store field."""
        from ms_rag.utils.credentials import resolve_credential  # noqa: PLC0415

        raw = str(config.connection_params.get(field_name, "") or "").strip()
        if raw and raw != field_name:
            return raw
        provider = config.store_type
        resolved = resolve_credential(field_name, self.credential_store, provider)
        return resolved or default

    def _resolved_sqlite_path(self, config: KeywordStoreConfig) -> Path:
        raw = self._resolve_param(config, "KEYWORD_SQLITE_PATH")
        if raw:
            return Path(raw).expanduser().resolve()
        return _default_sqlite_path(config).expanduser().resolve()

    def _persist_sqlite(self, config: KeywordStoreConfig, texts: list[str]) -> None:
        db_path = self._resolved_sqlite_path(config)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS keywords ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  collection_name TEXT NOT NULL,"
                "  chunk_index INTEGER NOT NULL,"
                "  text TEXT NOT NULL"
                ")"
            )
            conn.execute("DELETE FROM keywords WHERE collection_name = ?", (config.collection_name,))
            conn.executemany(
                "INSERT INTO keywords (collection_name, chunk_index, text) VALUES (?, ?, ?)",
                [(config.collection_name, i, text) for i, text in enumerate(texts)],
            )
            conn.commit()
        finally:
            conn.close()

    def _load_sqlite(self, config: KeywordStoreConfig) -> list[str]:
        db_path = self._resolved_sqlite_path(config)
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT text FROM keywords WHERE collection_name = ? ORDER BY chunk_index",
                (config.collection_name,),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def _persist_postgres(self, config: KeywordStoreConfig, texts: list[str]) -> None:
        table = _safe_sql_identifier(self._resolve_param(config, "KEYWORD_POSTGRES_TABLE", "ms_rag_keywords"))
        try:
            import psycopg2  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL keyword store requires psycopg2. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        conn_string = self._resolve_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")
        conn = psycopg2.connect(conn_string)  # type: ignore[arg-type]
        try:
            cur = conn.cursor()
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "  id SERIAL PRIMARY KEY,"
                "  collection_name TEXT NOT NULL,"
                "  chunk_index INTEGER NOT NULL,"
                "  text TEXT NOT NULL"
                ")"
            )
            cur.execute(f"DELETE FROM {table} WHERE collection_name = %s", (config.collection_name,))
            for i, text in enumerate(texts):
                cur.execute(
                    f"INSERT INTO {table} (collection_name, chunk_index, text) VALUES (%s, %s, %s)",
                    (config.collection_name, i, text),
                )
            conn.commit()
        finally:
            conn.close()

    def _load_postgres(self, config: KeywordStoreConfig) -> list[str]:
        try:
            import psycopg2  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL keyword store requires psycopg2. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        conn_string = self._resolve_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")
        table = _safe_sql_identifier(self._resolve_param(config, "KEYWORD_POSTGRES_TABLE", "ms_rag_keywords"))
        conn = psycopg2.connect(conn_string)  # type: ignore[arg-type]
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT text FROM {table} WHERE collection_name = %s ORDER BY chunk_index",
                (config.collection_name,),
            )
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def _persist_elasticsearch(
        self, config: KeywordStoreConfig, texts: list[str], via_opensearch: bool = False
    ) -> None:
        try:
            import requests  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"{'OpenSearch' if via_opensearch else 'Elasticsearch'} keyword store requires the requests package."
            ) from exc

        url = self._search_url(config, via_opensearch=via_opensearch)
        auth, headers = self._search_auth(config, via_opensearch=via_opensearch)

        index = _safe_collection_name(config.collection_name)
        delete_resp = requests.post(
            f"{url}/{index}/_delete_by_query",
            json={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"collection_name.keyword": config.collection_name}},
                            {"term": {"collection_name": config.collection_name}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            },
            auth=auth,
            headers=headers,
            timeout=10,
        )
        if delete_resp.status_code not in {200, 201, 404}:
            raise RuntimeError(
                f"Failed to clear existing keyword documents in {index}: "
                f"HTTP {delete_resp.status_code} {delete_resp.text[:300]}"
            )

        for i, text in enumerate(texts):
            resp = requests.post(
                f"{url}/{index}/_doc/{i}",
                json={"collection_name": config.collection_name, "chunk_index": i, "text": text},
                auth=auth,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()

    def _load_elasticsearch(self, config: KeywordStoreConfig, via_opensearch: bool = False) -> list[str]:
        try:
            import requests  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"{'OpenSearch' if via_opensearch else 'Elasticsearch'} keyword store requires the requests package."
            ) from exc

        url = self._search_url(config, via_opensearch=via_opensearch)
        auth, headers = self._search_auth(config, via_opensearch=via_opensearch)
        index = _safe_collection_name(config.collection_name)
        try:
            resp = requests.get(
                f"{url}/{index}/_search",
                json={
                    "size": 10000,
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"collection_name.keyword": config.collection_name}},
                                {"term": {"collection_name": config.collection_name}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                    "sort": [{"chunk_index": {"order": "asc"}}],
                },
                auth=auth,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            return [h["_source"]["text"] for h in hits if "_source" in h and "text" in h["_source"]]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load keyword texts from {'OpenSearch' if via_opensearch else 'Elasticsearch'} "
                f"index {index}: {exc}"
            ) from exc

    def _search_url(self, config: KeywordStoreConfig, *, via_opensearch: bool) -> str:
        field_name = "OPENSEARCH_URL" if via_opensearch else "ELASTICSEARCH_URL"
        return self._resolve_param(config, field_name, "http://localhost:9200").rstrip("/")

    def _search_auth(self, config: KeywordStoreConfig, *, via_opensearch: bool) -> tuple[tuple[str, str] | None, dict[str, str]]:
        headers: dict[str, str] = {}
        if via_opensearch:
            user = self._resolve_param(config, "OPENSEARCH_USERNAME")
            pwd = self._resolve_param(config, "OPENSEARCH_PASSWORD")
            return ((user, pwd) if user and pwd else None), headers

        api_key = self._resolve_param(config, "ELASTICSEARCH_API_KEY")
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
            return None, headers
        user = self._resolve_param(config, "ELASTICSEARCH_USERNAME")
        pwd = self._resolve_param(config, "ELASTICSEARCH_PASSWORD")
        return ((user, pwd) if user and pwd else None), headers
