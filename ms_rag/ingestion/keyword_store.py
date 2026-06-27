"""Persistent keyword/chunk text stores for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    import questionary
except ImportError:  # pragma: no cover
    questionary = None  # type: ignore[assignment]

from ms_rag.models import KeywordStoreConfig, RetrievalConfig


@dataclass(frozen=True)
class KeywordStoreInfo:
    store_type: str
    display_name: str
    description: str
    credential_fields: list[str]
    optional_fields: list[str] = field(default_factory=list)
    default_collection: str = "ms_rag_keywords"


KEYWORD_STORES: list[KeywordStoreInfo] = [
    KeywordStoreInfo(
        store_type="sqlite",
        display_name="SQLite FTS/local keyword store",
        description="Default local persistent chunk text store; good for single-server deployments.",
        credential_fields=[],
        optional_fields=["KEYWORD_SQLITE_PATH"],
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="postgres",
        display_name="PostgreSQL keyword store",
        description="Production text store using Postgres; works well beside PGVector or cloud vector DBs.",
        credential_fields=["KEYWORD_POSTGRES_CONNECTION_STRING"],
        optional_fields=[],
        default_collection="ms_rag_keywords",
    ),
    KeywordStoreInfo(
        store_type="elasticsearch",
        display_name="Elasticsearch keyword store",
        description="Managed/full-text keyword store for production hybrid retrieval.",
        credential_fields=["ELASTICSEARCH_URL"],
        optional_fields=["ELASTICSEARCH_USERNAME", "ELASTICSEARCH_PASSWORD", "ELASTICSEARCH_API_KEY"],
        default_collection="ms-rag-keywords",
    ),
    KeywordStoreInfo(
        store_type="opensearch",
        display_name="OpenSearch keyword store",
        description="OpenSearch/AWS-managed full-text keyword store for production hybrid retrieval.",
        credential_fields=["OPENSEARCH_URL"],
        optional_fields=["OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"],
        default_collection="ms-rag-keywords",
    ),
    KeywordStoreInfo(
        store_type="memory",
        display_name="Memory only",
        description="Development/testing only. Text is lost when the process exits.",
        credential_fields=[],
        optional_fields=[],
        default_collection="ms_rag_keywords",
    ),
]

KEYWORD_STORE_MAP = {store.store_type: store for store in KEYWORD_STORES}
KEYWORD_REQUIRED_STRATEGIES = {"keyword_bm25", "tfidf", "hybrid"}


def retrieval_needs_keyword_store(retrieval: RetrievalConfig | None) -> bool:
    if retrieval is None:
        return False
    if retrieval.strategy in KEYWORD_REQUIRED_STRATEGIES:
        return True
    if retrieval.strategy == "ensemble":
        return any(
            sub_id in KEYWORD_REQUIRED_STRATEGIES
            for sub_id in (retrieval.ensemble_sub_retrievers or ["dense_vector", "keyword_bm25"])
        )
    return False


class KeywordStoreConnector:
    """Prompt, test, write, and read persistent keyword stores."""

    def __init__(self, credential_store: object | None = None) -> None:
        self._credential_store = credential_store

    def prompt_and_configure(self, *, production_recommended: bool = False) -> KeywordStoreConfig:
        from ms_rag.ui.prompts import get_console, print_step, prompt_required_confirm, prompt_select, prompt_text  # noqa: PLC0415

        console = get_console()
        print_step(console, "11b", "Keyword / Chunk Text Store")
        console.print(
            "[dim]Hybrid, BM25, and TF-IDF need raw chunk text in addition to vectors. "
            "Cloud vector DBs such as Pinecone should use a persistent keyword store.[/dim]"
        )

        choices = []
        for store in KEYWORD_STORES:
            if production_recommended and store.store_type == "memory":
                title = f"{store.display_name}  —  {store.description} [dev only]"
            else:
                title = f"{store.display_name}  —  {store.description}"
            choices.append(questionary.Choice(title=title, value=store.store_type))

        store_type = prompt_select("Select keyword/chunk text store:", choices, console=console)
        info = KEYWORD_STORE_MAP[store_type]
        connection_params: dict[str, str] = {}

        for field_name in info.credential_fields:
            value = prompt_text(f"  {field_name}:", required=True, secret=_is_secret(field_name), console=console)
            connection_params[field_name] = str(value)
            if self._credential_store is not None:
                self._credential_store.set(store_type, field_name, str(value))  # type: ignore[union-attr]

        for field_name in info.optional_fields:
            value = prompt_text(f"  {field_name} (optional):", required=False, secret=_is_secret(field_name), console=console)
            if value:
                connection_params[field_name] = str(value)
                if self._credential_store is not None:
                    self._credential_store.set(store_type, field_name, str(value))  # type: ignore[union-attr]

        collection_name = prompt_text(
            f"  Keyword collection/table/index name (default: {info.default_collection}):",
            default=info.default_collection,
            required=True,
            console=console,
        )
        config = KeywordStoreConfig(
            store_type=store_type,
            connection_params=connection_params,
            collection_name=str(collection_name),
        )
        self.test_connection(config)
        prompt_required_confirm("Proceed with this keyword store?", console=console)
        return config

    def test_connection(self, config: KeywordStoreConfig) -> None:
        if config.store_type == "sqlite":
            path = _sqlite_path(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as conn:
                _ensure_sqlite_schema(conn, config.collection_name)
            return
        if config.store_type == "postgres":
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return
        if config.store_type == "elasticsearch":
            from elasticsearch import Elasticsearch  # noqa: PLC0415
            client = _elasticsearch_client(config, Elasticsearch)
            if not client.ping():
                raise RuntimeError("Elasticsearch keyword store connection failed.")
            return
        if config.store_type == "opensearch":
            from opensearchpy import OpenSearch  # noqa: PLC0415
            client = _opensearch_client(config, OpenSearch)
            if not client.ping():
                raise RuntimeError("OpenSearch keyword store connection failed.")
            return
        if config.store_type == "memory":
            return
        raise ValueError(f"Unsupported keyword store: {config.store_type}")

    def persist_documents(self, config: KeywordStoreConfig, documents: list) -> list[str]:
        records = [_record_from_doc(doc, index) for index, doc in enumerate(documents)]
        if config.store_type == "memory":
            return [record["text"] for record in records if record["text"].strip()]
        if config.store_type == "sqlite":
            with sqlite3.connect(_sqlite_path(config)) as conn:
                _ensure_sqlite_schema(conn, config.collection_name)
                conn.executemany(
                    f'INSERT OR REPLACE INTO "{config.collection_name}" (chunk_id, text, metadata_json) VALUES (?, ?, ?)',
                    [(r["chunk_id"], r["text"], json.dumps(r["metadata"], ensure_ascii=True)) for r in records],
                )
            return self.load_texts(config)
        if config.store_type == "postgres":
            self._persist_postgres(config, records)
            return self.load_texts(config)
        if config.store_type == "elasticsearch":
            from elasticsearch import Elasticsearch, helpers  # noqa: PLC0415
            client = _elasticsearch_client(config, Elasticsearch)
            helpers.bulk(client, [_elastic_action(config.collection_name, record) for record in records])
            return self.load_texts(config)
        if config.store_type == "opensearch":
            from opensearchpy import OpenSearch, helpers  # noqa: PLC0415
            client = _opensearch_client(config, OpenSearch)
            helpers.bulk(client, [_elastic_action(config.collection_name, record) for record in records])
            return self.load_texts(config)
        raise ValueError(f"Unsupported keyword store: {config.store_type}")

    def load_texts(self, config: KeywordStoreConfig) -> list[str]:
        if config.store_type == "sqlite":
            with sqlite3.connect(_sqlite_path(config)) as conn:
                _ensure_sqlite_schema(conn, config.collection_name)
                rows = conn.execute(f'SELECT text FROM "{config.collection_name}" WHERE text != ""').fetchall()
            return [row[0] for row in rows]
        if config.store_type == "postgres":
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")) as conn:
                with conn.cursor() as cur:
                    _ensure_postgres_schema(cur, config.collection_name)
                    cur.execute(f'SELECT text FROM "{config.collection_name}" WHERE text <> %s', ("",))
                    return [row[0] for row in cur.fetchall()]
        if config.store_type == "elasticsearch":
            from elasticsearch import Elasticsearch  # noqa: PLC0415
            return _load_elastic_texts(_elasticsearch_client(config, Elasticsearch), config.collection_name)
        if config.store_type == "opensearch":
            from opensearchpy import OpenSearch  # noqa: PLC0415
            return _load_elastic_texts(_opensearch_client(config, OpenSearch), config.collection_name)
        return []

    def _persist_postgres(self, config: KeywordStoreConfig, records: list[dict[str, Any]]) -> None:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(_param(config, "KEYWORD_POSTGRES_CONNECTION_STRING")) as conn:
            with conn.cursor() as cur:
                _ensure_postgres_schema(cur, config.collection_name)
                cur.executemany(
                    f'INSERT INTO "{config.collection_name}" (chunk_id, text, metadata_json) VALUES (%s, %s, %s) '
                    f'ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, metadata_json = EXCLUDED.metadata_json',
                    [(r["chunk_id"], r["text"], json.dumps(r["metadata"], ensure_ascii=True)) for r in records],
                )


def _is_secret(field_name: str) -> bool:
    upper = field_name.upper()
    if upper == "KEYWORD_SQLITE_PATH":
        return False
    return any(
        token in upper
        for token in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "CONNECTION_STRING")
    )


def _param(config: KeywordStoreConfig, key: str, default: str = "") -> str:
    value = config.connection_params.get(key, "")
    if value == key:
        return os.getenv(key, default)
    return value or os.getenv(key, default)


def _sqlite_path(config: KeywordStoreConfig) -> Path:
    return Path(_param(config, "KEYWORD_SQLITE_PATH", "./keyword_store/ms_rag_keywords.sqlite"))


def _ensure_sqlite_schema(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{table}" (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL)'
    )
    conn.commit()


def _ensure_postgres_schema(cur: object, table: str) -> None:
    cur.execute(
        f'CREATE TABLE IF NOT EXISTS "{table}" (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json JSONB NOT NULL)'
    )


def _record_from_doc(doc: object, index: int) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    chunk_id = metadata.get("ms_rag_child_id") or metadata.get("id") or f"chunk::{index}"
    return {
        "chunk_id": str(chunk_id),
        "text": str(getattr(doc, "page_content", "") or ""),
        "metadata": metadata,
    }


def _elastic_action(index_name: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "_op_type": "index",
        "_index": index_name,
        "_id": record["chunk_id"],
        "_source": record,
    }


def _elasticsearch_client(config: KeywordStoreConfig, cls: object) -> object:
    kwargs: dict[str, Any] = {}
    api_key = _param(config, "ELASTICSEARCH_API_KEY")
    username = _param(config, "ELASTICSEARCH_USERNAME")
    password = _param(config, "ELASTICSEARCH_PASSWORD")
    if api_key:
        kwargs["api_key"] = api_key
    elif username or password:
        kwargs["basic_auth"] = (username, password)
    return cls(_param(config, "ELASTICSEARCH_URL"), **kwargs)


def _opensearch_client(config: KeywordStoreConfig, cls: object) -> object:
    auth = None
    username = _param(config, "OPENSEARCH_USERNAME")
    password = _param(config, "OPENSEARCH_PASSWORD")
    if username or password:
        auth = (username, password)
    return cls(_param(config, "OPENSEARCH_URL"), http_auth=auth)


def _load_elastic_texts(client: object, index_name: str) -> list[str]:
    result = client.search(index=index_name, body={"query": {"match_all": {}}, "size": 10000})
    hits = result.get("hits", {}).get("hits", [])
    return [hit.get("_source", {}).get("text", "") for hit in hits if hit.get("_source", {}).get("text", "").strip()]
