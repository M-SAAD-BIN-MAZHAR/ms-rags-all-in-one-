"""Vector DB Connector for MS_RAG.

Handles vector database selection, credential prompting, connection
testing, vector store creation, and document ingestion.

Requirement 9:
- Display all 12 supported Vector_Databases (9.1)
- Prompt for DB-specific credentials (9.2)
- Show connection summary and confirm before testing (9.3)
- Confirm connection on success (9.4)
- Re-prompt on connection failure (9.5)
- Show progress indicator during ingestion (9.6)
- Display chunk count and collection name after ingestion (9.7)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

try:
    import questionary
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

from ms_rag.models import EmbeddingModelConfig, VectorDBConfig, IngestionResult
from ms_rag.utils.exceptions import ConnectionError as MSRAGConnectionError
from ms_rag.utils.metadata import sanitize_documents


# ---------------------------------------------------------------------------
# Vector DB definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorDBInfo:
    """Metadata for a single vector database type."""

    db_type: str
    display_name: str
    description: str
    credential_fields: list[str]          # required connection params
    optional_fields: list[str] = field(default_factory=list)
    default_collection: str = "ms_rag_collection"
    is_local: bool = False


VECTOR_DBS: list[VectorDBInfo] = [
    VectorDBInfo(
        db_type="chroma",
        display_name="ChromaDB (local)",
        description="Local embedded vector store — zero config, ideal for development",
        credential_fields=[],
        optional_fields=["CHROMA_PERSIST_DIRECTORY"],
        default_collection="ms_rag",
        is_local=True,
    ),
    VectorDBInfo(
        db_type="faiss",
        display_name="FAISS (local, in-memory)",
        description="Facebook AI Similarity Search — extremely fast, runs in RAM",
        credential_fields=[],
        optional_fields=["FAISS_INDEX_PATH"],
        default_collection="ms_rag_faiss",
        is_local=True,
    ),
    VectorDBInfo(
        db_type="pinecone",
        display_name="Pinecone",
        description="Fully managed vector database — serverless or pod-based",
        credential_fields=["PINECONE_API_KEY"],
        optional_fields=["PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME"],
        default_collection="ms-rag-index",
    ),
    VectorDBInfo(
        db_type="weaviate",
        display_name="Weaviate",
        description="Open-source vector DB with hybrid search and multi-tenancy",
        credential_fields=["WEAVIATE_URL"],
        optional_fields=["WEAVIATE_API_KEY"],
        default_collection="MsRagCollection",
    ),
    VectorDBInfo(
        db_type="qdrant",
        display_name="Qdrant",
        description="Rust-powered vector DB with rich filtering and cloud/local options",
        credential_fields=["QDRANT_URL"],
        optional_fields=["QDRANT_API_KEY", "QDRANT_PORT"],
        default_collection="ms_rag",
    ),
    VectorDBInfo(
        db_type="milvus",
        display_name="Milvus",
        description="Distributed vector DB — handles billions of vectors",
        credential_fields=["MILVUS_URI"],
        optional_fields=["MILVUS_TOKEN"],
        default_collection="ms_rag_collection",
    ),
    VectorDBInfo(
        db_type="redis",
        display_name="Redis (with RedisSearch / RediStack)",
        description="In-memory vector search using Redis VSS — ultra-low latency",
        credential_fields=["REDIS_URL"],
        optional_fields=["REDIS_INDEX_NAME"],
        default_collection="ms_rag_idx",
    ),
    VectorDBInfo(
        db_type="pgvector",
        display_name="PGVector (PostgreSQL)",
        description="Vector search extension for PostgreSQL — ACID compliant",
        credential_fields=["PGVECTOR_CONNECTION_STRING"],
        optional_fields=["PGVECTOR_COLLECTION_NAME"],
        default_collection="ms_rag_vectors",
    ),
    VectorDBInfo(
        db_type="elasticsearch",
        display_name="Elasticsearch",
        description="Full-text + vector search — ideal for hybrid RAG pipelines",
        credential_fields=["ELASTICSEARCH_URL"],
        optional_fields=["ELASTICSEARCH_USERNAME", "ELASTICSEARCH_PASSWORD",
                         "ELASTICSEARCH_API_KEY"],
        default_collection="ms-rag-index",
    ),
    VectorDBInfo(
        db_type="opensearch",
        display_name="OpenSearch",
        description="AWS-managed alternative to Elasticsearch with k-NN plugin",
        credential_fields=["OPENSEARCH_URL"],
        optional_fields=["OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"],
        default_collection="ms-rag-index",
    ),
    VectorDBInfo(
        db_type="azure_ai_search",
        display_name="Azure AI Search",
        description="Azure managed search service with vector and hybrid search",
        credential_fields=["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY"],
        optional_fields=["AZURE_SEARCH_INDEX_NAME"],
        default_collection="ms-rag-index",
    ),
    VectorDBInfo(
        db_type="mongodb_atlas",
        display_name="MongoDB Atlas Vector Search",
        description="MongoDB Atlas with vector search index on document collections",
        credential_fields=["MONGODB_ATLAS_CLUSTER_URI"],
        optional_fields=["MONGODB_ATLAS_DB_NAME", "MONGODB_ATLAS_COLLECTION_NAME"],
        default_collection="ms_rag_vectors",
    ),
]

VECTOR_DB_MAP: dict[str, VectorDBInfo] = {db.db_type: db for db in VECTOR_DBS}
VECTOR_DB_IDS: list[str] = [db.db_type for db in VECTOR_DBS]


@dataclass
class ConnectionResult:
    """Result of a vector DB connection test."""
    success: bool
    error_message: str | None = None


# ---------------------------------------------------------------------------
# VectorDBConnector
# ---------------------------------------------------------------------------


class VectorDBConnector:
    """Handles vector DB selection, credential prompting, connection
    testing, vector store creation, and document ingestion.

    Usage::

        connector = VectorDBConnector(credential_store)
        config = connector.prompt_and_configure()
        result = connector.test_connection(config)
        if result.success:
            store = connector.get_vector_store(config, embeddings)
            count = connector.ingest_documents(docs, store)
    """

    def __init__(self, credential_store: object | None = None) -> None:
        self._credential_store = credential_store

    def prompt_and_configure(
        self,
        embedding_model: EmbeddingModelConfig | None = None,
    ) -> VectorDBConfig:
        """Interactive flow: select DB → prompt credentials → confirm → return config.

        Requirement 9.1-9.3.
        """
        import questionary  # noqa: PLC0415
        from ms_rag.ui.prompts import get_console, print_step, prompt_checkbox  # noqa: PLC0415

        console = get_console()
        print_step(console, 9, "Select Vector Database")

        choices = [
            questionary.Choice(
                title=f"{db.display_name}  —  {db.description}",
                value=db.db_type,
            )
            for db in VECTOR_DBS
        ]

        from ms_rag.ui.prompts import prompt_select, prompt_text  # noqa: PLC0415

        db_type = prompt_select("Select vector database:", choices, console=console)

        db_info = VECTOR_DB_MAP[db_type]
        connection_params: dict[str, str] = {}
        embedding_dimension = self._embedding_dimension(embedding_model)
        self._display_embedding_compatibility(db_info, embedding_model, embedding_dimension, console)

        # Prompt for required credential fields
        if db_info.credential_fields:
            console.print(f"\n  [bold white]Credentials for {db_info.display_name}:[/bold white]")
            for field_name in db_info.credential_fields:
                value = self._prompt_credential(field_name, required=True, console=console)
                if value:
                    connection_params[field_name] = value
                    if self._credential_store is not None:
                        self._credential_store.set(db_type, field_name, value)  # type: ignore[union-attr]

        # Prompt for optional fields
        for field_name in db_info.optional_fields:
            value = self._prompt_credential(field_name, required=False, console=console)
            if value:
                connection_params[field_name] = value

        # Prompt for collection name
        collection_name = prompt_text(
            f"  Collection / index name (default: {db_info.default_collection}):",
            default=db_info.default_collection,
            required=True,
            console=console,
        )

        config = VectorDBConfig(
            db_type=db_type,
            connection_params=connection_params,
            collection_name=collection_name,
            dimension=embedding_dimension,
        )

        # Show summary (Req 9.3)
        self._display_config_summary(config, db_info, console)

        from ms_rag.ui.prompts import prompt_required_confirm  # noqa: PLC0415

        prompt_required_confirm("Proceed with these vector DB settings?", console=console)

        return config

    def reprompt_credentials(self, config: VectorDBConfig) -> VectorDBConfig:
        """Re-prompt required and optional credentials for an existing vector DB config."""
        console = Console()
        db_info = VECTOR_DB_MAP[config.db_type]
        connection_params = dict(config.connection_params)

        if db_info.credential_fields:
            console.print(
                f"\n  [bold white]Re-enter credentials for {db_info.display_name}:[/bold white]"
            )
            for field_name in db_info.credential_fields:
                value = self._prompt_credential(field_name, required=True, console=console)
                connection_params[field_name] = value
                if self._credential_store is not None:
                    self._credential_store.set(config.db_type, field_name, value)  # type: ignore[union-attr]

        for field_name in db_info.optional_fields:
            value = self._prompt_credential(field_name, required=False, console=console)
            if value:
                connection_params[field_name] = value

        updated = VectorDBConfig(
            db_type=config.db_type,
            connection_params=connection_params,
            collection_name=config.collection_name,
            dimension=config.dimension,
        )
        self._display_config_summary(updated, db_info, console)
        return updated

    def test_connection(self, config: VectorDBConfig) -> ConnectionResult:
        """Attempt a lightweight connection to the vector DB.

        Requirement 9.4-9.5.

        Returns:
            ConnectionResult with success flag and optional error message.
        """
        try:
            self._probe_connection(config)
            return ConnectionResult(success=True)
        except Exception as exc:  # noqa: BLE001
            return ConnectionResult(success=False, error_message=str(exc))

    def get_vector_store(
        self,
        config: VectorDBConfig,
        embeddings: object,
    ) -> object:
        """Return a LangChain VectorStore instance for the configured DB.

        Args:
            config:     The VectorDBConfig with connection params.
            embeddings: A LangChain Embeddings instance.

        Returns:
            A LangChain VectorStore (Chroma, PineconeVectorStore, etc.)

        Raises:
            ImportError: If the required package is not installed.
            ValueError:  If the db_type is not recognised.
        """
        db_type = config.db_type
        params = self._resolved_connection_params(config)

        if db_type == "chroma":
            from langchain_chroma import Chroma  # noqa: PLC0415
            persist_dir = (
                params.get("CHROMA_PERSIST_DIRECTORY")
                or params.get("CHROMA_PERSIST_DIR")
                or "./chroma_db"
            )
            return Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings,  # type: ignore[arg-type]
                persist_directory=persist_dir,
            )

        if db_type == "faiss":
            # FAISS requires documents at creation time; return a factory wrapper
            return _FAISSFactory(config=config, embeddings=embeddings)

        if db_type == "pinecone":
            from langchain_pinecone import PineconeVectorStore  # noqa: PLC0415
            import os  # noqa: PLC0415
            os.environ.setdefault("PINECONE_API_KEY", params.get("PINECONE_API_KEY", ""))
            return PineconeVectorStore(
                index_name=params.get("PINECONE_INDEX_NAME", config.collection_name),
                embedding=embeddings,  # type: ignore[arg-type]
            )

        if db_type == "qdrant":
            from langchain_qdrant import QdrantVectorStore  # noqa: PLC0415
            from qdrant_client import QdrantClient  # noqa: PLC0415
            from qdrant_client.http.models import Distance, VectorParams  # noqa: PLC0415
            client = QdrantClient(
                url=params.get("QDRANT_URL", "http://localhost:6333"),
                api_key=params.get("QDRANT_API_KEY"),
            )
            if not client.collection_exists(config.collection_name):
                client.create_collection(
                    collection_name=config.collection_name,
                    vectors_config=VectorParams(
                        size=int(config.dimension or 1536),
                        distance=Distance.COSINE,
                    ),
                )
            return QdrantVectorStore(
                client=client,
                collection_name=config.collection_name,
                embedding=embeddings,  # type: ignore[arg-type]
            )

        if db_type == "weaviate":
            from langchain_weaviate import WeaviateVectorStore  # noqa: PLC0415
            import weaviate  # noqa: PLC0415
            url = params.get("WEAVIATE_URL", "localhost").strip()
            api_key = params.get("WEAVIATE_API_KEY", "").strip()
            auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
            if "weaviate.cloud" in url:
                cluster_url = url if url.startswith(("http://", "https://")) else f"https://{url}"
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=cluster_url,
                    auth_credentials=auth,
                )
            else:
                clean = url.replace("https://", "").replace("http://", "")
                host = clean.split(":")[0]
                port = int(clean.split(":")[-1]) if ":" in clean else 8080
                secure = url.startswith("https://")
                client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=secure,
                    grpc_host=params.get("WEAVIATE_GRPC_HOST", host),
                    grpc_port=int(params.get("WEAVIATE_GRPC_PORT", "50051")),
                    grpc_secure=secure,
                    auth_credentials=auth,
                )
            return WeaviateVectorStore(
                client=client,
                index_name=config.collection_name,
                text_key="text",
                embedding=embeddings,  # type: ignore[arg-type]
            )

        if db_type == "pgvector":
            from langchain_postgres import PGVector  # noqa: PLC0415
            return PGVector(
                embeddings=embeddings,  # type: ignore[arg-type]
                collection_name=config.collection_name,
                connection=params.get("PGVECTOR_CONNECTION_STRING", ""),
            )

        if db_type == "milvus":
            from langchain_milvus import Milvus  # noqa: PLC0415
            from pymilvus import connections  # noqa: PLC0415
            milvus_uri = params.get("MILVUS_URI", "http://localhost:19530")
            connection_args = {
                "uri": milvus_uri,
                "secure": milvus_uri.startswith("https://"),
            }
            if params.get("MILVUS_TOKEN"):
                connection_args["token"] = params["MILVUS_TOKEN"]
            store = Milvus(
                embedding_function=embeddings,  # type: ignore[arg-type]
                collection_name=config.collection_name,
                connection_args=connection_args,
                text_field="page_content",
            )
            alias = getattr(store, "alias", None)
            if isinstance(alias, str) and not connections.has_connection(alias):
                connections.connect(alias=alias, **connection_args)
            return store

        if db_type == "elasticsearch":
            from langchain_elasticsearch import ElasticsearchStore  # noqa: PLC0415
            return ElasticsearchStore(
                index_name=config.collection_name,
                embedding=embeddings,  # type: ignore[arg-type]
                es_url=params.get("ELASTICSEARCH_URL", "http://localhost:9200"),
                es_user=params.get("ELASTICSEARCH_USERNAME"),
                es_password=params.get("ELASTICSEARCH_PASSWORD"),
                es_api_key=params.get("ELASTICSEARCH_API_KEY"),
            )

        if db_type == "opensearch":
            from langchain_community.vectorstores import OpenSearchVectorSearch  # noqa: PLC0415
            return OpenSearchVectorSearch(
                index_name=config.collection_name,
                embedding_function=embeddings,  # type: ignore[arg-type]
                opensearch_url=params.get("OPENSEARCH_URL", "http://localhost:9200"),
                http_auth=(
                    params.get("OPENSEARCH_USERNAME", "admin"),
                    params.get("OPENSEARCH_PASSWORD", "admin"),
                ),
                engine=params.get("OPENSEARCH_ENGINE", "faiss"),
            )

        if db_type == "azure_ai_search":
            from langchain_community.vectorstores import AzureSearch  # noqa: PLC0415
            return AzureSearch(
                azure_search_endpoint=params.get("AZURE_SEARCH_ENDPOINT", ""),
                azure_search_key=params.get("AZURE_SEARCH_KEY") or params.get("AZURE_SEARCH_API_KEY", ""),
                index_name=params.get("AZURE_SEARCH_INDEX_NAME", config.collection_name),
                embedding_function=embeddings,  # type: ignore[arg-type]
            )

        if db_type == "mongodb_atlas":
            from langchain_mongodb import MongoDBAtlasVectorSearch  # noqa: PLC0415
            from pymongo import MongoClient  # noqa: PLC0415
            client = MongoClient(
                params.get("MONGODB_ATLAS_CLUSTER_URI")
                or params.get("MONGODB_ATLAS_CONNECTION_STRING", "")
            )
            db = client[params.get("MONGODB_ATLAS_DB_NAME", "ms_rag_db")]
            collection = db[params.get("MONGODB_ATLAS_COLLECTION_NAME", config.collection_name)]
            return MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,  # type: ignore[arg-type]
                index_name=params.get("MONGODB_ATLAS_INDEX_NAME", "vector_index"),
                dimensions=int(config.dimension or 1536),
                auto_create_index=True,
                auto_index_timeout=int(params.get("MONGODB_ATLAS_AUTO_INDEX_TIMEOUT", "120")),
            )

        if db_type == "redis":
            from langchain_redis import RedisConfig, RedisVectorStore  # noqa: PLC0415

            redis_config = RedisConfig(
                index_name=params.get("REDIS_INDEX_NAME", config.collection_name),
                redis_url=params.get("REDIS_URL", "redis://localhost:6379"),
            )
            return RedisVectorStore(
                embeddings=embeddings,  # type: ignore[arg-type]
                config=redis_config,
            )

        raise ValueError(f"Unsupported vector DB type: {db_type!r}")

    def ingest_documents(
        self,
        docs: list,
        vector_store: object,
    ) -> int:
        """Add documents to the vector store with a Rich progress bar.

        Requirement 9.6-9.7.

        Args:
            docs:         List of LangChain Document objects.
            vector_store: A LangChain VectorStore instance.

        Returns:
            Number of documents successfully stored.
        """
        if not docs:
            return 0

        console = Console()
        stored = self._ingest_batch(docs, vector_store)

        console.print(
            f"[green]  ✓ Ingestion complete: [bold]{stored}[/bold] chunks stored.[/green]"
        )
        return stored

    def _ingest_batch(self, docs: list, vector_store: object) -> int:
        """Internal batched ingestion with progress bar.

        Separated for testability.
        """
        batch_size = 50
        total = len(docs)
        stored = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Ingesting documents...[/bold cyan]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} chunks"),
            console=Console(),
        ) as progress:
            task = progress.add_task("ingesting", total=total)

            for i in range(0, total, batch_size):
                batch = docs[i: i + batch_size]
                sanitize_documents(batch)
                vector_store.add_documents(batch)  # type: ignore[union-attr]
                stored += len(batch)
                progress.update(task, completed=stored)

        return stored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prompt_credential(
        self,
        field_name: str,
        required: bool,
        console: object,
    ) -> str | None:
        """Prompt for a single credential field."""
        suffix = "" if required else " (optional, press Enter to skip)"
        is_secret = any(
            field_name.upper().endswith(s)
            for s in ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")
        )
        prompt_text = f"    {field_name}{suffix}:"

        while True:
            if is_secret:
                value: str | None = questionary.password(prompt_text).ask()
            else:
                value = questionary.text(prompt_text, default="").ask()

            if value is None:
                value = ""
            value = value.strip()

            if not value:
                if required:
                    console.print(  # type: ignore[union-attr]
                        f"[red]  ✗ {field_name} is required.[/red]"
                    )
                    continue
                return None
            return value

    def _display_config_summary(
        self,
        config: VectorDBConfig,
        db_info: VectorDBInfo,
        console: object,
    ) -> None:
        """Display a Rich table summarising the connection parameters."""
        table = Table(title=f"Connection Parameters — {db_info.display_name}", border_style="cyan")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="green")

        table.add_row("Database Type", db_info.display_name)
        table.add_row("Collection / Index", config.collection_name)
        if config.dimension:
            table.add_row("Embedding Dimension", str(config.dimension))

        for key, val in config.connection_params.items():
            # Mask secret values
            is_secret = any(key.upper().endswith(s) for s in ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD"))
            display_val = "***" + val[-4:] if is_secret and len(val) > 4 else val
            table.add_row(key, display_val)

        console.print(table)  # type: ignore[union-attr]

    @staticmethod
    def _embedding_dimension(embedding_model: EmbeddingModelConfig | None) -> int | None:
        if embedding_model is None:
            return None
        from ms_rag.ingestion.vectorization_module import get_embedding_dimension  # noqa: PLC0415

        return get_embedding_dimension(embedding_model)

    @staticmethod
    def _display_embedding_compatibility(
        db_info: VectorDBInfo,
        embedding_model: EmbeddingModelConfig | None,
        embedding_dimension: int | None,
        console: object,
    ) -> None:
        """Explain dimension/index compatibility before the user confirms the DB."""
        model_name = embedding_model.model_id if embedding_model else "not selected"
        dimension = f"{embedding_dimension} dimensions" if embedding_dimension else "custom/unknown dimension"
        console.print(  # type: ignore[union-attr]
            "\n[bold white]Embedding compatibility note[/bold white]\n"
            f"  Model: [cyan]{model_name}[/cyan]\n"
            f"  Output: [cyan]{dimension}[/cyan]\n"
            f"  Target DB: [cyan]{db_info.display_name}[/cyan]\n"
            "  [yellow]If an existing collection/index was created with a different embedding dimension, "
            "create a new collection or re-ingest before querying.[/yellow]\n"
        )

    def _probe_connection(self, config: VectorDBConfig) -> None:
        """Lightweight connection probe — raises on failure."""
        db_type = config.db_type
        params = self._resolved_connection_params(config)

        if db_type in ("chroma", "faiss"):
            # Local DBs — always succeed
            return

        if db_type == "pinecone":
            from pinecone import Pinecone  # noqa: PLC0415
            pc = Pinecone(api_key=params.get("PINECONE_API_KEY", ""))
            pc.list_indexes()

        elif db_type == "qdrant":
            from qdrant_client import QdrantClient  # noqa: PLC0415
            client = QdrantClient(
                url=params.get("QDRANT_URL", "http://localhost:6333"),
                api_key=params.get("QDRANT_API_KEY"),
                timeout=5,
            )
            client.get_collections()

        elif db_type == "weaviate":
            import weaviate  # noqa: PLC0415
            url = params.get("WEAVIATE_URL", "http://localhost:8080")
            client = weaviate.connect_to_local(
                host=url.replace("http://", "").split(":")[0],
                port=int(url.split(":")[-1]) if ":" in url else 8080,
            )
            client.is_ready()
            client.close()

        elif db_type == "pgvector":
            import psycopg2  # noqa: PLC0415
            conn = psycopg2.connect(params.get("PGVECTOR_CONNECTION_STRING", ""))
            conn.close()

        else:
            # For other DBs, a simple import check is sufficient for the probe
            # Real connection test happens in get_vector_store()
            pass

    @staticmethod
    def _resolved_connection_params(config: VectorDBConfig) -> dict[str, str]:
        """Resolve sanitized env-var markers in connection_params to runtime values."""
        resolved: dict[str, str] = {}
        for key, value in config.connection_params.items():
            if value == key:
                resolved[key] = os.getenv(key, "")
            else:
                resolved[key] = value
        return resolved


class _FAISSFactory:
    """Wrapper that creates a FAISS index when documents are first added."""

    def __init__(self, config: VectorDBConfig, embeddings: object) -> None:
        self._config = config
        self._embeddings = embeddings
        self._store: object | None = None
        self._load_existing_index()

    def add_documents(self, docs: list) -> None:
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415

        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)  # type: ignore[arg-type]
        else:
            self._store.add_documents(docs)  # type: ignore[union-attr]
        self._save_index()

    def as_retriever(self, **kwargs: object) -> object:
        if self._store is None:
            index_path = self._index_path()
            suffix = (
                f" No persisted index was found at {index_path}."
                if index_path is not None
                else " Run ingestion first or configure FAISS_INDEX_PATH for session reloads."
            )
            raise RuntimeError(f"FAISS store has no documents yet.{suffix}")
        return self._store.as_retriever(**kwargs)  # type: ignore[union-attr]

    def similarity_search(self, query: str, k: int = 4) -> list:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)  # type: ignore[union-attr]

    def _index_path(self) -> Path | None:
        raw = self._config.connection_params.get("FAISS_INDEX_PATH")
        if not raw:
            raw = str(Path("./faiss_indexes") / self._config.collection_name)
            self._config.connection_params["FAISS_INDEX_PATH"] = raw
        return Path(raw).expanduser()

    def _load_existing_index(self) -> None:
        index_path = self._index_path()
        if index_path is None or not index_path.exists():
            return

        from langchain_community.vectorstores import FAISS  # noqa: PLC0415

        self._store = FAISS.load_local(
            str(index_path),
            self._embeddings,  # type: ignore[arg-type]
            allow_dangerous_deserialization=True,
        )

    def _save_index(self) -> None:
        index_path = self._index_path()
        if index_path is None or self._store is None:
            return
        index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(index_path))  # type: ignore[union-attr]
