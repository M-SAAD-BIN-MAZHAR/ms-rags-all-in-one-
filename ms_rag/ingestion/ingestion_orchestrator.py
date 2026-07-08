"""Ingestion Orchestrator for MS_RAG.

Coordinates the full ingestion pipeline:
  discover_documents → load → chunk → embed → store

Implements:
- Recursive directory discovery matching selected doc types (Req 20.2)
- Per-document failure isolation with error logging (Req 19.2)
- Exponential backoff retry for external API calls (Req 19.1)
- Progress display via Rich (Req 9.6)
- Post-ingestion summary (Req 9.7)
- Loader assignment by detected file type / MIME (Req 20.3)
- YouTube URL prompt for transcript language (Req 20.4)
"""

from __future__ import annotations

from copy import copy
from datetime import UTC, datetime
import time
from pathlib import Path
from typing import Callable
import warnings

from langchain_core.documents import Document

try:
    from rich.console import Console
except ImportError:
    Console = None  # type: ignore[assignment]

from ms_rag.ingestion.document_type_selector import EXTENSION_TO_DOCTYPE
from ms_rag.ingestion.chunking_engine import ChunkingEngine
from ms_rag.models import (
    ChunkingConfig,
    EmbeddingModelConfig,
    IngestionResult,
    VectorDBConfig,
)
from ms_rag.utils.exceptions import IngestionError
from ms_rag.utils.metadata import sanitize_documents
from ms_rag.utils.telemetry import TelemetryReporter


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def retry_with_backoff(
    fn: Callable,
    max_attempts: int = 3,
    delays: tuple[float, ...] = (1.0, 2.0),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> object:
    """Call *fn()* with exponential backoff, re-raising after *max_attempts*.

    Args:
        fn:           Zero-argument callable to invoke.
        max_attempts: Maximum number of attempts (default 3).
        delays:       Seconds to wait between attempts (default: 1s, 2s).
        on_retry:     Optional callback(attempt_number, exception) called before
                      each retry.

    Returns:
        The return value of *fn()* on success.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                if on_retry:
                    on_retry(attempt + 1, exc)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _empty_retrieval_state() -> dict[str, object]:
    """Return the in-memory state needed by advanced retrievers."""
    return {
        "parent_documents": {},
        "chunk_documents": [],
    }


def _coalesce_documents(docs: list, chunk_size: int) -> list:
    """Merge adjacent same-source element documents up to ``chunk_size``.

    Element-based loaders (Unstructured PDF/Word, HTML, etc.) return one
    Document per line/element. ``TextSplitter.split_documents`` only ever splits
    *within* a Document — it never merges across them — so those loaders would
    otherwise yield many ~100-character chunks that wreck retrieval. This
    coalesces consecutive elements from the same source into blocks near the
    target chunk size before splitting, so the splitter produces real
    section-sized chunks (and parent-child parents become real sections).

    Documents that are already at/above the target are left as their own block;
    the splitter still divides them normally afterwards.
    """
    if not docs:
        return docs
    target = chunk_size if isinstance(chunk_size, int) and chunk_size > 0 else 1500

    merged: list = []
    buf_texts: list[str] = []
    buf_len = 0
    buf_meta: dict | None = None
    buf_source: object = object()  # sentinel that never equals a real source

    def _flush() -> None:
        nonlocal buf_texts, buf_len, buf_meta
        if buf_texts:
            merged.append(Document(page_content="\n\n".join(buf_texts), metadata=dict(buf_meta or {})))
        buf_texts = []
        buf_len = 0
        buf_meta = None

    for doc in docs:
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        meta = dict(getattr(doc, "metadata", {}) or {})
        source = meta.get("source")
        # New source, or adding this element would exceed the target → flush.
        if buf_texts and (source != buf_source or buf_len + len(text) + 2 > target):
            _flush()
        if not buf_texts:
            buf_source = source
            buf_meta = meta
        buf_texts.append(text)
        buf_len += len(text) + 2
    _flush()

    return merged or docs


def _copy_document(doc: object) -> object:
    """Make a shallow document copy so metadata edits do not surprise loaders."""
    try:
        copied = copy(doc)
        copied.metadata = dict(getattr(doc, "metadata", {}) or {})
        return copied
    except Exception:  # noqa: BLE001
        return doc


def _document_text(doc: object) -> str:
    """Read text from LangChain, LlamaIndex, or simple document-like objects."""
    page_content = getattr(doc, "page_content", None)
    if page_content is not None:
        return str(page_content)

    get_content = getattr(doc, "get_content", None)
    if callable(get_content):
        try:
            return str(get_content() or "")
        except TypeError:
            return str(get_content(metadata_mode="none") or "")

    text = getattr(doc, "text", None)
    if text is not None:
        return str(text)

    return str(doc or "")


def _normalize_documents(docs: list, source: str, loader: str) -> list[Document]:
    """Convert loader output into LangChain Documents.

    Some parser SDKs, especially LlamaParse/LlamaIndex, return their own
    Document shape with `.text` or `.get_content()` instead of `.page_content`.
    Normalize at the ingestion boundary so chunkers, retrievers, and vector DBs
    receive one predictable document contract.
    """
    normalized: list[Document] = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.setdefault("source", source)
            metadata.setdefault("loader", loader)
            normalized.append(
                Document(
                    page_content=str(getattr(doc, "page_content") or ""),
                    metadata=metadata,
                )
            )
            continue

        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata.setdefault("source", source)
        metadata.setdefault("loader", loader)
        normalized.append(Document(page_content=_document_text(doc), metadata=metadata))
    return normalized


def _prepare_parent_documents(docs: list, source: str) -> dict[str, object]:
    """Attach stable parent IDs and timestamps to loaded documents."""
    parent_documents: dict[str, object] = {}
    prepared_docs: list = []
    ingested_at = datetime.now(UTC).isoformat()

    for index, doc in enumerate(docs):
        prepared = _copy_document(doc)
        metadata = dict(getattr(prepared, "metadata", {}) or {})
        parent_id = metadata.get("ms_rag_parent_id") or f"{source}::parent::{index}"
        metadata.update(
            {
                "source": metadata.get("source", source),
                "ms_rag_parent_id": parent_id,
                "ms_rag_ingested_at": metadata.get("ms_rag_ingested_at", ingested_at),
            }
        )
        prepared.metadata = metadata
        parent_documents[parent_id] = prepared
        prepared_docs.append(prepared)

    return {
        "documents": prepared_docs,
        "parent_documents": parent_documents,
    }


def _prepare_child_documents(chunks: list) -> None:
    """Attach child IDs, parent IDs, source IDs, and recency metadata to chunks."""
    ingested_at = datetime.now(UTC).isoformat()
    for index, chunk in enumerate(chunks):
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        parent_id = metadata.get("ms_rag_parent_id") or metadata.get("source") or f"unknown_parent::{index}"
        child_id = metadata.get("ms_rag_child_id") or f"{parent_id}::child::{index}"
        metadata.update(
            {
                "ms_rag_parent_id": parent_id,
                "ms_rag_child_id": child_id,
                "ms_rag_multi_vector_source_id": metadata.get("ms_rag_multi_vector_source_id", child_id),
                "ms_rag_ingested_at": metadata.get("ms_rag_ingested_at", ingested_at),
            }
        )
        chunk.metadata = metadata


def _attach_retrieval_state(vector_store: object, state: dict[str, object]) -> None:
    """Attach backend-independent advanced retrieval state to a vector store."""
    setattr(vector_store, "_ms_rag_parent_documents", state.get("parent_documents", {}))
    setattr(vector_store, "_ms_rag_chunk_documents", state.get("chunk_documents", []))


# ---------------------------------------------------------------------------
# IngestionOrchestrator
# ---------------------------------------------------------------------------


class IngestionOrchestrator:
    """Coordinates the full document ingestion pipeline.

    Usage::

        orchestrator = IngestionOrchestrator()
        sources = ["./docs/", "https://example.com/page"]
        paths = orchestrator.discover_documents(sources, ["pdf", "html"])
        result = orchestrator.ingest(
            sources=sources,
            loader_map={"pdf": "PyPDFLoader", "html": "BSHTMLLoader"},
            chunking_config=chunking_cfg,
            embedding_model=embedding_cfg,
            vector_db=vector_db_cfg,
            vector_store=store,
        )
    """

    def __init__(self, credential_store: object | None = None) -> None:
        self._credential_store = credential_store

    def discover_documents(
        self,
        sources: list[str],
        doc_types: list[str],
    ) -> list[Path | str]:
        """Recursively discover files and URLs matching selected doc types.

        Requirement 20.1-20.4.

        Args:
            sources:   File paths, directory paths, or URLs.
            doc_types: Selected doc type IDs from DocumentTypeSelector.

        Returns:
            Flat list of Path objects (for files) and raw strings (for URLs/YouTube).
        """
        discovered: list[Path | str] = []
        selected_extensions: set[str] = {
            ext
            for ext, dt in EXTENSION_TO_DOCTYPE.items()
            if dt in doc_types
        }

        for source in sources:
            source_str = str(source).strip()

            # YouTube URL
            if "youtube.com" in source_str or "youtu.be" in source_str:
                if "youtube" in doc_types:
                    discovered.append(source_str)
                continue

            # HTTP/HTTPS URL
            if source_str.startswith(("http://", "https://")):
                if "url" in doc_types:
                    discovered.append(source_str)
                continue

            path = Path(source_str)

            # Directory — recursive walk
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        ext = child.suffix.lower()
                        if ext in selected_extensions:
                            discovered.append(child)
                continue

            # Single file
            if path.is_file():
                ext = path.suffix.lower()
                if ext in selected_extensions:
                    discovered.append(path)

        return discovered

    def ingest(
        self,
        sources: list[str],
        loader_map: dict[str, str],
        chunking_config: ChunkingConfig,
        embedding_model: EmbeddingModelConfig,
        vector_db: VectorDBConfig,
        vector_store: object,
        embeddings: object | None = None,
        llm: object | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        youtube_language: str = "en",
    ) -> IngestionResult:
        """Run the full load → chunk → embed → store pipeline.

        Requirement 9.6, 19.2, 20.3.

        Per-document failures are isolated: failed documents are logged to
        IngestionResult.failed_documents and ingestion continues.

        Args:
            sources:           File paths, directories, or URLs.
            loader_map:        {doc_type_id: loader_class_name}
            chunking_config:   ChunkingConfig from ChunkingConfigurator.
            embedding_model:   EmbeddingModelConfig (not used directly here —
                               embeddings are already in the vector_store).
            vector_db:         VectorDBConfig.
            vector_store:      Initialised LangChain VectorStore instance.
            progress_callback: Optional callback(processed, total).
            youtube_language:  Language code for YouTube transcripts (default "en").

        Returns:
            IngestionResult with chunk_count, collection_name, failed_documents.
        """
        console = Console()
        telemetry = TelemetryReporter()
        doc_types = list(loader_map.keys())
        discovered = self.discover_documents(sources, doc_types)
        total = len(discovered)

        if total == 0:
            console.print("[yellow]  No documents discovered matching selected types.[/yellow]")
            return IngestionResult(
                chunk_count=0,
                collection_name=vector_db.collection_name,
                failed_documents=[],
            )

        # Display discovery summary (Req 20.5)
        console.print(
            f"\n  [bold cyan]Discovered {total} document(s).[/bold cyan] Starting ingestion...\n"
        )

        chunker = ChunkingEngine()
        splitter = chunker.get_splitter(chunking_config)
        if chunking_config.strategy == "semantic" and hasattr(splitter, "with_embeddings"):
            resolved_embeddings = embeddings or getattr(vector_store, "_ms_rag_embeddings", None)
            if resolved_embeddings is None:
                raise RuntimeError(
                    "Semantic chunking requires the selected embeddings object. "
                    "Re-run setup so ingestion can bind semantic chunking to the embedding model."
                )
            splitter = splitter.with_embeddings(resolved_embeddings)
        if chunking_config.strategy == "agentic" and hasattr(splitter, "with_llm"):
            if llm is None:
                raise RuntimeError(
                    "Agentic chunking requires the selected LLM during ingestion. "
                    "Choose a non-agentic chunking strategy or configure a generation model."
                )
            splitter = splitter.with_llm(llm)

        total_chunks = 0
        failed_documents: list[tuple[str, str]] = []
        keyword_corpus: list[str] = []
        retrieval_state = _empty_retrieval_state()

        with telemetry.span(
            "ingestion.run",
            source_count=total,
            collection_name=vector_db.collection_name,
            db_type=vector_db.db_type,
        ):
            for idx, source in enumerate(discovered):
                source_str = str(source)
                try:
                    with telemetry.span("ingestion.source", source=source_str):
                        # Load document(s)
                        docs = self._load_source(
                            source=source,
                            loader_map=loader_map,
                            youtube_language=youtube_language,
                        )
                        docs = _coalesce_documents(docs, chunking_config.chunk_size)
                        prepared_docs = _prepare_parent_documents(docs, source_str)
                        retrieval_state["parent_documents"].update(prepared_docs["parent_documents"])

                        # Chunk
                        chunks = splitter.split_documents(prepared_docs["documents"])
                        _prepare_child_documents(chunks)
                        sanitize_documents(chunks)
                        chunks = [
                            chunk
                            for chunk in chunks
                            if getattr(chunk, "page_content", "").strip()
                        ]
                        if not chunks:
                            raise IngestionError(
                                "No extractable text chunks were produced. "
                                "For scanned/image PDFs, install Poppler/Tesseract or use LlamaParse.",
                                document_path=source_str,
                            )
                        retrieval_state["chunk_documents"].extend(chunks)
                        keyword_corpus.extend(
                            chunk.page_content
                            for chunk in chunks
                            if getattr(chunk, "page_content", "").strip()
                        )

                        # Store (with retry)
                        def _store(c: list = chunks, vs: object = vector_store) -> None:
                            vs.add_documents(c)  # type: ignore[union-attr]

                        retry_with_backoff(_store)
                        total_chunks += len(chunks)

                except Exception as exc:  # noqa: BLE001
                    failed_documents.append((source_str, str(exc)))
                    telemetry.record_error("ingestion.source_failed", str(exc), source=source_str)
                    console.print(
                        f"[yellow]  ⚠ Skipped: {source_str} — {exc}[/yellow]"
                    )

                if progress_callback:
                    progress_callback(idx + 1, total)

        result = IngestionResult(
            chunk_count=total_chunks,
            collection_name=vector_db.collection_name,
            failed_documents=failed_documents,
        )
        setattr(vector_store, "_ms_rag_keyword_corpus", keyword_corpus)
        _attach_retrieval_state(vector_store, retrieval_state)

        # Post-ingestion summary (Req 9.7)
        self._display_summary(result, console)
        return result

    def build_keyword_corpus(
        self,
        *,
        sources: list[str],
        loader_map: dict[str, str],
        chunking_config: ChunkingConfig,
        youtube_language: str = "en",
    ) -> list[str]:
        """Load and chunk sources to build a backend-independent keyword corpus."""
        doc_types = list(loader_map.keys())
        discovered = self.discover_documents(sources, doc_types)
        splitter = ChunkingEngine().get_splitter(chunking_config)
        texts: list[str] = []

        for source in discovered:
            docs = self._load_source(
                source=source,
                loader_map=loader_map,
                youtube_language=youtube_language,
            )
            docs = _coalesce_documents(docs, chunking_config.chunk_size)
            chunks = splitter.split_documents(docs)
            texts.extend(
                chunk.page_content
                for chunk in chunks
                if getattr(chunk, "page_content", "").strip()
            )

        return texts

    def build_retrieval_state(
        self,
        *,
        sources: list[str],
        loader_map: dict[str, str],
        chunking_config: ChunkingConfig,
        youtube_language: str = "en",
    ) -> dict[str, object]:
        """Load and chunk sources to rebuild advanced retriever state.

        This state is backend-independent and is used by Parent-Child,
        Multi-Vector, and Time-Weighted retrieval when a saved session is loaded.
        """
        doc_types = list(loader_map.keys())
        discovered = self.discover_documents(sources, doc_types)
        splitter = ChunkingEngine().get_splitter(chunking_config)
        state = _empty_retrieval_state()

        for source in discovered:
            source_str = str(source)
            docs = self._load_source(
                source=source,
                loader_map=loader_map,
                youtube_language=youtube_language,
            )
            docs = _coalesce_documents(docs, chunking_config.chunk_size)
            prepared_docs = _prepare_parent_documents(docs, source_str)
            state["parent_documents"].update(prepared_docs["parent_documents"])
            chunks = splitter.split_documents(prepared_docs["documents"])
            _prepare_child_documents(chunks)
            sanitize_documents(chunks)
            state["chunk_documents"].extend(chunks)

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_source(
        self,
        source: Path | str,
        loader_map: dict[str, str],
        youtube_language: str = "en",
    ) -> list:
        """Load a single source using the appropriate LangChain loader.

        Requirement 20.3: loader is selected by file extension / source type.

        Args:
            source:           A Path or URL string.
            loader_map:       {doc_type_id: loader_class_name}
            youtube_language: Language code for YouTube transcripts.

        Returns:
            List of LangChain Document objects.

        Raises:
            IngestionError: If no compatible loader is found or loading fails.
        """
        source_str = str(source)

        # Determine doc_type
        if "youtube.com" in source_str or "youtu.be" in source_str:
            doc_type = "youtube"
        elif source_str.startswith(("http://", "https://")):
            doc_type = "url"
        elif isinstance(source, Path):
            ext = source.suffix.lower()
            from ms_rag.ingestion.document_type_selector import EXTENSION_TO_DOCTYPE  # noqa: PLC0415
            doc_type = EXTENSION_TO_DOCTYPE.get(ext, "txt")
        else:
            doc_type = "txt"

        loader_class_name = loader_map.get(doc_type)
        if not loader_class_name:
            raise IngestionError(
                f"No loader configured for doc_type {doc_type!r}",
                document_path=source_str,
            )

        docs = self._invoke_loader(loader_class_name, source_str, youtube_language)
        return _normalize_documents(docs, source_str, loader_class_name)

    def _invoke_loader(
        self,
        loader_class_name: str,
        source: str,
        youtube_language: str = "en",
    ) -> list:
        """Instantiate and invoke a LangChain loader by class name.

        Args:
            loader_class_name: e.g. "PyPDFLoader", "WebBaseLoader"
            source:            File path string or URL string.
            youtube_language:  Language for YoutubeLoader.

        Returns:
            List of Document objects.
        """
        # ── PDF loaders ──────────────────────────────────────────────
        if loader_class_name == "PyPDFLoader":
            from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
            return PyPDFLoader(source).load()

        if loader_class_name == "UnstructuredPDFLoader":
            try:
                from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
                return UnstructuredLoader(source).load()
            except Exception as exc:  # noqa: BLE001
                if "poppler" in str(exc).lower() or "page count" in str(exc).lower():
                    raise IngestionError(
                        "UnstructuredPDFLoader needs Poppler for this PDF. "
                        "Install Poppler and add it to PATH, or choose LlamaParse for cloud parsing.",
                        document_path=source,
                    ) from exc
                warnings.warn(
                    f"UnstructuredPDFLoader could not parse {source}; falling back to PyPDFLoader: {exc}",
                    stacklevel=2,
                )
                from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
                return PyPDFLoader(source).load()

        if loader_class_name == "PDFPlumberLoader":
            from langchain_community.document_loaders import PDFPlumberLoader  # noqa: PLC0415
            return PDFPlumberLoader(source).load()

        if loader_class_name in {"CamelotLoader", "TabulaLoader"}:
            try:
                if loader_class_name == "CamelotLoader":
                    camelot_docs = self._load_with_camelot(source)
                    if camelot_docs:
                        return camelot_docs
                    from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
                    return PyPDFLoader(source).load()
                from langchain_community.document_loaders import UnstructuredPDFLoader  # noqa: PLC0415
                return UnstructuredPDFLoader(source, mode="elements", strategy="fast").load()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"{loader_class_name} could not parse {source}; falling back to PyPDFLoader: {exc}",
                    stacklevel=2,
                )
                from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
                return PyPDFLoader(source).load()

        # ── DOCX loaders ─────────────────────────────────────────────
        if loader_class_name == "UnstructuredWordDocumentLoader":
            try:
                from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
                return UnstructuredLoader(source).load()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"UnstructuredWordDocumentLoader could not parse {source}; falling back to Docx2txtLoader: {exc}",
                    stacklevel=2,
                )
                from langchain_community.document_loaders import Docx2txtLoader  # noqa: PLC0415
                return Docx2txtLoader(source).load()

        if loader_class_name == "Docx2txtLoader":
            from langchain_community.document_loaders import Docx2txtLoader  # noqa: PLC0415
            return Docx2txtLoader(source).load()

        # ── CSV / Excel ───────────────────────────────────────────────
        if loader_class_name == "CSVLoader":
            from langchain_community.document_loaders.csv_loader import CSVLoader  # noqa: PLC0415
            return CSVLoader(source).load()

        if loader_class_name == "UnstructuredCSVLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        if loader_class_name == "UnstructuredExcelLoader":
            try:
                from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
                return UnstructuredLoader(source).load()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"UnstructuredExcelLoader could not parse {source}; falling back to pandas: {exc}",
                    stacklevel=2,
                )
                return self._load_dataframe(source)

        if loader_class_name == "PandasDataFrameLoader":
            return self._load_dataframe(source)

        # ── Plain text ────────────────────────────────────────────────
        if loader_class_name == "TextLoader":
            from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
            return TextLoader(source, encoding="utf-8").load()

        # ── PPTX ──────────────────────────────────────────────────────
        if loader_class_name == "UnstructuredPowerPointLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── HTML ──────────────────────────────────────────────────────
        if loader_class_name == "BSHTMLLoader":
            from langchain_community.document_loaders import BSHTMLLoader  # noqa: PLC0415
            return BSHTMLLoader(source, open_encoding="utf-8").load()

        if loader_class_name == "UnstructuredHTMLLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── Web URLs ──────────────────────────────────────────────────
        if loader_class_name == "WebBaseLoader":
            from langchain_community.document_loaders import WebBaseLoader  # noqa: PLC0415
            return WebBaseLoader(source).load()

        if loader_class_name == "AsyncHtmlLoader":
            from langchain_community.document_loaders import AsyncHtmlLoader  # noqa: PLC0415
            return AsyncHtmlLoader([source]).load()

        if loader_class_name == "FireCrawlLoader":
            from langchain_community.document_loaders import FireCrawlLoader  # noqa: PLC0415
            from ms_rag.utils.credentials import env_from_store, temporary_env  # noqa: PLC0415

            with temporary_env(env_from_store(self._credential_store, "firecrawl", ("FIRECRAWL_API_KEY",))):
                return FireCrawlLoader(url=source, mode="scrape").load()

        if loader_class_name == "ApifyWebScraper":
            if str(source).startswith(("http://", "https://")):
                raise RuntimeError(
                    "ApifyWebScraper needs an Apify dataset ID or a project-specific actor workflow, "
                    "not a direct URL. Use WebBaseLoader/FireCrawlLoader for direct URLs, or pass an "
                    "Apify dataset ID produced by your approved actor."
                )
            from langchain_community.document_loaders import ApifyDatasetLoader  # noqa: PLC0415
            from langchain_core.documents import Document  # noqa: PLC0415
            from ms_rag.utils.credentials import env_from_store, temporary_env  # noqa: PLC0415

            with temporary_env(env_from_store(self._credential_store, "apify", ("APIFY_API_TOKEN",))):
                return ApifyDatasetLoader(
                    dataset_id=source,
                    dataset_mapping_function=lambda item: Document(
                        page_content=str(item.get("text") or item.get("content") or item),
                        metadata={"source": source, "loader": "ApifyWebScraper"},
                    ),
                ).load()

        if loader_class_name == "DoclingLoader":
            try:
                from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]  # noqa: PLC0415
                from langchain_core.documents import Document  # noqa: PLC0415

                converter = DocumentConverter()
                result = converter.convert(source)
                docling_doc = getattr(result, "document", result)
                # DoclingDocument exposes structured exporters, NOT a `.text`
                # attribute. Using getattr(..., "text", str(doc)) silently dumps
                # the object repr (schema_name='DoclingDocument' ...) instead of
                # the actual content. Export to markdown (keeps headings), then
                # plain text, and only then fall back — never to the repr.
                text = ""
                for exporter in ("export_to_markdown", "export_to_text"):
                    fn = getattr(docling_doc, exporter, None)
                    if callable(fn):
                        try:
                            candidate = str(fn() or "")
                        except Exception:  # noqa: BLE001
                            continue
                        if candidate.strip():
                            text = candidate
                            break
                if not text.strip():
                    attr = getattr(docling_doc, "text", None)
                    if isinstance(attr, str) and attr.strip():
                        text = attr
                if not text.strip():
                    raise IngestionError(
                        "DoclingLoader converted the document but could not extract any text. "
                        "Try PyPDFLoader/Unstructured, or verify the file is not empty/scanned.",
                        document_path=source,
                    )
                return _normalize_documents(
                    [Document(page_content=text, metadata={"source": source})],
                    source,
                    loader_class_name,
                )
            except ImportError as exc:
                raise ImportError(
                    "DoclingLoader requires the docling package. "
                    "Install it with `pip install docling` or choose PyPDFLoader for basic text extraction."
                ) from exc

        if loader_class_name == "LlamaParseLoader":
            try:
                from llama_parse import LlamaParse  # type: ignore[import-not-found]  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "LlamaParseLoader requires the llama-parse package. "
                    "Install it with `pip install llama-parse` and set LLAMA_CLOUD_API_KEY."
                ) from exc
            from ms_rag.utils.credentials import env_from_store, temporary_env  # noqa: PLC0415

            with temporary_env(env_from_store(self._credential_store, "llamaparse", ("LLAMA_CLOUD_API_KEY",))):
                parser = LlamaParse(result_type="markdown")
                return _normalize_documents(
                    parser.load_data(source),
                    source,
                    loader_class_name,
                )

        # ── YouTube ───────────────────────────────────────────────────
        if loader_class_name == "YoutubeLoader":
            from langchain_community.document_loaders import YoutubeLoader  # noqa: PLC0415
            video_id = self._extract_youtube_id(source)
            return YoutubeLoader(
                video_id=video_id,
                language=youtube_language,
            ).load()

        # ── Markdown ──────────────────────────────────────────────────
        if loader_class_name == "UnstructuredMarkdownLoader":
            try:
                from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
                return UnstructuredLoader(source).load()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"UnstructuredMarkdownLoader could not parse {source}; falling back to TextLoader: {exc}",
                    stacklevel=2,
                )
                from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
                return TextLoader(source, encoding="utf-8").load()

        # ── JSON ──────────────────────────────────────────────────────
        if loader_class_name == "JSONLoader":
            try:
                from langchain_community.document_loaders import JSONLoader  # noqa: PLC0415
                return JSONLoader(file_path=source, jq_schema=".", text_content=False).load()
            except ImportError as exc:
                warnings.warn(
                    f"JSONLoader dependency jq is unavailable; using built-in JSON fallback: {exc}",
                    stacklevel=2,
                )
                return self._load_json_fallback(source)

        # ── XML ───────────────────────────────────────────────────────
        if loader_class_name == "UnstructuredXMLLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── Images / OCR ──────────────────────────────────────────────
        if loader_class_name == "UnstructuredImageLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── eBook / RTF ───────────────────────────────────────────────
        if loader_class_name in {"UnstructuredEPubLoader", "UnstructuredRTFLoader"}:
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── Source code ───────────────────────────────────────────────
        if loader_class_name == "GenericLoader":
            from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
            return TextLoader(source, encoding="utf-8").load()

        # ── SQL ───────────────────────────────────────────────────────
        if loader_class_name == "SQLDatabaseLoader":
            from langchain_community.document_loaders import SQLDatabaseLoader  # noqa: PLC0415
            from langchain_community.utilities import SQLDatabase  # noqa: PLC0415
            db = SQLDatabase.from_uri(source)
            loader = SQLDatabaseLoader(query="SELECT * FROM documents LIMIT 1000", db=db)
            return loader.load()

        # ── MongoDB ───────────────────────────────────────────────────
        if loader_class_name == "MongoDBAtlasLoader":
            from langchain_community.document_loaders import MongodbLoader  # noqa: PLC0415
            return MongodbLoader(connection_string=source, db_name="ms_rag", collection_name="docs").load()

        # ── Fallback: plain text ──────────────────────────────────────
        from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
        return TextLoader(source, encoding="utf-8").load()

    def _load_with_camelot(self, source: str) -> list[Document]:
        """Extract PDF tables directly with Camelot and return LangChain documents."""
        try:
            import camelot  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "CamelotLoader requires camelot-py. Install the production extra "
                "or choose PyPDFLoader/PDFPlumberLoader for general PDF text."
            ) from exc

        tables = camelot.read_pdf(source, pages="all")
        docs: list[Document] = []
        for index, table in enumerate(tables):
            dataframe = getattr(table, "df", None)
            if dataframe is None or dataframe.empty:
                continue
            docs.append(
                Document(
                    page_content=dataframe.to_csv(index=False),
                    metadata={
                        "source": source,
                        "loader": "CamelotLoader",
                        "table_index": index,
                        "page": int(getattr(table, "page", 0) or 0),
                    },
                )
            )
        return docs

    @staticmethod
    def _load_dataframe(source: str) -> list:
        import pandas as pd  # noqa: PLC0415
        from langchain_community.document_loaders import DataFrameLoader  # noqa: PLC0415

        if str(source).lower().endswith(".csv"):
            frame = pd.read_csv(source)
        else:
            frame = pd.read_excel(source)
        frame = frame.fillna("").astype(str)
        text_column = str(frame.columns[0])
        return DataFrameLoader(frame, page_content_column=text_column).load()

    @staticmethod
    def _load_json_fallback(source: str) -> list:
        import json  # noqa: PLC0415
        from langchain_core.documents import Document  # noqa: PLC0415

        with open(source, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [
                Document(
                    page_content=json.dumps(item, ensure_ascii=False),
                    metadata={"source": source, "row": index},
                )
                for index, item in enumerate(data)
            ]
        if isinstance(data, dict):
            return [
                Document(
                    page_content=json.dumps(value, ensure_ascii=False),
                    metadata={"source": source, "key": str(key)},
                )
                for key, value in data.items()
            ]
        return [
            Document(
                page_content=json.dumps(data, ensure_ascii=False),
                metadata={"source": source},
            )
        ]

    @staticmethod
    def _extract_youtube_id(url: str) -> str:
        """Extract YouTube video ID from URL."""
        import re  # noqa: PLC0415
        patterns = [
            r"youtube\.com/watch\?v=([^&]+)",
            r"youtu\.be/([^?]+)",
            r"youtube\.com/embed/([^?]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url  # fallback: return URL as-is

    @staticmethod
    def _display_summary(result: IngestionResult, console: object) -> None:
        """Display post-ingestion summary (Req 9.7)."""
        console.print(  # type: ignore[union-attr]
            f"\n[bold green]  ✓ Ingestion complete![/bold green]\n"
            f"  [bold white]Chunks stored:[/bold white] {result.chunk_count}\n"
            f"  [bold white]Collection:[/bold white] {result.collection_name}"
        )
        if result.failed_documents:
            console.print(  # type: ignore[union-attr]
                f"  [yellow]  Failed documents ({len(result.failed_documents)}):[/yellow]"
            )
            for path, err in result.failed_documents:
                console.print(f"    [dim]• {path}: {err}[/dim]")  # type: ignore[union-attr]
