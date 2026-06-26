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

import time
from pathlib import Path
from typing import Callable

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

        total_chunks = 0
        failed_documents: list[tuple[str, str]] = []

        for idx, source in enumerate(discovered):
            source_str = str(source)
            try:
                # Load document(s)
                docs = self._load_source(
                    source=source,
                    loader_map=loader_map,
                    youtube_language=youtube_language,
                )

                # Chunk
                chunks = splitter.split_documents(docs)
                sanitize_documents(chunks)

                # Store (with retry)
                def _store(c: list = chunks, vs: object = vector_store) -> None:
                    vs.add_documents(c)  # type: ignore[union-attr]

                retry_with_backoff(_store)
                total_chunks += len(chunks)

            except Exception as exc:  # noqa: BLE001
                failed_documents.append((source_str, str(exc)))
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

        # Post-ingestion summary (Req 9.7)
        self._display_summary(result, console)
        return result

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

        return self._invoke_loader(loader_class_name, source_str, youtube_language)

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
            except Exception:  # noqa: BLE001
                from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
                return PyPDFLoader(source).load()

        if loader_class_name == "PDFPlumberLoader":
            from langchain_community.document_loaders import PDFPlumberLoader  # noqa: PLC0415
            return PDFPlumberLoader(source).load()

        # ── DOCX loaders ─────────────────────────────────────────────
        if loader_class_name == "UnstructuredWordDocumentLoader":
            try:
                from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
                return UnstructuredLoader(source).load()
            except Exception:  # noqa: BLE001
                from langchain_community.document_loaders import Docx2txtLoader  # noqa: PLC0415
                return Docx2txtLoader(source).load()

        if loader_class_name == "Docx2txtLoader":
            from langchain_community.document_loaders import Docx2txtLoader  # noqa: PLC0415
            return Docx2txtLoader(source).load()

        # ── CSV / Excel ───────────────────────────────────────────────
        if loader_class_name == "CSVLoader":
            from langchain_community.document_loaders.csv_loader import CSVLoader  # noqa: PLC0415
            return CSVLoader(source).load()

        if loader_class_name == "UnstructuredExcelLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── Plain text ────────────────────────────────────────────────
        if loader_class_name == "TextLoader":
            from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
            return TextLoader(source, encoding="utf-8").load()

        # ── HTML ──────────────────────────────────────────────────────
        if loader_class_name == "BSHTMLLoader":
            from langchain_community.document_loaders import BSHTMLLoader  # noqa: PLC0415
            return BSHTMLLoader(source).load()

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
            return FireCrawlLoader(url=source, mode="scrape").load()

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
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── JSON ──────────────────────────────────────────────────────
        if loader_class_name == "JSONLoader":
            from langchain_community.document_loaders import JSONLoader  # noqa: PLC0415
            return JSONLoader(file_path=source, jq_schema=".", text_content=False).load()

        # ── XML ───────────────────────────────────────────────────────
        if loader_class_name == "UnstructuredXMLLoader":
            from langchain_unstructured import UnstructuredLoader  # noqa: PLC0415
            return UnstructuredLoader(source).load()

        # ── Images / OCR ──────────────────────────────────────────────
        if loader_class_name == "UnstructuredImageLoader":
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
