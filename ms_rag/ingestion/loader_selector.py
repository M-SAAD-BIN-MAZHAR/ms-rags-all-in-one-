"""Loader Selector for MS_RAG.

Presents document loaders filtered by the user's selected document types.
Handles credential prompting for paid loaders (LlamaParse, FireCrawl, Apify).
Enforces one loader per document type.

- Display loaders filtered by selected doc types (5.1)
- Include all required loaders per doc type (5.2)
- Prompt credentials for paid loaders; block selection if cancelled (5.3)
- One primary loader per selected document type (5.4)
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

from ms_rag.utils.exceptions import CredentialError


# ---------------------------------------------------------------------------
# Loader definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoaderInfo:
    """Metadata for a single document loader."""

    loader_class: str             # Python class name used in generated code
    display_name: str             # shown in checklist
    compatible_doc_types: list[str]  # doc_type_ids this loader supports
    description: str              # one-line description
    requires_credentials: bool = False
    credential_fields: list[str] = field(default_factory=list)
    # e.g. ["LLAMA_CLOUD_API_KEY"] for LlamaParse


ALL_LOADERS: list[LoaderInfo] = [
    # ── PDF ──────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="PyPDFLoader",
        display_name="PyPDFLoader",
        compatible_doc_types=["pdf"],
        description="Fast, pure-Python PDF text extractor (recommended default)",
    ),
    LoaderInfo(
        loader_class="UnstructuredPDFLoader",
        display_name="UnstructuredPDFLoader",
        compatible_doc_types=["pdf"],
        description="Unstructured.io PDF parser — handles scanned and complex PDFs",
    ),
    LoaderInfo(
        loader_class="PDFPlumberLoader",
        display_name="PDFPlumberLoader",
        compatible_doc_types=["pdf"],
        description="pdfplumber — excellent for tables and columnar layouts",
    ),
    LoaderInfo(
        loader_class="CamelotLoader",
        display_name="CamelotLoader (tables)",
        compatible_doc_types=["pdf"],
        description="Camelot — specialised PDF table extractor",
    ),
    LoaderInfo(
        loader_class="TabulaLoader",
        display_name="TabulaLoader (tables)",
        compatible_doc_types=["pdf"],
        description="Tabula — Java-based PDF table extractor via tabula-py",
    ),
    LoaderInfo(
        loader_class="DoclingLoader",
        display_name="DoclingLoader",
        compatible_doc_types=["pdf", "docx"],
        description="Docling (DS4SD) — structured document parsing with layout analysis, tables, and OCR",
    ),
    LoaderInfo(
        loader_class="LlamaParseLoader",
        display_name="LlamaParse (cloud, paid)",
        compatible_doc_types=["pdf", "docx", "pptx"],
        description="LlamaIndex cloud parser — best quality for complex PDFs",
        requires_credentials=True,
        credential_fields=["LLAMA_CLOUD_API_KEY"],
    ),
    # ── DOCX / DOC ────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredWordDocumentLoader",
        display_name="UnstructuredWordDocumentLoader",
        compatible_doc_types=["docx"],
        description="Unstructured.io Word loader — preserves heading structure",
    ),
    LoaderInfo(
        loader_class="Docx2txtLoader",
        display_name="Docx2txtLoader",
        compatible_doc_types=["docx"],
        description="docx2txt — lightweight plain text extraction from .docx",
    ),
    # ── CSV ───────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="CSVLoader",
        display_name="CSVLoader",
        compatible_doc_types=["csv"],
        description="LangChain CSVLoader — each row becomes a Document",
    ),
    LoaderInfo(
        loader_class="UnstructuredCSVLoader",
        display_name="UnstructuredCSVLoader",
        compatible_doc_types=["csv"],
        description="Unstructured CSV loader with metadata extraction",
    ),
    # ── XLSX ──────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredExcelLoader",
        display_name="UnstructuredExcelLoader",
        compatible_doc_types=["xlsx"],
        description="Unstructured Excel loader — sheet-level extraction",
    ),
    LoaderInfo(
        loader_class="PandasDataFrameLoader",
        display_name="PandasDataFrameLoader",
        compatible_doc_types=["xlsx", "csv"],
        description="Pandas-based loader — full control via DataFrame",
    ),
    # ── PPTX ──────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredPowerPointLoader",
        display_name="UnstructuredPowerPointLoader",
        compatible_doc_types=["pptx"],
        description="Unstructured PPTX loader — slide-level extraction",
    ),
    # ── HTML ──────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="BSHTMLLoader",
        display_name="BSHTMLLoader (BeautifulSoup)",
        compatible_doc_types=["html"],
        description="BeautifulSoup HTML loader — strips tags, preserves text",
    ),
    LoaderInfo(
        loader_class="UnstructuredHTMLLoader",
        display_name="UnstructuredHTMLLoader",
        compatible_doc_types=["html"],
        description="Unstructured HTML loader — structure-aware extraction",
    ),
    # ── MARKDOWN ──────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredMarkdownLoader",
        display_name="UnstructuredMarkdownLoader",
        compatible_doc_types=["markdown"],
        description="Unstructured Markdown loader — header-aware extraction",
    ),
    # ── JSON ──────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="JSONLoader",
        display_name="JSONLoader",
        compatible_doc_types=["json"],
        description="LangChain JSONLoader — configurable jq-path extraction",
    ),
    # ── XML ───────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredXMLLoader",
        display_name="UnstructuredXMLLoader",
        compatible_doc_types=["xml"],
        description="Unstructured XML loader — tag-filtered text extraction",
    ),
    # ── Web URLs ──────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="WebBaseLoader",
        display_name="WebBaseLoader",
        compatible_doc_types=["url"],
        description="LangChain web scraper using requests + BeautifulSoup",
    ),
    LoaderInfo(
        loader_class="AsyncHtmlLoader",
        display_name="AsyncHtmlLoader",
        compatible_doc_types=["url"],
        description="Async parallel web scraper for multiple URLs",
    ),
    LoaderInfo(
        loader_class="FireCrawlLoader",
        display_name="FireCrawlLoader (cloud, paid)",
        compatible_doc_types=["url"],
        description="FireCrawl — JavaScript-rendered scraping with clean markdown",
        requires_credentials=True,
        credential_fields=["FIRECRAWL_API_KEY"],
    ),
    LoaderInfo(
        loader_class="ApifyWebScraper",
        display_name="ApifyWebScraper (cloud, paid)",
        compatible_doc_types=["url"],
        description="Apify — scalable web scraping with actor-based extraction",
        requires_credentials=True,
        credential_fields=["APIFY_API_TOKEN"],
    ),
    # ── YouTube ───────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="YoutubeLoader",
        display_name="YoutubeLoader",
        compatible_doc_types=["youtube"],
        description="YouTube transcript API loader — language-selectable captions",
    ),
    # ── Images / OCR ──────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredImageLoader",
        display_name="UnstructuredImageLoader (Tesseract OCR)",
        compatible_doc_types=["image_ocr"],
        description="Tesseract-based OCR via Unstructured.io",
    ),
    # ── Source Code ───────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="GenericLoader",
        display_name="GenericLoader (language-aware)",
        compatible_doc_types=["code"],
        description="LangChain GenericLoader with language-specific parsers",
    ),
    LoaderInfo(
        loader_class="TextLoader",
        display_name="TextLoader (plain)",
        compatible_doc_types=["code", "txt", "markdown"],
        description="Simple plain-text loader — no parsing, raw content",
    ),
    # ── SQL ───────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="SQLDatabaseLoader",
        display_name="SQLDatabaseLoader",
        compatible_doc_types=["sql"],
        description="SQLAlchemy-based DB loader — each row becomes a Document",
    ),
    # ── MongoDB ───────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="MongoDBAtlasLoader",
        display_name="MongoDBAtlasLoader",
        compatible_doc_types=["mongodb"],
        description="MongoDB Atlas loader — collection-level document extraction",
    ),
    # ── eBook ─────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredEPubLoader",
        display_name="UnstructuredEPubLoader",
        compatible_doc_types=["epub"],
        description="Unstructured ePub loader — chapter-level extraction",
    ),
    # ── RTF ───────────────────────────────────────────────────────────────
    LoaderInfo(
        loader_class="UnstructuredRTFLoader",
        display_name="UnstructuredRTFLoader",
        compatible_doc_types=["rtf"],
        description="Unstructured RTF loader — plain text extraction",
    ),
]

# ── Lookup helpers ────────────────────────────────────────────────────────

LOADER_MAP: dict[str, LoaderInfo] = {lo.loader_class: lo for lo in ALL_LOADERS}

# doc_type_id -> list of compatible LoaderInfo
LOADER_COMPATIBILITY: dict[str, list[LoaderInfo]] = {}
for _loader in ALL_LOADERS:
    for _dt in _loader.compatible_doc_types:
        LOADER_COMPATIBILITY.setdefault(_dt, []).append(_loader)

# Paid loaders that require credentials
CREDENTIAL_REQUIRED_LOADERS: frozenset[str] = frozenset(
    lo.loader_class for lo in ALL_LOADERS if lo.requires_credentials
)


# ---------------------------------------------------------------------------
# LoaderSelector
# ---------------------------------------------------------------------------


class LoaderSelector:
    """Interactive loader selection — one loader per selected document type.

    Usage::

        selector = LoaderSelector(credential_store)
        loader_map = selector.display_filtered_loaders(["pdf", "docx", "url"])
        # {"pdf": "PyPDFLoader", "docx": "UnstructuredWordDocumentLoader", ...}
    """

    def __init__(self, credential_store: object | None = None) -> None:
        # credential_store is a CredentialStore instance (typed loosely to
        # avoid circular import with models.py)
        self._credential_store = credential_store

    def display_filtered_loaders(
        self,
        doc_types: list[str],
    ) -> dict[str, str]:
        """Show compatible loaders per doc type; return {doc_type: loader_class}.

        For each selected document type, shows only the loaders that are
        compatible with that type.  Credential-requiring loaders prompt for
        credentials; if the user cancels, that loader is blocked.

        Args:
            doc_types: List of doc_type_ids from DocumentTypeSelector.

        Returns:
            Dict mapping doc_type_id -> selected loader_class name.
        """
        console = Console()
        loader_map: dict[str, str] = {}

        console.print(
            "\n[bold cyan]Step 5 — Select Document Loaders[/bold cyan]\n"
        )

        for doc_type in doc_types:
            compatible = LOADER_COMPATIBILITY.get(doc_type, [])
            if not compatible:
                console.print(
                    f"[yellow]  No loaders available for '{doc_type}'. Skipping.[/yellow]"
                )
                continue

            from ms_rag.ingestion.document_type_selector import DOCUMENT_TYPE_MAP  # noqa: PLC0415
            dt_display = DOCUMENT_TYPE_MAP.get(doc_type)
            dt_name = dt_display.display_name if dt_display else doc_type

            console.print(f"\n  [bold white]Loader for: {dt_name}[/bold white]")

            choices = [
                questionary.Choice(
                    title=f"{lo.display_name}  —  {lo.description}"
                          + (" [credentials required]" if lo.requires_credentials else ""),
                    value=lo.loader_class,
                )
                for lo in compatible
            ]

            while True:
                selected_class = questionary.select(
                    f"Select loader for {dt_name}:",
                    choices=choices,
                ).ask()
                if selected_class is None:
                    console.print(
                        "[yellow]  Selection cancelled — please choose a loader.[/yellow]"
                    )
                    continue

                loader_info = LOADER_MAP[selected_class]

                if loader_info.requires_credentials:
                    ok = self._prompt_loader_credentials(loader_info, console)
                    if not ok:
                        console.print(
                            f"[red]  ✗ Credentials required for {loader_info.display_name}. "
                            f"Please select a different loader.[/red]"
                        )
                        continue

                loader_map[doc_type] = selected_class
                break

        return loader_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prompt_loader_credentials(
        self,
        loader_info: LoaderInfo,
        console: object,
    ) -> bool:
        """Prompt for and store credentials required by a paid loader.

        Returns:
            True if all credentials were provided; False if user cancelled.
        """
        for field_name in loader_info.credential_fields:
            value: str = questionary.password(
                f"    {loader_info.display_name} — {field_name}:",
            ).ask()

            if not value or not value.strip():
                return False  # user cancelled or entered empty

            if self._credential_store is not None:
                # Store under a synthetic provider ID derived from loader name
                provider_id = loader_info.loader_class.lower()
                self._credential_store.set(provider_id, field_name, value.strip())  # type: ignore[union-attr]

        return True
