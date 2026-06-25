"""Document Type Selector for MS_RAG.

Presents all supported document/source types as a multi-select checklist,
validates that at least one type is selected, and stores the result in
PipelineConfig.

Requirement 4:
- Display checklist of all 16+ Document_Types (4.1)
- Allow multi-select (4.2)
- Store selections in PipelineConfig (4.3)
- Re-present checklist if user selects none (4.4)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import questionary  # noqa: F401 — imported at module level so tests can patch it
    from rich.console import Console  # noqa: F401
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Document type definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentTypeInfo:
    """Metadata for a single supported document type."""

    doc_type_id: str        # internal ID used in loader_map keys
    display_name: str       # shown in the checklist
    extensions: list[str]   # file extensions that map to this type
    description: str        # one-line description shown in the checklist


DOCUMENT_TYPES: list[DocumentTypeInfo] = [
    DocumentTypeInfo(
        doc_type_id="pdf",
        display_name="PDF (.pdf)",
        extensions=[".pdf"],
        description="PDF files — supports text, tables, and scanned pages",
    ),
    DocumentTypeInfo(
        doc_type_id="txt",
        display_name="Plain Text (.txt)",
        extensions=[".txt"],
        description="Plain text files — UTF-8 or ASCII",
    ),
    DocumentTypeInfo(
        doc_type_id="docx",
        display_name="Microsoft Word (.docx / .doc)",
        extensions=[".docx", ".doc"],
        description="Word documents — preserves headings and formatting",
    ),
    DocumentTypeInfo(
        doc_type_id="csv",
        display_name="CSV (.csv)",
        extensions=[".csv"],
        description="Comma-separated values — each row becomes a document",
    ),
    DocumentTypeInfo(
        doc_type_id="xlsx",
        display_name="Excel (.xlsx / .xls)",
        extensions=[".xlsx", ".xls"],
        description="Excel spreadsheets — sheet-level document extraction",
    ),
    DocumentTypeInfo(
        doc_type_id="pptx",
        display_name="PowerPoint (.pptx / .ppt)",
        extensions=[".pptx", ".ppt"],
        description="Presentation slides — each slide becomes a document",
    ),
    DocumentTypeInfo(
        doc_type_id="html",
        display_name="HTML (.html / .htm)",
        extensions=[".html", ".htm"],
        description="HTML files — strips tags and preserves text structure",
    ),
    DocumentTypeInfo(
        doc_type_id="markdown",
        display_name="Markdown (.md / .mdx)",
        extensions=[".md", ".mdx"],
        description="Markdown files — structure-aware splitting by headers",
    ),
    DocumentTypeInfo(
        doc_type_id="json",
        display_name="JSON (.json)",
        extensions=[".json"],
        description="JSON files — configurable jq-path extraction",
    ),
    DocumentTypeInfo(
        doc_type_id="xml",
        display_name="XML (.xml)",
        extensions=[".xml"],
        description="XML files — text extraction with tag filtering",
    ),
    DocumentTypeInfo(
        doc_type_id="url",
        display_name="Web URLs (HTTP/HTTPS)",
        extensions=[],
        description="Web pages scraped via BeautifulSoup or FireCrawl",
    ),
    DocumentTypeInfo(
        doc_type_id="youtube",
        display_name="YouTube Video Transcripts",
        extensions=[],
        description="YouTube transcripts fetched via the transcript API",
    ),
    DocumentTypeInfo(
        doc_type_id="image_ocr",
        display_name="Images with OCR (.png / .jpg / .jpeg / .tiff / .bmp)",
        extensions=[".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"],
        description="Images processed via Tesseract OCR or LLM vision",
    ),
    DocumentTypeInfo(
        doc_type_id="code",
        display_name="Source Code (.py / .js / .ts / .java / .cpp / .go / ...)",
        extensions=[
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".c", ".cs", ".go",
            ".rb", ".rs", ".swift", ".kt", ".php",
            ".sh", ".bash", ".sql", ".r",
        ],
        description="Source code files — language-aware AST-boundary splitting",
    ),
    DocumentTypeInfo(
        doc_type_id="sql",
        display_name="SQL Database Tables",
        extensions=[],
        description="Relational DB tables fetched via SQLAlchemy connection",
    ),
    DocumentTypeInfo(
        doc_type_id="mongodb",
        display_name="MongoDB Collections",
        extensions=[],
        description="MongoDB documents fetched via Atlas or local connection",
    ),
    DocumentTypeInfo(
        doc_type_id="epub",
        display_name="eBook (.epub)",
        extensions=[".epub"],
        description="eBook files — chapter-level extraction",
    ),
    DocumentTypeInfo(
        doc_type_id="rtf",
        display_name="Rich Text Format (.rtf)",
        extensions=[".rtf"],
        description="RTF files — plain text extraction",
    ),
]

# Lookup helpers
DOCUMENT_TYPE_MAP: dict[str, DocumentTypeInfo] = {
    d.doc_type_id: d for d in DOCUMENT_TYPES
}

DOCUMENT_TYPE_IDS: list[str] = [d.doc_type_id for d in DOCUMENT_TYPES]

# Extension → doc_type_id mapping (for ingestion orchestrator)
EXTENSION_TO_DOCTYPE: dict[str, str] = {}
for _dt in DOCUMENT_TYPES:
    for _ext in _dt.extensions:
        EXTENSION_TO_DOCTYPE[_ext.lower()] = _dt.doc_type_id


# ---------------------------------------------------------------------------
# DocumentTypeSelector
# ---------------------------------------------------------------------------


class DocumentTypeSelector:
    """Interactive multi-select checklist for document types.

    Usage::

        selector = DocumentTypeSelector()
        selected_ids = selector.display_checklist()
        # e.g. ["pdf", "docx", "url"]
    """

    def display_checklist(self) -> list[str]:
        """Show multi-select checklist and return selected doc type IDs.

        Re-presents the checklist if the user confirms with zero selections
        (Requirement 4.4).

        Returns:
            Non-empty list of selected document type IDs.
        """
        console = Console()

        console.print(
            "\n[bold cyan]Step 4 — Select Document Types[/bold cyan]\n"
            "  Use [bold white]Space[/bold white] to select, "
            "[bold white]Enter[/bold white] to confirm.\n"
        )

        choices = [
            questionary.Choice(
                title=f"{dt.display_name}  —  {dt.description}",
                value=dt.doc_type_id,
            )
            for dt in DOCUMENT_TYPES
        ]

        while True:
            selected: list[str] = questionary.checkbox(
                "Which document types do you have?",
                choices=choices,
            ).ask()

            if selected is None:
                selected = []

            if not selected:
                console.print(
                    "[red]  ✗ Please select at least one document type.[/red]"
                )
                continue

            return selected
