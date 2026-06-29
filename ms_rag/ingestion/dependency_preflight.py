"""External runtime dependency checks for document extraction.

These checks cover non-Python tools that loaders may need at runtime, such as
Poppler for scanned PDFs. They are advisory but surfaced before ingestion so
users understand what must be installed for their selected files and loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from ms_rag.ingestion.document_type_selector import EXTENSION_TO_DOCTYPE


@dataclass(frozen=True)
class DependencyCheck:
    """Result of checking one external command-line dependency."""

    tool: str
    command: str
    required: bool
    installed: bool
    needed_for: str
    install_hint: str


def selected_sources_include_doc_type(sources: list[str], doc_type: str) -> bool:
    """Return True when selected sources likely contain a given document type."""
    selected_extensions = {
        ext
        for ext, mapped_doc_type in EXTENSION_TO_DOCTYPE.items()
        if mapped_doc_type == doc_type
    }
    for raw_source in sources:
        source = str(raw_source).strip()
        if source.startswith(("http://", "https://")):
            continue
        path = Path(source)
        if path.is_file() and path.suffix.lower() in selected_extensions:
            return True
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in selected_extensions:
                    return True
    return False


def build_dependency_checks(loader_map: dict[str, str], sources: list[str]) -> list[DependencyCheck]:
    """Build external dependency checks for selected loaders and sources."""
    checks: list[DependencyCheck] = []
    has_pdf_sources = selected_sources_include_doc_type(sources, "pdf")
    pdf_loader = loader_map.get("pdf")

    if pdf_loader == "UnstructuredPDFLoader" and has_pdf_sources:
        checks.append(
            DependencyCheck(
                tool="Poppler",
                command="pdfinfo",
                required=True,
                installed=shutil.which("pdfinfo") is not None,
                needed_for="scanned/image-heavy PDF page inspection used by Unstructured",
                install_hint=(
                    "Install Poppler for Windows and add its bin folder to PATH, "
                    "or choose LlamaParse for cloud parsing."
                ),
            )
        )
        checks.append(
            DependencyCheck(
                tool="Tesseract OCR",
                command="tesseract",
                required=False,
                installed=shutil.which("tesseract") is not None,
                needed_for="OCR on scanned pages when PDFs contain images instead of text",
                install_hint="Install Tesseract OCR and add it to PATH for local OCR quality.",
            )
        )

    if pdf_loader == "TabulaLoader" and has_pdf_sources:
        checks.append(
            DependencyCheck(
                tool="Java",
                command="java",
                required=True,
                installed=shutil.which("java") is not None,
                needed_for="Tabula PDF table extraction",
                install_hint="Install a Java runtime and add java to PATH.",
            )
        )

    if pdf_loader == "CamelotLoader" and has_pdf_sources:
        checks.append(
            DependencyCheck(
                tool="Ghostscript",
                command="gswin64c",
                required=False,
                installed=shutil.which("gswin64c") is not None or shutil.which("gs") is not None,
                needed_for="Camelot lattice table extraction from PDFs",
                install_hint="Install Ghostscript for better Camelot table extraction.",
            )
        )

    return checks


def missing_required_dependencies(checks: list[DependencyCheck]) -> list[DependencyCheck]:
    """Return required checks that are not installed."""
    return [check for check in checks if check.required and not check.installed]
