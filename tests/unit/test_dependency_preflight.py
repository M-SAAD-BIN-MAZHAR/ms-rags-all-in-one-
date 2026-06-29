"""Tests for external dependency preflight checks."""

from __future__ import annotations

from pathlib import Path

from ms_rag.ingestion.dependency_preflight import (
    build_dependency_checks,
    missing_required_dependencies,
    selected_sources_include_doc_type,
)


def test_pdf_source_discovery_inside_directory(tmp_path: Path) -> None:
    (tmp_path / "scan.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("fake", encoding="utf-8")

    assert selected_sources_include_doc_type([str(tmp_path)], "pdf") is True
    assert selected_sources_include_doc_type([str(tmp_path)], "docx") is False


def test_unstructured_pdf_reports_poppler_and_tesseract(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("fake", encoding="utf-8")

    monkeypatch.setattr("shutil.which", lambda _command: None)

    checks = build_dependency_checks({"pdf": "UnstructuredPDFLoader"}, [str(pdf)])

    assert [check.tool for check in checks] == ["Poppler", "Tesseract OCR"]
    assert checks[0].required is True
    assert checks[0].installed is False
    assert "PATH" in checks[0].install_hint
    assert [check.tool for check in missing_required_dependencies(checks)] == ["Poppler"]
