"""Property-based tests for DocumentTypeSelector.

Properties covered:
    Property 6: Document Type Multi-Select Round-Trip (Req 4.2, 4.3)

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.ingestion.document_type_selector import (
    DOCUMENT_TYPES,
    DOCUMENT_TYPE_IDS,
    DOCUMENT_TYPE_MAP,
    EXTENSION_TO_DOCTYPE,
    DocumentTypeSelector,
)


# ---------------------------------------------------------------------------
# Property 6: Document Type Multi-Select Round-Trip
# ---------------------------------------------------------------------------


@given(
    selected=st.frozensets(
        st.sampled_from(DOCUMENT_TYPE_IDS),
        min_size=1,
        max_size=len(DOCUMENT_TYPE_IDS),
    )
)
@settings(max_examples=100)
def test_document_type_multi_select_round_trip(selected: frozenset[str]) -> None:
    """Feature: ms-rag, Property 6: Document Type Multi-Select Round-Trip."""
    selector = DocumentTypeSelector()
    selected_list = list(selected)

    with patch("ms_rag.ingestion.document_type_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.document_type_selector.Console"):
        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = selected_list
        mock_q.checkbox.return_value = mock_checkbox
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = selector.display_checklist()

    assert set(result) == selected
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# Empty selection re-prompt (Requirement 4.4)
# ---------------------------------------------------------------------------


def test_empty_selection_reprompts_once() -> None:
    """Requirement 4.4: if user selects nothing, checklist must be re-presented."""
    selector = DocumentTypeSelector()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        mock = MagicMock()
        call_count["n"] += 1
        mock.ask.return_value = [] if call_count["n"] == 1 else ["pdf"]
        return mock

    with patch("ms_rag.ingestion.document_type_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.document_type_selector.Console"):
        mock_q.checkbox.side_effect = side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)
        result = selector.display_checklist()

    assert result == ["pdf"]
    assert call_count["n"] == 2


def test_none_selection_treated_as_empty_and_reprompts() -> None:
    """None return (Ctrl+C) must be treated as empty and re-prompted."""
    selector = DocumentTypeSelector()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        mock = MagicMock()
        call_count["n"] += 1
        mock.ask.return_value = None if call_count["n"] == 1 else ["txt"]
        return mock

    with patch("ms_rag.ingestion.document_type_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.document_type_selector.Console"):
        mock_q.checkbox.side_effect = side_effect
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)
        result = selector.display_checklist()

    assert result == ["txt"]
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Structural completeness tests (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestDocumentTypeListCompleteness:
    def test_at_least_16_document_types_defined(self) -> None:
        assert len(DOCUMENT_TYPES) >= 16

    def test_all_required_types_present(self) -> None:
        required = {
            "pdf", "txt", "docx", "csv", "xlsx", "pptx",
            "html", "markdown", "json", "xml", "url",
            "youtube", "image_ocr", "code", "sql", "mongodb",
        }
        defined = set(DOCUMENT_TYPE_IDS)
        missing = required - defined
        assert not missing, f"Missing document types: {missing}"

    def test_no_duplicate_type_ids(self) -> None:
        assert len(DOCUMENT_TYPE_IDS) == len(set(DOCUMENT_TYPE_IDS))

    def test_document_type_map_matches_list(self) -> None:
        assert set(DOCUMENT_TYPE_MAP.keys()) == set(DOCUMENT_TYPE_IDS)

    def test_all_types_have_display_names(self) -> None:
        for dt in DOCUMENT_TYPES:
            assert len(dt.display_name.strip()) > 0

    def test_all_types_have_descriptions(self) -> None:
        for dt in DOCUMENT_TYPES:
            assert len(dt.description.strip()) > 0

    def test_pdf_has_correct_extension(self) -> None:
        assert ".pdf" in DOCUMENT_TYPE_MAP["pdf"].extensions

    def test_docx_has_both_extensions(self) -> None:
        exts = DOCUMENT_TYPE_MAP["docx"].extensions
        assert ".docx" in exts
        assert ".doc" in exts

    def test_url_and_youtube_have_no_extensions(self) -> None:
        assert DOCUMENT_TYPE_MAP["url"].extensions == []
        assert DOCUMENT_TYPE_MAP["youtube"].extensions == []


class TestExtensionToDocTypeMapping:
    def test_pdf_extension_maps_to_pdf(self) -> None:
        assert EXTENSION_TO_DOCTYPE[".pdf"] == "pdf"

    def test_docx_extension_maps_to_docx(self) -> None:
        assert EXTENSION_TO_DOCTYPE[".docx"] == "docx"

    def test_py_extension_maps_to_code(self) -> None:
        assert EXTENSION_TO_DOCTYPE[".py"] == "code"

    def test_md_extension_maps_to_markdown(self) -> None:
        assert EXTENSION_TO_DOCTYPE[".md"] == "markdown"

    def test_png_extension_maps_to_image_ocr(self) -> None:
        assert EXTENSION_TO_DOCTYPE[".png"] == "image_ocr"

    def test_all_extensions_lowercase(self) -> None:
        for ext in EXTENSION_TO_DOCTYPE:
            assert ext == ext.lower(), f"Extension {ext!r} is not lowercase"
