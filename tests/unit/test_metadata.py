"""Unit tests for document metadata sanitization."""

from __future__ import annotations

from ms_rag.utils.metadata import sanitize_metadata


class TestSanitizeMetadata:
    def test_scalar_values_preserved(self) -> None:
        meta = sanitize_metadata({"source": "doc.pdf", "page": 1, "score": 0.9, "active": True})
        assert meta == {"source": "doc.pdf", "page": 1, "score": 0.9, "active": True}

    def test_complex_links_serialized_to_json(self) -> None:
        links = [
            {"text": "github.com/foo", "url": "https://github.com/foo", "start_index": 74},
        ]
        meta = sanitize_metadata({"source": "resume.docx", "links": links})
        assert isinstance(meta["links"], str)
        assert "github.com/foo" in meta["links"]

    def test_none_values_dropped(self) -> None:
        meta = sanitize_metadata({"source": "doc.pdf", "category": None})
        assert meta == {"source": "doc.pdf"}

    def test_homogeneous_string_list_preserved(self) -> None:
        meta = sanitize_metadata({"tags": ["a", "b"]})
        assert meta["tags"] == ["a", "b"]
