"""Unit tests for import audit script."""

from __future__ import annotations

from scripts.audit_imports import collect_imports


def test_collect_imports_ignores_invalid_symbols() -> None:
    pairs = collect_imports()
    symbols = {symbol for _, symbol in pairs if symbol}
    assert "compressor" not in symbols
    assert "texts" not in symbols
    assert "splitter" not in symbols
    assert "RecursiveCharacterTextSplitter" in symbols
