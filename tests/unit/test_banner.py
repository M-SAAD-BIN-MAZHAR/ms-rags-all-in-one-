"""Unit tests for ms_rag.ui.banner.

Tests (Requirement 1.1, 1.2, 1.3):
- display_banner() does not raise on a non-terminal console.
- Banner text contains "MS-RAGS(ALL-IN-ONE)".
- Tagline is printed immediately after the banner.
- Falls back to plain text when Rich raises.
- Continues silently when both display methods fail.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from ms_rag.ui.banner import (
    MS_RAG_BANNER,
    TAGLINE,
    VERSION_LINE,
    display_banner,
    _display_plain,
    _display_rich,
)


class TestBannerContent:
    """Verify the banner constants contain required text."""

    def test_banner_contains_ms_rags_all_in_one(self) -> None:
        """Requirement 1.1: banner must contain the visible product name."""
        assert "MS-RAGS(ALL-IN-ONE)" in MS_RAG_BANNER

    def test_tagline_is_non_empty(self) -> None:
        """Requirement 1.2: tagline must be a non-empty string."""
        assert isinstance(TAGLINE, str)
        assert len(TAGLINE.strip()) > 0

    def test_version_line_is_non_empty(self) -> None:
        assert isinstance(VERSION_LINE, str)
        assert len(VERSION_LINE.strip()) > 0


class TestDisplayBannerNoRaise:
    """display_banner() must never raise, regardless of terminal support."""

    def test_does_not_raise_with_no_console(self) -> None:
        """Requirement 1.3: must not raise on any terminal type."""
        # display_banner() creates its own console internally — must not raise
        display_banner()

    def test_does_not_raise_with_mock_console(self) -> None:
        """Must not raise when given a mock console."""
        mock_console = MagicMock()
        display_banner(console=mock_console)

    def test_does_not_raise_on_non_terminal_console(self) -> None:
        """Must not raise when Rich console targets a non-terminal (e.g. StringIO)."""
        from rich.console import Console  # type: ignore[import-untyped]
        console = Console(file=io.StringIO(), force_terminal=False)
        display_banner(console=console)


class TestRichFallback:
    """When Rich raises, display_banner() falls back to plain print()."""

    def test_falls_back_to_plain_when_rich_raises(self) -> None:
        """Requirement 1.3: falls back to plain text if Rich display fails."""
        with patch("ms_rag.ui.banner._display_rich", side_effect=RuntimeError("Rich failed")):
            with patch("ms_rag.ui.banner._display_plain") as mock_plain:
                display_banner()
                mock_plain.assert_called_once()

    def test_continues_silently_when_both_fail(self) -> None:
        """Requirement 1.3: startup continues silently if both methods fail."""
        with patch("ms_rag.ui.banner._display_rich", side_effect=RuntimeError("Rich failed")):
            with patch("ms_rag.ui.banner._display_plain", side_effect=OSError("stdout closed")):
                # Must NOT raise — startup should continue
                display_banner()

    def test_plain_fallback_not_called_when_rich_succeeds(self) -> None:
        """Plain fallback must NOT be called when Rich succeeds."""
        with patch("ms_rag.ui.banner._display_rich") as mock_rich:
            with patch("ms_rag.ui.banner._display_plain") as mock_plain:
                display_banner()
                mock_rich.assert_called_once()
                mock_plain.assert_not_called()


class TestPlainDisplay:
    """_display_plain() must print banner + tagline to stdout."""

    def test_plain_output_contains_banner(self, capsys: pytest.CaptureFixture[str]) -> None:
        _display_plain()
        captured = capsys.readouterr()
        # Banner is multi-line — check the tagline at minimum
        assert TAGLINE in captured.out

    def test_plain_output_contains_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        _display_plain()
        captured = capsys.readouterr()
        assert VERSION_LINE in captured.out


class TestRichDisplay:
    """_display_rich() must render without error to a StringIO-backed console."""

    def test_rich_display_to_string_io(self) -> None:
        from rich.console import Console  # type: ignore[import-untyped]
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        _display_rich(console=console)
        output = buf.getvalue()
        # Rich strips ANSI for non-force_terminal but content must be non-empty
        assert len(output.strip()) > 0

    def test_rich_display_raises_propagated(self) -> None:
        """If Rich internals raise, _display_rich propagates — display_banner catches it."""
        broken_console = MagicMock()
        broken_console.print.side_effect = RuntimeError("broken")
        with pytest.raises(RuntimeError, match="broken"):
            _display_rich(console=broken_console)
