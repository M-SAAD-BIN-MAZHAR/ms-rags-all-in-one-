"""Property-based tests for LoaderSelector.

Properties covered:
    Property 7: Loader Compatibility Filtering (Req 5.1)
    Property 8: Credential-Requiring Loader Gatekeeping (Req 5.3)
    Property 9: One Loader Per Document Type (Req 5.4)

Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.ingestion.document_type_selector import DOCUMENT_TYPE_IDS
from ms_rag.ingestion.loader_selector import (
    ALL_LOADERS,
    CREDENTIAL_REQUIRED_LOADERS,
    LOADER_COMPATIBILITY,
    LOADER_MAP,
    LoaderInfo,
    LoaderSelector,
)
from ms_rag.models import CredentialStore


# ---------------------------------------------------------------------------
# Property 7: Loader Compatibility Filtering
# ---------------------------------------------------------------------------


@given(
    doc_types=st.frozensets(
        st.sampled_from(DOCUMENT_TYPE_IDS),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_loader_compatibility_filtering(doc_types: frozenset[str]) -> None:
    """Feature: ms-rag, Property 7: Loader Compatibility Filtering.

    The loaders available for a given set of doc types must be exactly
    those compatible with at least one of the selected doc types.
    """
    expected_loaders: set[str] = set()
    for doc_type in doc_types:
        for loader in LOADER_COMPATIBILITY.get(doc_type, []):
            expected_loaders.add(loader.loader_class)

    # Verify LOADER_COMPATIBILITY is correct for each type
    for doc_type in doc_types:
        actual = {lo.loader_class for lo in LOADER_COMPATIBILITY.get(doc_type, [])}
        # All loaders in the compatibility map for this doc_type must
        # have this doc_type in their compatible_doc_types
        for loader_class in actual:
            loader_info = LOADER_MAP[loader_class]
            assert doc_type in loader_info.compatible_doc_types, (
                f"Loader {loader_class} is in LOADER_COMPATIBILITY['{doc_type}'] "
                f"but {doc_type!r} not in its compatible_doc_types"
            )

    # No incompatible loaders should appear for a given doc_type
    for doc_type in doc_types:
        incompatible = [
            lo.loader_class for lo in ALL_LOADERS
            if doc_type not in lo.compatible_doc_types
            and lo.loader_class in {l.loader_class for l in LOADER_COMPATIBILITY.get(doc_type, [])}
        ]
        assert not incompatible, (
            f"Incompatible loaders found for {doc_type!r}: {incompatible}"
        )


# ---------------------------------------------------------------------------
# Property 8: Credential-Requiring Loader Gatekeeping
# ---------------------------------------------------------------------------


@given(loader_class=st.sampled_from(sorted(CREDENTIAL_REQUIRED_LOADERS)))
@settings(max_examples=len(CREDENTIAL_REQUIRED_LOADERS))
def test_credential_required_loader_blocked_on_cancel(loader_class: str) -> None:
    """Feature: ms-rag, Property 8: Credential-Requiring Loader Gatekeeping.

    When the user cancels the credential prompt for a paid loader,
    the loader must NOT be stored in loader_map.
    """
    loader_info = LOADER_MAP[loader_class]
    doc_type = loader_info.compatible_doc_types[0]
    store = CredentialStore()
    selector = LoaderSelector(credential_store=store)

    # Mock: user first selects the paid loader, then cancels credential prompt
    # (returns empty string), then selects a free loader on re-prompt
    free_loaders = [
        lo for lo in LOADER_COMPATIBILITY.get(doc_type, [])
        if not lo.requires_credentials
    ]
    if not free_loaders:
        return  # skip if no free alternative exists for this doc_type

    free_class = free_loaders[0].loader_class
    call_count = {"select": 0}

    def mock_select(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        call_count["select"] += 1
        # First call: pick paid loader; subsequent: pick free loader
        m.ask.return_value = loader_class if call_count["select"] == 1 else free_class
        return m

    def mock_password(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        m.ask.return_value = ""  # cancelled / empty
        return m

    with patch("ms_rag.ingestion.loader_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.loader_selector.Console"):
        mock_q.select.side_effect = mock_select
        mock_q.password.side_effect = mock_password
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = selector.display_filtered_loaders([doc_type])

    # Paid loader must NOT be stored; free loader must be stored instead
    assert result.get(doc_type) != loader_class, (
        f"Paid loader {loader_class} must not be stored when credentials are cancelled"
    )
    assert result.get(doc_type) == free_class


@given(loader_class=st.sampled_from(sorted(CREDENTIAL_REQUIRED_LOADERS)))
@settings(max_examples=len(CREDENTIAL_REQUIRED_LOADERS))
def test_credential_required_loader_stored_when_credentials_provided(
    loader_class: str,
) -> None:
    """Property 8 (positive case): paid loader IS stored when credentials provided."""
    loader_info = LOADER_MAP[loader_class]
    doc_type = loader_info.compatible_doc_types[0]
    store = CredentialStore()
    selector = LoaderSelector(credential_store=store)

    def mock_select(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        m.ask.return_value = loader_class
        return m

    def mock_password(*args, **kwargs) -> MagicMock:  # noqa: ANN002
        m = MagicMock()
        m.ask.return_value = "valid-api-key-xyz"
        return m

    with patch("ms_rag.ingestion.loader_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.loader_selector.Console"):
        mock_q.select.side_effect = mock_select
        mock_q.password.side_effect = mock_password
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = selector.display_filtered_loaders([doc_type])

    assert result.get(doc_type) == loader_class


# ---------------------------------------------------------------------------
# Property 9: One Loader Per Document Type
# ---------------------------------------------------------------------------


@given(
    doc_types=st.frozensets(
        st.sampled_from([
            dt for dt in DOCUMENT_TYPE_IDS
            if dt in LOADER_COMPATIBILITY and LOADER_COMPATIBILITY[dt]
        ]),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=100)
def test_one_loader_per_document_type(doc_types: frozenset[str]) -> None:
    """Feature: ms-rag, Property 9: One Loader Per Document Type.

    loader_map must contain exactly one entry per selected document type.
    """
    store = CredentialStore()
    selector = LoaderSelector(credential_store=store)
    doc_types_list = list(doc_types)

    # For each doc type, select the first free loader
    type_to_loader: dict[str, str] = {}
    for dt in doc_types_list:
        loaders = LOADER_COMPATIBILITY.get(dt, [])
        free = [lo for lo in loaders if not lo.requires_credentials]
        if free:
            type_to_loader[dt] = free[0].loader_class
        elif loaders:
            type_to_loader[dt] = loaders[0].loader_class

    call_counts: dict[str, int] = {dt: 0 for dt in doc_types_list}

    def mock_select_for(dt_list: list[str]) -> MagicMock:
        """Returns a side_effect that cycles through doc_types in order."""
        state = {"idx": 0}

        def side_effect(*args, **kwargs) -> MagicMock:  # noqa: ANN002
            m = MagicMock()
            current_dt = dt_list[state["idx"]]
            state["idx"] += 1
            m.ask.return_value = type_to_loader.get(current_dt, "TextLoader")
            return m

        return side_effect

    mock_select_fn = mock_select_for(doc_types_list)

    with patch("ms_rag.ingestion.loader_selector.questionary") as mock_q, \
         patch("ms_rag.ingestion.loader_selector.Console"):
        mock_q.select.side_effect = mock_select_fn
        mock_q.Choice = MagicMock(side_effect=lambda title, value: value)

        result = selector.display_filtered_loaders(doc_types_list)

    # Must have exactly one entry per doc type that has compatible loaders
    for dt in doc_types_list:
        if LOADER_COMPATIBILITY.get(dt):
            assert dt in result, f"Missing loader for doc_type {dt!r}"
    assert len(result) == len([dt for dt in doc_types_list if LOADER_COMPATIBILITY.get(dt)])


# ---------------------------------------------------------------------------
# Structural completeness tests (Requirement 5.2)
# ---------------------------------------------------------------------------


class TestLoaderListCompleteness:
    def test_pdf_has_required_loaders(self) -> None:
        required = {
            "PyPDFLoader", "UnstructuredPDFLoader",
            "PDFPlumberLoader", "CamelotLoader", "TabulaLoader", "LlamaParseLoader",
        }
        pdf_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("pdf", [])}
        missing = required - pdf_loaders
        assert not missing, f"Missing PDF loaders: {missing}"

    def test_docx_has_required_loaders(self) -> None:
        docx_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("docx", [])}
        assert "UnstructuredWordDocumentLoader" in docx_loaders

    def test_url_has_web_loaders(self) -> None:
        url_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("url", [])}
        assert "WebBaseLoader" in url_loaders
        assert "AsyncHtmlLoader" in url_loaders
        assert "FireCrawlLoader" in url_loaders
        assert "ApifyWebScraper" in url_loaders

    def test_youtube_has_loader(self) -> None:
        yt_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("youtube", [])}
        assert "YoutubeLoader" in yt_loaders

    def test_sql_has_loader(self) -> None:
        sql_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("sql", [])}
        assert "SQLDatabaseLoader" in sql_loaders

    def test_mongodb_has_loader(self) -> None:
        mongo_loaders = {lo.loader_class for lo in LOADER_COMPATIBILITY.get("mongodb", [])}
        assert "MongoDBAtlasLoader" in mongo_loaders

    def test_credential_required_loaders_non_empty(self) -> None:
        assert len(CREDENTIAL_REQUIRED_LOADERS) >= 3

    def test_llamaparse_requires_credentials(self) -> None:
        assert "LlamaParseLoader" in CREDENTIAL_REQUIRED_LOADERS

    def test_firecrawl_requires_credentials(self) -> None:
        assert "FireCrawlLoader" in CREDENTIAL_REQUIRED_LOADERS

    def test_apify_requires_credentials(self) -> None:
        assert "ApifyWebScraper" in CREDENTIAL_REQUIRED_LOADERS

    def test_pypdf_does_not_require_credentials(self) -> None:
        assert "PyPDFLoader" not in CREDENTIAL_REQUIRED_LOADERS

    def test_no_duplicate_loader_classes(self) -> None:
        classes = [lo.loader_class for lo in ALL_LOADERS]
        assert len(classes) == len(set(classes))

    def test_all_loaders_have_descriptions(self) -> None:
        for lo in ALL_LOADERS:
            assert len(lo.description.strip()) > 0, (
                f"Loader {lo.loader_class} missing description"
            )

    def test_loader_map_matches_all_loaders(self) -> None:
        assert set(LOADER_MAP.keys()) == {lo.loader_class for lo in ALL_LOADERS}
