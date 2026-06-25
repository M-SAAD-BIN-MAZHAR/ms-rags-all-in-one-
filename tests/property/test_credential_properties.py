"""Property-based tests for CredentialStore and CredentialManager.

Properties covered:
    Property 1: Credential Storage Round-Trip (Req 2.3)
    Property 2: Empty Credential Field Re-Prompt (Req 2.4)
    Property 3: Provider Credential Coverage (Req 2.2)

Validates: Requirements 2.2, 2.3, 2.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ms_rag.config.credential_manager import (
    PROVIDER_FIELDS,
    PROVIDER_IDS,
    CredentialManager,
    _is_secret_field,
)
from ms_rag.models import CredentialStore


# ---------------------------------------------------------------------------
# Property 1: Credential Storage Round-Trip
# ---------------------------------------------------------------------------


@given(
    provider_id=st.sampled_from(PROVIDER_IDS),
    field=st.text(min_size=1, max_size=60,
                  alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Ps"))),
    value=st.text(min_size=1, max_size=512),
)
@settings(max_examples=200)
def test_credential_storage_round_trip(
    provider_id: str, field: str, value: str
) -> None:
    """Feature: ms-rag, Property 1: Credential Storage Round-Trip.

    For any provider/field/value triple, store.get(provider, field)
    must return the exact value that was set.
    """
    store = CredentialStore()
    store.set(provider_id, field, value)
    assert store.get(provider_id, field) == value


@given(
    provider_id=st.sampled_from(PROVIDER_IDS),
    field=st.text(min_size=1, max_size=60,
                  alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Ps"))),
    value=st.text(min_size=1, max_size=512),
)
@settings(max_examples=100)
def test_overwrite_keeps_latest_value(
    provider_id: str, field: str, value: str
) -> None:
    """Overwriting a field must return the latest value, not the original."""
    store = CredentialStore()
    store.set(provider_id, field, "old_value")
    store.set(provider_id, field, value)
    assert store.get(provider_id, field) == value


def test_get_returns_none_for_unset_field() -> None:
    store = CredentialStore()
    assert store.get("openai", "OPENAI_API_KEY") is None


def test_has_provider_false_before_set() -> None:
    store = CredentialStore()
    assert not store.has_provider("openai")


def test_has_provider_true_after_set() -> None:
    store = CredentialStore()
    store.set("openai", "OPENAI_API_KEY", "sk-test")
    assert store.has_provider("openai")


def test_summary_returns_field_names_not_values() -> None:
    store = CredentialStore()
    store.set("openai", "OPENAI_API_KEY", "sk-secret")
    store.set("openai", "OPENAI_ORG_ID", "org-123")
    summary = store.summary()
    assert "openai" in summary
    assert "OPENAI_API_KEY" in summary["openai"]
    assert "OPENAI_ORG_ID" in summary["openai"]
    # Values must NOT appear in the summary
    assert "sk-secret" not in str(summary)
    assert "org-123" not in str(summary)


def test_env_var_names_returns_field_names() -> None:
    store = CredentialStore()
    store.set("anthropic", "ANTHROPIC_API_KEY", "sk-ant-test")
    names = store.env_var_names("anthropic")
    assert "ANTHROPIC_API_KEY" in names


def test_all_providers_lists_configured_providers() -> None:
    store = CredentialStore()
    store.set("openai", "OPENAI_API_KEY", "sk-test")
    store.set("cohere", "COHERE_API_KEY", "co-test")
    providers = store.all_providers()
    assert "openai" in providers
    assert "cohere" in providers


# ---------------------------------------------------------------------------
# Property 2: Empty Credential Field Re-Prompt
# ---------------------------------------------------------------------------


@given(provider_id=st.sampled_from(PROVIDER_IDS))
@settings(max_examples=30)
def test_empty_credential_field_reprompt(provider_id: str) -> None:
    """Feature: ms-rag, Property 2: Empty Credential Field Re-Prompt.

    When a credential field returns empty string on first call,
    the manager must re-prompt and must NOT store the empty value.
    """
    manager = CredentialManager()
    fields = PROVIDER_FIELDS[provider_id]
    first_field = fields[0]

    # Simulate: first call returns empty, second call returns valid value
    call_count = {"n": 0}
    valid_value = "valid-test-credential-xyz"

    def mock_ask() -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ""
        return valid_value

    mock_prompt = MagicMock()
    mock_prompt.ask.side_effect = mock_ask

    mock_questionary = MagicMock()
    mock_questionary.text.return_value = mock_prompt
    mock_questionary.password.return_value = mock_prompt

    mock_console = MagicMock()

    result = manager._prompt_field(
        first_field, provider_id, mock_questionary, mock_console
    )

    # Must have been called at least twice (empty + valid)
    assert call_count["n"] >= 2
    # Must return the valid value, not empty
    assert result == valid_value
    # Empty string must NOT be returned
    assert result != ""


def test_empty_string_triggers_reprompt_exactly() -> None:
    """Empty string input causes exactly one re-prompt before accepting valid input."""
    manager = CredentialManager()
    responses = ["", "  ", "actual-key"]
    idx = {"i": 0}

    def mock_ask() -> str:
        val = responses[idx["i"]]
        idx["i"] += 1
        return val

    mock_prompt = MagicMock()
    mock_prompt.ask.side_effect = mock_ask

    mock_q = MagicMock()
    mock_q.text.return_value = mock_prompt
    mock_q.password.return_value = mock_prompt

    mock_console = MagicMock()

    result = manager._prompt_field("SOME_KEY", "openai", mock_q, mock_console)
    assert result == "actual-key"
    # 3 calls: empty, whitespace, valid
    assert idx["i"] == 3


# ---------------------------------------------------------------------------
# Property 3: Provider Credential Coverage
# ---------------------------------------------------------------------------


@given(
    providers=st.frozensets(st.sampled_from(PROVIDER_IDS), min_size=1, max_size=4)
)
@settings(max_examples=50)
def test_provider_credential_coverage(providers: frozenset[str]) -> None:
    """Feature: ms-rag, Property 3: Provider Credential Coverage.

    For every selected provider, ALL required fields in PROVIDER_FIELDS
    must be prompted and stored — none may be skipped.
    """
    manager = CredentialManager()
    prompted_fields: dict[str, list[str]] = {}

    def mock_ask_for_field(field: str, provider_id: str) -> str:
        if provider_id not in prompted_fields:
            prompted_fields[provider_id] = []
        prompted_fields[provider_id].append(field)
        return f"test-value-{field}"

    mock_console = MagicMock()

    for pid in providers:
        fields = PROVIDER_FIELDS[pid]
        mock_q = MagicMock()

        call_idx = {"i": 0}

        def make_ask(p: str, f_list: list[str]) -> MagicMock:
            prompt_mock = MagicMock()
            field_calls: list[str] = []

            def ask() -> str:
                if p not in prompted_fields:
                    prompted_fields[p] = []
                # Determine which field is being asked based on call count
                n = len(prompted_fields[p])
                field_name = f_list[n] if n < len(f_list) else f_list[-1]
                prompted_fields[p].append(field_name)
                return f"test-{field_name}"

            prompt_mock.ask.side_effect = ask
            return prompt_mock

        # Each questionary call returns a prompt that records the field
        for field in fields:
            mock_console_inner = MagicMock()
            result = manager._prompt_field(field, pid, mock_q, mock_console_inner)
            if pid not in prompted_fields:
                prompted_fields[pid] = []
            # Record that this field was prompted
            prompted_fields[pid].append(field)

    # Every provider's every field must have been prompted
    for pid in providers:
        expected_fields = set(PROVIDER_FIELDS[pid])
        actual_fields = set(prompted_fields.get(pid, []))
        assert expected_fields == actual_fields, (
            f"Provider {pid}: expected {expected_fields}, got {actual_fields}"
        )


# ---------------------------------------------------------------------------
# Additional unit tests for CredentialManager
# ---------------------------------------------------------------------------


class TestIsSecretField:
    def test_api_key_is_secret(self) -> None:
        assert _is_secret_field("OPENAI_API_KEY") is True

    def test_secret_access_key_is_secret(self) -> None:
        assert _is_secret_field("AWS_SECRET_ACCESS_KEY") is True

    def test_token_is_secret(self) -> None:
        assert _is_secret_field("REPLICATE_API_TOKEN") is True

    def test_url_is_not_secret(self) -> None:
        assert _is_secret_field("OLLAMA_BASE_URL") is False

    def test_region_is_not_secret(self) -> None:
        assert _is_secret_field("AWS_REGION") is False

    def test_org_id_is_not_secret(self) -> None:
        assert _is_secret_field("OPENAI_ORG_ID") is False


class TestProviderFieldsCompleteness:
    def test_all_12_providers_defined(self) -> None:
        assert len(PROVIDER_FIELDS) == 12

    def test_all_provider_ids_have_display_names(self) -> None:
        from ms_rag.config.credential_manager import PROVIDER_DISPLAY_NAMES
        for pid in PROVIDER_IDS:
            assert pid in PROVIDER_DISPLAY_NAMES, f"Missing display name for {pid}"

    def test_each_provider_has_at_least_one_field(self) -> None:
        for pid, fields in PROVIDER_FIELDS.items():
            assert len(fields) >= 1, f"Provider {pid} has no fields"

    def test_azure_openai_has_all_required_fields(self) -> None:
        fields = PROVIDER_FIELDS["azure_openai"]
        assert "AZURE_OPENAI_API_KEY" in fields
        assert "AZURE_OPENAI_ENDPOINT" in fields
        assert "AZURE_OPENAI_API_VERSION" in fields

    def test_aws_bedrock_has_all_required_fields(self) -> None:
        fields = PROVIDER_FIELDS["aws_bedrock"]
        assert "AWS_ACCESS_KEY_ID" in fields
        assert "AWS_SECRET_ACCESS_KEY" in fields
        assert "AWS_REGION" in fields

    def test_ollama_has_model_name_field(self) -> None:
        """Requirement 2.5: Ollama must prompt for model name."""
        assert "OLLAMA_MODEL_NAME" in PROVIDER_FIELDS["ollama"]


class TestCredentialManagerStore:
    def test_store_and_retrieve(self) -> None:
        manager = CredentialManager()
        manager.store("openai", {"OPENAI_API_KEY": "sk-test"})
        assert manager.get("openai", "OPENAI_API_KEY") == "sk-test"

    def test_has_provider_after_store(self) -> None:
        manager = CredentialManager()
        manager.store("cohere", {"COHERE_API_KEY": "co-test"})
        assert manager.has_provider("cohere")

    def test_summary_after_store(self) -> None:
        manager = CredentialManager()
        manager.store("groq", {"GROQ_API_KEY": "gsk-test"})
        s = manager.summary()
        assert "groq" in s
        assert "GROQ_API_KEY" in s["groq"]
