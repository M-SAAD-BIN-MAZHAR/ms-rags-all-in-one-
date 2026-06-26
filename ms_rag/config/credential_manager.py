"""Credential Manager for MS_RAG.

Handles all LLM provider credential collection, in-memory storage,
and optional encrypted persistence to disk.

Requirement 2:
- Display all 12 supported LLM providers (2.1)
- Prompt for provider-specific credential fields (2.2)
- Store in CredentialStore; offer encrypted persistence (2.3)
- Re-prompt on empty required fields (2.4)
- Extra prompt for Ollama local model name/path (2.5)
- Show summary and confirm/edit before proceeding (2.6)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from ms_rag.models import CredentialStore
from ms_rag.utils.exceptions import CredentialError

# ---------------------------------------------------------------------------
# Provider field definitions
# ---------------------------------------------------------------------------

PROVIDER_REQUIRED_FIELDS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "huggingface": ["HUGGINGFACEHUB_API_TOKEN"],
    "google_gemini": ["GOOGLE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "together_ai": ["TOGETHER_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "replicate": ["REPLICATE_API_TOKEN"],
    "azure_openai": [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ],
    "aws_bedrock": [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ],
    "ollama": ["OLLAMA_MODEL_NAME"],
}

PROVIDER_OPTIONAL_FIELDS: dict[str, list[str]] = {
    "openai": ["OPENAI_ORG_ID"],
    "azure_openai": ["AZURE_OPENAI_API_VERSION"],
    "aws_bedrock": ["AWS_REGION"],
    "ollama": ["OLLAMA_BASE_URL"],
}

# All fields (required + optional) — used for coverage tests and summaries
PROVIDER_FIELDS: dict[str, list[str]] = {
    pid: PROVIDER_REQUIRED_FIELDS[pid] + PROVIDER_OPTIONAL_FIELDS.get(pid, [])
    for pid in PROVIDER_REQUIRED_FIELDS
}

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic (Claude)",
    "cohere": "Cohere",
    "huggingface": "HuggingFace Inference API",
    "google_gemini": "Google Gemini",
    "mistral": "Mistral AI",
    "together_ai": "Together AI",
    "groq": "Groq",
    "replicate": "Replicate",
    "azure_openai": "Azure OpenAI",
    "aws_bedrock": "AWS Bedrock",
    "ollama": "Ollama (Local)",
}

# Fields that should be masked during display (passwords / secret keys)
_SECRET_SUFFIXES = ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")

PROVIDER_IDS: list[str] = list(PROVIDER_FIELDS.keys())


def _is_secret_field(field_name: str) -> bool:
    """Return True if the field value should be masked in display."""
    return any(field_name.upper().endswith(suffix) for suffix in _SECRET_SUFFIXES)


# ---------------------------------------------------------------------------
# CredentialManager
# ---------------------------------------------------------------------------


class CredentialManager:
    """Interactive manager for LLM provider credentials.

    Usage::

        manager = CredentialManager()
        selected = manager.prompt_providers()          # show numbered list
        for pid in selected:
            creds = manager.collect_credentials(pid)   # prompt fields
            manager.store(pid, creds)
        manager.display_summary_and_confirm()
    """

    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self._store = credential_store or CredentialStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt_providers(self) -> list[str]:
        """Display a numbered multi-select list of LLM providers.

        Returns:
            List of selected provider IDs.
        """
        import questionary  # noqa: PLC0415
        from ms_rag.ui.prompts import prompt_checkbox  # noqa: PLC0415

        choices = [
            questionary.Choice(
                title=f"{PROVIDER_DISPLAY_NAMES[pid]}",
                value=pid,
            )
            for pid in PROVIDER_IDS
        ]

        return prompt_checkbox(
            "Select the LLM providers you want to use:",
            choices,
            min_selections=1,
        )

    def collect_credentials(self, provider_id: str) -> dict[str, str]:
        """Prompt the user for all required credential fields for *provider_id*.

        Re-prompts on empty values (Requirement 2.4).

        Args:
            provider_id: One of the keys in PROVIDER_FIELDS.

        Returns:
            Dict mapping field_name -> value.

        Raises:
            CredentialError: If provider_id is not recognised.
        """
        import questionary  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415

        console = Console()

        if provider_id not in PROVIDER_FIELDS:
            raise CredentialError(
                f"Unknown provider: {provider_id}",
                provider_id=provider_id,
            )

        required_fields = PROVIDER_REQUIRED_FIELDS[provider_id]
        optional_fields = PROVIDER_OPTIONAL_FIELDS.get(provider_id, [])
        display_name = PROVIDER_DISPLAY_NAMES[provider_id]
        collected: dict[str, str] = {}

        console.print(f"\n[bold cyan]  {display_name}[/bold cyan]")

        for field in required_fields:
            value = self._prompt_field(
                field, provider_id, questionary, console, required=True
            )
            collected[field] = value

        for field in optional_fields:
            value = self._prompt_field(
                field, provider_id, questionary, console, required=False
            )
            if value:
                collected[field] = value

        return collected

    def store(self, provider_id: str, credentials: dict[str, str]) -> None:
        """Store credentials dict in the in-memory CredentialStore."""
        for field, value in credentials.items():
            self._store.set(provider_id, field, value)

    def get(self, provider_id: str, field: str) -> str | None:
        """Retrieve a specific credential value."""
        return self._store.get(provider_id, field)

    def has_provider(self, provider_id: str) -> bool:
        """Return True if credentials exist for the provider."""
        return self._store.has_provider(provider_id)

    def summary(self) -> dict[str, list[str]]:
        """Return dict mapping provider_id -> list of configured field names."""
        return self._store.summary()

    def display_summary_and_confirm(self) -> bool:
        """Show credential summary and ask user to confirm or go back.

        Requirement 2.6.

        Returns:
            True if user confirms, False if user wants to edit.
        """
        import questionary  # noqa: PLC0415
        from ms_rag.ui.prompts import prompt_confirm  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        console = Console()
        summary = self._store.summary()

        if not summary:
            console.print("[yellow]  No providers configured.[/yellow]")
            return True

        table = Table(title="Configured Providers", border_style="cyan")
        table.add_column("Provider", style="bold white")
        table.add_column("Fields Configured", style="green")

        for pid, fields in summary.items():
            display = PROVIDER_DISPLAY_NAMES.get(pid, pid)
            table.add_row(display, ", ".join(fields))

        console.print(table)

        return prompt_confirm("Proceed with these credentials?", default=True, console=console)

    # ------------------------------------------------------------------
    # Encrypted persistence
    # ------------------------------------------------------------------

    def persist_encrypted(self, file_path: Path, passphrase: str) -> None:
        """Encrypt all credentials and write to *file_path*.

        Uses Fernet symmetric encryption with a PBKDF2-derived key.

        Args:
            file_path:  Destination file path.
            passphrase: User-supplied passphrase for key derivation.
        """
        from cryptography.fernet import Fernet  # noqa: PLC0415
        from cryptography.hazmat.primitives import hashes  # noqa: PLC0415
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: PLC0415

        salt = os.urandom(16)
        key = self._derive_key(passphrase, salt)
        fernet = Fernet(key)

        plaintext = json.dumps(self._store.summary()).encode()
        # summary() returns field names only — we need values too
        full_data: dict[str, dict[str, str]] = {}
        for pid in self._store.all_providers():
            full_data[pid] = {
                field: (self._store.get(pid, field) or "")
                for field in self._store.env_var_names(pid)
            }

        ciphertext = fernet.encrypt(json.dumps(full_data).encode())

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(salt + ciphertext)

    def load_encrypted(self, file_path: Path, passphrase: str) -> None:
        """Decrypt credentials from *file_path* and populate the store.

        Args:
            file_path:  Path to the encrypted credential file.
            passphrase: The passphrase used when the file was created.

        Raises:
            CredentialError: If the file cannot be read or decryption fails.
        """
        from cryptography.fernet import Fernet, InvalidToken  # noqa: PLC0415

        file_path = Path(file_path)
        if not file_path.exists():
            raise CredentialError(
                f"Credential file not found: {file_path}",
                provider_id="",
                field="",
            )

        raw = file_path.read_bytes()
        salt, ciphertext = raw[:16], raw[16:]

        key = self._derive_key(passphrase, salt)
        fernet = Fernet(key)

        try:
            plaintext = fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise CredentialError(
                "Failed to decrypt credential file — wrong passphrase?",
                original=exc,
            ) from exc

        data: dict[str, dict[str, str]] = json.loads(plaintext.decode())
        for pid, fields in data.items():
            for field, value in fields.items():
                self._store.set(pid, field, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes) -> bytes:
        """Derive a 32-byte Fernet-compatible key using PBKDF2-HMAC-SHA256."""
        from cryptography.hazmat.primitives import hashes  # noqa: PLC0415
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: PLC0415

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        raw_key = kdf.derive(passphrase.encode())
        return base64.urlsafe_b64encode(raw_key)

    def _prompt_field(
        self,
        field: str,
        provider_id: str,
        questionary_module: object,
        console: object,
        required: bool = True,
    ) -> str:
        """Prompt for a single credential field, re-prompting on empty required input.

        Requirement 2.4: re-prompt when user enters empty value for required fields.
        """
        q = questionary_module  # type: ignore[assignment]
        is_secret = _is_secret_field(field)
        suffix = "" if required else " (optional, press Enter to skip)"

        while True:
            if is_secret:
                value: str = q.password(
                    f"    {field}{suffix}:",
                ).ask()
            else:
                value = q.text(
                    f"    {field}{suffix}:",
                    default="" if not required else None,
                ).ask()

            if value is None:
                # User pressed Ctrl+C / cancelled
                value = ""

            value = value.strip()

            if not value:
                if not required:
                    return ""
                console.print(  # type: ignore[union-attr]
                    f"[red]  ✗ {field} is required. Please enter a value.[/red]"
                )
                continue

            return value

    @property
    def credential_store(self) -> CredentialStore:
        """Expose the underlying CredentialStore (read-only reference)."""
        return self._store
