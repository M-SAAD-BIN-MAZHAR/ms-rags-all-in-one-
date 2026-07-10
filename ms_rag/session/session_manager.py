"""Session Manager for MS_RAG.

Handles /save command and --load CLI argument for PipelineConfig
serialization and deserialization.

- /save serialises PipelineConfig to JSON (18.1)
- --load deserialises and skips to query loop (18.2)
- Descriptive error + fallback on missing/invalid file (18.3)
- JSON includes schema_version field (18.4)
"""

from __future__ import annotations

import json
from pathlib import Path

from ms_rag.models import PipelineConfig
from ms_rag.utils.exceptions import SessionLoadError


class SessionManager:
    """Serialises and deserialises PipelineConfig to/from JSON.

    Usage::

        manager = SessionManager()

        # Save
        manager.save(config, Path("session.json"))

        # Load
        config = manager.load(Path("session.json"))
    """

    def save(self, config: PipelineConfig, file_path: Path) -> None:
        """Serialise PipelineConfig to a JSON file.

        The output always contains ``schema_version`` for forward compatibility.
        Credentials are intentionally excluded — only Pipeline_Config data is written.

        Args:
            config:    The PipelineConfig to persist.
            file_path: Destination file path. Parent directories are created
                       if they do not exist.

        Raises:
            OSError: If the file cannot be written.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        json_str = config.to_json()
        file_path.write_text(json_str, encoding="utf-8")

    def load(self, file_path: Path) -> PipelineConfig:
        """Deserialise a PipelineConfig from a JSON file.

        Args:
            file_path: Path to the saved config JSON file.

        Returns:
            A reconstructed PipelineConfig.

        Raises:
            SessionLoadError: If the file does not exist or cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise SessionLoadError(
                f"Session config file not found: {file_path}. "
                "Falling back to interactive setup.",
                file_path=str(file_path),
            )

        try:
            json_str = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionLoadError(
                f"Cannot read session config file: {file_path} — {exc}",
                file_path=str(file_path),
                original=exc,
            ) from exc

        if not json_str.strip():
            raise SessionLoadError(
                f"Session config file is empty: {file_path}",
                file_path=str(file_path),
            )

        try:
            config = PipelineConfig.from_json(json_str)
        except Exception as exc:  # noqa: BLE001
            raise SessionLoadError(
                f"Failed to parse session config at {file_path}: {exc}. "
                "Falling back to interactive setup.",
                file_path=str(file_path),
                original=exc,
            ) from exc

        return config
