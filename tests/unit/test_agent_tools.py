from __future__ import annotations

from pathlib import Path

import pytest

from ms_rag.agent.tools import AgentToolRuntime, ToolExecutionError
from ms_rag.models import AgentToolConfig, CredentialStore, PipelineConfig


def test_agent_tool_config_round_trip() -> None:
    config = PipelineConfig(
        agent_tools=AgentToolConfig(
            enabled_tools=["memory", "url_fetch"],
            tool_settings={
                "memory": {"memory_types": ["short_term"], "path": "./agent_memory/test.json"},
                "url_fetch": {"allowed_domains": ["example.com"]},
            },
        )
    )

    restored = PipelineConfig.from_json(config.to_json())

    assert restored.agent_tools is not None
    assert restored.agent_tools.enabled_tools == ["memory", "url_fetch"]
    assert restored.agent_tools.tool_settings["url_fetch"]["allowed_domains"] == ["example.com"]


def test_file_read_requires_allowlisted_path(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    blocked = tmp_path / "blocked"
    allowed.mkdir()
    blocked.mkdir()
    allowed_file = allowed / "note.txt"
    blocked_file = blocked / "secret.txt"
    allowed_file.write_text("ok", encoding="utf-8")
    blocked_file.write_text("no", encoding="utf-8")
    runtime = AgentToolRuntime(
        AgentToolConfig(
            enabled_tools=["file_read"],
            tool_settings={"file_read": {"allowed_paths": [str(allowed)], "max_bytes": 100}},
        )
    )

    assert runtime.read_file(str(allowed_file)) == "ok"
    with pytest.raises(ToolExecutionError, match="not allowlisted"):
        runtime.read_file(str(blocked_file))


def test_memory_store_persists_enabled_memory_types(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.json"
    config = AgentToolConfig(
        enabled_tools=["memory"],
        tool_settings={
            "memory": {
                "memory_types": ["long_term", "semantic"],
                "path": str(memory_path),
            }
        },
    )
    runtime = AgentToolRuntime(config)

    runtime.remember("long_term", "Use Pinecone for managed vector search.", {"source": "test"})
    recalled = runtime.recall_memory("Pinecone vector")

    assert "Pinecone" in recalled
    assert memory_path.exists()
    with pytest.raises(ToolExecutionError, match="not enabled"):
        runtime.remember("episodic", "not allowed")


def test_api_request_requires_method_and_base_allowlists() -> None:
    runtime = AgentToolRuntime(
        AgentToolConfig(
            enabled_tools=["api_request"],
            tool_settings={
                "api_request": {
                    "allowed_base_urls": ["https://api.example.com"],
                    "allowed_methods": ["GET"],
                }
            },
        )
    )

    with pytest.raises(ToolExecutionError, match="method"):
        runtime.api_request("POST", "https://api.example.com/v1/items")
    with pytest.raises(ToolExecutionError, match="base URL"):
        runtime.api_request("GET", "https://evil.example/v1/items")


def test_web_search_requires_provider_credential() -> None:
    store = CredentialStore()
    runtime = AgentToolRuntime(
        AgentToolConfig(
            enabled_tools=["web_search"],
            tool_settings={"web_search": {"provider": "tavily"}},
        ),
        credential_store=store,
    )

    with pytest.raises(ToolExecutionError, match="TAVILY_API_KEY"):
        runtime.web_search("rag frameworks")
