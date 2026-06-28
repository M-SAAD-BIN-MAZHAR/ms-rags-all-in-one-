from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ms_rag.ingestion.graph_store import GraphStoreConnector
from ms_rag.models import GraphStoreConfig, PipelineConfig


def _config(path: Path) -> GraphStoreConfig:
    return GraphStoreConfig(
        store_type="local_json",
        connection_params={"GRAPH_STORE_PATH": str(path)},
        graph_name="test_graph",
        query_mode="hybrid",
    )


def test_local_graph_store_builds_persists_and_loads_graph(tmp_path: Path) -> None:
    connector = GraphStoreConnector()
    config = _config(tmp_path / "graph.json")
    docs = [
        Document(
            page_content="Muhammad Saad builds MS_RAG with GraphRAG support.",
            metadata={"ms_rag_child_id": "chunk-1", "source": "resume.txt"},
        ),
        Document(
            page_content="MS_RAG supports Neo4j Aura and local graph storage.",
            metadata={"ms_rag_child_id": "chunk-2", "source": "docs.txt"},
        ),
    ]

    graph = connector.build_graph_index(docs, llm=None)
    connector.persist_graph(config, graph)
    loaded = connector.load_graph(config)

    assert loaded["nodes"]
    assert loaded["communities"]
    assert loaded["source_chunks"][0]["chunk_id"] == "chunk-1"


def test_graph_context_supports_local_and_global_modes(tmp_path: Path) -> None:
    connector = GraphStoreConnector()
    config = _config(tmp_path / "graph.json")
    docs = [
        Document(
            page_content="Neo4j Aura stores persistent graph entities for GraphRAG.",
            metadata={"ms_rag_child_id": "chunk-1"},
        )
    ]
    graph = connector.build_graph_index(docs, llm=None)
    connector.persist_graph(config, graph)

    config.query_mode = "local"
    local_context = connector.retrieve_graph_context(config, "What does Neo4j Aura store?", llm=None)
    config.query_mode = "global"
    global_context = connector.retrieve_graph_context(config, "What does Neo4j Aura store?", llm=None)

    assert "Neo4j Aura" in local_context or "neo4j_aura" in local_context.lower()
    assert "Community" in global_context


def test_graph_store_config_round_trip_sanitizes_secrets() -> None:
    config = PipelineConfig(
        graph_store=GraphStoreConfig(
            store_type="neo4j",
            connection_params={
                "NEO4J_URI": "neo4j+s://example.databases.neo4j.io",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "secret",
                "NEO4J_DATABASE": "neo4j",
            },
            graph_name="prod_graph",
            query_mode="hybrid",
        )
    )

    json_text = config.to_json()
    restored = PipelineConfig.from_json(json_text)

    assert "secret" not in json_text
    assert restored.graph_store is not None
    assert restored.graph_store.connection_params["NEO4J_PASSWORD"] == "NEO4J_PASSWORD"
    assert restored.graph_store.query_mode == "hybrid"
