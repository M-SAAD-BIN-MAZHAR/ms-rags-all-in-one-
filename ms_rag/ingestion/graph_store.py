"""Graph store and GraphRAG index builder.

The graph layer is explicit and credential-gated. Local JSON works anywhere;
Neo4j supports managed Neo4j Aura or self-hosted Neo4j; Kuzu supports embedded
local graph persistence when the optional package is installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_rag.models import CredentialStore, GraphStoreConfig
from ms_rag.ui.prompts import (
    get_console,
    print_hint,
    print_step,
    print_success,
    prompt_confirm,
    prompt_select,
    prompt_text,
)


@dataclass(frozen=True)
class GraphStoreInfo:
    store_type: str
    display_name: str
    description: str
    credential_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()


GRAPH_STORES: list[GraphStoreInfo] = [
    GraphStoreInfo(
        store_type="local_json",
        display_name="Local JSON graph store",
        description="Portable local graph index persisted as JSON. Works in Docker and local deployments.",
        optional_fields=("GRAPH_STORE_PATH",),
    ),
    GraphStoreInfo(
        store_type="neo4j",
        display_name="Neo4j / Neo4j Aura",
        description="Managed or self-hosted graph database for persistent production GraphRAG.",
        credential_fields=("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"),
        optional_fields=("NEO4J_DATABASE",),
    ),
    GraphStoreInfo(
        store_type="kuzu",
        display_name="Kuzu embedded graph DB",
        description="Embedded graph database persisted to a local folder or mounted volume.",
        optional_fields=("KUZU_DATABASE_PATH",),
    ),
]

GRAPH_STORE_MAP = {item.store_type: item for item in GRAPH_STORES}


def _graph_path(config: GraphStoreConfig) -> Path:
    raw = config.connection_params.get("GRAPH_STORE_PATH") or f"./graph_indexes/{config.graph_name}.json"
    return Path(raw).expanduser().resolve()


def _normalise_entity(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    return cleaned[:120]


def _entity_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:100] or "entity"


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_extract(text: str) -> dict[str, Any]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,4}\b", text)
    entities = [{"name": _normalise_entity(item), "type": "Concept"} for item in dict.fromkeys(candidates[:12])]
    relationships = []
    for left, right in zip(entities, entities[1:]):
        relationships.append({
            "source": left["name"],
            "target": right["name"],
            "type": "RELATED_TO",
            "description": "Co-mentioned in the same chunk.",
        })
    return {"entities": entities, "relationships": relationships}


class GraphStoreConnector:
    """Prompt, test, persist, and load GraphRAG graph indexes."""

    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self.credential_store = credential_store or CredentialStore()

    def prompt_and_configure(self) -> GraphStoreConfig:
        con = get_console()
        print_step(con, "3g", "GraphRAG Graph Store")
        print_hint(
            con,
            "GraphRAG builds a persistent knowledge graph during ingestion. Choose where to store it.",
        )
        selected = prompt_select(
            "  Select graph database/store:",
            choices=[
                {
                    "name": f"{item.display_name} - {item.description}",
                    "value": item.store_type,
                }
                for item in GRAPH_STORES
            ],
            console=con,
        )
        info = GRAPH_STORE_MAP[selected]
        params: dict[str, str] = {}
        for field in info.credential_fields:
            value = prompt_text(f"  {field}:", required=True, secret="PASSWORD" in field, console=con)
            params[field] = str(value)
            self.credential_store.set(selected, field, str(value))
        for field in info.optional_fields:
            default = ""
            if field == "GRAPH_STORE_PATH":
                default = "./graph_indexes/ms_rag_graph.json"
            elif field == "KUZU_DATABASE_PATH":
                default = "./graph_indexes/kuzu"
            value = prompt_text(
                f"  {field} (optional):",
                default=default,
                required=False,
                secret="PASSWORD" in field,
                console=con,
            )
            if value:
                params[field] = str(value)
                self.credential_store.set(selected, field, str(value))
        graph_name = prompt_text("  Graph name:", default="ms_rag_graph", required=True, console=con)
        query_mode = prompt_select(
            "  GraphRAG query mode:",
            choices=[
                {"name": "Hybrid - local entity context plus global community summaries", "value": "hybrid"},
                {"name": "Local - nearest entities and relationship neighborhoods", "value": "local"},
                {"name": "Global - community summaries for corpus-level questions", "value": "global"},
            ],
            console=con,
        )
        config = GraphStoreConfig(
            store_type=selected,
            connection_params=params,
            graph_name=str(graph_name),
            query_mode=query_mode,
        )
        self.test_connection(config)
        print_success(con, f"Graph store ready: {info.display_name} / {config.graph_name}")
        return config

    def reprompt_credentials(self, config: GraphStoreConfig) -> GraphStoreConfig:
        info = GRAPH_STORE_MAP.get(config.store_type)
        if info is None:
            raise ValueError(f"Unsupported graph store: {config.store_type}")
        con = get_console()
        params = dict(config.connection_params)
        for field in info.credential_fields:
            value = prompt_text(f"  {field}:", required=True, secret="PASSWORD" in field, console=con)
            params[field] = str(value)
            self.credential_store.set(config.store_type, field, str(value))
        config.connection_params = params
        self.test_connection(config)
        return config

    def test_connection(self, config: GraphStoreConfig) -> None:
        if config.store_type == "local_json":
            path = _graph_path(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            return
        if config.store_type == "neo4j":
            try:
                from neo4j import GraphDatabase  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Neo4j graph store requires the 'neo4j' package. Install the graph extra.") from exc
            uri = config.connection_params.get("NEO4J_URI", "")
            user = config.connection_params.get("NEO4J_USERNAME", "")
            password = config.connection_params.get("NEO4J_PASSWORD", "")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            try:
                driver.verify_connectivity()
            finally:
                driver.close()
            return
        if config.store_type == "kuzu":
            try:
                import kuzu  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Kuzu graph store requires the 'kuzu' package. Install the graph extra.") from exc
            path = Path(config.connection_params.get("KUZU_DATABASE_PATH") or "./graph_indexes/kuzu")
            path.mkdir(parents=True, exist_ok=True)
            return
        raise ValueError(f"Unsupported graph store: {config.store_type}")

    def persist_graph(self, config: GraphStoreConfig, graph: dict[str, Any]) -> None:
        if config.store_type == "local_json":
            path = _graph_path(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        if config.store_type == "neo4j":
            self._persist_neo4j(config, graph)
            return
        if config.store_type == "kuzu":
            self._persist_kuzu_json(config, graph)
            return
        raise ValueError(f"Unsupported graph store: {config.store_type}")

    def load_graph(self, config: GraphStoreConfig) -> dict[str, Any]:
        if config.store_type == "local_json":
            path = _graph_path(config)
            if not path.exists():
                raise RuntimeError(f"Graph index not found: {path}")
            return json.loads(path.read_text(encoding="utf-8"))
        if config.store_type == "kuzu":
            path = Path(config.connection_params.get("KUZU_DATABASE_PATH") or "./graph_indexes/kuzu") / "graph.json"
            if not path.exists():
                raise RuntimeError(f"Kuzu graph export not found: {path}")
            return json.loads(path.read_text(encoding="utf-8"))
        if config.store_type == "neo4j":
            return self._load_neo4j(config)
        raise ValueError(f"Unsupported graph store: {config.store_type}")

    def build_graph_index(self, documents: list, llm: object | None = None) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        source_chunks: list[dict[str, str]] = []
        for index, doc in enumerate(documents):
            text = str(getattr(doc, "page_content", "") or "")
            metadata = dict(getattr(doc, "metadata", {}) or {})
            chunk_id = str(metadata.get("ms_rag_child_id") or f"chunk::{index}")
            source_chunks.append({"chunk_id": chunk_id, "text": text[:2000], "source": str(metadata.get("source", ""))})
            extraction = self._extract_entities_and_relations(text, llm)
            for entity in extraction.get("entities", []):
                name = _normalise_entity(str(entity.get("name", "")))
                if not name:
                    continue
                node_id = _entity_id(name)
                node = nodes.setdefault(
                    node_id,
                    {"id": node_id, "name": name, "type": entity.get("type", "Entity"), "chunk_ids": []},
                )
                if chunk_id not in node["chunk_ids"]:
                    node["chunk_ids"].append(chunk_id)
            for rel in extraction.get("relationships", []):
                src = _entity_id(str(rel.get("source", "")))
                dst = _entity_id(str(rel.get("target", "")))
                if not src or not dst or src == dst:
                    continue
                rel_type = re.sub(r"[^A-Z0-9_]+", "_", str(rel.get("type", "RELATED_TO")).upper()) or "RELATED_TO"
                key = (src, dst, rel_type)
                edge = edges.setdefault(
                    key,
                    {"source": src, "target": dst, "type": rel_type, "descriptions": [], "chunk_ids": []},
                )
                desc = str(rel.get("description", "")).strip()
                if desc and desc not in edge["descriptions"]:
                    edge["descriptions"].append(desc[:500])
                if chunk_id not in edge["chunk_ids"]:
                    edge["chunk_ids"].append(chunk_id)
        communities = self._build_communities(nodes, list(edges.values()), source_chunks, llm)
        return {
            "schema_version": "1.0",
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "communities": communities,
            "source_chunks": source_chunks,
        }

    def retrieve_graph_context(self, config: GraphStoreConfig, query: str, llm: object | None = None) -> str:
        graph = self.load_graph(config)
        query_entities = self._extract_entities_and_relations(query, llm).get("entities", [])
        names = {_entity_id(str(item.get("name", ""))) for item in query_entities}
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        chunks = {item["chunk_id"]: item for item in graph.get("source_chunks", [])}
        communities = graph.get("communities", [])
        local_lines: list[str] = []
        for node in nodes:
            node_id = node.get("id")
            if node_id in names or any(str(node.get("name", "")).lower() in query.lower() for _ in [0]):
                local_lines.append(f"Entity: {node.get('name')} ({node.get('type')})")
                for edge in edges:
                    if edge.get("source") == node_id or edge.get("target") == node_id:
                        local_lines.append(
                            f"Relationship: {edge.get('source')} -[{edge.get('type')}]-> {edge.get('target')}; "
                            f"{'; '.join(edge.get('descriptions', [])[:2])}"
                        )
                for chunk_id in node.get("chunk_ids", [])[:3]:
                    chunk = chunks.get(chunk_id)
                    if chunk:
                        local_lines.append(f"Evidence chunk {chunk_id}: {chunk.get('text', '')[:800]}")
        global_lines = [
            f"Community {item.get('id')}: {item.get('summary')}"
            for item in communities[:8]
        ]
        mode = config.query_mode
        if mode == "local":
            return "\n".join(local_lines)
        if mode == "global":
            return "\n".join(global_lines)
        return "\n\n".join(part for part in ("\n".join(local_lines), "\n".join(global_lines)) if part.strip())

    def _extract_entities_and_relations(self, text: str, llm: object | None) -> dict[str, Any]:
        if llm is None:
            return _fallback_extract(text)
        prompt = (
            "Extract a knowledge graph from the text. Return only JSON with keys "
            "entities and relationships. entities: [{name,type}]. relationships: "
            "[{source,target,type,description}]. Text:\n\n"
            f"{text[:6000]}"
        )
        try:
            result = llm.invoke(prompt)  # type: ignore[attr-defined]
            raw = getattr(result, "content", str(result))
            parsed = _extract_json_object(raw)
            if not isinstance(parsed.get("entities", []), list):
                return _fallback_extract(text)
            if not isinstance(parsed.get("relationships", []), list):
                parsed["relationships"] = []
            return parsed
        except Exception:
            return _fallback_extract(text)

    def _build_communities(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        chunks: list[dict[str, str]],
        llm: object | None,
    ) -> list[dict[str, Any]]:
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in edges:
            src = str(edge.get("source", ""))
            dst = str(edge.get("target", ""))
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        seen: set[str] = set()
        communities: list[dict[str, Any]] = []
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        for node_id in nodes:
            if node_id in seen:
                continue
            stack = [node_id]
            component: list[str] = []
            seen.add(node_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbour in adjacency.get(current, set()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            evidence_ids: list[str] = []
            for member in component:
                evidence_ids.extend(nodes[member].get("chunk_ids", [])[:2])
            evidence = "\n".join(chunk_by_id.get(chunk_id, {}).get("text", "")[:500] for chunk_id in evidence_ids[:8])
            names = [nodes[member]["name"] for member in component[:12]]
            summary = self._summarize_community(names, evidence, llm)
            communities.append({"id": f"community_{len(communities)}", "node_ids": component, "summary": summary})
        return communities

    @staticmethod
    def _summarize_community(names: list[str], evidence: str, llm: object | None) -> str:
        if llm is None:
            return f"Entities: {', '.join(names)}. Evidence: {evidence[:700]}"
        prompt = (
            "Summarize this knowledge graph community for GraphRAG global retrieval. "
            "Mention key entities, relationships, and useful facts.\n\n"
            f"Entities: {', '.join(names)}\nEvidence:\n{evidence}"
        )
        try:
            result = llm.invoke(prompt)  # type: ignore[attr-defined]
            return str(getattr(result, "content", result))[:1500]
        except Exception:
            return f"Entities: {', '.join(names)}. Evidence: {evidence[:700]}"

    def _persist_neo4j(self, config: GraphStoreConfig, graph: dict[str, Any]) -> None:
        from neo4j import GraphDatabase  # type: ignore

        database = config.connection_params.get("NEO4J_DATABASE") or None
        driver = GraphDatabase.driver(
            config.connection_params["NEO4J_URI"],
            auth=(config.connection_params["NEO4J_USERNAME"], config.connection_params["NEO4J_PASSWORD"]),
        )
        try:
            with driver.session(database=database) as session:
                session.run("MERGE (g:MSRAGGraph {name: $name})", name=config.graph_name)
                for node in graph.get("nodes", []):
                    session.run(
                        """
                        MERGE (e:MSRAGEntity {graph: $graph, id: $id})
                        SET e.name = $name, e.type = $type, e.chunk_ids = $chunk_ids
                        """,
                        graph=config.graph_name,
                        **node,
                    )
                for edge in graph.get("edges", []):
                    session.run(
                        """
                        MATCH (s:MSRAGEntity {graph: $graph, id: $source})
                        MATCH (t:MSRAGEntity {graph: $graph, id: $target})
                        MERGE (s)-[r:MSRAG_RELATED {graph: $graph, type: $type}]->(t)
                        SET r.descriptions = $descriptions, r.chunk_ids = $chunk_ids
                        """,
                        graph=config.graph_name,
                        **edge,
                    )
                for community in graph.get("communities", []):
                    session.run(
                        """
                        MERGE (c:MSRAGCommunity {graph: $graph, id: $id})
                        SET c.node_ids = $node_ids, c.summary = $summary
                        """,
                        graph=config.graph_name,
                        **community,
                    )
        finally:
            driver.close()

    def _load_neo4j(self, config: GraphStoreConfig) -> dict[str, Any]:
        from neo4j import GraphDatabase  # type: ignore

        database = config.connection_params.get("NEO4J_DATABASE") or None
        driver = GraphDatabase.driver(
            config.connection_params["NEO4J_URI"],
            auth=(config.connection_params["NEO4J_USERNAME"], config.connection_params["NEO4J_PASSWORD"]),
        )
        try:
            with driver.session(database=database) as session:
                nodes = [dict(r["e"]) | {"id": r["e"]["id"]} for r in session.run(
                    "MATCH (e:MSRAGEntity {graph: $graph}) RETURN e", graph=config.graph_name
                )]
                edges = [dict(r["r"]) | {"source": r["source"], "target": r["target"]} for r in session.run(
                    """
                    MATCH (s:MSRAGEntity {graph: $graph})-[r:MSRAG_RELATED {graph: $graph}]->(t:MSRAGEntity {graph: $graph})
                    RETURN s.id AS source, t.id AS target, r
                    """,
                    graph=config.graph_name,
                )]
                communities = [dict(r["c"]) | {"id": r["c"]["id"]} for r in session.run(
                    "MATCH (c:MSRAGCommunity {graph: $graph}) RETURN c", graph=config.graph_name
                )]
        finally:
            driver.close()
        return {"schema_version": "1.0", "nodes": nodes, "edges": edges, "communities": communities, "source_chunks": []}

    @staticmethod
    def _persist_kuzu_json(config: GraphStoreConfig, graph: dict[str, Any]) -> None:
        path = Path(config.connection_params.get("KUZU_DATABASE_PATH") or "./graph_indexes/kuzu")
        path.mkdir(parents=True, exist_ok=True)
        (path / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
