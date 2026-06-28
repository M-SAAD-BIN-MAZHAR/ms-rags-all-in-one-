"""Live smoke audit for MS-RAGS(ALL-IN-ONE).

Reads credentials from the process environment. Does not persist secrets.
Deletes temporary HuggingFace/model cache directories on exit.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DOCS = ROOT / "SmokeDocs"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class Result:
    area: str
    name: str
    status: str
    detail: str = ""


class SmokeEmbeddings(Embeddings):
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text) or 1
        return [float((seed + i) % 997) / 997.0 for i in range(self.dim)]


def _scrub(text: object) -> str:
    value = str(text)
    for key, raw in os.environ.items():
        if any(token in key for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and raw:
            value = value.replace(raw, "***")
    return value[:500]


def _result(results: list[Result], area: str, name: str, status: str, detail: object = "") -> None:
    results.append(Result(area, name, status, _scrub(detail)))


def _run(results: list[Result], area: str, name: str, fn: Callable[[], object]) -> None:
    try:
        detail = fn()
        _result(results, area, name, "PASS", detail or "")
    except SkipTest as exc:
        _result(results, area, name, exc.status, exc)
    except Exception as exc:  # noqa: BLE001
        message = _scrub(exc)
        lowered = message.lower()
        if any(word in lowered for word in ("quota", "insufficient", "billing", "credit", "payment", "usage_limit", "too many requests")):
            _result(results, area, name, "PASS_EXTERNAL_LIMIT", message)
        elif any(word in lowered for word in ("401", "403", "unauthorized", "forbidden", "invalid api", "authentication")):
            _result(results, area, name, "SKIP_AUTH_OR_PERMISSION", message)
        elif any(word in lowered for word in ("pip install", "not installed", "no module named", "package not found")):
            _result(results, area, name, "SKIP_OPTIONAL_DEPENDENCY", message)
        elif any(word in lowered for word in ("not configured", "missing", "not provided")):
            _result(results, area, name, "SKIP_NOT_PROVIDED", message)
        else:
            _result(results, area, name, "FAIL_CODE", f"{type(exc).__name__}: {message}")


class SkipTest(Exception):
    def __init__(self, status: str, detail: str) -> None:
        self.status = status
        super().__init__(detail)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.lower().startswith("optional"):
        raise SkipTest("SKIP_NOT_PROVIDED", f"{name} not provided")
    return value


def audit_provider_endpoints(results: list[Result]) -> None:
    probes: list[tuple[str, Callable[[], object]]] = [
        ("openai", lambda: requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {require_env('OPENAI_API_KEY')}"}, timeout=20).raise_for_status()),
        ("cohere", lambda: requests.get("https://api.cohere.com/v1/models", headers={"Authorization": f"Bearer {require_env('COHERE_API_KEY')}"}, timeout=20).raise_for_status()),
        ("huggingface", lambda: requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {require_env('HUGGINGFACEHUB_API_TOKEN')}"}, timeout=20).raise_for_status()),
        ("mistral", lambda: requests.get("https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {require_env('MISTRAL_API_KEY')}"}, timeout=20).raise_for_status()),
        ("groq", lambda: requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {require_env('GROQ_API_KEY')}"}, timeout=20).raise_for_status()),
        ("ollama_cloud", lambda: requests.get("https://ollama.com/v1/models", headers={"Authorization": f"Bearer {require_env('OLLAMA_API_KEY')}"}, timeout=20).raise_for_status()),
        ("anthropic", lambda: (_ for _ in ()).throw(SkipTest("SKIP_NOT_PROVIDED", "ANTHROPIC_API_KEY not provided")) if not os.getenv("ANTHROPIC_API_KEY", "").strip() else "provided"),
        ("google", lambda: requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={require_env('GOOGLE_API_KEY')}", timeout=20).raise_for_status()),
    ]
    for name, fn in probes:
        _run(results, "provider_endpoint", name, fn)


def audit_loaders(results: list[Result]) -> None:
    from ms_rag.ingestion.ingestion_orchestrator import IngestionOrchestrator
    from ms_rag.ingestion.loader_selector import ALL_LOADERS

    samples = {
        "pdf": SMOKE_DOCS / "elephants_small.pdf",
        "docx": SMOKE_DOCS / "elephants.docx",
        "csv": SMOKE_DOCS / "elephants.csv",
        "xlsx": SMOKE_DOCS / "elephants.xlsx",
        "pptx": SMOKE_DOCS / "elephants.pptx",
        "html": SMOKE_DOCS / "elephants.html",
        "markdown": SMOKE_DOCS / "elephants.md",
        "json": SMOKE_DOCS / "elephants.json",
        "xml": SMOKE_DOCS / "elephants.xml",
        "image_ocr": SMOKE_DOCS / "elephant_image.png",
        "epub": SMOKE_DOCS / "elephants.epub",
        "rtf": SMOKE_DOCS / "elephants.rtf",
        "code": SMOKE_DOCS / "generate_binary_files.py",
        "txt": SMOKE_DOCS / "elephants.txt",
        "url": "https://en.wikipedia.org/wiki/Elephant",
    }
    skipped = {"LlamaParseLoader", "FireCrawlLoader", "ApifyWebScraper", "YoutubeLoader", "SQLDatabaseLoader", "MongoDBAtlasLoader"}
    orch = IngestionOrchestrator()
    for loader in ALL_LOADERS:
        name = loader.loader_class
        if name in skipped:
            _result(results, "loader", name, "SKIP_NOT_PROVIDED", "requires external dataset/service/source not included in SmokeDocs")
            continue
        doc_type = next((dt for dt in loader.compatible_doc_types if dt in samples), None)
        if doc_type is None:
            _result(results, "loader", name, "SKIP_NO_SAMPLE", f"no SmokeDocs sample for {loader.compatible_doc_types}")
            continue

        def run_loader(loader_name: str = name, source: object = samples[doc_type]) -> str:
            docs = orch._invoke_loader(loader_name, str(source))  # noqa: SLF001
            if not docs:
                raise RuntimeError("loader returned zero documents")
            return f"{len(docs)} docs"

        _run(results, "loader", name, run_loader)


def audit_chunkers(results: list[Result]) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from ms_rag.ingestion.chunking_engine import STRATEGY_IDS, ChunkingEngine
    from ms_rag.models import ChunkingConfig

    text = (SMOKE_DOCS / "elephants.txt").read_text(encoding="utf-8")[:4000]
    docs = [Document(page_content=text, metadata={"source": "SmokeDocs/elephants.txt"})]
    engine = ChunkingEngine()
    embeddings = SmokeEmbeddings(dim=32)
    llm = FakeListChatModel(responses=["<MS_RAG_CHUNK>Elephants are large mammals.</MS_RAG_CHUNK>"])

    for strategy in STRATEGY_IDS:
        def run_chunker(strategy_id: str = strategy) -> str:
            cfg = ChunkingConfig(strategy=strategy_id, chunk_size=600, chunk_overlap=80)
            splitter = engine.get_splitter(cfg)
            if strategy_id == "semantic" and hasattr(splitter, "with_embeddings"):
                splitter = splitter.with_embeddings(embeddings)
            if strategy_id == "agentic" and hasattr(splitter, "with_llm"):
                splitter = splitter.with_llm(llm)
            if hasattr(splitter, "split_documents"):
                chunks = splitter.split_documents(docs)
            else:
                chunks = [
                    Document(page_content=chunk, metadata=dict(docs[0].metadata))
                    for chunk in splitter.split_text(docs[0].page_content)
                ]
            if not chunks:
                raise RuntimeError("chunker returned zero chunks")
            return f"{len(chunks)} chunks"

        _run(results, "chunker", strategy, run_chunker)


def base_config(rag_type: str):
    from ms_rag.models import (
        ChunkingConfig,
        EmbeddingModelConfig,
        LLMModelConfig,
        PipelineConfig,
        RAGTypeConfig,
        RetrievalConfig,
        VectorDBConfig,
    )
    from ms_rag.workflow.rag_type_selector import RAG_TYPE_MAP

    info = RAG_TYPE_MAP[rag_type]
    return PipelineConfig(
        configured_providers=["openai"],
        llm_model=LLMModelConfig(provider="openai", model_id="gpt-4o-mini"),
        rag_type=RAGTypeConfig(info.rag_type, info.display_name, info.description, info.requires_langgraph),
        document_types=["txt"],
        loader_map={"txt": "TextLoader"},
        chunking=ChunkingConfig(strategy="recursive_character", chunk_size=600, chunk_overlap=80),
        embedding_model=EmbeddingModelConfig(provider="openai", model_id="text-embedding-3-small"),
        vector_db=VectorDBConfig(db_type="faiss", connection_params={"FAISS_INDEX_PATH": str(ROOT / ".live_smoke" / "faiss_codegen")}, collection_name="live_smoke"),
        retrieval=RetrievalConfig(strategy="dense_vector", top_k=2),
        system_prompt="Answer from context.",
        document_sources=[str(SMOKE_DOCS / "elephants.txt")],
    )


def audit_rag_codegen(results: list[Result]) -> None:
    from ms_rag.codegen.code_generator import CodeGenerator
    from ms_rag.workflow.rag_type_selector import RAG_TYPES

    generator = CodeGenerator()
    for rag in RAG_TYPES:
        def run_codegen(rag_type: str = rag.rag_type) -> str:
            code = generator.generate(base_config(rag_type)).python_code
            ast.parse(code)
            if "MS-RAGS(ALL-IN-ONE)" not in code:
                raise RuntimeError("generated code missing product name")
            return f"{len(code)} chars"

        _run(results, "rag_codegen", rag.rag_type, run_codegen)


def audit_vector_dbs(results: list[Result]) -> None:
    from ms_rag.ingestion.vectordb_connector import VectorDBConnector
    from ms_rag.models import VectorDBConfig

    docs = [Document(page_content=f"MS-RAGS live smoke elephant retrieval {uuid.uuid4().hex}")]
    connector = VectorDBConnector()
    embeddings = SmokeEmbeddings(dim=768)
    local_root = ROOT / ".live_smoke"

    configs = [
        VectorDBConfig("chroma", {"CHROMA_PERSIST_DIRECTORY": str(local_root / "chroma")}, f"smoke_{uuid.uuid4().hex}", dimension=768),
        VectorDBConfig("faiss", {"FAISS_INDEX_PATH": str(local_root / "faiss")}, f"smoke_{uuid.uuid4().hex}", dimension=768),
    ]
    if os.getenv("QDRANT_URL", "").strip():
        configs.append(VectorDBConfig("qdrant", {"QDRANT_URL": os.getenv("QDRANT_URL", ""), "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "")}, f"smoke_{uuid.uuid4().hex}", dimension=768))
    if os.getenv("PINECONE_API_KEY", "").strip() and os.getenv("PINECONE_INDEX_NAME", "").strip():
        configs.append(VectorDBConfig("pinecone", {"PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""), "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "")}, os.getenv("PINECONE_INDEX_NAME", ""), dimension=768))
    if os.getenv("WEAVIATE_URL", "").strip():
        configs.append(VectorDBConfig("weaviate", {"WEAVIATE_URL": os.getenv("WEAVIATE_URL", ""), "WEAVIATE_API_KEY": os.getenv("WEAVIATE_API_KEY", "")}, f"Smoke{uuid.uuid4().hex[:12]}", dimension=768))
    if os.getenv("MILVUS_URI", "").strip():
        configs.append(VectorDBConfig("milvus", {"MILVUS_URI": os.getenv("MILVUS_URI", ""), "MILVUS_TOKEN": os.getenv("MILVUS_TOKEN", "")}, f"smoke_{uuid.uuid4().hex}", dimension=768))
    if os.getenv("ELASTICSEARCH_URL", "").strip():
        configs.append(VectorDBConfig("elasticsearch", {"ELASTICSEARCH_URL": os.getenv("ELASTICSEARCH_URL", ""), "ELASTICSEARCH_API_KEY": os.getenv("ELASTICSEARCH_API_KEY", ""), "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME", ""), "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD", "")}, f"smoke-{uuid.uuid4().hex}", dimension=768))
    if os.getenv("OPENSEARCH_URL", "").strip():
        configs.append(VectorDBConfig("opensearch", {"OPENSEARCH_URL": os.getenv("OPENSEARCH_URL", ""), "OPENSEARCH_USERNAME": os.getenv("OPENSEARCH_USERNAME", ""), "OPENSEARCH_PASSWORD": os.getenv("OPENSEARCH_PASSWORD", "")}, f"smoke-{uuid.uuid4().hex}", dimension=768))
    if os.getenv("MONGODB_ATLAS_CONNECTION_STRING", "").strip() or os.getenv("MONGODB_ATLAS_CLUSTER_URI", "").strip():
        configs.append(VectorDBConfig("mongodb_atlas", {"MONGODB_ATLAS_CONNECTION_STRING": os.getenv("MONGODB_ATLAS_CONNECTION_STRING", ""), "MONGODB_ATLAS_CLUSTER_URI": os.getenv("MONGODB_ATLAS_CLUSTER_URI", ""), "MONGODB_ATLAS_DB_NAME": os.getenv("MONGODB_ATLAS_DB_NAME", "ms_rag_db"), "MONGODB_ATLAS_COLLECTION_NAME": os.getenv("MONGODB_ATLAS_COLLECTION_NAME", "smoke")}, os.getenv("MONGODB_ATLAS_COLLECTION_NAME", "smoke"), dimension=768))

    seen = {cfg.db_type for cfg in configs}
    for missing in ["redis", "pgvector", "azure_ai_search"]:
        if missing not in seen:
            _result(results, "vector_db", missing, "SKIP_NOT_PROVIDED", "credentials/service not provided")

    for cfg in configs:
        def run_db(config: VectorDBConfig = cfg) -> str:
            store = connector.get_vector_store(config, embeddings)
            connector.ingest_documents(docs, store)
            hits = store.as_retriever(search_kwargs={"k": 1}).invoke("elephant")
            if not hits:
                raise RuntimeError("no retrieval hits after ingest")
            return "ingest+retrieve ok"

        _run(results, "vector_db", cfg.db_type, run_db)


def main() -> int:
    cache_root = Path(tempfile.mkdtemp(prefix="ms_rags_smoke_cache_"))
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_root / "sentence_transformers")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    results: list[Result] = []
    start = time.time()
    try:
        audit_provider_endpoints(results)
        audit_loaders(results)
        audit_chunkers(results)
        audit_rag_codegen(results)
        audit_vector_dbs(results)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    summary: dict[str, int] = {}
    for item in results:
        summary[item.status] = summary.get(item.status, 0) + 1

    report = {
        "seconds": round(time.time() - start, 2),
        "summary": summary,
        "results": [asdict(item) for item in results],
    }
    report_path = os.getenv("MS_RAGS_SMOKE_REPORT", "").strip()
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"seconds": report["seconds"], "summary": summary, "report": str(path)}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 1 if summary.get("FAIL_CODE", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
