"""Audit LangChain-related imports used across ms_rag."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS_RAG = ROOT / "ms_rag"


def _symbol_name(node: ast.expr) -> str | None:
    """Return a valid Python identifier from an import alias node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.alias):
        return node.asname or node.name.split(".")[-1]
    return None


def collect_imports() -> set[tuple[str, str | None]]:
    """Return (module, symbol) pairs from ms_rag Python files using AST."""
    pairs: set[tuple[str, str | None]] = set()

    for path in MS_RAG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pairs.add((alias.name, None))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    symbol = alias.asname or alias.name
                    if symbol.isidentifier():
                        pairs.add((module, symbol))

    return pairs


def check_import(module: str, symbol: str | None) -> tuple[bool, str]:
    try:
        if symbol is None:
            importlib.import_module(module)
            return True, "OK"
        mod = importlib.import_module(module)
        getattr(mod, symbol)
        return True, "OK"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    pairs = sorted(collect_imports())
    lang_pairs = [
        (m, s)
        for m, s in pairs
        if m.startswith(
            (
                "langchain",
                "langgraph",
                "ragas",
                "trulens",
                "deepeval",
                "flashrank",
                "cohere",
                "sentence_transformers",
            )
        )
    ]

    failures: list[tuple[str, str | None, str]] = []
    optional_modules = {
        "langchain_pinecone",
        "langchain_qdrant",
        "langchain_weaviate",
        "langchain_postgres",
        "langchain_milvus",
        "langchain_elasticsearch",
        "langchain_mongodb",
        "langchain_aws",
        "langchain_redis",
        "ragas",
        "deepeval",
        "trulens",
        "trulens.apps.langchain",
        "flashrank",
    }

    print(f"Checking {len(lang_pairs)} LangChain-related imports...\n")
    for module, symbol in lang_pairs:
        ok, msg = check_import(module, symbol)
        label = f"{module}.{symbol}" if symbol else module
        if ok:
            print(f"  OK  {label}")
        else:
            is_optional = module.split(".")[0] in optional_modules
            status = "WARN (optional)" if is_optional else "FAIL"
            print(f"  {status}  {label} -> {msg}")
            if not is_optional:
                failures.append((module, symbol, msg))

    print(f"\n{len(failures)} required import failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
