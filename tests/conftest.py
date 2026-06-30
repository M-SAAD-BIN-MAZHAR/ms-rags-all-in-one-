"""Shared pytest memory hygiene for MS-RAGS tests.

The framework supports local embedding/reranking models and many optional
provider SDKs. A single pytest process can otherwise retain large model caches
or vector-store objects between tests until the full run exits.
"""

from __future__ import annotations

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def _ms_rags_memory_cleanup() -> None:
    """Release common heavyweight caches after each test."""
    yield

    try:
        from ms_rag.query.reranking_module import clear_reranker_model_cache

        clear_reranker_model_cache()
    except Exception:
        pass

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            cuda = getattr(torch_module, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                cuda.empty_cache()
        except Exception:
            pass

    gc.collect()
