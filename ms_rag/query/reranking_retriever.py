"""Reranking retriever wrapper compatible with LangChain LCEL chains."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RerankingRetriever(BaseRetriever):
    """Wraps a base retriever and applies post-retrieval reranking."""

    base_retriever: object
    reranking_module: object
    config: object
    llm: object | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)  # type: ignore[union-attr]
        self._ms_rag_pre_rerank_count = len(docs)
        reranked = self.reranking_module.rerank(  # type: ignore[union-attr]
            query,
            docs,
            self.config,
            llm=self.llm,
        )
        self._ms_rag_post_rerank_count = len(reranked)
        return reranked
