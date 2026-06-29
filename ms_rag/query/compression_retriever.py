"""Safe contextual compression retriever wrapper."""

from __future__ import annotations

import warnings

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever


class SafeCompressionRetriever(BaseRetriever):
    """Wrap a retriever and compressor without allowing empty silent context."""

    base_retriever: object
    base_compressor: object
    min_retained_documents: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)  # type: ignore[union-attr]
        self._ms_rag_pre_compression_count = len(docs)
        if not docs:
            self._ms_rag_post_compression_count = 0
            self._ms_rag_compression_fallback = False
            return []

        compressed = _compress_or_transform_documents(self.base_compressor, docs, query)
        compressed_docs = list(compressed or [])
        self._ms_rag_post_compression_count = len(compressed_docs)
        configured_minimum = max(int(self.min_retained_documents), 1)
        minimum = min(configured_minimum, len(docs))
        should_guard = not compressed_docs or len(docs) >= configured_minimum
        if compressed_docs and (not should_guard or len(compressed_docs) >= minimum):
            self._ms_rag_compression_fallback = False
            return compressed_docs

        warnings.warn(
            "Context compression returned too little context "
            f"({len(compressed_docs)}/{len(docs)} documents); using the top {minimum} "
            "uncompressed retrieved documents instead. Lower the compression threshold, "
            "increase retrieval top_k, or disable compression if this repeats.",
            stacklevel=2,
        )
        self._ms_rag_compression_fallback = True
        self._ms_rag_post_compression_count = minimum
        return list(docs)[:minimum]


def _compress_or_transform_documents(
    compressor: object,
    documents: list[Document],
    query: str,
) -> list[Document]:
    """Run either a LangChain compressor or document transformer.

    Some LangChain building blocks used for context compression, such as
    EmbeddingsRedundantFilter, are document transformers. They implement
    transform_documents() instead of compress_documents().
    """
    if hasattr(compressor, "compress_documents"):
        return list(compressor.compress_documents(documents, query) or [])  # type: ignore[attr-defined]
    if hasattr(compressor, "transform_documents"):
        return list(compressor.transform_documents(documents) or [])  # type: ignore[attr-defined]
    raise TypeError(
        f"Compression component {type(compressor).__name__} does not implement "
        "compress_documents() or transform_documents()."
    )


class LLMSummaryCompressor(BaseDocumentCompressor):
    """Summarize each retrieved document with the configured LLM."""

    llm: object

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: object | None = None,
    ) -> list[Document]:
        from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
        from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize the document chunk for the user's query. Keep only facts relevant to the query. "
                    "Do not add outside knowledge.",
                ),
                ("human", "Query:\n{query}\n\nDocument chunk:\n{document}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()  # type: ignore[operator]
        summarized: list[Document] = []
        for doc in documents:
            text = str(getattr(doc, "page_content", "") or "")
            if not text.strip():
                continue
            summary = str(chain.invoke({"query": query, "document": text})).strip()
            if summary:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata["ms_rag_compression"] = "summary_compression"
                summarized.append(Document(page_content=summary, metadata=metadata))
        return summarized
