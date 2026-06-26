"""Chunking Engine for MS_RAG.

Factory that returns the correct LangChain TextSplitter based on the
user's selected ChunkingConfig.  Also holds strategy descriptions and
defaults used by ChunkingConfigurator (Task 8).

Requirement 6:
- Support all 11 chunking strategies (6.1)
- Provide 1-3 sentence descriptions per strategy (6.2)
- Accept exactly one strategy per session (6.3)

Design note:
    SemanticChunker comes from langchain-experimental and requires an
    Embeddings instance.  All other splitters are imported lazily inside
    get_splitter() so that missing optional packages only fail at runtime
    when actually requested, not at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ms_rag.models import ChunkingConfig

# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingStrategyInfo:
    """Metadata for a single chunking strategy."""

    strategy_id: str
    display_name: str
    description: str          # 1-3 sentences shown to user on selection
    default_chunk_size: int   # pre-filled default when prompting user
    default_overlap: int      # pre-filled default when prompting user
    supports_separators: bool = False  # whether to prompt for custom separators
    requires_tokenizer: bool = False   # whether to prompt for tokenizer
    requires_language: bool = False    # whether to prompt for programming language
    requires_llm: bool = False         # True for agentic / LLM-driven chunking


STRATEGY_DESCRIPTIONS: dict[str, ChunkingStrategyInfo] = {
    "recursive_character": ChunkingStrategyInfo(
        strategy_id="recursive_character",
        display_name="Recursive Character Text Splitter",
        description=(
            "Recursively splits text using a priority list of separators "
            "(paragraphs → sentences → words → characters) until every chunk "
            "is within the target size. "
            "Best for general-purpose use on any text document — the recommended default."
        ),
        default_chunk_size=1000,
        default_overlap=200,
        supports_separators=True,
    ),
    "fixed_size": ChunkingStrategyInfo(
        strategy_id="fixed_size",
        display_name="Fixed-Size Text Splitter",
        description=(
            "Splits text into chunks of exactly the specified character count, "
            "regardless of sentence or paragraph boundaries. "
            "Best for uniform embeddings where consistent chunk size matters more "
            "than semantic coherence."
        ),
        default_chunk_size=1000,
        default_overlap=0,
    ),
    "semantic": ChunkingStrategyInfo(
        strategy_id="semantic",
        display_name="Semantic Chunker (embedding-similarity-based)",
        description=(
            "Uses an embedding model to detect semantic boundaries: chunks are "
            "split where the cosine similarity between adjacent sentences drops "
            "below a configurable threshold. "
            "Best for documents where topic transitions matter more than fixed size."
        ),
        default_chunk_size=0,   # not size-based — controlled by threshold
        default_overlap=0,
        requires_llm=False,     # requires embeddings, not LLM
    ),
    "sentence": ChunkingStrategyInfo(
        strategy_id="sentence",
        display_name="Sentence-Based Splitter",
        description=(
            "Splits text around sentence-style boundaries, then groups sentences "
            "into chunks that fit within the configured size. "
            "Best for documents where preserving full sentences is critical, such "
            "as legal or medical texts."
        ),
        default_chunk_size=256,
        default_overlap=32,
    ),
    "paragraph": ChunkingStrategyInfo(
        strategy_id="paragraph",
        display_name="Paragraph-Based Splitter",
        description=(
            "Splits on double newlines (paragraph breaks) and groups consecutive "
            "paragraphs until the chunk size limit is reached. "
            "Best for well-structured documents like articles or books with "
            "clear paragraph formatting."
        ),
        default_chunk_size=1000,
        default_overlap=100,
    ),
    "token_based": ChunkingStrategyInfo(
        strategy_id="token_based",
        display_name="Token-Based Splitter (tiktoken / HuggingFace tokenizer)",
        description=(
            "Splits text based on token count using a specified tokenizer "
            "(tiktoken for OpenAI models, or a HuggingFace tokenizer identifier). "
            "Best when you need precise token budgets aligned with the embedding "
            "or generation model's context window."
        ),
        default_chunk_size=512,
        default_overlap=64,
        requires_tokenizer=True,
    ),
    "markdown_aware": ChunkingStrategyInfo(
        strategy_id="markdown_aware",
        display_name="Markdown-Aware Splitter",
        description=(
            "Splits Markdown documents at heading boundaries (H1, H2, H3 etc.) "
            "and preserves the heading hierarchy as chunk metadata. "
            "Best for documentation, wikis, and README files."
        ),
        default_chunk_size=1000,
        default_overlap=100,
    ),
    "html_aware": ChunkingStrategyInfo(
        strategy_id="html_aware",
        display_name="HTML-Aware Splitter (HTMLSectionSplitter)",
        description=(
            "Splits HTML documents at tag-defined section boundaries and stores "
            "section titles as metadata for each chunk. "
            "Best for scraped web pages where HTML structure reflects content "
            "organisation."
        ),
        default_chunk_size=1000,
        default_overlap=100,
    ),
    "code_aware": ChunkingStrategyInfo(
        strategy_id="code_aware",
        display_name="Code-Aware Splitter (language-specific AST boundaries)",
        description=(
            "Splits source code at language-specific boundaries such as function "
            "definitions, class declarations, and block structures using "
            "LangChain's language-aware recursive splitter. "
            "Best for code search, documentation generation, and code review pipelines."
        ),
        default_chunk_size=1000,
        default_overlap=100,
        supports_separators=False,
        requires_language=True,
    ),
    "agentic": ChunkingStrategyInfo(
        strategy_id="agentic",
        display_name="Agentic Chunking (LLM-driven boundary detection)",
        description=(
            "Uses an LLM to identify semantically meaningful chunk boundaries "
            "by analysing document structure and topic flow. "
            "Produces the highest-quality chunks at the cost of additional LLM "
            "calls per document — best for high-value documents where chunk quality "
            "directly impacts retrieval accuracy."
        ),
        default_chunk_size=1000,
        default_overlap=0,
        requires_llm=True,
    ),
    "document_aware": ChunkingStrategyInfo(
        strategy_id="document_aware",
        display_name="Document-Aware Splitter (heading hierarchy)",
        description=(
            "Splits documents by respecting their heading hierarchy (H1 → H2 → H3) "
            "and keeps heading context attached to each child chunk as metadata. "
            "Best for structured documents like reports, manuals, or textbooks where "
            "section context improves retrieval relevance."
        ),
        default_chunk_size=1000,
        default_overlap=100,
    ),
}

STRATEGY_IDS: list[str] = list(STRATEGY_DESCRIPTIONS.keys())

# Supported language identifiers for code_aware splitter
SUPPORTED_LANGUAGES: list[str] = [
    "python", "javascript", "typescript", "java", "cpp", "c",
    "csharp", "go", "ruby", "rust", "swift", "kotlin", "php",
    "scala", "html", "markdown", "latex", "sol",
]


# ---------------------------------------------------------------------------
# ChunkingEngine
# ---------------------------------------------------------------------------


class ChunkingEngine:
    """Factory that creates the correct LangChain TextSplitter from a ChunkingConfig.

    All LangChain imports are deferred to get_splitter() so that the
    chunking engine module itself can be imported without requiring
    LangChain to be installed (useful for unit tests and task 1-6 coverage).
    """

    def get_splitter(self, config: ChunkingConfig) -> object:  # -> TextSplitter
        """Return the appropriate LangChain TextSplitter for *config*.

        Args:
            config: A fully-populated ChunkingConfig from ChunkingConfigurator.

        Returns:
            A LangChain TextSplitter instance ready to call .split_documents().

        Raises:
            ImportError: If the required LangChain package for the strategy
                         is not installed.
            ValueError:  If the strategy_id is not recognised.
        """
        strategy = config.strategy

        if strategy == "recursive_character":
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
            kwargs: dict = {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
            if config.separators:
                kwargs["separators"] = config.separators
            return RecursiveCharacterTextSplitter(**kwargs)

        if strategy == "fixed_size":
            from langchain_text_splitters import CharacterTextSplitter  # noqa: PLC0415
            return CharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separator="",
            )

        if strategy == "semantic":
            # Requires embeddings — caller must provide via config or separately
            from langchain_experimental.text_splitter import SemanticChunker  # noqa: PLC0415
            # SemanticChunker requires an Embeddings object; we return a
            # factory function wrapper here because embeddings aren't known
            # at chunking-config time.  The ingestion orchestrator will call
            # get_semantic_splitter(embeddings) instead.
            return _SemanticChunkerFactory(
                breakpoint_threshold_type="percentile",
            )

        if strategy == "sentence":
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
            return RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=[". ", "? ", "! ", "\n\n", "\n", " ", ""],
            )

        if strategy == "paragraph":
            from langchain_text_splitters import CharacterTextSplitter  # noqa: PLC0415
            return CharacterTextSplitter(
                separator="\n\n",
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

        if strategy == "token_based":
            from langchain_text_splitters import TokenTextSplitter  # noqa: PLC0415
            kwargs = {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
            if config.tokenizer:
                kwargs["encoding_name"] = config.tokenizer
            return TokenTextSplitter(**kwargs)

        if strategy == "markdown_aware":
            from langchain_text_splitters import MarkdownTextSplitter  # noqa: PLC0415
            return MarkdownTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

        if strategy == "html_aware":
            from langchain_text_splitters import HTMLSectionSplitter  # noqa: PLC0415
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
            return HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

        if strategy == "code_aware":
            from langchain_text_splitters import Language, RecursiveCharacterTextSplitter  # noqa: PLC0415
            lang_str = (config.language or "python").lower()
            try:
                lang_enum = Language[lang_str.upper()]
            except KeyError:
                lang_enum = Language.PYTHON
            return RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

        if strategy == "agentic":
            return _AgenticChunker(
                chunk_size=config.chunk_size,
            )

        if strategy == "document_aware":
            return _DocumentAwareChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Valid strategies: {STRATEGY_IDS}"
        )


# ---------------------------------------------------------------------------
# Placeholder classes for custom strategies (not in LangChain core)
# ---------------------------------------------------------------------------


class _SemanticChunkerFactory:
    """Deferred factory for SemanticChunker — requires embeddings at runtime."""

    def __init__(self, breakpoint_threshold_type: str = "percentile") -> None:
        self.breakpoint_threshold_type = breakpoint_threshold_type

    def with_embeddings(self, embeddings: object) -> object:
        """Return a SemanticChunker bound to the given embeddings model."""
        from langchain_experimental.text_splitter import SemanticChunker  # noqa: PLC0415
        return SemanticChunker(
            embeddings=embeddings,  # type: ignore[arg-type]
            breakpoint_threshold_type=self.breakpoint_threshold_type,
        )


class _AgenticChunker:
    """LLM-driven boundary detection chunker (custom implementation).

    Calls an LLM to identify semantically meaningful chunk boundaries.
    The actual LLM call is made in the ingestion orchestrator once the
    LLM is initialised.
    """

    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size
        self.strategy_id = "agentic"

    def split_documents(self, documents: list) -> list:
        """Split documents using LLM-identified boundaries.

        Fallback: uses RecursiveCharacterTextSplitter if LLM not available.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
        )
        return fallback.split_documents(documents)


class _DocumentAwareChunker:
    """Heading-hierarchy-aware chunker for structured documents."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_id = "document_aware"

    def split_documents(self, documents: list) -> list:
        """Split on heading boundaries, falling back to recursive splitting."""
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # noqa: PLC0415
        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        result = []
        for doc in documents:
            splits = header_splitter.split_text(doc.page_content)
            result.extend(recursive.split_documents(splits))
        return result
