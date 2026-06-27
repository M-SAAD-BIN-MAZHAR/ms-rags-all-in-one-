"""Query Enhancement Module for MS_RAG.

Provides interactive configuration and runtime application of all
supported query enhancement techniques.

Requirement 11:
- Ask yes/no for query enhancement (11.1)
- Show all 7 techniques as a checkbox (11.2)
- Allow multi-select (11.3)
- Store and persist selections across all queries in the session (11.4)
- If no: proceed with raw query (11.5)
- If HyDE selected: prompt which LLM to use (11.6)
"""

from __future__ import annotations

import warnings

from ms_rag.utils.error_formatting import format_provider_error

try:
    import questionary
    from rich.console import Console
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Technique registry
# ---------------------------------------------------------------------------

QUERY_ENHANCEMENT_TECHNIQUES: list[dict] = [
    {
        "id": "query_rewriting",
        "display": "Query Rewriting (LLM-based)",
        "description": "Rewrites the query using an LLM to be clearer and more retrieval-friendly",
    },
    {
        "id": "query_expansion",
        "display": "Query Expansion (synonyms / related terms)",
        "description": "Expands the query with synonyms and related terms to improve recall",
    },
    {
        "id": "hyde",
        "display": "HyDE — Hypothetical Document Embeddings",
        "description": "Generates a hypothetical ideal answer, then embeds it for retrieval",
    },
    {
        "id": "multi_query",
        "display": "Multi-Query Generation",
        "description": "Generates N query variants and retrieves documents for each",
    },
    {
        "id": "step_back_prompting",
        "display": "Step-Back Prompting",
        "description": "Generates a higher-level background question before retrieval",
    },
    {
        "id": "sub_question_decomposition",
        "display": "Sub-Question Decomposition",
        "description": "Breaks complex queries into simpler sub-questions for targeted retrieval",
    },
    {
        "id": "rag_fusion",
        "display": "RAG-Fusion Query Generation",
        "description": "Generates multiple search queries fused via Reciprocal Rank Fusion",
    },
]

TECHNIQUE_IDS: list[str] = [t["id"] for t in QUERY_ENHANCEMENT_TECHNIQUES]


class QueryEnhancer:
    """Interactive configuration and runtime enhancement of user queries.

    Usage::

        enhancer = QueryEnhancer()
        selected = enhancer.configure(configured_providers=["openai"])
        # selected: ["query_rewriting", "hyde"]

        enhanced_queries = enhancer.enhance(
            query="What is RAG?",
            techniques=selected,
            llm=llm_instance,
        )
    """

    def __init__(self) -> None:
        self.hyde_llm_provider: str | None = None

    def configure(self, configured_providers: list[str] | None = None) -> list[str]:
        """Interactive yes/no → technique checkbox → optional HyDE LLM selection.

        Requirement 11.1-11.6.

        Args:
            configured_providers: Available provider IDs for HyDE LLM selection.

        Returns:
            List of selected technique IDs (may be empty if user says no).
        """
        from ms_rag.ui.prompts import prompt_checkbox, prompt_confirm, prompt_select  # noqa: PLC0415

        console = Console()

        console.print("\n[bold cyan]Step 10 — Query Enhancement[/bold cyan]\n")

        wants_enhancement = prompt_confirm(
            "  Do you want to enable query enhancement techniques?",
            default=True,
            console=console,
        )

        if not wants_enhancement:
            console.print("  [dim]Skipping query enhancement — raw query will be used.[/dim]")
            return []

        choices = [
            questionary.Choice(
                title=f"{t['display']}  —  {t['description']}",
                value=t["id"],
            )
            for t in QUERY_ENHANCEMENT_TECHNIQUES
        ]

        selected = prompt_checkbox(
            "  Select query enhancement techniques:",
            choices=choices,
            min_selections=0,
            console=console,
        )

        if not selected:
            console.print("  [dim]No techniques selected — raw query will be used.[/dim]")
            return []

        # HyDE requires LLM selection (Req 11.6)
        hyde_provider: str | None = None
        if "hyde" in selected:
            hyde_provider = self._select_hyde_llm(configured_providers or [], console)
            self.hyde_llm_provider = hyde_provider
            console.print(
                f"  [green]HyDE LLM: [bold]{hyde_provider or 'default'}[/bold][/green]"
            )

        console.print(
            f"  [green]✓ Query enhancement enabled: "
            f"[bold]{', '.join(selected)}[/bold][/green]"
        )

        return selected

    def enhance(
        self,
        query: str,
        techniques: list[str],
        llm: object | None = None,
        hyde_provider: str | None = None,
        num_queries: int = 3,
    ) -> list[str]:
        """Apply selected enhancement techniques to the query.

        Args:
            query:        The original user query.
            techniques:   List of technique IDs to apply.
            llm:          A LangChain BaseChatModel / BaseLLM instance.
            hyde_provider: Provider used for HyDE (informational).
            num_queries:   Number of variant queries for multi_query / rag_fusion.

        Returns:
            List of enhanced query strings.  For most techniques this is a
            single string; for multi_query and rag_fusion it may be N strings.
        """
        if not techniques:
            return [query]

        result_queries: list[str] = [query]

        for technique in techniques:
            try:
                result_queries = self._apply_technique(
                    technique=technique,
                    queries=result_queries,
                    llm=llm,
                    num_queries=num_queries,
                )
            except Exception as exc:  # noqa: BLE001
                # Technique failure is non-fatal — continue with current queries
                import warnings  # noqa: PLC0415
                warnings.warn(
                    f"Query enhancement technique {technique!r} failed: {exc}",
                    stacklevel=2,
                )

        return result_queries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_technique(
        self,
        technique: str,
        queries: list[str],
        llm: object | None,
        num_queries: int,
    ) -> list[str]:
        """Apply a single technique to the current list of queries."""

        if technique == "query_rewriting":
            return [self._rewrite_query(q, llm) for q in queries]

        if technique == "query_expansion":
            return [self._expand_query(q, llm) for q in queries]

        if technique == "hyde":
            # HyDE returns a hypothetical document for embedding
            return [self._generate_hypothetical_document(q, llm) for q in queries]

        if technique == "multi_query":
            expanded: list[str] = []
            for q in queries:
                expanded.extend(self._generate_multi_queries(q, llm, num_queries))
            return list(dict.fromkeys(expanded))  # deduplicate preserving order

        if technique == "step_back_prompting":
            step_back = [self._generate_step_back(q, llm) for q in queries]
            return queries + step_back  # combine original + step-back

        if technique == "sub_question_decomposition":
            sub_qs: list[str] = []
            for q in queries:
                sub_qs.extend(self._decompose_query(q, llm))
            return sub_qs if sub_qs else queries

        if technique == "rag_fusion":
            fused: list[str] = []
            for q in queries:
                fused.extend(self._generate_fusion_queries(q, llm, num_queries))
            return list(dict.fromkeys(fused))

        warnings.warn(
            f"Unknown query enhancement technique {technique!r}; using current queries unchanged.",
            stacklevel=2,
        )
        return queries

    def _rewrite_query(self, query: str, llm: object | None) -> str:
        """Rewrite query to be clearer and more retrieval-friendly."""
        if llm is None:
            return query
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that rewrites user queries to be "
                           "clearer and more suitable for document retrieval. "
                           "Return only the rewritten query, nothing else."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            return chain.invoke({"query": query}).strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Query rewriting failed; using original query: {exc}", stacklevel=2)
            return query

    def _expand_query(self, query: str, llm: object | None) -> str:
        """Expand query with synonyms and related terms."""
        if llm is None:
            return query
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Expand the following query by adding synonyms and related terms "
                           "to improve search recall. Return the expanded query only."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            return chain.invoke({"query": query}).strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Query expansion failed; using original query: {format_provider_error(exc)}",
                stacklevel=2,
            )
            return query

    def _generate_hypothetical_document(self, query: str, llm: object | None) -> str:
        """Generate a hypothetical document that would answer the query (HyDE)."""
        if llm is None:
            return query
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Write a short passage (2-3 sentences) that would be the ideal "
                           "answer to the following question. This will be used as a "
                           "hypothetical document for embedding-based retrieval."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            return chain.invoke({"query": query}).strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"HyDE generation failed; using original query: {format_provider_error(exc)}",
                stacklevel=2,
            )
            return query

    def _generate_multi_queries(
        self, query: str, llm: object | None, n: int
    ) -> list[str]:
        """Generate N rephrased variants of the query."""
        if llm is None:
            return [query]
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Generate {n} different phrasings of the following question. "
                           f"Return one per line, no numbering or bullets."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            raw = chain.invoke({"query": query})
            variants = [line.strip() for line in raw.splitlines() if line.strip()]
            return variants[:n] if variants else [query]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Multi-query generation failed; using original query: {exc}", stacklevel=2)
            return [query]

    def _generate_step_back(self, query: str, llm: object | None) -> str:
        """Generate a higher-level step-back question."""
        if llm is None:
            return query
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a more abstract, higher-level version of the question "
                           "that asks about the underlying concepts. Return only the question."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            return chain.invoke({"query": query}).strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Step-back prompting failed; using original query: {exc}", stacklevel=2)
            return query

    def _decompose_query(self, query: str, llm: object | None) -> list[str]:
        """Break complex query into simpler sub-questions."""
        if llm is None:
            return [query]
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Break the following complex question into 2-4 simpler sub-questions. "
                           "Return one sub-question per line, no numbering or bullets."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            raw = chain.invoke({"query": query})
            parts = [line.strip() for line in raw.splitlines() if line.strip()]
            return parts if parts else [query]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Sub-question decomposition failed; using original query: {exc}", stacklevel=2)
            return [query]

    def _generate_fusion_queries(
        self, query: str, llm: object | None, n: int
    ) -> list[str]:
        """Generate N search queries for RAG-Fusion."""
        if llm is None:
            return [query]
        try:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
            from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Generate {n} diverse search queries to find documents related "
                           f"to the following question. Each query should approach the topic "
                           f"from a different angle. Return one per line."),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()  # type: ignore[operator]
            raw = chain.invoke({"query": query})
            variants = [line.strip() for line in raw.splitlines() if line.strip()]
            return ([query] + variants[:n - 1]) if variants else [query]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RAG-Fusion query generation failed; using original query: {exc}", stacklevel=2)
            return [query]

    def _select_hyde_llm(
        self, configured_providers: list[str], console: object
    ) -> str | None:
        """Prompt user to select which LLM to use for HyDE generation."""
        if not configured_providers:
            console.print(  # type: ignore[union-attr]
                "  [yellow]No LLM providers configured — HyDE will use the default LLM.[/yellow]"
            )
            return None

        from ms_rag.ui.prompts import prompt_select  # noqa: PLC0415

        choices = [
            questionary.Choice(title=p.replace("_", " ").title(), value=p)
            for p in configured_providers
        ]

        selected = prompt_select(
            "  Select LLM provider for HyDE hypothetical document generation:",
            choices=choices,
            console=console,  # type: ignore[arg-type]
        )

        return selected
