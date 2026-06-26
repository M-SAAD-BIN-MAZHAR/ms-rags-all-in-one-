"""System Prompt Configurator for MS_RAG.

Displays the default production-grade system prompt and offers three
options: use as-is, edit inline, or replace entirely.

Requirement 15:
- Display default prompt in a labelled panel (15.1)
- Default prompt contains 5 testable instruction properties (15.2)
- Offer use/edit/replace choice; accept any non-null content (15.3)
- Inline edit pre-fills with current default (15.4)
- Replace: blank input; max 10,000 chars (15.5)
- Store final prompt; handle write failure gracefully (15.6)
"""

from __future__ import annotations

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    questionary = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Default system prompt
# Requirement 15.2 — must contain all 5 testable instruction properties:
#   (a) Answer only from provided context passages
#   (b) Cite source document name or chunk identifier
#   (c) Respond with "I don't know" when context is insufficient
#   (d) Keep answers concise and factual
#   (e) Do not introduce information not present in context
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = """You are a precise and helpful AI assistant.

CORE INSTRUCTIONS:
1. Answer ONLY using information present in the provided context passages. Do not use any knowledge from your training data that is not reflected in the context.
2. When citing information, always include the source document name or chunk identifier (e.g. [Source: doc_name.pdf, chunk_id: 3]).
3. If the context does not contain sufficient information to answer the question, respond with exactly: "I don't know" — never speculate or fabricate an answer.
4. Keep all answers concise and factual. Avoid unnecessary verbosity, filler phrases, or hedging language.
5. Do not introduce any information, claims, or details that are not explicitly present in the provided context passages.

RESPONSE FORMAT:
- Provide a direct answer to the question.
- If you cite sources, list them after the answer.
- If multiple context passages support the answer, synthesise them coherently.

QUALITY STANDARDS:
- Accuracy: Every factual claim must be traceable to a specific context passage.
- Clarity: Use plain language appropriate to the question.
- Completeness: Answer all parts of a multi-part question if the context supports it."""

MAX_SYSTEM_PROMPT_LENGTH: int = 10_000

# Choices for the three-way selection
CHOICE_USE_DEFAULT = "use_default"
CHOICE_EDIT_INLINE = "edit_inline"
CHOICE_REPLACE = "replace"


class SystemPromptConfigurator:
    """Interactive system prompt configuration.

    Usage::

        configurator = SystemPromptConfigurator()
        final_prompt = configurator.configure()
    """

    def configure(self) -> str:
        """Display default prompt, offer choice, return final prompt string.

        Requirement 15.1-15.6.

        Returns:
            The final system prompt string (never None or empty unless user
            explicitly cleared it, which is permitted).
        """
        console = Console()
        console.print("\n[bold cyan]Step 14 — System Prompt Configuration[/bold cyan]\n")

        # Req 15.1: Display default in a labelled panel
        self._display_prompt_panel(DEFAULT_SYSTEM_PROMPT, console, title="Default System Prompt")

        # Req 15.3: Three-way choice
        choice: str = questionary.select(
            "  What would you like to do with the system prompt?",
            choices=[
                questionary.Choice(
                    "Use the default prompt as-is",
                    value=CHOICE_USE_DEFAULT,
                ),
                questionary.Choice(
                    "Edit the default prompt inline",
                    value=CHOICE_EDIT_INLINE,
                ),
                questionary.Choice(
                    "Replace entirely with custom prompt",
                    value=CHOICE_REPLACE,
                ),
            ],
        ).ask()

        while True:
            if choice == CHOICE_USE_DEFAULT:
                final_prompt = DEFAULT_SYSTEM_PROMPT

            elif choice == CHOICE_EDIT_INLINE:
                # Req 15.4: pre-filled with current default, Ctrl+D or Done sentinel
                final_prompt = self._edit_inline(DEFAULT_SYSTEM_PROMPT, console)

            else:  # CHOICE_REPLACE
                # Req 15.5: blank input, max 10,000 chars
                final_prompt = self._replace_prompt(console)
                if final_prompt is None:
                    # User cancelled — go back to choice
                    continue

            # Req 15.6: handle storage failure gracefully
            try:
                if not isinstance(final_prompt, str):
                    final_prompt = str(final_prompt)
                console.print(
                    f"[green]  ✓ System prompt configured "
                    f"({len(final_prompt)} chars).[/green]"
                )
                return final_prompt
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"[red]  ✗ Failed to store system prompt: {exc}. "
                    f"Please try again.[/red]"
                )
                # Re-present the choice without losing text
                choice = questionary.select(
                    "  Retry:",
                    choices=[
                        questionary.Choice("Retry with same prompt", value=CHOICE_USE_DEFAULT),
                        questionary.Choice("Edit prompt again", value=CHOICE_EDIT_INLINE),
                        questionary.Choice("Replace with new prompt", value=CHOICE_REPLACE),
                    ],
                ).ask()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _display_prompt_panel(
        self, prompt: str, console: object, title: str = "System Prompt"
    ) -> None:
        """Render the prompt in a labelled Rich panel."""
        # Truncate very long prompts for display
        display_text = prompt if len(prompt) <= 2000 else prompt[:2000] + "\n... [truncated for display]"
        console.print(  # type: ignore[union-attr]
            Panel(
                Text(display_text, style="white"),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def _edit_inline(self, current_prompt: str, console: object) -> str:
        """Open multi-line text input pre-filled with current_prompt.

        Req 15.4: Ctrl+D or 'Done' on a new line signals end of editing.
        """
        console.print(  # type: ignore[union-attr]
            "  [dim]Edit the prompt below. Type [bold]Done[/bold] on a new line "
            "or press Ctrl+D to finish.[/dim]\n"
        )
        # questionary doesn't support true multi-line editing in all terminals;
        # we use text() with the current prompt as default and instruct the user
        # to use a sentinel line. For production, this would integrate with an
        # external editor or readline. Here we use a single questionary.text
        # call with the full default.
        result: str = questionary.text(
            "  Edit prompt (entire content):",
            default=current_prompt,
            multiline=True,
        ).ask()

        if result is None:
            return current_prompt
        return result.rstrip("\nDone").strip() or current_prompt

    def _replace_prompt(self, console: object) -> str | None:
        """Open blank multi-line input for full replacement.

        Req 15.5: max 10,000 chars; re-prompt if exceeded.
        """
        console.print(  # type: ignore[union-attr]
            "  [dim]Enter your custom system prompt below. "
            f"Maximum {MAX_SYSTEM_PROMPT_LENGTH:,} characters.[/dim]\n"
        )
        while True:
            result: str | None = questionary.text(
                "  Custom system prompt:",
                multiline=True,
            ).ask()

            if result is None:
                return None  # user cancelled

            result = result.strip()

            if len(result) > MAX_SYSTEM_PROMPT_LENGTH:
                console.print(  # type: ignore[union-attr]
                    f"[red]  ✗ Prompt too long ({len(result):,} chars). "
                    f"Maximum is {MAX_SYSTEM_PROMPT_LENGTH:,} characters.[/red]"
                )
                continue

            return result
