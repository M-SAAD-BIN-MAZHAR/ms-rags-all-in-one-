"""Entry point for `python -m ms_rag`, `ms-rags`, and `ms-rag` CLI commands."""

import click


def main() -> None:
    """MS-RAGS(ALL-IN-ONE) CLI entry point — implemented in ms_rag.cli.main."""
    # Deferred import to avoid circular imports at package load time
    from ms_rag.cli.main import run  # noqa: PLC0415
    run()


if __name__ == "__main__":
    main()
