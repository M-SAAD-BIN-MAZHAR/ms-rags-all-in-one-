# Security Guide for MS-RAGS(ALL-IN-ONE)

This project is permission-first and secret-aware. Before public release, keep the
following controls in place.

## Secret Handling

- Never commit real API keys, tokens, passwords, or cloud connection strings.
- Store credentials in `CredentialStore` at runtime and serialize only safe markers.
- Use environment variables or a secret manager for generated standalone apps.
- Rotate any secret that has ever been exposed in a commit, log, screenshot, or pasted text.

## User Permission

- Ask before ingestion, external web fetches, file reads, API calls, and cloud service connections.
- Show the exact provider, backend, path, domain, or URL before the user approves it.
- Do not silently switch models, databases, or tools when a configured choice fails.

## Input Safety

- Validate file paths, URLs, collection names, and numeric limits.
- Keep file and URL allowlists narrow for agentic tools.
- Reject path traversal and unsafe API methods.

## Logging

- Keep logs structured.
- Redact secret-like values.
- Avoid logging raw credentials, full tokens, or sensitive document contents unless explicitly needed.

## Deployment

- Use pinned dependencies for production builds.
- Prefer isolated runtime environments for generated apps.
- Keep `docs-deploy/` and other generated site output out of version control when not needed.

## Review Checklist

- `README.md` has no secrets.
- `AGENTS.md` has no private notes.
- Generated `.env` files contain placeholders only.
- Security-sensitive warnings are visible in the terminal.
- Agent tools remain deny-by-default with explicit allowlists.
