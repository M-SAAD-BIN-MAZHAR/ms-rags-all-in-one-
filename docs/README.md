# MS-RAGS(ALL-IN-ONE) Docs Site

This folder contains the deployable multi-page documentation site for MS-RAGS(ALL-IN-ONE).

It is intentionally static: Vercel can deploy it without installing a JavaScript
framework or build tool.

## Deploy On Vercel

1. Push this repository to GitHub.
2. Import the repository in Vercel.
3. Keep the framework preset as `Other`.
4. Leave the build command empty.
5. Leave the output directory empty.
6. Deploy.

The root `vercel.json` rewrites clean routes such as `/rag-types`, `/pipeline`,
and `/production` to the matching static HTML files in this folder.

## Local Preview

Open `docs/index.html` in a browser, or serve the repo root with any static file
server.

```bash
python -m http.server 3000
```

Then visit `http://localhost:3000`.
