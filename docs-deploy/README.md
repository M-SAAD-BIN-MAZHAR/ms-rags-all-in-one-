# MS-RAGS(ALL-IN-ONE) Docs

This folder is the standalone static documentation bundle for MS-RAGS(ALL-IN-ONE).
It can be deployed independently from the Python framework code.

## Deploy On Vercel

1. Push this `docs-deploy/` folder to the repository or use it as the Vercel project root.
2. In Vercel, set the framework preset to `Other`.
3. Leave build command empty.
4. Set output directory to `.`.

The production docs URL is:

https://ms-rags-all-in-one.vercel.app/

## Local Preview

```bash
python -m http.server 4174 --directory docs-deploy
```

Then open:

```text
http://localhost:4174/
```
