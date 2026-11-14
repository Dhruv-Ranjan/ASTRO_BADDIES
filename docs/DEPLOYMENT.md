# Deployment

## Netlify frontend

`frontend/netlify.toml` configures a dependency-free static deployment with
the publish directory set to `frontend/`. In Netlify, set the repository base
directory to `frontend` (or move the contents to the repository root) and use
an empty build command. The live reference is
https://aurora-sr.netlify.app/.

## Local backend

The Python API is intentionally separate from Netlify:

```powershell
.\.venv\Scripts\python.exe .\backend\mission_api.py
```

It loads the selected checkpoint from
`FINAL_PROJECT_NASA/artifacts/spectral_realworld_production.pt`. Do not expose
this development server publicly without production authentication, request
limits, and a managed deployment.

