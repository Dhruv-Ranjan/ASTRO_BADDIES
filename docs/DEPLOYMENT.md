# Deployment

## Netlify static site

Deploy the `netlify/` folder. For a repository-connected site, configure:

```text
Base directory: ASTRO_BADDIES/netlify
Build command: leave blank
Publish directory: .
```

The static site serves the Spectral Lab, regional mission dashboard, bundled
sample, and research figures. It does not host the Python inference API or
private model checkpoint.

It also does not fetch climate or imagery datasets at runtime. NASA POWER is
consumed by the separate AURORA research pipeline documented in
[`DATA_SOURCES.md`](DATA_SOURCES.md); it supplies daily precipitation and
temperature features there, with Sentinel-2 scene statistics as complementary
surface inputs.

## Local inference API

The API runs separately for development:

```powershell
python .\backend\mission_api.py
```

It loads the selected checkpoint from
`FINAL_PROJECT_NASA/artifacts/spectral_realworld_production.pt`. Do not expose
this development server publicly without production authentication, request
limits, and a managed deployment.
