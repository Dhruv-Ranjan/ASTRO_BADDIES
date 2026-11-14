# ASTRO BADDIES static site

This folder is the deployable Netlify site.

## Netlify

Upload this folder as the site directory, or connect the repository and set:

```text
Base directory: netlify
Build command: leave blank
Publish directory: .
```

The site includes the bundled drought snapshot and a verified precomputed
spectral reconstruction demonstration. The private PyTorch checkpoint is not
included, so arbitrary image upload inference remains a local-only feature.

## Local preview

```powershell
python -m http.server 8080
```

Open `http://127.0.0.1:8080/mission-control.html`.
