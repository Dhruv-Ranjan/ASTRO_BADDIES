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

## Included figures

The upload also includes real project outputs under `graphs/` and
`screenshots/`:

- nominal, battery-stress, cloud-stress, and storage-stress RMSE plots;
- mission observation and information-gain plots;
- drought held-out forecast and observed-versus-predicted figures;
- a research-results dashboard capture.

These are recorded experiment outputs, not live spacecraft telemetry.

After deployment, open `graphs.html` for the browsable results gallery.

## Local preview

```powershell
python -m http.server 8080
```

Open `http://127.0.0.1:8080/mission-control.html`.
