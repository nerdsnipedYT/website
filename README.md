# Nerdsniped

A small static research blog (HTML + CSS, interactive Plotly figures).

## Deploy to GitHub Pages
1. Push this directory to a GitHub repo.
2. Settings → Pages → Build from branch → `main` / root.
3. Visit `https://<user>.github.io/<repo>/`.

`.nojekyll` is included so all files (incl. `assets/`) are served as-is.
Plotly/KaTeX load via CDN; no build step required.

## Structure
- `index.html` — landing / post list
- `posts/dynamic-reweighting.html` — the first post
- `css/style.css`
- `assets/` — interactive figures (`*.html`), UMAP explorers, diagram
- `_gen/` — Python scripts that regenerate the figures (not needed to serve)
# website
