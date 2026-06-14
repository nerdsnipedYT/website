# AGENTS.md — Nerdsniped website

Instructions for any agent working on this repository.

## Anonymity — hard rule

**Never write the author's real name (or any personal name) anywhere on the
website.** The author publishes anonymously under the **Nerdsniped** brand only.

- Bylines, sign-offs, "about" text, captions, alt text, commit-authored content,
  metadata — all must use **"Nerdsniped"**, never a personal name.
- Do **not** link to anything that exposes identity (e.g. personal GitHub
  profiles or repos under a personal username). Use the anonymous handle
  **`github.com/nerdsnipedYT`** and the channel **`youtube.com/@nerdsnipedYT`**.
- When adapting source material (Google Docs, video scripts, etc.) that contains
  real names or affiliations (schools, programmes, etc.), strip them and write in
  the neutral "Nerdsniped"/"we"/"I" voice.
- If a task would require publishing a name to satisfy it, don't — keep it
  anonymous and flag it.

## Project notes

- Static site (HTML + CSS + a little vanilla JS), served as-is via GitHub Pages
  (`.nojekyll`). No build step is required to serve.
- Posts live in `posts/`; the listing/cards are in `index.html`.
- `_gen/` holds Python scripts that regenerate figures and the search index
  (`build_search_index.py` -> `search-index.json`). Re-run after adding/editing a
  post so search and tags stay correct.
- After CSS/JS changes, bump the `?v=N` cache-busting query on the
  `style.css` / `search.js` links so browsers (and phones) pick up the update.
