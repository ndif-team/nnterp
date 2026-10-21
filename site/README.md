# nnterp site

The documentation site for nnterp, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
`mkdocs.yml` is the config, `docs/` the pages, `build/` the output (not committed).

## Setup

From the repository root:

```bash
pip install -e ".[docs]"
```

## Development

```bash
mkdocs serve -f site/mkdocs.yml            # http://127.0.0.1:8000, rebuilds on every change
mkdocs serve -f site/mkdocs.yml --dirtyreload   # faster; rebuilds only the changed page
```

## Build

```bash
mkdocs build -f site/mkdocs.yml            # writes site/build/
```

The GitHub Actions workflow `.github/workflows/build-docs.yml` runs the same build and deploys
`site/build/` to GitHub Pages.

## The Documentation tab

`docs/documentation/` holds one page per module of the `nnterp` package, each a
[mkdocstrings](https://mkdocstrings.github.io/) directive, and its section of the nav in
`mkdocs.yml`. Both are written by `gen_reference.py`; run it after adding, moving or removing a
module, and commit the result:

```bash
python site/gen_reference.py
```

## The encyclopedia

The encyclopedia (`encyclopedia/build.py`) is built locally, since it reads model configs from the
Hub cache, and its output is published on the orphan branch `encyclopedia-site`, which the deploy
workflow copies to `/encyclopedia/`. To update it:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py
git worktree add --detach /tmp/enc HEAD && cd /tmp/enc
git checkout --orphan encyclopedia-site && git rm -rfq . && cp -r ../path/to/encyclopedia/site/. .
git add -A && git commit -m "encyclopedia: built site" && git push -f origin encyclopedia-site
```

For the dev server, copy `encyclopedia/site/` to `site/docs/encyclopedia/` (gitignored).
