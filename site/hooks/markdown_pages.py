"""Serve every hand-written page as Markdown beside its HTML, for agents and their tools.

A page at ``guide/loading/`` is also served at ``guide/loading.md``; a section index at
``guide/`` at ``guide/index.md``. The home page and the generated reference under
``documentation/`` are not served this way. ``llms-full.txt`` holds every served page in nav
order, and each served page carries ``page.meta.markdown_url`` for the template's
"View as Markdown" control.
"""

from __future__ import annotations

import os

_PAGES: dict[str, tuple[object, str]] = {}


def _markdown_path(page) -> str | None:
    src = page.file.src_uri
    if src == "index.md" or src.startswith("documentation/"):
        return None
    return src


def on_page_markdown(markdown, page, config, files):
    path = _markdown_path(page)
    if path is None:
        return markdown
    page.meta["markdown_url"] = path
    _PAGES[path] = (page, markdown)
    return markdown


def _nav_order(nav, out):
    for item in nav:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                _nav_order(value if isinstance(value, list) else [value], out)


def on_post_build(config):
    site = config["site_dir"]
    for path, (page, markdown) in _PAGES.items():
        out = os.path.join(site, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(markdown)
    order: list[str] = []
    _nav_order(config["nav"], order)
    parts = []
    for path in order:
        if path in _PAGES:
            page, markdown = _PAGES[path]
            parts.append(f"<!-- {config['site_url']}{path} -->\n{markdown.strip()}\n")
    with open(os.path.join(site, "llms-full.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    _PAGES.clear()
