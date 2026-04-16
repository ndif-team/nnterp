import os
import sys
import time
import re
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "nnterp"
copyright = "2025, Clément Dumas"
author = "Clément Dumas"

extensions = [
    "sphinx.ext.autodoc",  # Auto documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Boostrap design components
    "nbsphinx",  # Jupyter notebook support
    "sphinx.ext.viewcode",  # Add source links to the generated HTML files
    "sphinx.ext.extlinks",  # Add external links
]

templates_path = ["_templates"]
fixed_sidebar = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "nnterp"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["source-link-navbar.js"]
html_show_sphinx = False
html_show_sourcelink = True
html_copy_source = True
html_sourcelink_suffix = ".txt"

# Important for GitHub Pages
# Allow override via environment variable for dev/staging deployments
html_baseurl = os.environ.get(
    "SPHINX_HTML_BASEURL", "https://ndif-team.github.io/nnterp/"
)
html_extra_path = [".nojekyll", "llms.txt"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
add_module_names = False
html_context = {
    "default_mode": "dark",
    "version_identifier": str(int(time.time())),
}


html_theme_options = {
    "show_nav_level": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ndif-team/nnterp",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_prev_next": False,
    "pygments_dark_style": "monokai",
}

# Hide empty left sidebar on all pages
html_sidebars = {"**": []}


def parse_toctree(index_rst_path):
    """Extract toctree structure from index.rst"""
    with open(index_rst_path) as f:
        content = f.read()

    toctree_pattern = r"\.\. toctree::\s*\n((?:   .*\n|\s*\n)*?)(?=\n\S|\Z)"

    sections = []
    for match in re.finditer(toctree_pattern, content, re.MULTILINE):
        block = match.group(0)

        caption_match = re.search(r":caption:\s*(.+?):", block)
        caption = caption_match.group(1).strip() if caption_match else None

        paths = []
        for line in block.split("\n"):
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith(":")
                and not stripped.startswith("..")
                and line.startswith("   ")
            ):
                paths.append(stripped)

        if paths:
            sections.append({"caption": caption, "files": paths})

    return sections


def parse_rst_metadata(rst_file):
    """Extract title and llm-description from RST file"""
    if not rst_file.exists():
        return None, None

    with open(rst_file) as f:
        content = f.read()

    title_match = re.search(r"^(.+)\n[=\-~]+", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    desc_match = re.search(
        r":llm-description:\s+(.+?)(?:\n\n|\n\.\.|\Z)", content, re.DOTALL
    )
    description = desc_match.group(1).strip() if desc_match else None

    return title, description


def generate_llms_txt(source_dir, output_file, base_url=None):
    """Generate llms.txt from RST metadata"""
    source_path = Path(source_dir)
    index_rst = source_path / "index.rst"
    header_file = source_path / "llms_header.txt"

    with open(header_file) as f:
        lines = [f.read().rstrip(), ""]

    sections = parse_toctree(index_rst)

    for section in sections:
        if section["caption"]:
            lines.append(f"## {section['caption']}")
            lines.append("")

        for rst_path in section["files"]:
            full_path = (
                source_path / f"{rst_path}.rst"
                if not rst_path.endswith(".rst")
                else source_path / rst_path
            )

            title, description = parse_rst_metadata(full_path)
            if title and description:
                normalized_path = (
                    rst_path if rst_path.endswith(".rst") else f"{rst_path}.rst"
                )
                # Use base_url if provided, otherwise use relative path
                if base_url:
                    source_link = (
                        f"{base_url.rstrip('/')}/_sources/{normalized_path}.txt"
                    )
                else:
                    source_link = f"/_sources/{normalized_path}.txt"
                lines.append(f"- [{title}]({source_link}): {description}")

        lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def setup(app):
    """Sphinx setup hook"""
    source_dir = app.srcdir
    output_file = Path(source_dir) / "llms.txt"

    # Pass the html_baseurl to generate_llms_txt
    generate_llms_txt(source_dir, output_file, html_baseurl)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
