"""Sphinx configuration for rustmatrix documentation.

The site uses MyST-MD source (`.md` and `.ipynb`) rendered through Sphinx
with the PyData theme — the standard scientific-Python documentation
stack (NumPy / SciPy / xarray). Notebook tutorials in `tutorials/`
execute on build with caching via myst-nb.
"""

from __future__ import annotations

from datetime import datetime
from importlib import metadata as _metadata

# rustmatrix is expected to be installed into the build environment
# (RTD's `pip install ".[docs]"` runs maturin, which compiles `_core`).
# Never insert the `python/` source tree onto sys.path — that shadows
# the installed package and hides the compiled extension.

# -- Project information -----------------------------------------------------

project = "rustmatrix"
author = "Stephen Nesbitt"
copyright = f"{datetime.now().year}, {author}"

try:
    release = _metadata.version("rustmatrix")
except _metadata.PackageNotFoundError:  # not installed in this env
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

# myst_nb auto-registers both `.md` and `.ipynb`; no explicit
# source_suffix needed.

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

templates_path = ["_templates"]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Notebook execution: cache so unchanged notebooks aren't re-run.
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_excludepatterns: list[str] = []
nb_merge_streams = True

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Don't fail the build if a tutorial imports an optional dep that isn't
# present in the autodoc-only environment.
autodoc_mock_imports: list[str] = []

# Re-exports from `rustmatrix/__init__.py` create legitimate duplicate
# Python cross-reference targets (e.g. `rustmatrix.Scatterer` and
# `rustmatrix.scatterer.Scatterer`). Silence that warning class; the
# rendered pages still link to the correct module page.
suppress_warnings = [
    "ref.python",          # re-exports (e.g. rustmatrix.Scatterer)
    "docutils",            # `|` in docstrings reads as a substitution ref
    "autosectionlabel.*",
]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- BibTeX ------------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/icon-128.png"
html_title = f"{project} {release}"

html_theme_options = {
    "github_url": "https://github.com/swnesbitt/rustmatrix",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/rustmatrix/",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "crates.io",
            "url": "https://crates.io/crates/rustmatrix",
            "icon": "fa-brands fa-rust",
        },
    ],
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "swnesbitt",
    "github_repo": "rustmatrix",
    "github_version": "main",
    "doc_path": "docs",
}
