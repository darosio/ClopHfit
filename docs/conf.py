"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../..").resolve()))


# -- Project information -----------------------------------------------------

project = "ClopHfit"
copyright = "2023, Daniele Arosio"  # noqa: A001
author = "Daniele Arosio"
release = "0.10.8"
html_title = "ClopHfit"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.plantuml",
    "nbsphinx",
    "sphinx_click",
    # "myst_parser",
    # 'IPython.sphinxext.ipython_console_highlighting',
]

# Napoleon settings to Default
napoleon_use_ivar = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "autosummary": True,
}

autodoc_typehints = "description"

# The suffix of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    # ".md": "markdown",
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    # Additional preamble content
    "preamble": r"""
\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
\newunicodechar{█}{\rule{1ex}{1ex}}
""",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
