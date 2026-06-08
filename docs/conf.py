# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import strong_sphinx_theme

sys.path.insert(0, os.path.abspath('../torchcast'))

# -- Project information -----------------------------------------------------

project = 'torchcast'
copyright = '2025, Strong Analytics'
author = 'Jacob Dink'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', '_html', 'conf.py']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'strong_sphinx_theme'
html_theme_path = strong_sphinx_theme.html_theme_path()

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "project_url": "https://strong.io",
    "globaltoc_depth": 4
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

#
autodoc_inherit_docstrings = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
}

nbsphinx_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}
