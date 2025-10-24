# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent)]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MetaChat'
copyright = '2025, Songhao Luo'
author = 'Songhao Luo'
release = '0.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_mdinclude',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_size',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'myst_nb'
]
myst_enable_extensions = [
    'html_admonition',
    'colon_fence',
    
]
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
nbsphinx_allow_errors = True
nbsphinx_execute = 'never' # auto before
nb_execution_mode = "off"
sphinx_rtd_size_width = "75%"
jupyter_execute_notebooks = 'off' # new
sphinx_gallery_conf = {
    'run_stale_examples': False,
    'execute_scripts': False,
} # new

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
    "css/override.css",
]

html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#357473",
        "color-brand-content": "#357473",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/SonghaoLuo/metachat",
            "html": "",
            "class": "fab fa-github",
        },
    ],
}
html_logo = "_static/image/logo.png"