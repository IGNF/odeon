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
from os.path import abspath, dirname

sys.path.insert(0, abspath(dirname(dirname(__file__))))
print(sys.path)

from sphinx_markdown_parser.parser import MarkdownParser
from sphinx_markdown_parser.transform import AutoStructify


# -- Project information -----------------------------------------------------

project = 'odeon-landcover'
copyright = '2020, IGN'
author = 'IGN'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
   'sphinxcontrib.details.directive',
   'sphinx_tabs.tabs',
   'sphinxcontrib.mermaid',
   'sphinx.ext.napoleon',
   'sphinx.ext.autosummary',
   'sphinx.ext.autodoc'
]

autosummary_generate = True
autodoc_mock_imports = [
    "sklearn", "skimage", "torch", "fiona", "shapely", "tqdm", "rasterio", "pandas",
    "geopandas", "torchvision", "gdal", "jsonschema", "matplotlib"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', 'assets']
html_sidebars = {
    "**": ["globaltoc.html", "localtoc.html", "searchbox.html"]
}
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'Odeon Lancover',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://gitlab.com/dai-projets/odeon-landcover',

    # Set the color and the accent color
    'color_primary': 'blue',
    'color_accent': 'light-blue',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://gitlab.com/dai-projets/odeon-landcover',
    'repo_name': 'odeon-landcover',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 1,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}

# for MarkdownParser
import pymdownx.superfences
def setup(app):
    app.add_source_parser(MarkdownParser)
    app.add_config_value('markdown_parser_config', {
        'auto_toc_tree_section': 'Content',
        'enable_auto_toc_tree': True,
        'enable_eval_rst': True,
        'extensions': [
            'extra',
            'nl2br',
            'sane_lists',
            'smarty',
            'pymdownx.highlight',
            'pymdownx.inlinehilite',
            'pymdownx.superfences',
            'pymdownx.details'
        ],
        "extension_configs": { "pymdownx.superfences": {
                "custom_fences": [
                    {
                        'name': 'mermaid',
                        'class': 'mermaid',
                        'format': pymdownx.superfences.fence_div_format
                    }
                ]
            }
        }
    }, True)

