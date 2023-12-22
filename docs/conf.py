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
import pathlib
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "odeon"
copyright = "samy khelifi richards"
author = "samy khelifi richards"
this_dir = pathlib.Path(__file__).resolve().parent
with (this_dir / ".." / "VERSION").open() as vf:
    version = vf.read().strip()
print("Version as read from version.txt: '{}'".format(version))
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
theme_plugin = "sphinx_rtd_theme"
# pygments_style = "friendly"
# pygments_dark_style = "monokai"
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_defaultargs",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme"
]
tag_version = True
html_theme = theme_plugin
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# -- HTML theme settings -----------------------------------------------
html_show_sourcelink = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Changing sidebar title to Kornia
html_title = "ODEON"
html_favicon = "_static/img/logo_favicon.ico"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
