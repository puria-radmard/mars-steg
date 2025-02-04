# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'MARS Steg'
copyright = '2025, J.S'
author = 'J.S'

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']

extensions = [
    'sphinx.ext.napoleon',
    'myst_parser',

]
 
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

	
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
