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
#sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../src/algorithms/"))
sys.path.append(os.path.abspath("../../src/algorithms/dp/"))
sys.path.append(os.path.abspath("../../src/algorithms/td/"))
sys.path.append(os.path.abspath("../../src/algorithms/dummy/"))
sys.path.append(os.path.abspath("../../src/algorithms/pg/"))
sys.path.append(os.path.abspath("../../src/algorithms/dqn/"))
sys.path.append(os.path.abspath("../../src/algorithms/mc/"))
sys.path.append(os.path.abspath("../../src/algorithms/planning/"))
sys.path.append(os.path.abspath("../../src/optimization/"))
sys.path.append(os.path.abspath("../../src/parallel_utils/"))
sys.path.append(os.path.abspath("../../src/policies/"))
sys.path.append(os.path.abspath("../../src/utils/"))
sys.path.append(os.path.abspath("../../src/worlds/"))
sys.path.append(os.path.abspath("../../src/examples/dp/"))
sys.path.append(os.path.abspath("../../src/agents/"))
sys.path.append(os.path.abspath("../../src/agents/torch_agents/"))
sys.path.append(os.path.abspath("../../src/trainers/"))
sys.path.append(os.path.abspath("../../src/filtering/"))
print(sys.path)


# -- Project information -----------------------------------------------------

project = 'PyCubeAI'
copyright = '2022, Alexandros Giavaras'
author = 'Alexandros Giavaras'

# The full version, including alpha/beta/rc tags
release = 'v0.0.8-alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   #"numpydoc",
   'sphinx.ext.napoleon',
   #"breathe", 
   #"m2r2"
]

#extensions = ['sphinx.ext.napoleon']

#numpydoc_show_class_members = True

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' #'default' #'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
