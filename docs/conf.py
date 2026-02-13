import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'PyCorrAna'
copyright = '2025, Sidney Zhang'
author = 'Sidney Zhang'

release = '0.1.0'
version = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'

source_suffix = '.rst'
master_doc = 'index'

pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_logo = 'logo.png'
html_favicon = None

html_static_path = ['_static']

htmlhelp_basename = 'PyCorrAnadoc'

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'PyCorrAna.tex', 'PyCorrAna Documentation',
     'Sidney Zhang', 'manual'),
]

man_pages = [
    (master_doc, 'pycorrana', 'PyCorrAna Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'PyCorrAna', 'PyCorrAna Documentation',
     author, 'PyCorrAna', 'Python Correlation Analysis Toolkit.',
     'Miscellaneous'),
]

epub_title = project
epub_exclude_files = ['search.html']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
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

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autosummary_generate = True
autosummary_imported_members = True

todo_include_todos = True
