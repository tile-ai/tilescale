# General information about the project.
project = "TileScale"
author = "TileScale and TileLang Contributors"
copyright = f"2025-2026, {author}"

# Version information.
with open("../VERSION") as f:
    version = f.read().strip()
release = version

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../tilelang"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_keep_files = False  # Useful for debugging the generated rst files

autoapi_generate_api_docs = True

autodoc_typehints = "description"

autoapi_ignore = ["*language/ast*", "*version*", "*libinfo*", "*parser*"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["colon_fence", "deflist"]

redirects = {"get_started/try_out": "../index.html#getting-started"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**/*libinfo*", "**/*version*"]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
templates_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]
footer_copyright = "Copyright 2025-2026 TileScale and TileLang Contributors"
footer_note = " "

html_theme_options = {"light_logo": "img/logo-v2.png", "dark_logo": "img/logo-v2.png"}

header_links = [
    ("Home", "https://github.com/tile-ai/tilescale"),
    ("GitHub", "https://github.com/tile-ai/tilescale"),
    ("Upstream TileLang", "https://github.com/tile-ai/tilelang"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "tile-ai",
    "github_repo": "tilescale",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
}
