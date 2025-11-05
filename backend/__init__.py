"""Top-level package marker for the backend folder.

Creating this file makes `backend` a regular Python package so imports
such as `backend.app.services.*` resolve reliably in local development
and CI where pytest is run from the repository root.
"""

# Intentionally minimal — this file only marks the directory as a package.
