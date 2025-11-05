"""Compatibility shim so `import app.*` resolves to `backend/app` during local
development and CI runs.

This project uses both `app.*` and `backend.app.*` import styles in tests and
implementation. Adding this tiny package makes `app` a package whose path
includes the `backend/app` directory so absolute imports like
`app.models.profile` continue to work.
"""
import os
from pathlib import Path

# Calculate the repo root and the `backend/app` path relative to this file.
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
_backend_app = _repo_root / "backend" / "app"
if _backend_app.is_dir():
    # Prepend backend/app to the package search path for this package.
    __path__.insert(0, str(_backend_app))
