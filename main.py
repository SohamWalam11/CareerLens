"""Top-level entrypoint shim to expose `app` when tests import `main`.

Some tests import `from main import app`. The actual FastAPI app lives in
`backend/main.py`. Re-export it here so both import styles work.
"""
from backend.main import app  # type: ignore
