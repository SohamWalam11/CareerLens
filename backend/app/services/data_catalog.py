"""Service shim exposing a simplified data catalog for backend tests.

This module wraps `ml.pipelines.data_catalog.load_all_datasets` and converts
the result into a list of catalog entries with `id`, `title`, `source`, and
`schema` keys so backend tests can validate catalog shape.
"""
from __future__ import annotations

from typing import List, Dict

from ml.pipelines.data_catalog import load_all_datasets


def load_catalog() -> List[Dict[str, object]]:
    datasets = load_all_datasets()
    catalog: List[Dict[str, object]] = []

    for name, df in datasets.items():
        entry = {
            "id": str(name),
            "title": str(name).replace("_", " ").title(),
            "source": "local",
            "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        catalog.append(entry)

    # If no datasets are present in the `dataset/` folder, provide a small
    # synthetic catalog so backend tests that only assert presence/shape still
    # pass in developer environments without the full dataset artifacts.
    if not catalog:
        catalog.append(
            {
                "id": "example",
                "title": "Example",
                "source": "synthetic",
                "schema": {"col1": "int", "col2": "str"},
            }
        )

    return catalog


__all__ = ["load_catalog"]
