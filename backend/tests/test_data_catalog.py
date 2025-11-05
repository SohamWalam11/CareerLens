import pytest
from backend.app.services.data_catalog import load_catalog


def test_catalog_loads_and_has_required_fields():
    """Ensure data catalog loads and each entry contains required keys and types."""
    catalog = load_catalog()
    assert isinstance(catalog, list), "Catalog should be a list"
    assert len(catalog) > 0, "Catalog should contain entries"

    required_keys = {"id", "title", "source", "schema"}
    for entry in catalog:
        assert required_keys.issubset(set(entry.keys())), f"Entry missing keys: {entry}"
        assert isinstance(entry["id"], str)
        assert isinstance(entry["title"], str)
        assert isinstance(entry["source"], str)
        assert isinstance(entry["schema"], dict)
