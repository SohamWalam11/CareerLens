"""Skeleton preprocessing pipeline for CareerLens ML workflows."""

from pathlib import Path

import pandas as pd

DATA_ROOT = Path("dataset")


def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset by filename from the shared dataset directory."""
    path = DATA_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset {name} not found at {path}")
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform minimal cleaning to drop duplicates and normalize column names."""
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    return cleaned
