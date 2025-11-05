"""Data catalog loader for CareerLens datasets.

This module ingests curated local datasets, performs light normalization, and
exposes them as a mapping of dataset name to pandas DataFrame. Each dataset is
persisted to a Parquet snapshot under ``ml/artifacts/raw`` for downstream use.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "artifacts" / "raw"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_DELIMITERS = [",", ";", "\t", "|", ":"]
FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "latin1"]


def _slugify_name(path: Path) -> str:
    stem = path.stem.lower().replace(" ", "_")
    return stem.replace("-", "_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        col.strip()
        .replace("%", " percent ")
        .replace("/", " ")
        .replace("-", " ")
        .replace(".", " ")
        for col in normalized.columns
    ]
    normalized.columns = [
        "_".join(filter(None, "".join(ch if ch.isalnum() or ch == " " else " " for ch in col).split())).lower()
        for col in normalized.columns
    ]
    return normalized


def _detect_encoding(path: Path, encodings: Iterable[str] = FALLBACK_ENCODINGS) -> str:
    sample_size = 4096
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as handle:
                handle.read(sample_size)
            return enc
        except UnicodeDecodeError:
            continue
    LOGGER.warning("Falling back to latin1 for %s", path)
    return "latin1"


def _detect_delimiter(path: Path, encoding: str) -> str:
    sample_size = 4096
    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            sample = handle.read(sample_size)
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CSV_DELIMITERS))
        return dialect.delimiter
    except (csv.Error, UnicodeDecodeError):
        return ","


def _read_csv(path: Path) -> pd.DataFrame:
    encoding = _detect_encoding(path)
    delimiter = _detect_delimiter(path, encoding)
    try:
        df = pd.read_csv(path, sep=delimiter, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed default CSV read for %s (%s), retrying with python engine", path, exc)
        df = pd.read_csv(path, sep=delimiter, encoding=encoding, engine="python")
    return df


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def _persist_parquet(name: str, df: pd.DataFrame) -> None:
    output_path = ARTIFACT_ROOT / f"{name}.parquet"
    try:
        df.to_parquet(output_path, index=False)
        LOGGER.info("Wrote parquet snapshot to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to write parquet for %s: %s", name, exc)


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all configured datasets and return as a dictionary."""
    datasets: dict[str, pd.DataFrame] = {}

    file_specs: list[tuple[str, Path]] = [
        ("age_interest_career_dataset", DATASET_ROOT / "age_interest_career_dataset.csv"),
        (
            "ai_based_career_recommendation_system",
            DATASET_ROOT / "AI-based Career Recommendation System.csv",
        ),
        ("education_career_success", DATASET_ROOT / "education_career_success.csv"),
        ("dataset9000", DATASET_ROOT / "dataset9000.csv"),
        ("dataset9000_data", DATASET_ROOT / "dataset9000.data"),
        ("dataset_project_404", DATASET_ROOT / "Dataset Project 404.xlsx"),
    ]

    karrierewege_dir = DATASET_ROOT / "karrierewege"
    if karrierewege_dir.exists():
        for csv_path in sorted(karrierewege_dir.glob("*.csv")):
            file_specs.append((f"karrierewege_{_slugify_name(csv_path)}", csv_path))

    for name, path in file_specs:
        if not path.exists():
            LOGGER.info("Skipping missing dataset: %s", path)
            continue

        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = _read_excel(path)
        else:
            df = _read_csv(path)

        df = _normalize_columns(df)
        datasets[name] = df

        na_counts = df.isna().sum().sum()
        LOGGER.info("Loaded %s shape=%s missing_values=%s", name, df.shape, na_counts)
        _persist_parquet(name, df)

    return datasets


def _build_summary(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in datasets.items():
        total_cells = df.shape[0] * df.shape[1]
        missing = int(df.isna().sum().sum())
        rows.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing": missing,
                "missing_pct": round((missing / total_cells * 100) if total_cells else 0, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset") if rows else pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CareerLens data catalog utilities")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table of datasets loaded",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    datasets = load_all_datasets()

    if args.summary:
        summary = _build_summary(datasets)
        if summary.empty:
            print("No datasets loaded. Ensure files are present under the dataset/ directory.")
        else:
            with pd.option_context("display.max_colwidth", None):
                print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
