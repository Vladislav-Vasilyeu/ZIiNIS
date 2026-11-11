"""Helpers for writing experiment results in spreadsheet-friendly CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def write_experiment_csv(path: str | Path, rows: Iterable[Sequence[object]]) -> None:
    """Write semicolon-separated UTF-8 CSV with a BOM for Windows Excel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";", lineterminator="\n")
        writer.writerows(rows)
