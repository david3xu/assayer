"""The synthetic dataset's own file format.

Deliberately plain: a UTF-8 CSV with four named columns, written with `\\n`
endings and a fixed column order, so the same plan and seed produce the same
bytes on any machine (rule 2). A null is the empty field, which is how a
no-answer survives a round trip.

The format is the profile's published shape, not the source's. It is read back
by the same code the checks use, so a check measures the bytes that were written
and hashed rather than an in-memory object that happens to be lying around.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Sequence

from .reader import Sample

COLUMNS = ("depth_m", "valid", "swir_group", "tir_group")


def to_csv_bytes(samples: Sequence[Sample], depth_places: int = 5) -> bytes:
    """Encode rows. Depth is fixed-precision so the bytes are not float-formatting
    dependent, and `valid` is written as the words true/false rather than 1/0 so
    the column cannot be mistaken for the source's Final Mask."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for s in samples:
        writer.writerow([
            f"{s.depth_m:.{depth_places}f}",
            "true" if s.valid else "false",
            "" if s.swir_group is None else s.swir_group,
            "" if s.tir_group is None else s.tir_group,
        ])
    return buf.getvalue().encode("utf-8")


def write(samples: Sequence[Sample], path: str | Path, depth_places: int = 5) -> bytes:
    data = to_csv_bytes(samples, depth_places)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def read(path: str | Path) -> list[Sample]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh)
        if tuple(rows.fieldnames or ()) != COLUMNS:
            raise ValueError(f"expected columns {COLUMNS}, found {rows.fieldnames}")
        out = []
        for row in rows:
            out.append(Sample(
                depth_m=float(row["depth_m"]),
                valid=row["valid"] == "true",
                swir_group=row["swir_group"] or None,
                tir_group=row["tir_group"] or None,
            ))
    return out
