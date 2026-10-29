"""Stage 2 — source intake.

Nothing downstream is allowed to touch a live service. Intake pulls the source
once, writes it to disk, and records a manifest: the SHA-256 of the exact bytes,
where they came from, when, under what licence, and who must be credited. Every
later stage reads the frozen file and quotes the hash, so a run can be replayed
against the same bytes years later and a changed source shows up as a changed
hash rather than as a quietly different result.

The catalogue is served over plain HTTP, so the hash is taken at download time:
it is a record of what this run actually read, not a trusted checksum from the
publisher.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from engine.hashing import sha256_file

DATA_SERVICES = "http://geossdi.dmp.wa.gov.au/NVCLDataServices"

# The four logs the profile reads. The service names them with spaces; it
# returns them as CSV headers with the spaces replaced by underscores.
WANTED_LOGS = ("Final Mask", "Grp1 sTSAS", "Grp1 sjCLST")

# Recorded in provenance on every release. The source is credited whether or not
# the synthetic output is dedicated CC0 (07 §14 keeps that question open).
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "Geological Survey of Western Australia, National Virtual Core Library "
    "(NVCL) WA node, via AuScope. Licensed CC BY 4.0."
)


@dataclass(frozen=True)
class SourceManifest:
    """What a later reader needs to trust the file it is about to open."""

    dataset_id: str
    dataset_name: str
    borehole_uri: str
    retrieved_from: str
    retrieved_at: str
    path: str
    sha256: str
    size_bytes: int
    licence: str
    attribution: str
    rows: int | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "SourceManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def describe_dataset(dataset_id: str) -> dict:
    """The catalogue record for one dataset: depth range, logs, borehole."""
    url = (f"{DATA_SERVICES}/getDatasetCollection.html"
           f"?datasetid={urllib.parse.quote(dataset_id)}&outputformat=json")
    return _get_json(url)["datasetCollection"][0]


def log_ids(record: dict, wanted: tuple[str, ...] = WANTED_LOGS) -> dict[str, str]:
    """Resolve log names to ids.

    A dataset lists the same log name more than once (one entry per domain), and
    the ids differ. The first occurrence is the one the scalar service returns
    for the whole hole, so take it and ignore the rest rather than picking an
    arbitrary duplicate.
    """
    logs = (record.get("logCollection") or {}).get("logCollection") or []
    found: dict[str, str] = {}
    for entry in logs:
        name = entry.get("logName")
        if name in wanted and name not in found:
            found[name] = entry["logID"]
    missing = [name for name in wanted if name not in found]
    if missing:
        raise LookupError(
            f"dataset {record.get('datasetName')} has no log named "
            + ", ".join(missing)
        )
    return found


def fetch_scalars(dataset_id: str, dest_dir: str | Path,
                  timeout: int = 300) -> SourceManifest:
    """Freeze one hole's scalar logs to `dest_dir` and return its manifest.

    This is the light intake path. The full TSG package for a hole is ~1.5 GB,
    almost all of it spectral cube and imagery that this profile never reads;
    the scalar service returns the same four logs for the same hole in about a
    megabyte, which is what makes held-out holes affordable.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    record = describe_dataset(dataset_id)
    ids = log_ids(record)
    query = "&".join(f"logid={urllib.parse.quote(v)}" for v in ids.values())
    url = (f"{DATA_SERVICES}/downloadscalars.html"
           f"?datasetid={urllib.parse.quote(dataset_id)}&{query}")

    name = record.get("datasetName") or dataset_id
    path = dest_dir / f"{name}-scalars.csv"
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with urllib.request.urlopen(url, timeout=timeout) as fh, \
            open(path, "wb") as out:
        while chunk := fh.read(1024 * 1024):
            out.write(chunk)

    with open(path, "rb") as fh:
        rows = max(0, sum(1 for _ in fh) - 1)  # minus the header

    manifest = SourceManifest(
        dataset_id=dataset_id,
        dataset_name=name,
        borehole_uri=record.get("boreholeURI") or "",
        retrieved_from=url,
        retrieved_at=retrieved_at,
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        licence=LICENCE,
        attribution=ATTRIBUTION,
        rows=rows,
    )
    (dest_dir / f"{name}-source.json").write_text(
        json.dumps(manifest.to_json(), indent=1) + "\n", encoding="utf-8")
    return manifest


def verify(manifest: SourceManifest) -> None:
    """Raise if the frozen file is not the one the manifest describes."""
    path = Path(manifest.path)
    if not path.exists():
        raise FileNotFoundError(f"source file is gone: {path}")
    actual = sha256_file(path)
    if actual != manifest.sha256:
        raise ValueError(
            f"{path} hashes to {actual[:12]}, manifest says {manifest.sha256[:12]}")
