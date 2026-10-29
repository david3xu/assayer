"""Stage 2/3 boundary — read a frozen source into the profile's five fields.

Written fresh on the published NVCL interfaces and on pytsg. No raw row read
here ever reaches a model: the reader hands its records to `statistics.py`, and
only the summary that produces crosses the boundary into stage 4.

Two intake paths, one record shape:

  `read_scalar_csv`  the frozen CSV from the NVCL scalar service (continuous
                     holes, ~1 MB each, the path the profile uses)
  `read_tsg_package` a TSG package on disk, via pytsg (offline, and the path a
                     user with their own unpublished package needs)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# TSG records "the instrument looked and could not answer" several different
# ways, and the scalar service adds one more: it spells a missing categorical as
# the four characters `null`. All of them mean the same thing downstream and all
# of them become None, so a no-answer is never counted as a mineral group.
NO_ANSWER = frozenset({"", "null", "NULL", "INVALID", "ASPECTRAL", "NOTAROK",
                       "NONE", "NA", "N/A"})

# TSG writes an absent float as the smallest representable single-precision
# value rather than as a NaN.
TSG_FLOAT_MIN = -3.4028234663852886e38


@dataclass(frozen=True)
class Sample:
    """One logged depth. `valid` is TSG's own Final Mask, not our judgement."""

    depth_m: float
    valid: bool
    swir_group: str | None
    tir_group: str | None


def _group(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return None if text.upper() in {v.upper() for v in NO_ANSWER} else text


def _depth(raw: object) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value <= TSG_FLOAT_MIN / 2 or value != value:
        return None
    return value


def read_scalar_csv(path: str | Path) -> list[Sample]:
    """Read a frozen `downloadscalars` CSV.

    The service names its columns after the logs, replacing spaces with
    underscores, and emits `StartDepth`/`EndDepth` rather than TSG's
    `Depth (m)`. The two are equal for every row of a scalar log, so StartDepth
    is the depth. Rows with no usable depth are dropped and counted by the
    caller through the length difference; a row with a depth but no mineral
    answer is kept, because "logged here, no answer" is a fact about the hole
    and the statistics need it.
    """
    out: list[Sample] = []
    with open(path, encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh):
            depth = _depth(row.get("StartDepth"))
            if depth is None:
                continue
            out.append(Sample(
                depth_m=depth,
                valid=str(row.get("Final_Mask", "")).strip() == "1",
                swir_group=_group(row.get("Grp1_sTSAS")),
                tir_group=_group(row.get("Grp1_sjCLST")),
            ))
    return out


def read_tsg_package(directory: str | Path) -> list[Sample]:
    """Read a TSG package with pytsg.

    The four fields are split across two parts of the package, which the design
    docs did not say: `Grp1 sTSAS` is a NIR scalar and `Grp1 sjCLST` a TIR one,
    and each part carries its own `Depth (m)` and `Final Mask`. The two depth
    arrays are identical in every package checked, so the parts are joined on
    index after that equality is asserted rather than assumed.
    """
    from pytsg import parse_tsg  # imported here: the CSV path needs no pytsg

    ds = parse_tsg.read_tsg(str(directory))
    available = set(getattr(ds, "available", ()) or ())
    if not {"nir", "tir"} <= available:
        raise LookupError(
            f"package has parts {sorted(available)}; both nir and tir are needed")

    nir, tir = ds.nir.scalars, ds.tir.scalars
    if len(nir) != len(tir):
        raise ValueError(f"nir has {len(nir)} samples, tir has {len(tir)}")

    depth_nir = [_depth(v) for v in nir["Depth (m)"]]
    depth_tir = [_depth(v) for v in tir["Depth (m)"]]
    if depth_nir != depth_tir:
        raise ValueError("the nir and tir depth arrays disagree; cannot join on index")

    mask = nir["Final Mask"] if "Final Mask" in nir else tir["Final Mask"]
    swir = nir["Grp1 sTSAS"]
    tirg = tir["Grp1 sjCLST"]

    out: list[Sample] = []
    for i, depth in enumerate(depth_nir):
        if depth is None:
            continue
        out.append(Sample(
            depth_m=depth,
            valid=str(mask.iloc[i]).strip() in {"1", "1.0", "True"},
            swir_group=_group(swir.iloc[i]),
            tir_group=_group(tirg.iloc[i]),
        ))
    return out


def read(source: str | Path) -> list[Sample]:
    """Dispatch on what `source` is: a CSV file, or a package directory."""
    path = Path(source)
    if path.is_dir():
        return read_tsg_package(path)
    return read_scalar_csv(path)


def groups(samples: list[Sample]) -> Iterator[tuple[str | None, str | None]]:
    """The (swir, tir) pairs of the valid samples, in depth order."""
    for s in samples:
        if s.valid:
            yield s.swir_group, s.tir_group
