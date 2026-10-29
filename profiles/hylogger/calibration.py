"""Step 5 — turn thresholds into derived numbers.

A check is only a ruler if someone can say where its threshold came from. This
module derives them from measurement, and it reports two kinds of answer,
because measuring real holes against each other showed the statistics fall into
two groups that need different rulers.

**Invariants.** Some statistics barely move between holes: depth never runs
backwards, spacing sits in a narrow band, TIR almost always answers. A synthetic
hole outside that band is wrong, so these become absolute checks. The band is
the observed range over the calibration set, widened by how far one half of a
single hole can sit from the other — because a new hole's half may legitimately
sit that far outside anything yet seen.

**Composition.** The mineral statistics do not work this way, and the honest
result is that they cannot be checked against a universal threshold at all.
Measured over 38 WA holes, the marginal distribution of the top half of a hole
differs from its own bottom half by a total variation distance of 0.27 at the
median and 0.83 at the 95th percentile — as much as two different holes differ
from each other. A tolerance wide enough to accept real rock would accept almost
anything. So composition is checked against *this run's own source*, with the
tolerance derived from that source's own internal variation, and the threshold
is recomputed per run rather than fixed here. A run that cannot establish its
own tolerance reports that instead of borrowing someone else's.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Sequence

from .reader import Sample
from .statistics import channel, co_occurrence, geometry, validity

THRESHOLDS = Path(__file__).with_name("thresholds.json")

# Fractions live in [0, 1]; a derived band is clipped rather than allowed to
# imply an impossible value such as a negative proportion.
_FRACTIONS = frozenset({"valid_fraction", "tir_answer_rate", "swir_answer_rate",
                        "agreement_rate", "swir_top_share", "tir_top_share",
                        "repeated_depth_fraction",
                        "valid_fraction_among_repeated_depths"})

# A band this wide forbids nothing a fraction could do. Derivation still reports
# it, flagged, so the one statistic it failed to pin down is visible rather than
# shipped as a check that always passes.
_VACUOUS_SPAN = 0.95


@dataclass(frozen=True)
class Band:
    """An inclusive range a statistic must fall in, and where it came from."""

    statistic: str
    low: float
    high: float
    observed_min: float
    observed_max: float
    widened_by: float
    holes: int
    vacuous: bool = False

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def to_json(self) -> dict:
        return asdict(self)


def valid_fraction_among_repeated_depths(samples: Sequence[Sample]) -> float | None:
    """How often a row that repeats its predecessor's depth is masked in.

    Almost never, as it turns out. Over the calibration set this sits between
    0.000 and 0.113 with a median of 0.005, against an overall valid fraction of
    0.68-0.86: the instrument repeats a depth exactly where it has nothing to
    report. Nothing in the design documents said so, and it is the sharpest
    structural constraint found — a generator has to put its repeated depths
    inside masked-out stretches, not sprinkle them at the right overall rate.

    None when the hole has no repeated depth, where the statistic is undefined.
    """
    repeated = [b for a, b in zip(samples, samples[1:]) if b.depth_m == a.depth_m]
    if not repeated:
        return None
    return sum(s.valid for s in repeated) / len(repeated)


def hole_statistics(samples: Sequence[Sample]) -> dict[str, float]:
    """The scalar statistics calibration is derived from, for one hole.

    A statistic that is undefined for this hole is absent rather than zero, so
    derivation never widens a band with a value that was never measured.
    """
    g = geometry(samples)
    v = validity(samples)
    sw = channel(samples, "swir_group")
    ti = channel(samples, "tir_group")
    out = {
        "spacing_median_m": g["spacing_m"]["median"],
        "repeated_depth_fraction": g["repeated_depths"] / g["samples"],
        "backward_steps": float(g["backward_steps"]),
        "valid_fraction": v["valid_fraction"],
        "tir_answer_rate": ti["answer_rate"],
        "swir_answer_rate": sw["answer_rate"],
    }
    repeated_valid = valid_fraction_among_repeated_depths(samples)
    if repeated_valid is not None:
        out["valid_fraction_among_repeated_depths"] = repeated_valid
    return out


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Half the L1 distance between two marginal distributions, in [0, 1]."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def split_half(samples: Sequence[Sample]) -> tuple[Sequence[Sample], Sequence[Sample]]:
    mid = len(samples) // 2
    return samples[:mid], samples[mid:]


def internal_variation(samples: Sequence[Sample], min_half: int = 1000) -> dict:
    """How far one half of this hole sits from the other.

    This is the per-run tolerance for a composition check. It is an upper bound
    on acceptable wobble rather than a tight one: the two halves of a hole are
    different depths and so genuinely different rock, which is the point — a
    synthetic hole no further from the source than the source is from itself has
    not been caught doing anything the source does not also do.
    """
    a, b = split_half(samples)
    if min(len(a), len(b)) < min_half:
        return {"measurable": False, "reason":
                f"each half needs {min_half} samples; got {min(len(a), len(b))}"}
    out: dict[str, object] = {"measurable": True, "samples_per_half": min(len(a), len(b))}
    for field, key in (("swir_group", "swir"), ("tir_group", "tir")):
        out[f"{key}_marginal_tvd"] = round(
            total_variation(channel(a, field)["marginals"],
                            channel(b, field)["marginals"]), 6)
    sa, sb = hole_statistics(a), hole_statistics(b)
    out["scalar_delta"] = {k: round(abs(sa[k] - sb[k]), 6)
                           for k in set(sa) & set(sb)}
    return out


def derive(holes: dict[str, Sequence[Sample]]) -> dict:
    """Derive the invariant bands from a calibration set of real holes."""
    if len(holes) < 2:
        raise ValueError("calibration needs at least two holes")

    per_hole = {name: hole_statistics(s) for name, s in holes.items()}
    internal = {name: internal_variation(s) for name, s in holes.items()}

    # The widening term: the 95th percentile of within-hole half-to-half
    # movement for that statistic, over the holes where it could be measured.
    def widen(stat: str) -> float:
        deltas = sorted(v["scalar_delta"][stat] for v in internal.values()
                        if v.get("measurable") and stat in v["scalar_delta"])
        if not deltas:
            return 0.0
        return deltas[min(len(deltas) - 1, round(0.95 * (len(deltas) - 1)))]

    bands: dict[str, Band] = {}
    for stat in sorted({k for h in per_hole.values() for k in h}):
        values = [h[stat] for h in per_hole.values() if stat in h]
        lo, hi, w = min(values), max(values), widen(stat)
        low, high = lo - w, hi + w
        vacuous = False
        if stat in _FRACTIONS:
            low, high = max(0.0, low), min(1.0, high)
            vacuous = (high - low) >= _VACUOUS_SPAN
        if stat == "backward_steps":
            low = high = 0.0  # observed exactly zero on every hole; keep it exact
        bands[stat] = Band(statistic=stat, low=round(low, 8), high=round(high, 8),
                           observed_min=round(lo, 8), observed_max=round(hi, 8),
                           widened_by=round(w, 8), holes=len(values),
                           vacuous=vacuous)

    vocabulary = {
        field: sorted({g for s in holes.values()
                       for g in channel(s, field)["marginals"]})
        for field in ("swir_group", "tir_group")
    }

    tvds = {
        key: sorted(v[f"{key}_marginal_tvd"] for v in internal.values()
                    if v.get("measurable"))
        for key in ("swir", "tir")
    }
    return {
        "bands": {k: v.to_json() for k, v in bands.items()},
        "vocabulary": vocabulary,
        "composition_is_not_universally_checkable": {
            "reason": "within-hole variation is as large as between-hole variation",
            "within_hole_marginal_tvd": {
                key: {"median": round(median(v), 6), "p95": round(
                    v[min(len(v) - 1, round(0.95 * (len(v) - 1)))], 6), "holes": len(v)}
                for key, v in tvds.items() if v
            },
        },
        "calibration_holes": sorted(holes),
        "not_established": sorted(k for k, v in bands.items() if v.vacuous),
    }


def write(derived: dict, path: str | Path = THRESHOLDS) -> Path:
    path = Path(path)
    path.write_text(json.dumps(derived, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def load(path: str | Path = THRESHOLDS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bands(derived: dict, usable_only: bool = False) -> dict[str, Band]:
    out = {k: Band(**v) for k, v in derived["bands"].items()}
    return {k: v for k, v in out.items() if not v.vacuous} if usable_only else out
