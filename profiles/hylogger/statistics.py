"""Stage 3 — profile the source, as statistics and nothing else.

This module is the leakage boundary. Rule 1 says the model plans and never
writes data rows, and the plan it writes is informed only by what comes out of
here: counts, proportions, run lengths, spacings. No depth, no row, no sequence
of observed values is ever carried across. That is a licence obligation as much
as a leakage control — the source is CC BY 4.0 and its rows stay on disk.

Everything reported is a count or a function of counts, so the same source
always yields the same profile and a changed profile means changed bytes.
Proportions are rounded to six places to keep a profile byte-stable and
readable, which costs up to one unit in the last place per cell: a row of three
equal thirds sums to 0.999999. A consumer that needs an exact distribution
renormalises; the raw counts are reported alongside so it can.
"""
from __future__ import annotations

import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .reader import Sample

CHANNELS = ("swir_group", "tir_group")


def _quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile. Defined for a single value, unlike the stdlib's."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    i = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[i]


def _runs(labels: Iterable[object]) -> dict[object, list[int]]:
    """Run lengths, keyed by the value that repeats."""
    out: dict[object, list[int]] = defaultdict(list)
    current, length = object(), 0
    for value in labels:
        if value == current:
            length += 1
        else:
            if length:
                out[current].append(length)
            current, length = value, 1
    if length:
        out[current].append(length)
    return dict(out)


def _run_shape(lengths: Sequence[int]) -> dict[str, float | int]:
    return {
        "runs": len(lengths),
        "median": stats.median(lengths),
        "p90": _quantile(lengths, 0.90),
        "max": max(lengths),
        "mean": round(stats.fmean(lengths), 4),
    }


def geometry(samples: Sequence[Sample]) -> dict:
    """Depth extent and sampling spacing.

    Two properties of the real source that a generator gets wrong by default and
    that the design docs did not record: depth is non-decreasing but **not**
    strictly increasing — a continuous hole repeats a depth many times — and the
    spacing is a narrow band around 6.5 mm rather than a constant.
    """
    depths = [s.depth_m for s in samples]
    steps = [b - a for a, b in zip(depths, depths[1:])]
    forward = [s for s in steps if s > 0]
    return {
        "samples": len(samples),
        "depth_start_m": round(depths[0], 5),
        "depth_end_m": round(depths[-1], 5),
        "span_m": round(depths[-1] - depths[0], 5),
        "monotonic_non_decreasing": all(s >= 0 for s in steps),
        "strictly_increasing": all(s > 0 for s in steps),
        "repeated_depths": sum(1 for s in steps if s == 0),
        "backward_steps": sum(1 for s in steps if s < 0),
        "spacing_m": {
            "median": round(stats.median(forward), 6),
            "p05": round(_quantile(forward, 0.05), 6),
            "p95": round(_quantile(forward, 0.95), 6),
            "min": round(min(forward), 6),
            "max": round(max(forward), 6),
        } if forward else None,
    }


def validity(samples: Sequence[Sample]) -> dict:
    """How much of the hole TSG masked in, and in what sized stretches."""
    flags = [s.valid for s in samples]
    runs = _runs(flags)
    valid_runs = runs.get(True, [])
    invalid_runs = runs.get(False, [])
    return {
        "valid": sum(flags),
        "invalid": len(flags) - sum(flags),
        "valid_fraction": round(sum(flags) / len(flags), 6) if flags else 0.0,
        "valid_runs": _run_shape(valid_runs) if valid_runs else None,
        "invalid_runs": _run_shape(invalid_runs) if invalid_runs else None,
    }


def channel(samples: Sequence[Sample], field: str, min_support: int = 1) -> dict:
    """Marginals, persistence and first-order transitions for one channel.

    Computed over the **valid, answered** samples only, and the mass that is
    valid-but-unanswered is reported separately as `answer_rate` rather than
    folded into a category. A no-answer is the instrument declining, not a
    mineral group, and a generator that treats it as one reproduces the wrong
    thing (35% of the valid samples of the reference hole have no SWIR answer).

    `min_support` drops transition cells seen fewer times than that, with the
    proportion of transitions kept reported as `coverage`, so a pruned table
    never looks complete.

    Marginals, persistence and transitions are all measured on the same
    sequence: the answered samples in depth order, with the unanswered ones
    closed up. Measuring persistence on the valid sequence instead — letting a
    no-answer break a run — while transitions step over those gaps gives two
    statistics that describe different chains, and the thresholds derived from
    them in calibration would contradict each other. The gaps are not discarded;
    their own run structure is reported as `unanswered_runs`, so a generator has
    to reproduce where the instrument fell silent and for how long, separately
    from which minerals it reported.
    """
    valid = [s for s in samples if s.valid]
    values = [getattr(s, field) for s in valid]
    answered = [v for v in values if v is not None]
    counts = Counter(answered)

    runs = _runs(answered)
    persistence = {k: _run_shape(v) for k, v in sorted(runs.items())}
    gaps = _runs(v is None for v in values).get(True, [])

    pairs = Counter(
        (a, b) for a, b in zip(answered, answered[1:])
    )
    kept = {k: n for k, n in pairs.items() if n >= min_support}
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    totals = Counter()
    for (a, _), n in kept.items():
        totals[a] += n
    for (a, b), n in sorted(kept.items()):
        rows[a][b] = round(n / totals[a], 6)

    total_pairs = sum(pairs.values())
    return {
        "answered": len(answered),
        "unanswered": len(values) - len(answered),
        "answer_rate": round(len(answered) / len(values), 6) if values else 0.0,
        "groups": len(counts),
        "marginals": {k: round(n / len(answered), 6) for k, n in sorted(counts.items())}
        if answered else {},
        "counts": dict(sorted(counts.items())),
        "persistence": persistence,
        "unanswered_runs": _run_shape(gaps) if gaps else None,
        "transitions": {k: dict(v) for k, v in sorted(rows.items())},
        "transition_coverage": round(sum(kept.values()) / total_pairs, 6)
        if total_pairs else 0.0,
    }


def co_occurrence(samples: Sequence[Sample], min_support: int = 1) -> dict:
    """The joint distribution of the two channels where both answered.

    The channels are not independent — they are two instruments looking at the
    same rock — so a generator that samples them separately is wrong in a way
    the marginals cannot show.
    """
    both = [(s.swir_group, s.tir_group) for s in samples
            if s.valid and s.swir_group and s.tir_group]
    counts = Counter(both)
    kept = {k: n for k, n in counts.items() if n >= min_support}
    total = sum(counts.values())
    joint: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), n in sorted(kept.items()):
        joint[a][b] = round(n / total, 6)
    return {
        "pairs_observed": total,
        "distinct_pairs": len(counts),
        "joint": {k: dict(v) for k, v in sorted(joint.items())},
        "coverage": round(sum(kept.values()) / total, 6) if total else 0.0,
        "agreement_rate": round(
            sum(n for (a, b), n in counts.items() if a == b) / total, 6) if total else 0.0,
    }


@dataclass(frozen=True)
class Profile:
    """The whole statistical summary. This object is what stage 4 may see."""

    source_sha256: str
    source_name: str
    geometry: dict
    validity: dict
    channels: dict
    co_occurrence: dict

    def to_json(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "source_name": self.source_name,
            "geometry": self.geometry,
            "validity": self.validity,
            "channels": self.channels,
            "co_occurrence": self.co_occurrence,
        }


def build(samples: Sequence[Sample], source_sha256: str, source_name: str,
          min_support: int = 1) -> Profile:
    if not samples:
        raise ValueError("cannot profile an empty source")
    return Profile(
        source_sha256=source_sha256,
        source_name=source_name,
        geometry=geometry(samples),
        validity=validity(samples),
        channels={f: channel(samples, f, min_support) for f in CHANNELS},
        co_occurrence=co_occurrence(samples, min_support),
    )
