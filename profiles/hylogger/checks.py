"""Stage 6 — the deterministic checks.

Rule 3: every quality claim is a check result bound to an artifact hash. Each
check here is a pure function of the bytes that were written, the profile of the
source those bytes were fitted to, and the thresholds derived in calibration. No
model is consulted and no check consults the clock, the network or a random
number, so re-running a check on the same artifact gives the same verdict.

Two kinds, because calibration showed two kinds are needed:

  invariant  a band that holds on every real hole measured, so a violation is a
             generator defect
  fidelity   a comparison against this run's own source, with the tolerance
             taken from how far that source sits from itself

A check that cannot be evaluated returns `skipped` with the reason. It never
returns a pass it did not establish.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

from .calibration import (Band, bands, internal_variation, total_variation,
                           valid_fraction_among_repeated_depths)
from .reader import Sample
from .statistics import channel, geometry, validity


@dataclass(frozen=True)
class CheckResult:
    name: str
    kind: str
    status: str                      # "pass" | "fail" | "skip"
    observed: Any = None
    expected: str = ""
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CheckReport:
    """The verdict on one artifact. Bound to its hash, never to its filename."""

    artifact_sha256: str
    source_sha256: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def skipped(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "skip"]

    @property
    def verdict(self) -> str:
        return "fail" if self.failed else "pass"

    def to_json(self) -> dict:
        return {
            "artifact_sha256": self.artifact_sha256,
            "source_sha256": self.source_sha256,
            "verdict": self.verdict,
            "passed": sum(1 for r in self.results if r.status == "pass"),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "results": [r.to_json() for r in self.results],
        }


def _band_check(name: str, value: float, band: Band | None) -> CheckResult:
    if band is None:
        return CheckResult(name, "invariant", "skip",
                           detail="no threshold was derived for this statistic")
    if band.vacuous:
        return CheckResult(name, "invariant", "skip", observed=round(value, 8),
                           detail=f"calibration could not establish a band "
                                  f"({band.holes} holes spanned the whole range)")
    status = "pass" if band.contains(value) else "fail"
    return CheckResult(
        name, "invariant", status, observed=round(value, 8),
        expected=f"[{band.low:.6f}, {band.high:.6f}]",
        detail=(f"derived from {band.holes} real holes "
                f"(observed {band.observed_min:.6f}-{band.observed_max:.6f}, "
                f"widened by {band.widened_by:.6f})"))


def schema(samples: Sequence[Sample]) -> list[CheckResult]:
    """The artifact is shaped the way the profile says, before anything else."""
    out = []
    out.append(CheckResult(
        "schema.not_empty", "invariant", "pass" if samples else "fail",
        observed=len(samples), expected=">= 1 row"))
    bad_depth = sum(1 for s in samples if s.depth_m != s.depth_m or s.depth_m < 0)
    out.append(CheckResult(
        "schema.depth_is_a_non_negative_number", "invariant",
        "pass" if bad_depth == 0 else "fail", observed=bad_depth, expected="0 rows"))
    return out


def invariants(samples: Sequence[Sample], derived: dict) -> list[CheckResult]:
    b = bands(derived)
    g = geometry(samples)
    v = validity(samples)
    ti = channel(samples, "tir_group")
    sw = channel(samples, "swir_group")
    out: list[CheckResult] = []

    # The corrected form of the design's original check. Every one of the 38
    # calibration holes is non-decreasing and none takes a backward step, but
    # only some are strictly increasing, so strictness is not the invariant --
    # direction is.
    out.append(_band_check("geometry.no_backward_steps",
                           float(g["backward_steps"]), b.get("backward_steps")))
    out.append(CheckResult(
        "geometry.depth_non_decreasing", "invariant",
        "pass" if g["monotonic_non_decreasing"] else "fail",
        observed=g["monotonic_non_decreasing"], expected="True",
        detail="holds on all 38 calibration holes; strict increase does not"))

    if g["spacing_m"] is None:
        out.append(CheckResult("geometry.spacing_median_m", "invariant", "skip",
                               detail="no forward step to measure a spacing from"))
    else:
        out.append(_band_check("geometry.spacing_median_m",
                               g["spacing_m"]["median"], b.get("spacing_median_m")))
    out.append(_band_check("geometry.repeated_depth_fraction",
                           g["repeated_depths"] / g["samples"],
                           b.get("repeated_depth_fraction")))
    out.append(_band_check("validity.valid_fraction", v["valid_fraction"],
                           b.get("valid_fraction")))

    # The sharpest constraint calibration found: a repeated depth is a masked-out
    # depth. Reproducing the rate of repeated depths without reproducing where
    # they sit passes the band above and fails this one.
    repeated_valid = valid_fraction_among_repeated_depths(samples)
    if repeated_valid is None:
        out.append(CheckResult(
            "validity.repeated_depths_are_masked_out", "invariant", "skip",
            detail="the artifact repeats no depth, so the statistic is undefined"))
    else:
        out.append(_band_check("validity.repeated_depths_are_masked_out",
                               repeated_valid,
                               b.get("valid_fraction_among_repeated_depths")))
    out.append(_band_check("channel.tir_answer_rate", ti["answer_rate"],
                           b.get("tir_answer_rate")))
    out.append(_band_check("channel.swir_answer_rate", sw["answer_rate"],
                           b.get("swir_answer_rate")))

    for field_name, key in (("swir_group", "swir"), ("tir_group", "tir")):
        known = set(derived["vocabulary"][field_name])
        seen = set(channel(samples, field_name)["marginals"])
        unknown = sorted(seen - known)
        out.append(CheckResult(
            f"vocabulary.{key}_groups_are_known", "invariant",
            "pass" if not unknown else "fail", observed=unknown,
            expected=f"subset of {len(known)} groups seen in the calibration set",
            detail="a group outside the source taxonomy is an invented mineral"))
    return out


def fidelity(samples: Sequence[Sample], source: Sequence[Sample]) -> list[CheckResult]:
    """Compare the artifact to its own source, at the source's own tolerance."""
    out: list[CheckResult] = []
    tol = internal_variation(source)
    for field_name, key in (("swir_group", "swir"), ("tir_group", "tir")):
        observed = round(total_variation(channel(samples, field_name)["marginals"],
                                         channel(source, field_name)["marginals"]), 6)
        if not tol.get("measurable"):
            out.append(CheckResult(
                f"fidelity.{key}_marginal_tvd", "fidelity", "skip",
                observed=observed,
                detail=f"source cannot establish its own tolerance: {tol.get('reason')}"))
            continue
        limit = tol[f"{key}_marginal_tvd"]
        out.append(CheckResult(
            f"fidelity.{key}_marginal_tvd", "fidelity",
            "pass" if observed <= limit else "fail", observed=observed,
            expected=f"<= {limit:.6f}",
            detail=("the tolerance is this source's own split-half distance, so the "
                    "artifact is allowed to differ from the source by as much as the "
                    "source differs from itself and no more")))
    return out


def run(samples: Sequence[Sample], artifact_sha256: str, derived: dict,
        source: Sequence[Sample] | None = None,
        source_sha256: str = "") -> CheckReport:
    results = schema(samples)
    if any(r.status == "fail" for r in results):
        return CheckReport(artifact_sha256, source_sha256, results)
    results += invariants(samples, derived)
    if source is not None:
        results += fidelity(samples, source)
    return CheckReport(artifact_sha256, source_sha256, results)
