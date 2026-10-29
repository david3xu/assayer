"""Does the ruler catch anything?

The checks passing on real holes only shows they raise no false alarm. This
takes a real hole and breaks it in one specific way at a time, then asserts that
the check responsible for that property is the one that fails. A suite that
passes everything is not a ruler, and a suite where every perturbation trips
every check is not a diagnosis.

Run:  python evaluation/negative_control.py data/sources/PDD446-source.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.hashing import sha256_bytes
from profiles.hylogger import calibration, checks, dataset, reader, source
from profiles.hylogger.reader import Sample


def shuffled_depths(samples):
    """Rows reordered, as a generator that forgot to sort would emit them."""
    return samples[len(samples) // 2:] + samples[: len(samples) // 2]


def invented_mineral(samples):
    return [Sample(s.depth_m, s.valid, "UNOBTANIUM" if s.swir_group else None,
                   s.tir_group) for s in samples]


def everything_valid(samples):
    return [Sample(s.depth_m, True, s.swir_group, s.tir_group) for s in samples]


def depths_in_centimetres(samples):
    """A unit error: the right numbers, the wrong scale."""
    return [Sample(s.depth_m * 10, s.valid, s.swir_group, s.tir_group) for s in samples]


def tir_silent(samples):
    return [Sample(s.depth_m, s.valid, s.swir_group, None) for s in samples]


def one_mineral_everywhere(samples):
    """Marginals collapsed onto the modal group: plausible rows, wrong mixture."""
    return [Sample(s.depth_m, s.valid,
                   "CHLORITE" if s.swir_group else None,
                   "CHLORITE" if s.tir_group else None) for s in samples]


def depths_deduplicated(samples):
    """Rows at a repeated depth dropped.

    This is not a clean perturbation and the expectation below says so. Measured
    on PDD446, only 1.7% of repeated-depth rows are valid against 85.9%
    overall, so dropping them lifts the valid fraction from 0.859 to 0.963 and
    out of its band. The check that fires is the one about validity, not the one
    about depth, which is correct: the edit changed validity.
    """
    out, last = [], None
    for s in samples:
        if last is None or s.depth_m > last:
            out.append(s)
            last = s.depth_m
    return out


def depths_nudged_strictly_increasing(samples):
    """Strict increase, every row kept.

    The clean version of the test above, and the one that matters: the original
    design demanded strictly increasing depth, and the calibration set contains
    a hole with no repeated depth at all, so strict increase must be allowed.
    """
    out, last = [], None
    for s in samples:
        depth = s.depth_m
        if last is not None and depth <= last:
            depth = last + 1e-5
        out.append(Sample(depth, s.valid, s.swir_group, s.tir_group))
        last = depth
    return out


def repeated_depths_misplaced(samples):
    """Repeated depths at the right overall rate, in the wrong places.

    Every repeat is moved onto a valid row. The rate of repeated depths is
    preserved, so the band check on it still passes; what fails is the
    constraint that a repeated depth is a masked-out depth.
    """
    cleaned = depths_nudged_strictly_increasing(samples)
    budget = sum(1 for a, b in zip(samples, samples[1:]) if b.depth_m == a.depth_m)
    out, last = [], None
    for s in cleaned:
        if budget and s.valid and last is not None:
            out.append(Sample(last, s.valid, s.swir_group, s.tir_group))
            budget -= 1
            continue
        out.append(s)
        last = s.depth_m
    return out


PERTURBATIONS = [
    ("rows reordered",            shuffled_depths,      "geometry.no_backward_steps"),
    ("a mineral group invented",  invented_mineral,     "vocabulary.swir_groups_are_known"),
    ("every row marked valid",    everything_valid,     "validity.valid_fraction"),
    ("depths in the wrong unit",  depths_in_centimetres, "geometry.spacing_median_m"),
    ("TIR never answers",         tir_silent,           "channel.tir_answer_rate"),
    ("one mineral everywhere",    one_mineral_everywhere, "fidelity.swir_marginal_tvd"),
    ("repeats in valid stretches", repeated_depths_misplaced,
     "validity.repeated_depths_are_masked_out"),
    ("repeated depths dropped",   depths_deduplicated,  "validity.valid_fraction"),
    ("depths made strict",        depths_nudged_strictly_increasing, None),
]


def main(manifest_path: str) -> int:
    manifest = source.SourceManifest.load(manifest_path)
    source.verify(manifest)
    truth = reader.read(manifest.path)
    derived = calibration.load()

    def report(samples):
        data = dataset.to_csv_bytes(samples)
        return checks.run(samples, sha256_bytes(data), derived,
                          source=truth, source_sha256=manifest.sha256)

    baseline = report(truth)
    print(f"source {manifest.dataset_name}: {baseline.verdict} "
          f"({len(baseline.failed)} failures)\n")
    if baseline.failed:
        print("the unperturbed source does not pass; the ruler is wrong, not the rock")
        for r in baseline.failed:
            print("   ", r.name, r.observed, r.expected)
        return 1

    print(f"{'perturbation':<26} {'verdict':<8} {'expected check':<36} {'caught'}")
    bad = 0
    for label, perturb, expected in PERTURBATIONS:
        rep = report(perturb(truth))
        names = {r.name for r in rep.failed}
        if expected is None:
            ok = rep.verdict == "pass"
            shown = "(must still pass)"
        else:
            ok = expected in names
            shown = expected
        bad += not ok
        print(f"{label:<26} {rep.verdict:<8} {shown:<36} {'yes' if ok else 'NO'}")
        if names:
            print(f"{'':<26} failed: {', '.join(sorted(names))}")

    print()
    print(f"{len(PERTURBATIONS) - bad}/{len(PERTURBATIONS)} perturbations "
          f"behaved as the design says they should")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "data/sources/PDD446-source.json"))
