"""Stage 6 (generator) — turn a statistics Profile into synthetic rows.

Seeded code only. No model is called anywhere in this module (rule 1/2): the
input is a `statistics.Profile` — marginals, transitions, validity, geometry,
co-occurrence — plus a plan dict carrying at least `seed` and `n_rows`. Raw
source rows never enter here; the only source-derived numbers consumed are
the aggregates statistics already computed at fit time inside this process.

Method (working base, deliberately simple but honest):

  * Depths start at the profile geometry `depth_start_m` and step forward at
    the source median spacing. A depth is repeated only on an invalid row:
    calibration found a repeated depth is a masked-out depth, so repeats are
    placed inside masked-out stretches and the repeat rate among invalid rows
    reproduces the source repeated-depth fraction.
  * Valid flags come from a two-state Markov chain whose transition
    probabilities are 1 / mean run length of the source valid and invalid
    runs, so the stationary valid fraction matches the source.
  * Invalid rows carry no answer in either channel. For valid rows, the
    (swir, tir) answer pattern is drawn from the source answer structure —
    both answered, swir only, tir only, neither — at the rates implied by
    the profile counts. Both-answered pairs are drawn from the source
    co-occurrence joint; single-channel answers from that channel's
    marginals. That respects the marginals and the joint where both
    instruments answered, instead of sampling the channels independently.

Determinism: one explicit `random.Random(seed)` drives every draw, rows are
emitted through `dataset.to_csv_bytes`, so the same profile, plan and seed
always produce identical bytes (rule 2).
"""
from __future__ import annotations

import random
from bisect import bisect_left

from . import dataset
from .reader import Sample
from .statistics import Profile


def _pick(rng: random.Random, weights: list[float]) -> int:
    """Index of a draw from a categorical weight table (need not sum to 1)."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("cannot draw from an empty distribution")
    x = rng.random() * total
    cumulative: list[float] = []
    running = 0.0
    for w in weights:
        running += w
        cumulative.append(running)
    return min(bisect_left(cumulative, x), len(weights) - 1)


def _table(dist: dict[str, float]) -> tuple[list[str], list[float]]:
    keys = sorted(dist)
    return keys, [max(0.0, dist[k]) for k in keys]


def _markov_params(profile: Profile) -> tuple[float, float, float]:
    """(initial valid probability, P(valid->invalid), P(invalid->valid))."""
    validity = profile.validity
    valid_runs = validity.get("valid_runs") or {}
    invalid_runs = validity.get("invalid_runs") or {}
    p_valid_to_invalid = 1.0 / valid_runs["mean"] if valid_runs.get("mean") else 0.0
    p_invalid_to_valid = (
        1.0 / invalid_runs["mean"] if invalid_runs.get("mean") else 0.0
    )
    return validity["valid_fraction"], p_valid_to_invalid, p_invalid_to_valid


def _repeat_probability(profile: Profile) -> float:
    """P(repeat the previous depth | this row is invalid)."""
    geometry, validity = profile.geometry, profile.validity
    invalid_fraction = 1.0 - validity["valid_fraction"]
    if invalid_fraction <= 0 or not geometry["samples"]:
        return 0.0
    repeated_fraction = geometry["repeated_depths"] / geometry["samples"]
    return min(1.0, repeated_fraction / invalid_fraction)


def _channel_draws(profile: Profile):
    """Pre-build the categorical tables for one valid row's answers."""
    swir = profile.channels["swir_group"]
    tir = profile.channels["tir_group"]
    n_valid = profile.validity["valid"]
    pairs = profile.co_occurrence["pairs_observed"]

    categories = ["both", "swir_only", "tir_only", "neither"]
    if n_valid > 0:
        weights = [
            pairs / n_valid,
            max(0, swir["answered"] - pairs) / n_valid,
            max(0, tir["answered"] - pairs) / n_valid,
            max(0, n_valid - swir["answered"] - tir["answered"] + pairs) / n_valid,
        ]
    else:
        weights = [0.0, 0.0, 0.0, 1.0]

    joint_keys: list[tuple[str, str]] = []
    joint_weights: list[float] = []
    for swir_group in sorted(profile.co_occurrence["joint"]):
        row = profile.co_occurrence["joint"][swir_group]
        for tir_group in sorted(row):
            joint_keys.append((swir_group, tir_group))
            joint_weights.append(max(0.0, row[tir_group]))

    return (
        categories,
        weights,
        joint_keys,
        joint_weights,
        _table(swir["marginals"]),
        _table(tir["marginals"]),
        swir["answer_rate"],
        tir["answer_rate"],
    )


def _draw_answers(rng: random.Random, draws) -> tuple[str | None, str | None]:
    (categories, weights, joint_keys, joint_weights,
     (swir_keys, swir_weights), (tir_keys, tir_weights),
     swir_rate, tir_rate) = draws
    if not joint_keys:
        # The source never answered in both channels at once; fall back to
        # independent per-channel answers at the source answer rates.
        swir_group = (
            swir_keys[_pick(rng, swir_weights)]
            if swir_keys and rng.random() < swir_rate else None
        )
        tir_group = (
            tir_keys[_pick(rng, tir_weights)]
            if tir_keys and rng.random() < tir_rate else None
        )
        return swir_group, tir_group
    category = categories[_pick(rng, weights)]
    if category == "both":
        return joint_keys[_pick(rng, joint_weights)]
    if category == "swir_only":
        return (swir_keys[_pick(rng, swir_weights)] if swir_keys else None), None
    if category == "tir_only":
        return None, (tir_keys[_pick(rng, tir_weights)] if tir_keys else None)
    return None, None


def generate(profile: Profile, plan: dict) -> bytes:
    """Synthetic CSV bytes for `profile` under `plan` ({seed, n_rows})."""
    if "seed" not in plan or "n_rows" not in plan:
        raise ValueError("plan must carry at least 'seed' and 'n_rows'")
    seed, n_rows = plan["seed"], plan["n_rows"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("plan 'seed' must be an int")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValueError("plan 'n_rows' must be a positive int")

    rng = random.Random(seed)
    geometry = profile.geometry
    spacing = geometry["spacing_m"]
    step = spacing["median"] if spacing else 0.0
    if step <= 0:
        # A source with no forward step to measure; keep depths advancing at
        # the span-implied spacing rather than repeating every depth.
        step = geometry["span_m"] / geometry["samples"] if geometry["samples"] else 0.0
    start = geometry["depth_start_m"]
    p_initial_valid, p_v2i, p_i2v = _markov_params(profile)
    p_repeat = _repeat_probability(profile)
    draws = _channel_draws(profile)

    samples: list[Sample] = []
    valid = rng.random() < p_initial_valid
    forward_steps = 0
    for i in range(n_rows):
        if i > 0:
            valid = rng.random() >= (p_v2i if valid else 1.0 - p_i2v)
            if not valid and rng.random() < p_repeat:
                pass  # repeat the previous depth: a masked-out depth
            else:
                forward_steps += 1
        depth = start + forward_steps * step
        if valid:
            swir_group, tir_group = _draw_answers(rng, draws)
        else:
            swir_group, tir_group = None, None
        samples.append(Sample(depth, valid, swir_group, tir_group))
    return dataset.to_csv_bytes(samples)
