import json

import pytest

from engine.hashing import canonical_json
from profiles.hylogger.reader import Sample
from profiles.hylogger.statistics import build, channel, co_occurrence, geometry, validity


def s(depth, valid=True, swir=None, tir=None):
    return Sample(depth_m=depth, valid=valid, swir_group=swir, tir_group=tir)


def test_geometry_reports_repeated_depths_rather_than_hiding_them():
    g = geometry([s(1.0), s(1.0), s(1.01), s(1.02)])
    assert g["samples"] == 4
    assert g["monotonic_non_decreasing"] is True
    assert g["strictly_increasing"] is False
    assert g["repeated_depths"] == 1
    assert g["backward_steps"] == 0
    assert g["span_m"] == pytest.approx(0.02)
    assert g["spacing_m"]["median"] == pytest.approx(0.01)


def test_geometry_flags_a_backward_step():
    g = geometry([s(2.0), s(1.0)])
    assert g["monotonic_non_decreasing"] is False
    assert g["backward_steps"] == 1


def test_validity_measures_runs_not_just_a_fraction():
    samples = [s(i / 100, valid=(i % 10 < 7)) for i in range(100)]
    v = validity(samples)
    assert v["valid"] == 70
    assert v["valid_fraction"] == pytest.approx(0.70)
    assert v["valid_runs"]["runs"] == 10
    assert v["valid_runs"]["median"] == 7
    assert v["invalid_runs"]["median"] == 3


def test_marginals_and_transition_rows_are_distributions():
    # Proportions are rounded to six places so a profile is byte-stable and
    # readable, which costs up to one unit in the last place per cell: a row of
    # three thirds sums to 0.999999, not 1. A consumer renormalises; the
    # tolerance here is the accumulated rounding, not slack.
    seq = ["A", "A", "B", "A", "C", "C", "B"]
    ch = channel([s(i / 100, swir=g) for i, g in enumerate(seq)], "swir_group")
    assert sum(ch["marginals"].values()) == pytest.approx(1.0, abs=1e-5)
    for row in ch["transitions"].values():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)
        assert all(0.0 <= v <= 1.0 for v in row.values())
    assert ch["counts"] == {"A": 3, "B": 2, "C": 2}
    assert ch["groups"] == 3


def test_a_no_answer_is_not_a_group_and_is_reported_as_a_gap():
    seq = ["A", None, None, "A", "B"]
    ch = channel([s(i / 100, swir=g) for i, g in enumerate(seq)], "swir_group")
    assert "None" not in ch["marginals"] and None not in ch["marginals"]
    assert ch["answered"] == 3
    assert ch["unanswered"] == 2
    assert ch["answer_rate"] == pytest.approx(0.6)
    assert ch["unanswered_runs"]["runs"] == 1
    assert ch["unanswered_runs"]["max"] == 2


def test_invalid_samples_are_excluded_entirely():
    samples = [s(0.0, valid=True, swir="A"), s(0.01, valid=False, swir="Z")]
    ch = channel(samples, "swir_group")
    assert ch["counts"] == {"A": 1}


def test_persistence_and_transitions_describe_the_same_chain():
    # Mean run length is 1/(1 - p_self) when both are measured on the same
    # sequence. If they ever disagree, one of them is measured over a different
    # subsequence and the thresholds derived from them in calibration would
    # contradict each other.
    seq = list("AAABAACAAAABBAC")
    ch = channel([s(i / 100, swir=g) for i, g in enumerate(seq)], "swir_group")
    for group, shape in ch["persistence"].items():
        p_self = ch["transitions"].get(group, {}).get(group, 0.0)
        if p_self < 1.0:
            assert shape["mean"] == pytest.approx(1 / (1 - p_self), rel=1e-6)


def test_min_support_prunes_and_says_so():
    seq = ["A", "A", "A", "A", "B", "A"]
    full = channel([s(i / 100, swir=g) for i, g in enumerate(seq)], "swir_group")
    pruned = channel([s(i / 100, swir=g) for i, g in enumerate(seq)],
                     "swir_group", min_support=2)
    assert full["transition_coverage"] == pytest.approx(1.0)
    assert pruned["transition_coverage"] < 1.0


def test_co_occurrence_is_a_joint_distribution_over_answered_pairs():
    samples = [s(0.0, swir="A", tir="A"), s(0.01, swir="A", tir="B"),
               s(0.02, swir="B", tir=None)]
    co = co_occurrence(samples)
    assert co["pairs_observed"] == 2
    assert co["agreement_rate"] == pytest.approx(0.5)
    assert sum(sum(row.values()) for row in co["joint"].values()) == pytest.approx(1.0)


def test_profile_carries_no_source_row_across_the_boundary():
    # Rule 1: the model plans from statistics only. The depths below are
    # distinctive; none of them may appear in what stage 4 is handed, apart
    # from the two aggregate endpoints that geometry reports on purpose.
    depths = [700.123456 + i * 0.00731 for i in range(40)]
    samples = [s(d, swir="A" if i % 3 else "B", tir="CHLORITE")
               for i, d in enumerate(depths)]
    text = canonical_json(build(samples, "deadbeef", "HOLE1").to_json()).decode()
    for d in depths[1:-1]:
        assert f"{d:.5f}" not in text
    assert json.loads(text)["geometry"]["depth_start_m"] == pytest.approx(depths[0])


def test_an_empty_source_is_refused():
    with pytest.raises(ValueError):
        build([], "deadbeef", "HOLE1")
