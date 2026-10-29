import pytest

from engine.hashing import sha256_bytes
from profiles.hylogger import calibration, checks, dataset
from profiles.hylogger.reader import Sample


def hole(n=4000, step=0.0074, start=100.0, valid_every=1, swir="A", tir="T"):
    out, depth = [], start
    for i in range(n):
        out.append(Sample(depth, i % valid_every == 0 if valid_every > 1 else True,
                          swir, tir))
        depth += step
    return out


@pytest.fixture
def derived():
    return calibration.derive({"a": hole(), "b": hole(step=0.0080),
                               "c": hole(step=0.0076)})


def report(samples, derived, source=None):
    data = dataset.to_csv_bytes(samples)
    return checks.run(samples, sha256_bytes(data), derived,
                      source=source, source_sha256="src")


def test_a_clean_hole_passes_and_the_verdict_is_bound_to_the_bytes(derived):
    samples = hole(step=0.0076)
    rep = report(samples, derived)
    assert rep.verdict == "pass"
    assert rep.artifact_sha256 == sha256_bytes(dataset.to_csv_bytes(samples))
    assert rep.failed == []


def test_an_empty_artifact_fails_the_schema_and_stops_there(derived):
    rep = report([], derived)
    assert rep.verdict == "fail"
    assert [r.name for r in rep.failed] == ["schema.not_empty"]
    # Nothing downstream is reported, because nothing downstream was measured.
    assert all(r.name.startswith("schema.") for r in rep.results)


def test_a_backward_step_fails(derived):
    samples = hole(step=0.0076)
    samples[500] = Sample(samples[0].depth_m - 1, True, "A", "T")
    names = {r.name for r in report(samples, derived).failed}
    assert "geometry.no_backward_steps" in names
    assert "geometry.depth_non_decreasing" in names


def test_an_invented_group_fails_the_vocabulary(derived):
    samples = hole(step=0.0076, swir="UNOBTANIUM")
    names = {r.name for r in report(samples, derived).failed}
    assert "vocabulary.swir_groups_are_known" in names


def test_spacing_outside_the_band_fails(derived):
    names = {r.name for r in report(hole(step=0.074), derived).failed}
    assert "geometry.spacing_median_m" in names


def test_a_statistic_calibration_could_not_pin_down_is_skipped_not_passed():
    wide = calibration.derive({
        "a": hole(),
        "b": [Sample(s.depth_m, s.valid, None, s.tir_group) for s in hole()],
    })
    rep = report(hole(step=0.0076), wide)
    skipped = {r.name for r in rep.skipped}
    assert "channel.swir_answer_rate" in skipped
    # A skip is not a pass, and it carries its reason.
    result, = [r for r in rep.results if r.name == "channel.swir_answer_rate"]
    assert result.passed is False
    assert "could not establish" in result.detail


def test_fidelity_compares_against_the_source_at_the_sources_own_tolerance(derived):
    source = hole(step=0.0076, swir="A")
    same = report(source, derived, source=source)
    assert same.verdict == "pass"

    # A source that is uniform has a split-half distance of zero, so any
    # departure at all is a failure. That is the tolerance working, not slack.
    other = [Sample(s.depth_m, s.valid, "KAOLIN", s.tir_group) for s in source]
    names = {r.name for r in report(other, derived, source=source).failed}
    assert "fidelity.swir_marginal_tvd" in names


def test_fidelity_is_skipped_when_the_source_cannot_establish_a_tolerance(derived):
    tiny = hole(n=100, step=0.0076)
    rep = report(hole(step=0.0076), derived, source=tiny)
    skipped = {r.name for r in rep.skipped}
    assert "fidelity.swir_marginal_tvd" in skipped
    assert "fidelity.tir_marginal_tvd" in skipped


def test_the_report_counts_what_it_says_it_counts(derived):
    j = report(hole(step=0.0076), derived).to_json()
    assert j["passed"] + j["failed"] + j["skipped"] == len(j["results"])
    assert j["verdict"] == ("fail" if j["failed"] else "pass")
