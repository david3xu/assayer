import pytest

from profiles.hylogger import calibration
from profiles.hylogger.reader import Sample


def hole(n, valid_every=1, swir="A", repeat_invalid=0, start=100.0, step=0.0074):
    """A synthetic hole: `n` samples, optionally with masked-out repeated depths."""
    out, depth = [], start
    for i in range(n):
        valid = (i % valid_every == 0) if valid_every > 1 else True
        out.append(Sample(depth, valid, swir, "T"))
        depth += step
    for _ in range(repeat_invalid):
        out.append(Sample(out[-1].depth_m, False, None, None))
    return out


def test_needs_more_than_one_hole():
    with pytest.raises(ValueError, match="at least two"):
        calibration.derive({"only": hole(100)})


def test_a_band_is_widened_by_within_hole_movement():
    derived = calibration.derive({"a": hole(4000), "b": hole(4000, step=0.0080)})
    band = calibration.bands(derived)["spacing_median_m"]
    assert band.low <= band.observed_min
    assert band.high >= band.observed_max
    assert band.holes == 2
    assert band.contains(band.observed_min) and band.contains(band.observed_max)


def test_a_fraction_band_is_never_impossible():
    derived = calibration.derive({"a": hole(4000, valid_every=2),
                                  "b": hole(4000, valid_every=3)})
    for name, band in calibration.bands(derived).items():
        if "fraction" in name or "rate" in name:
            assert band.low >= 0.0, name
            assert band.high <= 1.0, name


def test_a_band_spanning_everything_is_flagged_not_shipped():
    # One hole always answers, the other never does: the derived band covers the
    # whole of [0, 1] and forbids nothing.
    derived = calibration.derive({
        "a": hole(4000, swir="A"),
        "b": [Sample(s.depth_m, s.valid, None, s.tir_group) for s in hole(4000)],
    })
    band = calibration.bands(derived)["swir_answer_rate"]
    assert band.vacuous is True
    assert "swir_answer_rate" in derived["not_established"]
    assert "swir_answer_rate" not in calibration.bands(derived, usable_only=True)


def test_vocabulary_is_the_union_of_what_was_seen():
    derived = calibration.derive({"a": hole(4000, swir="KAOLIN"),
                                  "b": hole(4000, swir="CHLORITE")})
    assert derived["vocabulary"]["swir_group"] == ["CHLORITE", "KAOLIN"]


def test_repeated_depth_validity_is_undefined_without_a_repeat():
    assert calibration.valid_fraction_among_repeated_depths(hole(100)) is None
    assert calibration.valid_fraction_among_repeated_depths(
        hole(100, repeat_invalid=10)) == 0.0


def test_internal_variation_refuses_a_source_too_small_to_split():
    out = calibration.internal_variation(hole(50))
    assert out["measurable"] is False
    assert "1000" in out["reason"]


def test_total_variation_is_zero_for_equal_distributions_and_one_for_disjoint():
    assert calibration.total_variation({"A": 1.0}, {"A": 1.0}) == 0.0
    assert calibration.total_variation({"A": 1.0}, {"B": 1.0}) == pytest.approx(1.0)
