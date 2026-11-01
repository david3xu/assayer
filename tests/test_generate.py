import hashlib
from pathlib import Path

import pytest

from engine.hashing import sha256_bytes
from profiles.hylogger import calibration, checks, dataset, generate, reader, statistics

SOURCE = Path(__file__).resolve().parent.parent / "data" / "sources" / "PDD446-scalars.csv"
PLAN = {"seed": 446, "n_rows": 39300}


@pytest.fixture(scope="module")
def fitted():
    source = reader.read(SOURCE)
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    profile = statistics.build(source, source_sha256, SOURCE.stem)
    return source, source_sha256, profile


def test_same_seed_same_hash(fitted):
    _, _, profile = fitted
    first = generate.generate(profile, PLAN)
    second = generate.generate(profile, dict(PLAN))
    assert sha256_bytes(first) == sha256_bytes(second)
    assert first == second


def test_different_seed_differs(fitted):
    _, _, profile = fitted
    assert generate.generate(profile, PLAN) != generate.generate(
        profile, {"seed": PLAN["seed"] + 1, "n_rows": PLAN["n_rows"]})


def test_generated_bytes_pass_the_checks(fitted, tmp_path):
    source, source_sha256, profile = fitted
    data = generate.generate(profile, PLAN)
    path = tmp_path / "synthetic.csv"
    path.write_bytes(data)
    samples = dataset.read(path)
    report = checks.run(samples, sha256_bytes(data), calibration.load(),
                        source=source, source_sha256=source_sha256)
    assert report.verdict == "pass"
    assert report.failed == []
    # The one skip is expected: calibration could not establish a band for
    # the SWIR answer rate, so the check reports that instead of passing.
    assert [r.name for r in report.skipped] == ["channel.swir_answer_rate"]


def test_a_plan_without_seed_or_rows_is_refused(fitted):
    _, _, profile = fitted
    with pytest.raises(ValueError):
        generate.generate(profile, {"n_rows": 10})
    with pytest.raises(ValueError):
        generate.generate(profile, {"seed": 1})
    with pytest.raises(ValueError):
        generate.generate(profile, {"seed": 1, "n_rows": 0})
