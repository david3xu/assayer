import hashlib
import inspect
import json
import random

import pytest

from models.base import Completion, Usage
from profiles.hylogger import calibration, plan as plan_module, repair, statistics
from profiles.hylogger.reader import Sample

# A one-row plan can only have a valid fraction of 0 or 1, both outside the
# derived band, so it fails deterministically whatever the seed. A full-size
# plan from the same profile passes (verified for the seeds used here).
BAD_PLAN = {"seed": 1, "n_rows": 1}
GOOD_PLAN = {"seed": 1, "n_rows": 4000}


def _hole(seed, p_valid, step, n=4000, start=100.0):
    rng = random.Random(seed)
    out, depth = [], start
    for _ in range(n):
        valid = rng.random() < p_valid
        out.append(Sample(depth, valid, "A" if valid else None,
                          "T" if valid else None))
        depth += step
    return out


@pytest.fixture(scope="module")
def fitted():
    source = _hole(seed=11, p_valid=0.90, step=0.0074)
    source_sha256 = hashlib.sha256(b"synthetic-source").hexdigest()
    profile = statistics.build(source, source_sha256, "SYNTH")
    derived = calibration.derive({
        "a": _hole(seed=21, p_valid=0.85, step=0.0074),
        "b": _hole(seed=22, p_valid=0.90, step=0.0080),
        "c": _hole(seed=23, p_valid=0.95, step=0.0076),
    })
    return profile, source, source_sha256, derived


class FakeModel:
    """A StructuredModel that never touches the network.

    Returns the queued plans in order and records every prompt and schema
    it was handed, so tests can count repair calls and inspect prompts.
    """

    name = "fake"

    def __init__(self, values):
        self._values = list(values)
        self.prompts = []
        self.schemas = []

    def complete(self, prompt, schema, *, system=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        value = self._values.pop(0)
        return Completion(value=value,
                          usage=Usage(model="fake", input_tokens=11,
                                      cached_tokens=2, output_tokens=4,
                                      reasoning_tokens=1, calls=1),
                          raw_text=json.dumps(value))


def _run(fitted, initial_plan, model, max_attempts):
    profile, source, source_sha256, derived = fitted
    return repair.repair(profile, source, derived, initial_plan, model,
                         max_attempts=max_attempts,
                         source_sha256=source_sha256)


def test_first_plan_fails_repaired_plan_passes(fitted):
    model = FakeModel([GOOD_PLAN])
    result = _run(fitted, BAD_PLAN, model, max_attempts=3)

    assert result.passed is True
    assert result.report.verdict == "pass"
    assert result.report.failed == []
    assert result.plan == GOOD_PLAN
    assert result.artifact_sha256 == result.report.artifact_sha256
    assert len(result.data) > 0

    # Two attempts: the caller's plan, then the model's revision. Exactly
    # one repair call happened, under the strict plan schema.
    assert model.prompts and len(model.prompts) == 1
    assert model.schemas == [plan_module.PLAN_SCHEMA]
    assert len(result.history) == 2
    first, second = result.history
    assert first.plan == BAD_PLAN and first.verdict == "fail"
    assert first.failed_checks == ["validity.valid_fraction"]
    assert first.usage is None
    assert second.plan == GOOD_PLAN and second.verdict == "pass"
    assert second.failed_checks == []
    assert second.artifact_sha256 == result.artifact_sha256
    assert second.usage is not None
    assert second.usage.input_tokens == 11 and second.usage.calls == 1


def test_exhaustion_with_one_attempt_makes_no_repair_call(fitted):
    model = FakeModel([GOOD_PLAN])
    result = _run(fitted, BAD_PLAN, model, max_attempts=1)

    # The failing verdict is returned as-is, never converted to a pass.
    assert result.passed is False
    assert result.report.verdict == "fail"
    assert [r.name for r in result.report.failed] == ["validity.valid_fraction"]
    assert result.plan == BAD_PLAN
    # The bound was honoured: one attempt, and the model was never asked.
    assert len(result.history) == 1
    assert result.history[0].verdict == "fail"
    assert model.prompts == []


def test_attempts_never_exceed_the_bound(fitted):
    model = FakeModel([BAD_PLAN, BAD_PLAN, GOOD_PLAN])
    result = _run(fitted, BAD_PLAN, model, max_attempts=2)

    assert result.report.verdict == "fail"
    assert len(result.history) == 2
    # One repair call produced attempt 2; no call was made past the bound,
    # so the queued passing plan was never consumed.
    assert len(model.prompts) == 1
    assert result.history[1].usage is not None


def test_repair_prompt_carries_failed_checks_plan_and_statistics_only(fitted):
    model = FakeModel([GOOD_PLAN])
    _run(fitted, BAD_PLAN, model, max_attempts=2)
    prompt, = model.prompts

    # The failed check is named, as CheckResult JSON.
    assert "validity.valid_fraction" in prompt
    # The current plan and the statistics profile are present.
    assert '"n_rows": 1' in prompt
    profile, _, _, _ = fitted
    assert json.dumps(profile.to_json(), indent=2, sort_keys=True) in prompt
    assert "SYNTH" in prompt
    # Only the failed checks were sent: a check that passed is absent.
    assert "schema.not_empty" not in prompt
    # No raw row content: no dataset CSV header and no per-row encoding of
    # a source sample, in either the synthetic or the source CSV shape.
    assert "depth_m,valid" not in prompt
    assert "100.00000,true,A,T" not in prompt
    assert "100.00000,1" not in prompt


def test_max_attempts_has_no_default(fitted):
    parameter = inspect.signature(repair.repair).parameters["max_attempts"]
    assert parameter.default is inspect.Parameter.empty
    profile, source, source_sha256, derived = fitted
    with pytest.raises(TypeError):
        repair.repair(profile, source, derived, BAD_PLAN, FakeModel([GOOD_PLAN]),
                      source_sha256=source_sha256)


def test_a_nonconformant_revision_is_rejected_not_tried(fitted):
    model = FakeModel([{"seed": 1, "n_rows": 10, "extra": 1}])
    with pytest.raises(plan_module.PlanError):
        _run(fitted, BAD_PLAN, model, max_attempts=3)
