"""Stage 9 (repair) — revise the plan until the checks pass, or stop.

The repairer is `muse-spark-1.3` reached through
`models.meta_cli.MuseCLIModel` with contributor mode on by default, the
same product model role as the planner. This module never constructs that
model and never changes its defaults: the caller passes any
`StructuredModel`, and in production that is the default `MuseCLIModel`.
What the repairer may see is bounded exactly like the planner's input. A
repair prompt carries only three things: the failed `CheckResult` JSON, the
current plan, and the statistics `Profile` JSON. No raw rows, depths or
observed sequences ever appear in it (rule: no raw rows reach a model). The
source samples are used only locally, as the `checks.run` comparison input;
they are never serialised into a prompt.

The loop is deliberately mechanical:

  generate bytes with `generate.generate`, read them back with
  `dataset.read`, run `checks.run`; on pass, return; on fail, ask the model
  for a revised plan under `plan.PLAN_SCHEMA`, validate it with
  `plan.validate_plan`, and try again.

`max_attempts` is a required caller-supplied bound with no default in this
code. How many repairs are worth paying for is an unmeasured open decision,
so no value is guessed here. Attempts never exceed the bound. When the
bound is exhausted the last failing report is returned with verdict
``fail``: a failure is never converted to a pass, and the caller must gate
on `RepairResult.report.verdict` (or `.passed`) and ship `data` only on a
pass.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from engine.hashing import sha256_bytes
from models.base import StructuredModel, Usage

from . import checks, dataset, generate
from .checks import CheckReport, CheckResult
from .plan import PLAN_SCHEMA, validate_plan
from .reader import Sample
from .statistics import Profile

_REPAIR_INSTRUCTIONS = (
    "You are the repairer for a synthetic drill-hole mineralogy dataset. "
    "The generation plan below was tried and the deterministic checks "
    "listed below failed. Return only a revised JSON object matching the "
    "supplied schema: an integer \"seed\" and a positive integer "
    "\"n_rows\". Change the plan so the failed checks can pass — for "
    "example, a row count far from the source sample count in the profile "
    "statistics cannot reproduce the source proportions. Do not invent "
    "any other fields.\n\n"
    "Failed checks (CheckResult JSON):\n"
)


@dataclass(frozen=True)
class Attempt:
    """One generate/read/check cycle, recorded whether it passed or not."""

    attempt: int
    plan: dict
    artifact_sha256: str
    verdict: str
    failed_checks: list[str] = field(default_factory=list)
    # The model call that produced this attempt's plan. None for the first
    # attempt, whose plan was supplied by the caller, not by a repair call.
    usage: Usage | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "plan": self.plan,
            "artifact_sha256": self.artifact_sha256,
            "verdict": self.verdict,
            "failed_checks": list(self.failed_checks),
            "usage": self.usage.to_json() if self.usage is not None else None,
        }


@dataclass(frozen=True)
class RepairResult:
    """The outcome of the loop. Ship `data` only when `passed` is true."""

    plan: dict
    artifact_sha256: str
    data: bytes
    report: CheckReport
    history: list[Attempt] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.report.verdict

    @property
    def passed(self) -> bool:
        return self.report.verdict == "pass"

    def to_json(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "artifact_sha256": self.artifact_sha256,
            "verdict": self.verdict,
            "report": self.report.to_json(),
            "history": [a.to_json() for a in self.history],
        }


def build_repair_prompt(profile: Profile, current_plan: dict,
                        failed: Sequence[CheckResult]) -> str:
    """Prompt with exactly three payloads, and no raw source content.

    The failed `CheckResult` JSON, the current plan JSON, and the
    statistics `Profile` JSON. The fixed instruction text above carries no
    source values, and nothing derived from individual source rows is
    included: `failed` holds aggregate check outcomes, and the profile is
    aggregate statistics by construction.
    """
    failed_json = json.dumps([r.to_json() for r in failed],
                             indent=2, sort_keys=True)
    plan_json = json.dumps(current_plan, indent=2, sort_keys=True)
    profile_json = json.dumps(profile.to_json(), indent=2, sort_keys=True)
    return (
        _REPAIR_INSTRUCTIONS
        + failed_json
        + "\n\nCurrent plan:\n" + plan_json
        + "\n\nProfile statistics (the only source-derived information "
          "available):\n" + profile_json
    )


# Alias matching the verb used elsewhere for builders in this package.
build_prompt = build_repair_prompt


def _read_back(data: bytes) -> list[Sample]:
    """Read generated bytes back through `dataset.read`, as checks require.

    The checks must measure the bytes that were written and hashed, not an
    in-memory object, so the bytes go to a temporary file and return
    through the same reader a reviewer would use.
    """
    with tempfile.NamedTemporaryFile(
            "wb", suffix=".csv", delete=False) as fh:
        fh.write(data)
        path = Path(fh.name)
    try:
        return dataset.read(path)
    finally:
        path.unlink(missing_ok=True)


def repair(profile: Profile, source_samples: Sequence[Sample] | None,
           derived: dict, initial_plan: dict, model: StructuredModel, *,
           max_attempts: int, source_sha256: str = "") -> RepairResult:
    """Generate, check, and repair `initial_plan` at most `max_attempts` times.

    `initial_plan` must already conform to `plan.PLAN_SCHEMA`; it is
    re-validated here and a nonconformant value raises `PlanError` rather
    than flowing into the generator. Each revised plan the model returns
    is validated the same way, and a nonconformant revision raises
    `PlanError` instead of consuming an attempt as a bad plan.

    `max_attempts` counts generate/check cycles, including the first one:
    `max_attempts=1` tries the initial plan once and makes no model call.
    On exhaustion the returned result carries the last failing report with
    verdict ``fail`` and the full attempt history.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("max_attempts must be an int")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    current_plan = dict(validate_plan(initial_plan))

    history: list[Attempt] = []
    producing_usage: Usage | None = None
    result: RepairResult | None = None
    for attempt in range(1, max_attempts + 1):
        data = generate.generate(profile, current_plan)
        artifact_sha256 = sha256_bytes(data)
        samples = _read_back(data)
        report = checks.run(samples, artifact_sha256, derived,
                            source=source_samples,
                            source_sha256=source_sha256)
        history.append(Attempt(
            attempt=attempt,
            plan=dict(current_plan),
            artifact_sha256=artifact_sha256,
            verdict=report.verdict,
            failed_checks=[r.name for r in report.failed],
            usage=producing_usage,
        ))
        result = RepairResult(plan=dict(current_plan),
                              artifact_sha256=artifact_sha256, data=data,
                              report=report, history=history)
        if report.verdict == "pass":
            return result
        if attempt == max_attempts:
            break
        prompt = build_repair_prompt(profile, current_plan, report.failed)
        completion = model.complete(prompt, PLAN_SCHEMA)
        validate_plan(completion.value)
        producing_usage = completion.usage
        current_plan = dict(completion.value)
    assert result is not None  # max_attempts >= 1, so the loop ran once
    return result
