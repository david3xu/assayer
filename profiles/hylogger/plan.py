"""Stage 8 (plan) — ask the planner model for a generation plan.

The planner is `muse-spark-1.3` reached through
`models.meta_cli.MuseCLIModel` with contributor mode on by default
(product Spark calls use contributor mode; a caller may pass
`contributor=False` to `MuseCLIModel` to explicitly disable it). This module never
constructs that call with raw source data: the only source-derived content in
the prompt is the statistics `Profile` JSON (`statistics.Profile.to_json`) —
counts, proportions, run lengths and spacings. No raw rows, depths or observed
sequences ever appear in the prompt (rule: no raw rows reach a model).

The returned value is not trusted because the CLI is schema-guided. It is
validated here with `jsonschema` against `PLAN_SCHEMA`, and a nonconformant
value raises `PlanError` instead of flowing into `generate.generate` as a bad
plan.
"""
from __future__ import annotations

import json
from typing import Any

import jsonschema

from models.base import Completion, StructuredModel

from .statistics import Profile

# Strict schema for `muse exec --output-schema`: exactly the fields
# `generate.generate` requires, nothing else. `seed` may be any integer
# (it seeds `random.Random`); `n_rows` must be a positive integer.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seed": {"type": "integer"},
        "n_rows": {"type": "integer", "minimum": 1},
    },
    "required": ["seed", "n_rows"],
    "additionalProperties": False,
}

_INSTRUCTIONS = (
    "You are the planner for a synthetic drill-hole mineralogy dataset. "
    "Choose a generation plan for the seeded generator that will follow. "
    "Return only a JSON object matching the supplied schema: an integer "
    "\"seed\" for the generator's random number generator and a positive "
    "integer \"n_rows\" for how many rows to generate. Pick n_rows close to "
    "the source sample count reported in the profile statistics below so the "
    "synthetic hole is comparable in scale to the source. Do not invent any "
    "other fields.\n\n"
    "Profile statistics (the only source-derived information available):\n"
)


class PlanError(ValueError):
    """The planner returned a value that does not conform to PLAN_SCHEMA."""


def build_prompt(profile: Profile) -> str:
    """Prompt whose only source-derived content is the Profile JSON.

    The fixed instruction text above carries no source values. Everything
    derived from the source enters solely as `profile.to_json()`, which by
    construction contains only aggregate statistics.
    """
    return _INSTRUCTIONS + json.dumps(profile.to_json(), indent=2, sort_keys=True)


# Alias matching the verb used elsewhere for builders in this package.
build_plan_prompt = build_prompt


def validate_plan(value: Any) -> dict:
    """Return `value` if it conforms to PLAN_SCHEMA, else raise PlanError."""
    try:
        jsonschema.validate(instance=value, schema=PLAN_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise PlanError(
            f"planner returned a nonconformant plan: {exc.message} "
            f"(at {list(exc.absolute_path) or ['<root>']}); "
            f"plan must match PLAN_SCHEMA with integer 'seed' and positive "
            f"integer 'n_rows' and no additional properties"
        ) from exc
    if not isinstance(value, dict):  # jsonschema already enforces this
        raise PlanError(f"planner returned a nonconformant plan: {value!r} is not an object")
    return value


def propose(model: StructuredModel, profile: Profile) -> Completion:
    """Call `model` for a plan and validate it before anyone may use it.

    Returns the model's `Completion`, whose `value` is the validated plan
    dict and whose `usage` records what the call cost. Raises `PlanError`
    on nonconformance instead of returning a bad plan.
    """
    completion = model.complete(build_prompt(profile), PLAN_SCHEMA)
    validate_plan(completion.value)
    return completion


def propose_plan(model: StructuredModel, profile: Profile) -> dict:
    """Validated plan dict from `model` for `profile` (see `propose`)."""
    return propose(model, profile).value
