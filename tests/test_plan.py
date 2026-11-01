import json

import pytest

from models import meta_cli as meta_cli_module
from models.base import Completion, Usage
from models.meta_cli import MuseCLIModel
from profiles.hylogger import plan as plan_module
from profiles.hylogger.statistics import Profile


def _profile() -> Profile:
    return Profile(
        source_sha256="abc123",
        source_name="FAKEHOLE",
        geometry={"samples": 120, "depth_start_m": 10.0, "depth_end_m": 20.0,
                  "span_m": 10.0, "spacing_m": {"median": 0.083}},
        validity={"valid": 100, "invalid": 20, "valid_fraction": 0.833333},
        channels={
            "swir_group": {"answered": 60, "answer_rate": 0.6, "marginals": {"A": 1.0}},
            "tir_group": {"answered": 90, "answer_rate": 0.9, "marginals": {"B": 1.0}},
        },
        co_occurrence={"pairs_observed": 50, "joint": {"A": {"B": 1.0}}},
    )


class FakeModel:
    """A StructuredModel that never touches the network."""

    name = "fake"

    def __init__(self, value):
        self._value = value
        self.prompts = []
        self.schemas = []

    def complete(self, prompt, schema, *, system=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return Completion(value=self._value,
                          usage=Usage(model="fake", input_tokens=3,
                                      output_tokens=1, calls=1),
                          raw_text=json.dumps(self._value))


def test_plan_schema_is_strict():
    assert plan_module.PLAN_SCHEMA["additionalProperties"] is False
    assert plan_module.PLAN_SCHEMA["required"] == ["seed", "n_rows"]
    assert plan_module.PLAN_SCHEMA["properties"]["seed"]["type"] == "integer"
    assert plan_module.PLAN_SCHEMA["properties"]["n_rows"]["minimum"] == 1


def test_conformant_plan_is_accepted():
    model = FakeModel({"seed": 7, "n_rows": 120})
    completion = plan_module.propose(model, _profile())
    assert completion.value == {"seed": 7, "n_rows": 120}
    assert completion.usage.input_tokens == 3
    assert plan_module.propose_plan(FakeModel({"seed": 7, "n_rows": 120}),
                                    _profile()) == {"seed": 7, "n_rows": 120}
    # The model was handed the strict schema, not an ad-hoc one.
    assert model.schemas == [plan_module.PLAN_SCHEMA]


@pytest.mark.parametrize("bad", [
    {"seed": 7},                                  # missing n_rows
    {"n_rows": 10},                               # missing seed
    {"seed": 7, "n_rows": 0},                     # n_rows not positive
    {"seed": 7, "n_rows": -3},
    {"seed": "7", "n_rows": 10},                  # seed not an integer
    {"seed": 7, "n_rows": 10.5},                  # n_rows not an integer
    {"seed": True, "n_rows": 10},                 # bool is not an integer plan
    {"seed": 7, "n_rows": 10, "extra": 1},        # additionalProperties: false
    "not-a-plan",
    None,
])
def test_nonconformant_plan_is_rejected(bad):
    with pytest.raises(plan_module.PlanError):
        plan_module.propose(FakeModel(bad), _profile())
    with pytest.raises(plan_module.PlanError):
        plan_module.validate_plan(bad)


def test_contributor_mode_on_by_default_and_explicitly_disableable():
    # Product Spark calls use contributor mode by default.
    assert MuseCLIModel(model="muse-spark-1.3").contributor is True
    assert MuseCLIModel().contributor is True
    # A caller can still explicitly disable it.
    assert MuseCLIModel(model="muse-spark-1.3", contributor=False).contributor is False
    assert MuseCLIModel(model="muse-spark-1.3", contributor=True).contributor is True


def _fake_cli_run(captured):
    def fake_run(argv, **kwargs):
        captured.append(argv)
        stdout = "\n".join([
            json.dumps({"payload_type": "run.usage", "payload": {
                "models": [{"model": "muse-spark-1.3", "input_tokens": 10,
                            "cached_tokens": 0, "output_tokens": 5,
                            "reasoning_tokens": 0, "calls": 1}],
                "completed_calls": 1}}),
            json.dumps({"payload_type": "run.terminal.completed", "payload": {
                "terminal": "completed",
                "text": json.dumps({"seed": 1, "n_rows": 5})}}),
        ])

        class Proc:
            returncode = 0
            stderr = ""
        proc = Proc()
        proc.stdout = stdout
        return proc
    return fake_run


def test_contributor_default_reaches_cli_flag_and_meta(monkeypatch):
    captured = []
    monkeypatch.setattr(meta_cli_module.subprocess, "run", _fake_cli_run(captured))
    completion = MuseCLIModel(model="muse-spark-1.3").complete(
        "prompt", plan_module.PLAN_SCHEMA)
    assert completion.value == {"seed": 1, "n_rows": 5}
    assert completion.meta["contributor"] is True
    argv = captured[0]
    assert argv[argv.index("--contributor-mode") + 1] == "on"


def test_contributor_explicit_off_reaches_cli_flag_and_meta(monkeypatch):
    captured = []
    monkeypatch.setattr(meta_cli_module.subprocess, "run", _fake_cli_run(captured))
    completion = MuseCLIModel(
        model="muse-spark-1.3", contributor=False).complete(
        "prompt", plan_module.PLAN_SCHEMA)
    assert completion.meta["contributor"] is False
    argv = captured[0]
    assert argv[argv.index("--contributor-mode") + 1] == "off"


def test_prompt_carries_statistics_and_no_raw_rows():
    profile = _profile()
    prompt = plan_module.build_prompt(profile)
    # The profile statistics are in the prompt, as JSON.
    assert json.dumps(profile.to_json(), indent=2, sort_keys=True) in prompt
    assert "FAKEHOLE" in prompt and "0.833333" in prompt
    # Nothing that looks like a raw source row is in the prompt: no CSV
    # header, no per-row depth/answer sequence.
    assert "depth_m,valid" not in prompt
    assert "Depth (m)" not in prompt
    for raw_row in ("10.00000,true,A,B", "10.08300,true,A,B"):
        assert raw_row not in prompt
