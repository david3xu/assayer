"""Loop 4 CLI end to end with a fake planner (no network).

Every test drives `engine.cli.main` against the real PDD446 source CSV
with a fake StructuredModel, a tmp --out dir, and simulated stdin.
"""
import hashlib
import io
import json
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from engine import cli as cli_module
from engine.runlog import RunLog
from models.base import Completion, Usage

SOURCE = (Path(__file__).resolve().parent.parent
          / "data" / "sources" / "PDD446-scalars.csv")

# Proven outcomes on PDD446 (see Loop 1-3): 4000 rows pass, one row fails
# validity.valid_fraction whatever the seed.
GOOD_PLAN = {"seed": 1, "n_rows": 4000}
BAD_PLAN = {"seed": 1, "n_rows": 1}

REQUIRED_MEMBERS = {"data.csv", "validation-report.json", "plan.json",
                    "seed.txt", "provenance.md", "checksums.sha256"}


class FakeModel:
    """A StructuredModel that never touches the network."""

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
                          usage=Usage(model="fake", input_tokens=5,
                                      output_tokens=1, calls=1),
                          raw_text=json.dumps(value))


def _run(argv, model, stdin_text=""):
    stdin = io.StringIO(stdin_text)
    stdout = io.StringIO()
    code = cli_module.main(argv, model=model, stdin=stdin, stdout=stdout)
    return code, stdout.getvalue()


def _only_run_dir(out):
    dirs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_accept_writes_release_zip_with_required_members(tmp_path):
    out = tmp_path / "runs"
    code, text = _run(["run", str(SOURCE), "--max-attempts", "1",
                       "--out", str(out)],
                      FakeModel([GOOD_PLAN]),
                      "accept\nlooks good: 4000 rows, ship it.\n")

    assert code == cli_module.EXIT_OK
    run_dir = _only_run_dir(out)
    data = (run_dir / "data.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == run_dir.name

    zip_path = run_dir / "release.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as handle:
        assert set(handle.namelist()) == REQUIRED_MEMBERS
        for name in REQUIRED_MEMBERS:
            assert handle.read(name) == (run_dir / name).read_bytes()
    assert f"[release] zip={zip_path}" in text
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() in text

    report = json.loads((run_dir / "validation-report.json").read_text())
    assert report["verdict"] == "pass"
    assert report["artifact_sha256"] == run_dir.name
    assert json.loads((run_dir / "plan.json").read_text()) == GOOD_PLAN
    assert (run_dir / "seed.txt").read_text() == "1\n"

    log = RunLog(run_dir / "run.jsonl")
    log.verify()
    review = log.events_of("human_review")
    assert len(review) == 1
    assert review[0].payload["decision"] == "accept"
    assert review[0].payload["note"] == "looks good: 4000 rows, ship it."
    intake = log.events_of("source_intake")[0].payload
    assert intake["sha256"] == (
        "9705c4b6494d34fea6392dd25483fdad952ef2cf4103305ab67196f5dc2349d7")
    assert intake["licence"] == "CC-BY-4.0"
    assert "Geological Survey of Western Australia" in intake["attribution"]
    assert [e.event_type for e in log] == [
        "source_intake", "profile_built", "plan_proposed",
        "repair_completed", "validation_report", "human_review",
        "release_written"]


def test_reject_writes_no_release_but_keeps_log(tmp_path):
    out = tmp_path / "runs"
    code, text = _run(["run", str(SOURCE), "--max-attempts", "1",
                       "--out", str(out)],
                      FakeModel([GOOD_PLAN]),
                      "reject\nnot convincing -- redo.\n")

    assert code == cli_module.EXIT_NO_RELEASE
    run_dir = _only_run_dir(out)
    assert not (run_dir / "release.zip").exists()
    assert (run_dir / "data.csv").exists()

    log = RunLog(run_dir / "run.jsonl")
    log.verify()
    review = log.events_of("human_review")
    assert len(review) == 1
    assert review[0].payload["decision"] == "reject"
    assert review[0].payload["note"] == "not convincing -- redo."
    assert log.events_of("release_written") == []
    assert "[release]" not in text


def test_modify_writes_no_release(tmp_path):
    out = tmp_path / "runs"
    code, _ = _run(["run", str(SOURCE), "--max-attempts", "1",
                    "--out", str(out)],
                   FakeModel([GOOD_PLAN]),
                   "modify\n  fewer rows please  \n")

    assert code == cli_module.EXIT_MODIFY
    run_dir = _only_run_dir(out)
    assert not (run_dir / "release.zip").exists()
    log = RunLog(run_dir / "run.jsonl")
    log.verify()
    # The note is verbatim: surrounding spaces are kept, not stripped.
    assert log.events_of("human_review")[0].payload["note"] == \
        "  fewer rows please  "


def test_failing_checks_exit_nonzero_with_no_prompt_and_no_release(tmp_path):
    out = tmp_path / "runs"
    code, text = _run(["run", str(SOURCE), "--max-attempts", "1",
                       "--out", str(out)],
                      FakeModel([BAD_PLAN]),
                      stdin_text="")

    assert code == cli_module.EXIT_NO_RELEASE
    assert "validity.valid_fraction" in text
    assert "decision [accept/modify/reject]" not in text
    run_dir = _only_run_dir(out)
    assert not (run_dir / "release.zip").exists()
    log = RunLog(run_dir / "run.jsonl")
    log.verify()
    assert log.events_of("human_review") == []
    assert log.events_of("validation_report")[0].payload["report"][
        "verdict"] == "fail"


def test_same_plan_same_seed_regenerates_identical_bytes(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        code, _ = _run(["run", str(SOURCE), "--max-attempts", "1",
                        "--out", str(out)],
                       FakeModel([dict(GOOD_PLAN)]),
                       "accept\nok\n")
        assert code == cli_module.EXIT_OK
    a = next(first.iterdir()) / "data.csv"
    b = next(second.iterdir()) / "data.csv"
    assert a.read_bytes() == b.read_bytes()
    assert a.parent.name == b.parent.name


def test_no_raw_source_rows_reach_the_model(tmp_path):
    model = FakeModel([GOOD_PLAN])
    code, _ = _run(["run", str(SOURCE), "--max-attempts", "1",
                    "--out", str(tmp_path)],
                   model,
                   "accept\nok\n")
    assert code == cli_module.EXIT_OK
    assert model.prompts
    blob = "\n".join(model.prompts)
    assert "depth_m,valid" not in blob
    assert "StartDepth" not in blob


def test_missing_source_is_nonzero(tmp_path):
    code, text = _run(["run", str(tmp_path / "gone.csv"),
                       "--max-attempts", "1",
                       "--out", str(tmp_path)],
                      FakeModel([GOOD_PLAN]))
    assert code == cli_module.EXIT_NO_RELEASE
    assert "not found" in text


def test_max_attempts_is_required():
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["run", str(SOURCE)], model=FakeModel([GOOD_PLAN]),
                        stdin=io.StringIO("accept\nok\n"),
                        stdout=io.StringIO())
    assert exc.value.code == 2


def test_console_entry_point_is_main():
    assert callable(cli_module.main)
    with redirect_stdout(io.StringIO()):
        assert cli_module.build_parser().parse_args(
            ["run", "x.csv", "--max-attempts", "3"]).max_attempts == 3
