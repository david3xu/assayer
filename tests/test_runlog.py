import json

import pytest

from engine.runlog import GENESIS, RunLog


def test_appends_chain_from_genesis(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    assert log.head == GENESIS
    a = log.append("intake", "source_hashed", sha256="abc")
    b = log.append("plan", "plan_accepted", n_rows=100)
    assert a.prev_hash == GENESIS
    assert b.prev_hash == a.hash
    assert len(log) == 2
    log.verify()


def test_reopening_recovers_the_chain(tmp_path):
    path = tmp_path / "run.jsonl"
    first = RunLog(path)
    first.append("intake", "source_hashed", sha256="abc")
    reopened = RunLog(path)
    assert len(reopened) == 1
    assert reopened.head == first.head
    reopened.append("plan", "plan_accepted", n_rows=1)
    reopened.verify()


def test_edited_payload_fails_verification(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path)
    log.append("plan", "plan_accepted", n_rows=100)
    log.append("generate", "artifact_written", sha256="def")

    lines = path.read_text().splitlines()
    doctored = json.loads(lines[0])
    doctored["payload"]["n_rows"] = 999          # a claim quietly changed
    lines[0] = json.dumps(doctored)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="does not match its hash"):
        RunLog(path).verify()


def test_truncating_the_middle_fails_verification(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path)
    for i in range(3):
        log.append("validate", "check_ran", check_id=f"c{i}")
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(ValueError):
        RunLog(path).verify()


def test_events_of_filters_by_type(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    log.append("validate", "check_ran", check_id="a")
    log.append("validate", "check_ran", check_id="b")
    log.append("review", "accepted", by="human")
    assert [e.payload["check_id"] for e in log.events_of("check_ran")] == ["a", "b"]
