"""Meta models through the Muse Code CLI.

Why the CLI rather than the HTTP API: `muse exec --output-schema FILE` is
schema-guaranteed structured output on the meta provider, and it authenticates
through the account login, so it needs no API key. The final answer arrives as a
`run.terminal.completed` event on the `--json` stream and token counts as
`run.usage`, which is what `Usage` is built from.

The harness carries real overhead — about 18.5k input tokens per call under
`--preset empty`, against roughly 30k on the default preset — so the lean flags
below are deliberate, not incidental.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import Completion, ModelError, StructuredModel, Usage

DEFAULT_MODEL = "muse-spark-1.3"

# Trimming the agent harness down to what a single structured call needs.
LEAN_FLAGS = [
    "--preset", "empty",
    "--disable-web-tools",
    "--disable-reminders",
    "--no-foreign-personal-context",
    "--disable-request-user-input",
    "--no-session-log",
]


class MuseCLIModel(StructuredModel):
    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str = "max",
                 contributor: bool = True, timeout: int = 600,
                 binary: str = "muse"):
        self.name = model
        self.effort = effort
        self.contributor = contributor
        self.timeout = timeout
        self.binary = binary

    def complete(self, prompt: str, schema: dict[str, Any], *,
                 system: str | None = None) -> Completion:
        text = prompt if system is None else f"{system}\n\n{prompt}"
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            argv = [
                self.binary, "exec", "--json",
                "--model", self.name,
                "--effort", self.effort,
                "--output-schema", str(schema_path),
                "--contributor-mode", "on" if self.contributor else "off",
                *LEAN_FLAGS,
                text,
            ]
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=self.timeout, cwd=tmp,
            )

        if proc.returncode != 0:
            raise ModelError(
                f"{self.binary} exec exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )

        answer, usage, terminal = _parse_events(proc.stdout, self.name)
        if terminal != "completed":
            raise ModelError(f"run ended as {terminal!r}, not 'completed'")
        if answer is None:
            raise ModelError("no run.terminal.completed event in the stream")

        try:
            value = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise ModelError(f"answer was not JSON: {exc}") from exc

        return Completion(value=value, usage=usage, raw_text=answer,
                          meta={"contributor": self.contributor,
                                "effort": self.effort})


def _parse_events(stdout: str, model: str) -> tuple[str | None, Usage, str]:
    answer: str | None = None
    terminal = "unknown"
    usage = Usage(model=model)
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # the CLI also prints human-readable notices
        payload_type = event.get("payload_type")
        payload = event.get("payload") or {}
        if payload_type == "run.terminal.completed":
            terminal = payload.get("terminal", "unknown")
            answer = payload.get("text")
        elif payload_type == "run.usage":
            usage = _usage_from(payload, model)
    return answer, usage, terminal


def _usage_from(payload: dict[str, Any], model: str) -> Usage:
    models = payload.get("models") or []
    row = next((m for m in models if m.get("model") == model), None)
    row = row or (models[0] if models else {})
    return Usage(
        model=row.get("model", model),
        input_tokens=row.get("input_tokens", 0),
        cached_tokens=row.get("cached_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
        reasoning_tokens=row.get("reasoning_tokens", 0),
        calls=payload.get("completed_calls", row.get("calls", 0)),
    )
