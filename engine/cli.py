"""Steps 10+12+13 (minimal) — one command runs intake through release.

Pipeline, in order, each stage appended to `engine.runlog.RunLog` in the
run directory and printed to stdout:

  source intake  sha256 of the source file; licence/attribution from the
                 sibling ``*-source.json`` manifest when present
  profile        `statistics.build` only — aggregate counts and proportions
  plan           `plan.propose` via `models.meta_cli.MuseCLIModel` default
                 (muse-spark-1.3, contributor on, effort max)
  repair         `repair.repair` with the caller-supplied --max-attempts
                 (required, no default); artifact bytes land in
                 ``<out>/<artifact_sha256>/`` with the validation report
                 JSON (`checks` report `to_json`) and the plan JSON
  review         step 12, rule 4: a human on the CLI types accept/modify/
                 reject plus a verbatim note; both are recorded exactly
                 as typed. A failing-checks run stops before review and
                 never ships; reject and modify write no release.
  release        step 13: on accept, a ZIP with the data CSV, validation
                 report, plan, seed, provenance and checksums; its path
                 and SHA-256 are printed.

No raw source rows ever reach a model: the planner and repairer see only
the statistics profile JSON, the current plan, and failed check results.
Same source + same plan + same seed regenerates identical data bytes
(the generator is seeded code; this layer writes its bytes untouched).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from engine import hashing
from engine.runlog import RunLog
from models.base import StructuredModel, Usage
from models.meta_cli import MuseCLIModel
from profiles.hylogger import calibration
from profiles.hylogger import plan as plan_module
from profiles.hylogger import reader, repair, statistics
from profiles.hylogger import source as source_module

EXIT_OK = 0
EXIT_NO_RELEASE = 2  # reject, or checks still failing after repairs
EXIT_MODIFY = 3      # human asked for changes; re-run to apply them

RELEASE_MEMBERS = ("data.csv", "validation-report.json", "plan.json",
                   "seed.txt", "provenance.md", "checksums.sha256")

# Fixed ZIP member timestamps so identical inputs give identical ZIP bytes.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assayer",
        description="Source scalars in, human-accepted synthetic data out.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the lifecycle end to end")
    run.add_argument("source",
                     help="frozen source scalar CSV "
                          "(e.g. data/sources/PDD446-scalars.csv)")
    run.add_argument("--max-attempts", required=True, type=int,
                     help="generate/check cycles to attempt, including the "
                          "first one. Required with no default: how many "
                          "repairs are worth paying for is unmeasured.")
    run.add_argument("--out", default="runs",
                     help="output root (default: runs); the run lands in "
                          "<out>/<artifact_sha256>/")
    run.add_argument("--seed-note", nargs="?", const="", default=None,
                     metavar="TEXT",
                     help="free-text note recorded with intake")
    return parser


def _sibling_manifest(source: Path) -> Path | None:
    """The ``*-source.json`` next to a source CSV, when present."""
    name = source.name
    if name.endswith("-scalars.csv"):
        candidate = source.with_name(
            name[: -len("-scalars.csv")] + "-source.json")
    elif source.suffix == ".csv":
        candidate = source.with_name(source.stem + "-source.json")
    else:
        return None
    return candidate if candidate.exists() else None


def _licence_from(manifest_path: Path | None) -> tuple[str, str, dict | None]:
    licence, attribution = source_module.LICENCE, source_module.ATTRIBUTION
    manifest_json: dict | None = None
    if manifest_path is not None:
        try:
            manifest_json = json.loads(manifest_path.read_text(
                encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest_json = None
        if isinstance(manifest_json, dict):
            licence = manifest_json.get("licence", licence)
            attribution = manifest_json.get("attribution", attribution)
    return licence, attribution, manifest_json


def _source_name(source: Path, manifest_json: dict | None) -> str:
    if isinstance(manifest_json, dict) and manifest_json.get("dataset_name"):
        return str(manifest_json["dataset_name"])
    if source.name.endswith("-scalars.csv"):
        return source.name[: -len("-scalars.csv")]
    return source.stem


def _format_usage(usage: Usage) -> str:
    return (f"model={usage.model} input={usage.input_tokens} "
            f"cached={usage.cached_tokens} output={usage.output_tokens} "
            f"reasoning={usage.reasoning_tokens} calls={usage.calls}")


def _provenance_bytes(source_name: str, source_sha256: str,
                      licence: str, attribution: str) -> tuple[bytes, str]:
    doc = Path(__file__).resolve().parent.parent / "docs" / "provenance.md"
    if doc.exists():
        return doc.read_bytes(), "docs/provenance.md copy"
    statement = (
        "# Provenance (release statement)\n\n"
        f"Source {source_name} sha256:{source_sha256} {licence}. "
        f"{attribution} Synthetic output CC0 1.0.\n"
    )
    return statement.encode("utf-8"), "statement"


def _read_line(prompt: str, stream: Any, out: Any) -> str:
    out.write(prompt)
    out.flush()
    line = stream.readline()
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def main(argv: list[str] | None = None, *,
         model: StructuredModel | None = None,
         stdin: Any = None, stdout: Any = None) -> int:
    """Entry point (``assayer = engine.cli:main``). Returns an exit code.

    `model` injects a `StructuredModel` for tests; production uses the
    default `MuseCLIModel`. `stdin`/`stdout` default to the real streams.
    """
    args = build_parser().parse_args(argv)
    if args.max_attempts is True or args.max_attempts < 1:
        print("error: --max-attempts must be a positive int", file=sys.stderr)
        return EXIT_NO_RELEASE
    return _run(args, model=model, stdin=stdin, stdout=stdout)


def _run(args: argparse.Namespace, *, model: StructuredModel | None,
         stdin: Any, stdout: Any) -> int:
    out = stdout if stdout is not None else sys.stdout
    inp = stdin if stdin is not None else sys.stdin

    def say(message: str) -> None:
        print(message, file=out)

    src_path = Path(args.source)
    if not src_path.exists():
        say(f"error: source not found: {src_path}")
        return EXIT_NO_RELEASE
    if model is None:
        model = MuseCLIModel()

    source_sha256 = hashing.sha256_file(src_path)
    manifest_path = _sibling_manifest(src_path)
    licence, attribution, manifest_json = _licence_from(manifest_path)
    source_name = _source_name(src_path, manifest_json)
    say(f"[source] sha256={source_sha256}")
    say(f"[source] licence={licence} "
        f"manifest={manifest_path if manifest_path else 'absent'}")
    say(f"[source] attribution={attribution}")
    if args.seed_note is not None:
        say(f"[source] seed-note={args.seed_note}")

    samples = reader.read(src_path)
    say(f"[profile] samples={len(samples)}")
    profile = statistics.build(samples, source_sha256, source_name)
    say(f"[profile] valid_fraction={profile.validity['valid_fraction']} "
        f"span_m={profile.geometry['span_m']}")

    completion = plan_module.propose(model, profile)
    initial_plan = dict(completion.value)
    say(f"[plan] seed={initial_plan['seed']} n_rows={initial_plan['n_rows']}")
    say(f"[plan] usage={_format_usage(completion.usage)} "
        f"meta={json.dumps(completion.meta, sort_keys=True)}")

    derived = calibration.load()
    result = repair.repair(profile, samples, derived, initial_plan, model,
                           max_attempts=args.max_attempts,
                           source_sha256=source_sha256)
    say(f"[repair] attempts={len(result.history)} verdict={result.verdict} "
        f"artifact={result.artifact_sha256}")

    run_dir = Path(args.out) / result.artifact_sha256
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data.csv").write_bytes(result.data)
    (run_dir / "validation-report.json").write_text(
        json.dumps(result.report.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (run_dir / "plan.json").write_text(
        json.dumps(result.plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{result.plan['seed']}\n",
                                      encoding="utf-8")
    provenance_bytes, provenance_kind = _provenance_bytes(
        source_name, source_sha256, licence, attribution)
    (run_dir / "provenance.md").write_bytes(provenance_bytes)
    member_bytes = {
        "data.csv": result.data,
        "validation-report.json": (run_dir / "validation-report.json")
        .read_bytes(),
        "plan.json": (run_dir / "plan.json").read_bytes(),
        "seed.txt": (run_dir / "seed.txt").read_bytes(),
        "provenance.md": provenance_bytes,
    }
    checksums = "".join(
        f"{hashing.sha256_bytes(member_bytes[name])}  {name}\n"
        for name in RELEASE_MEMBERS if name != "checksums.sha256")
    (run_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")
    member_bytes["checksums.sha256"] = checksums.encode("utf-8")
    say(f"[artifact] dir={run_dir}")

    log = RunLog(run_dir / "run.jsonl")
    log.append("source", "source_intake", path=str(src_path),
               sha256=source_sha256, licence=licence,
               attribution=attribution,
               manifest=str(manifest_path) if manifest_path else None,
               seed_note=args.seed_note)
    log.append("profile", "profile_built", source_sha256=source_sha256,
               source_name=source_name, samples=len(samples),
               profile=profile.to_json())
    log.append("plan", "plan_proposed", plan=initial_plan,
               usage=completion.usage.to_json(), meta=completion.meta)
    log.append("generate", "repair_completed", plan=result.plan,
               artifact_sha256=result.artifact_sha256,
               verdict=result.verdict,
               history=[a.to_json() for a in result.history])
    report = result.report
    log.append("validate", "validation_report", report=report.to_json())
    say(f"[validate] verdict={report.verdict} "
        f"passed={report.to_json()['passed']} "
        f"failed={len(report.failed)} skipped={len(report.skipped)}")
    log.verify()
    say(f"[log] {log.path} events={len(log)} chain ok")

    if report.verdict != "pass":
        for failed in report.failed:
            say(f"[validate:fail] {failed.name} observed={failed.observed} "
                f"expected={failed.expected} {failed.detail}")
        say("run stops here with no release: checks still fail after "
            "repairs (never ship a failing run)")
        return EXIT_NO_RELEASE

    say(f"Human review (step 12, rule 4). Artifact "
        f"{result.artifact_sha256[:12]} verdict pass "
        f"({len(report.results) - len(report.failed) - len(report.skipped)} "
        f"passed, {len(report.failed)} failed, {len(report.skipped)} "
        f"skipped). The model never approves its own output: type a "
        f"decision and a verbatim note.")
    decision = _read_line("decision [accept/modify/reject]: ", inp, out)
    note = _read_line("note (recorded verbatim; may be empty): ", inp, out)
    log.append("review", "human_review", decision=decision, note=note)
    say(f"[review] decision={decision} note={note}")

    normalized = decision.strip().lower()
    if normalized == "accept":
        zip_path = run_dir / "release.zip"
        with zipfile.ZipFile(zip_path, "w",
                             compression=zipfile.ZIP_DEFLATED) as handle:
            for name in RELEASE_MEMBERS:
                handle.writestr(zipfile.ZipInfo(name, date_time=_ZIP_DATE),
                                member_bytes[name])
        zip_sha256 = hashing.sha256_file(zip_path)
        log.append("release", "release_written", zip=str(zip_path),
                   sha256=zip_sha256, members=list(RELEASE_MEMBERS),
                   provenance=provenance_kind)
        say(f"[release] zip={zip_path} sha256={zip_sha256}")
        return EXIT_OK
    if normalized == "modify":
        say("modifications noted; no release written "
            "(re-run to apply them as a new run)")
        return EXIT_MODIFY
    if normalized == "reject":
        say("rejected by the human reviewer: no release written")
    else:
        say(f"unrecognized decision {decision!r}: treated as reject; "
            f"no release written")
    return EXIT_NO_RELEASE
