"""The append-only run log.

Rule 3 of the thesis is that deterministic checks prove quality, and a proof is
only worth as much as the record of it. Every stage appends one event here and
nothing ever rewrites one. Each event carries the hash of its predecessor, so a
truncated or edited log fails `verify()` rather than passing quietly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .hashing import canonical_json, sha256_bytes

GENESIS = "0" * 64


@dataclass(frozen=True)
class Event:
    seq: int
    stage: str
    event_type: str
    recorded_at: float
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "stage": self.stage,
            "event_type": self.event_type,
            "recorded_at": self.recorded_at,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


def _digest(seq: int, stage: str, event_type: str, recorded_at: float,
            payload: dict[str, Any], prev_hash: str) -> str:
    return sha256_bytes(canonical_json({
        "seq": seq,
        "stage": stage,
        "event_type": event_type,
        "recorded_at": recorded_at,
        "payload": payload,
        "prev_hash": prev_hash,
    }))


class RunLog:
    """A hash-chained JSONL log. Open it, append to it, never edit it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = []
        if self.path.exists():
            self._events = list(_read(self.path))

    @property
    def head(self) -> str:
        return self._events[-1].hash if self._events else GENESIS

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def append(self, stage: str, event_type: str, **payload: Any) -> Event:
        seq = len(self._events)
        recorded_at = time.time()
        prev = self.head
        event = Event(
            seq=seq,
            stage=stage,
            event_type=event_type,
            recorded_at=recorded_at,
            payload=payload,
            prev_hash=prev,
            hash=_digest(seq, stage, event_type, recorded_at, payload, prev),
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
        self._events.append(event)
        return event

    def events_of(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]

    def verify(self) -> None:
        """Raise if the chain has been edited, reordered or truncated mid-file."""
        prev = GENESIS
        for i, e in enumerate(self._events):
            if e.seq != i:
                raise ValueError(f"event {i} carries seq {e.seq}")
            if e.prev_hash != prev:
                raise ValueError(f"event {i} does not follow its predecessor")
            expected = _digest(e.seq, e.stage, e.event_type, e.recorded_at,
                               e.payload, e.prev_hash)
            if e.hash != expected:
                raise ValueError(f"event {i} payload does not match its hash")
            prev = e.hash


def _read(path: Path) -> Iterator[Event]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Event(
                seq=d["seq"],
                stage=d["stage"],
                event_type=d["event_type"],
                recorded_at=d["recorded_at"],
                payload=d["payload"],
                prev_hash=d["prev_hash"],
                hash=d["hash"],
            )
