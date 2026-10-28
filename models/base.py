"""The contract every model adapter meets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Usage:
    """Token accounting for one call, recorded so the report can cost a run."""
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "calls": self.calls,
        }


@dataclass(frozen=True)
class Completion:
    """A structured answer plus what it cost to get it."""
    value: Any
    usage: Usage
    raw_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class StructuredModel(Protocol):
    """A model that returns JSON conforming to a supplied schema.

    The plan and repair stages depend on this being schema-guaranteed rather
    than best-effort: a malformed plan must fail loudly, not produce data.
    """

    name: str

    def complete(self, prompt: str, schema: dict[str, Any], *,
                 system: str | None = None) -> Completion:
        ...


class ModelError(RuntimeError):
    """The model did not return a usable answer."""
