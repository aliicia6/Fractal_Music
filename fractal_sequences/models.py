"""Data models for sequence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Number = float | complex
AttractorKind = Literal["fixed_point", "cycle", "unknown"]
BehaviorKind = Literal[
    "fixed_point",
    "periodic",
    "divergent",
    "chaotic_or_unresolved",
    "insufficient_data",
    "calculation_error",
]


@dataclass(frozen=True)
class SequenceTerm:
    """One term in a real or complex recurrence sequence."""

    index: int
    value: Number


@dataclass(frozen=True)
class Attractor:
    """A detected fixed point or periodic cycle."""

    values: list[Number]
    kind: AttractorKind
    period: int
    tolerance: float
    start_index: int | None = None


@dataclass(frozen=True)
class SequenceAnalysisResult:
    """Full result returned by the recurrence analyzer."""

    terms: list[SequenceTerm] = field(default_factory=list)
    attractors: list[Attractor] = field(default_factory=list)
    diverged: bool = False
    reason: str = ""
    behavior: BehaviorKind = "insufficient_data"
