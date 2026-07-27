"""Reusable configuration objects for reproducible explorations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .discretization import ContinuousRange, UniformSubdivision


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters used to calculate and classify a recurrence."""

    max_terms: int = 1000
    tolerance: float = 1e-7
    divergence_limit: float = 1e6
    attractor_window: int | None = None
    max_period: int = 16
    min_repetitions: int = 5
    transient_terms: int = 100
    stability_ratio: float = 0.98
    cluster_tolerance: float | None = None


@dataclass(frozen=True)
class AxisRangeConfig:
    """Bounded axis range plus its uniform subdivision count."""

    minimum: float
    maximum: float
    steps: int

    def to_subdivision(self) -> UniformSubdivision:
        return UniformSubdivision(
            ContinuousRange(self.minimum, self.maximum),
            steps=self.steps,
        )


@dataclass(frozen=True)
class RecurrenceConfig:
    """Named recurrence plus numeric parameters."""

    name: str
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscretizationConfig:
    """Coordinate system name plus axis ranges."""

    coordinate_system: str
    ranges: dict[str, AxisRangeConfig]


@dataclass(frozen=True)
class ExplorerConfig:
    """Complete reproducible configuration for one exploration."""

    recurrence: RecurrenceConfig
    analysis: AnalysisConfig
    discretization: DiscretizationConfig


def default_concept_config() -> ExplorerConfig:
    """Main concept case: complex quadratic map in polar discretization."""

    return ExplorerConfig(
        recurrence=RecurrenceConfig(
            name="quadratic_complex",
            parameters={
                "c_real": -0.123,
                "c_imag": 0.745,
                "z0_real": 0.0,
                "z0_imag": 0.0,
            },
        ),
        analysis=AnalysisConfig(
            max_terms=1000,
            tolerance=1e-7,
            max_period=16,
            transient_terms=100,
            stability_ratio=0.98,
        ),
        discretization=DiscretizationConfig(
            coordinate_system="polar",
            ranges={
                "r": AxisRangeConfig(0.0, 2.0, 12),
                "theta": AxisRangeConfig(0.0, 2 * math.pi, 24),
            },
        ),
    )
