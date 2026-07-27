"""Uniform discretization of bounded coordinate domains."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousRange:
    """Closed bounded interval [minimum, maximum]."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.maximum <= self.minimum:
            raise ValueError("maximum must be greater than minimum")

    @property
    def length(self) -> float:
        return self.maximum - self.minimum

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class UniformSubdivision:
    """Uniform subdivision of a continuous range."""

    range: ContinuousRange
    steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")

    @property
    def step_size(self) -> float:
        return self.range.length / self.steps

    def edges(self) -> list[float]:
        return [
            self.range.minimum + index * self.step_size
            for index in range(self.steps + 1)
        ]

    def centers(self) -> list[float]:
        half_step = self.step_size / 2
        return [
            self.range.minimum + half_step + index * self.step_size
            for index in range(self.steps)
        ]

    def index_of(self, value: float) -> int | None:
        if not self.range.contains(value):
            return None
        if value == self.range.maximum:
            return self.steps - 1
        return int((value - self.range.minimum) / self.step_size)

    def center_of(self, index: int) -> float:
        if index < 0 or index >= self.steps:
            raise IndexError("subdivision index out of range")
        return self.range.minimum + (index + 0.5) * self.step_size


@dataclass(frozen=True)
class DiscretizationGrid:
    """Multidimensional uniform discretization by named axis."""

    subdivisions: dict[str, UniformSubdivision]

    def __post_init__(self) -> None:
        if not self.subdivisions:
            raise ValueError("grid requires at least one subdivision")

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.subdivisions.keys())

    def indices_of(self, coordinates: dict[str, float]) -> dict[str, int] | None:
        indices: dict[str, int] = {}
        for axis, subdivision in self.subdivisions.items():
            if axis not in coordinates:
                raise KeyError(f"missing coordinate axis: {axis}")

            index = subdivision.index_of(coordinates[axis])
            if index is None:
                return None
            indices[axis] = index

        return indices

    def centers_of(self, indices: dict[str, int]) -> dict[str, float]:
        return {
            axis: subdivision.center_of(indices[axis])
            for axis, subdivision in self.subdivisions.items()
        }

    def cell_id(self, indices: dict[str, int]) -> int:
        cell_id = 0
        multiplier = 1
        for axis, subdivision in reversed(tuple(self.subdivisions.items())):
            cell_id += indices[axis] * multiplier
            multiplier *= subdivision.steps
        return cell_id
