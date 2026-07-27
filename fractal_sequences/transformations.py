"""Transform sequence values and attractors into discrete domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .coordinates import CoordinateMap, CoordinateSystem
from .discretization import DiscretizationGrid
from .models import Attractor, Number, SequenceAnalysisResult, SequenceTerm


@dataclass(frozen=True)
class DiscretePoint:
    """Result of projecting and discretizing a single value."""

    value: Number
    coordinates: CoordinateMap
    indices: dict[str, int] | None
    discrete_coordinates: CoordinateMap | None
    cell_id: int | None
    inside_domain: bool


@dataclass(frozen=True)
class DiscreteTerm:
    """A sequence term plus its discrete projection."""

    term: SequenceTerm
    point: DiscretePoint


@dataclass(frozen=True)
class DiscreteAttractor:
    """An attractor plus the discrete projection of its values."""

    attractor: Attractor
    points: list[DiscretePoint]


@dataclass(frozen=True)
class DiscreteAnalysisResult:
    """Sequence analysis and its discrete representation."""

    sequence_result: SequenceAnalysisResult
    transformer: "DiscreteTransformer"
    terms: list[DiscreteTerm]
    attractors: list[DiscreteAttractor]
    config: Any | None = None

    @classmethod
    def from_analysis(
        cls,
        sequence_result: SequenceAnalysisResult,
        transformer: "DiscreteTransformer",
        config: Any | None = None,
    ) -> "DiscreteAnalysisResult":
        return cls(
            sequence_result=sequence_result,
            transformer=transformer,
            terms=transformer.transform_terms(sequence_result.terms),
            attractors=transformer.transform_attractors(sequence_result.attractors),
            config=config,
        )


class DiscreteTransformer:
    """Project values into coordinates and quantize them on a uniform grid."""

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        grid: DiscretizationGrid,
    ) -> None:
        missing_axes = set(grid.axes) - set(coordinate_system.axes)
        if missing_axes:
            missing = ", ".join(sorted(missing_axes))
            raise ValueError(f"coordinate system does not provide axes: {missing}")

        self.coordinate_system = coordinate_system
        self.grid = grid

    def transform_value(self, value: Number) -> DiscretePoint:
        coordinates = self.coordinate_system.coordinates_from(value)
        indices = self.grid.indices_of(coordinates)

        if indices is None:
            return DiscretePoint(
                value=value,
                coordinates=coordinates,
                indices=None,
                discrete_coordinates=None,
                cell_id=None,
                inside_domain=False,
            )

        return DiscretePoint(
            value=value,
            coordinates=coordinates,
            indices=indices,
            discrete_coordinates=self.grid.centers_of(indices),
            cell_id=self.grid.cell_id(indices),
            inside_domain=True,
        )

    def transform_terms(self, terms: list[SequenceTerm]) -> list[DiscreteTerm]:
        return [
            DiscreteTerm(term=term, point=self.transform_value(term.value))
            for term in terms
        ]

    def transform_attractor(self, attractor: Attractor) -> DiscreteAttractor:
        return DiscreteAttractor(
            attractor=attractor,
            points=[self.transform_value(value) for value in attractor.values],
        )

    def transform_attractors(
        self,
        attractors: list[Attractor],
    ) -> list[DiscreteAttractor]:
        return [self.transform_attractor(attractor) for attractor in attractors]


def transform_terms(
    terms: list[SequenceTerm],
    transformer: DiscreteTransformer,
) -> list[DiscreteTerm]:
    return transformer.transform_terms(terms)


def transform_attractors(
    attractors: list[Attractor],
    transformer: DiscreteTransformer,
) -> list[DiscreteAttractor]:
    return transformer.transform_attractors(attractors)
