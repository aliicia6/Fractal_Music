"""High-level pipeline for reproducible sequence explorations."""

from __future__ import annotations

from .analyzer import analyze_recurrence
from .config import AnalysisConfig, DiscretizationConfig, ExplorerConfig, RecurrenceConfig
from .coordinates import (
    AngleProjectionCoordinates,
    ComplexCartesianCoordinates,
    ComplexPolarCoordinates,
    CoordinateSystem,
    ProjectionXCoordinates,
    ProjectionYCoordinates,
    RadiusProjectionCoordinates,
)
from .discretization import DiscretizationGrid
from .examples import cosine_fixed_point, logistic_map, quadratic_complex_map
from .recurrence import Recurrence
from .transformations import DiscreteAnalysisResult, DiscreteTransformer


def run_exploration(config: ExplorerConfig) -> DiscreteAnalysisResult:
    """Run recurrence analysis and discrete transformation from one config."""

    recurrence = build_recurrence(config.recurrence)
    coordinate_system = build_coordinate_system(config.discretization.coordinate_system)
    grid = build_grid(config.discretization)
    transformer = DiscreteTransformer(coordinate_system, grid)
    analysis = config.analysis

    result = analyze_recurrence(
        recurrence,
        max_terms=analysis.max_terms,
        tolerance=analysis.tolerance,
        divergence_limit=analysis.divergence_limit,
        attractor_window=_attractor_window(analysis),
        max_period=analysis.max_period,
        min_repetitions=analysis.min_repetitions,
        transient_terms=analysis.transient_terms,
        stability_ratio=analysis.stability_ratio,
        cluster_tolerance=analysis.cluster_tolerance,
    )

    return DiscreteAnalysisResult.from_analysis(
        sequence_result=result,
        transformer=transformer,
        config=config,
    )


def build_recurrence(config: RecurrenceConfig) -> Recurrence:
    params = config.parameters
    if config.name == "cosine":
        return cosine_fixed_point(x0=params.get("x0", 1.0))
    if config.name == "logistic":
        return logistic_map(
            r=params.get("r", 3.2),
            x0=params.get("x0", 0.2),
        )
    if config.name == "quadratic_complex":
        c = complex(params.get("c_real", -0.123), params.get("c_imag", 0.745))
        z0 = complex(params.get("z0_real", 0.0), params.get("z0_imag", 0.0))
        return quadratic_complex_map(c=c, z0=z0)
    raise ValueError(f"unknown recurrence config: {config.name}")


def build_coordinate_system(name: str) -> CoordinateSystem:
    if name == "x":
        return ProjectionXCoordinates()
    if name == "cartesian":
        return ComplexCartesianCoordinates()
    if name == "polar":
        return ComplexPolarCoordinates()
    if name == "y":
        return ProjectionYCoordinates()
    if name == "radius":
        return RadiusProjectionCoordinates()
    if name == "angle":
        return AngleProjectionCoordinates()
    raise ValueError(f"unknown coordinate system config: {name}")


def build_grid(config: DiscretizationConfig) -> DiscretizationGrid:
    coordinate_system = build_coordinate_system(config.coordinate_system)
    missing_axes = set(coordinate_system.axes) - set(config.ranges)
    if missing_axes:
        missing = ", ".join(sorted(missing_axes))
        raise ValueError(f"missing discretization ranges for axes: {missing}")

    return DiscretizationGrid(
        {
            axis: config.ranges[axis].to_subdivision()
            for axis in coordinate_system.axes
        }
    )


def _attractor_window(config: AnalysisConfig) -> int:
    if config.attractor_window is not None:
        return config.attractor_window
    return max(50, config.max_terms // 3)
