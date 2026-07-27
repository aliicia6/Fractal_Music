"""Core tools for real and complex recurrence sequences."""

from .analyzer import analyze_recurrence, detect_attractors
from .config import (
    AnalysisConfig,
    AxisRangeConfig,
    DiscretizationConfig,
    ExplorerConfig,
    RecurrenceConfig,
    default_concept_config,
)
from .coordinates import (
    AngleProjectionCoordinates,
    ComplexCartesianCoordinates,
    ComplexPolarCoordinates,
    CoordinateSystem,
    ProjectionXCoordinates,
    ProjectionYCoordinates,
    RadiusProjectionCoordinates,
    RealLineCoordinates,
)
from .discretization import ContinuousRange, DiscretizationGrid, UniformSubdivision
from .models import Attractor, SequenceAnalysisResult, SequenceTerm
from .pipeline import (
    build_coordinate_system,
    build_grid,
    build_recurrence,
    run_exploration,
)
from .recurrence import Recurrence
from .transformations import (
    DiscreteAnalysisResult,
    DiscreteAttractor,
    DiscretePoint,
    DiscreteTerm,
    DiscreteTransformer,
    transform_attractors,
    transform_terms,
)

__all__ = [
    "AngleProjectionCoordinates",
    "AnalysisConfig",
    "Attractor",
    "AxisRangeConfig",
    "ComplexCartesianCoordinates",
    "ComplexPolarCoordinates",
    "ContinuousRange",
    "CoordinateSystem",
    "DiscreteAnalysisResult",
    "DiscreteAttractor",
    "DiscretePoint",
    "DiscreteTerm",
    "DiscreteTransformer",
    "DiscretizationConfig",
    "DiscretizationGrid",
    "ExplorerConfig",
    "ProjectionXCoordinates",
    "ProjectionYCoordinates",
    "RadiusProjectionCoordinates",
    "RealLineCoordinates",
    "Recurrence",
    "RecurrenceConfig",
    "SequenceAnalysisResult",
    "SequenceTerm",
    "UniformSubdivision",
    "analyze_recurrence",
    "build_coordinate_system",
    "build_grid",
    "build_recurrence",
    "default_concept_config",
    "detect_attractors",
    "run_exploration",
    "transform_attractors",
    "transform_terms",
]
