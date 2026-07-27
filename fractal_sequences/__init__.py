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
from .conjugacy import (
    LogisticParameters,
    QuadraticParameters,
    logistic_to_quadratic,
    quadratic_to_logistic,
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
from .music import (
    MUSIC_CRITERIA,
    MUSIC_SEGMENTATIONS,
    SCALES,
    MusicMappingConfig,
    MusicalAttractor,
    StrudelPattern,
    generate_strudel_pattern,
)
from .pipeline import (
    build_coordinate_system,
    build_grid,
    build_recurrence,
    run_exploration,
)
from .recurrence import Recurrence
from .explorer_session import (
    ExplorationCase,
    ImportedSession,
    build_session_data,
    parse_session_data,
    read_session,
    write_results_csv,
    write_session,
)
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
    "ExplorationCase",
    "ImportedSession",
    "LogisticParameters",
    "MUSIC_CRITERIA",
    "MUSIC_SEGMENTATIONS",
    "MusicMappingConfig",
    "MusicalAttractor",
    "ProjectionXCoordinates",
    "ProjectionYCoordinates",
    "RadiusProjectionCoordinates",
    "RealLineCoordinates",
    "Recurrence",
    "RecurrenceConfig",
    "QuadraticParameters",
    "SequenceAnalysisResult",
    "SequenceTerm",
    "SCALES",
    "StrudelPattern",
    "UniformSubdivision",
    "analyze_recurrence",
    "build_session_data",
    "build_coordinate_system",
    "build_grid",
    "build_recurrence",
    "default_concept_config",
    "detect_attractors",
    "generate_strudel_pattern",
    "logistic_to_quadratic",
    "parse_session_data",
    "quadratic_to_logistic",
    "read_session",
    "run_exploration",
    "transform_attractors",
    "transform_terms",
    "write_results_csv",
    "write_session",
]
