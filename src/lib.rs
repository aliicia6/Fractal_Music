pub mod analyzer;
pub mod audio;
pub mod config;
pub mod conjugacy;
pub mod coordinates;
pub mod discretization;
pub mod examples;
pub mod explorer_session;
pub mod mandelbrot_renderer;
pub mod models;
pub mod music;
pub mod pipeline;
pub mod recurrence;
pub mod transformations;
pub mod ui;

pub use mandelbrot_renderer::MandelbrotCallback;

pub use analyzer::{analyze_recurrence, classify_behavior, detect_attractor, detect_attractors};
pub use config::{
    AnalysisConfig, AxisRangeConfig, DiscretizationConfig, ExplorerConfig, RecurrenceConfig,
    default_concept_config,
};
pub use conjugacy::{
    LogisticParameters, QuadraticParameters, logistic_to_quadratic, quadratic_to_logistic,
};
pub use coordinates::{
    AngleProjectionCoordinates, ComplexCartesianCoordinates, ComplexPolarCoordinates,
    ProjectionXCoordinates, ProjectionYCoordinates, RadiusProjectionCoordinates,
    RealLineCoordinates,
};
pub use discretization::{ContinuousRange, DiscretizationGrid, UniformSubdivision};
pub use examples::{cosine_fixed_point, logistic_map, quadratic_complex_map};
pub use explorer_session::{
    ExplorationCase, ImportedSession, build_session_data, parse_session_data, read_session,
    write_results_csv, write_session,
};
pub use models::{
    Attractor, AttractorKind, BehaviorKind, Number, SequenceAnalysisResult, SequenceTerm,
};
pub use music::{
    MUSIC_CRITERIA, MUSIC_SEGMENTATIONS, SCALES, MusicMappingConfig, MusicalAttractor,
    StrudelPattern, generate_strudel_pattern,
};
pub use pipeline::{
    build_coordinate_system, build_grid, build_recurrence, run_exploration,
};
pub use recurrence::Recurrence;
pub use transformations::{
    DiscreteAnalysisResult, DiscreteAttractor, DiscretePoint, DiscreteTerm, DiscreteTransformer,
    transform_attractors, transform_terms,
};
