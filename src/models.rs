use num_complex::Complex64;
use serde::{Deserialize, Serialize};

pub type Number = Complex64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttractorKind {
    FixedPoint,
    Cycle,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorKind {
    FixedPoint,
    Periodic,
    Divergent,
    ChaoticOrUnresolved,
    InsufficientData,
    CalculationError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceTerm {
    pub index: usize,
    pub value: Number,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attractor {
    pub values: Vec<Number>,
    pub kind: AttractorKind,
    pub period: usize,
    pub tolerance: f64,
    pub start_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceAnalysisResult {
    pub terms: Vec<SequenceTerm>,
    pub attractors: Vec<Attractor>,
    pub diverged: bool,
    pub reason: String,
    pub behavior: BehaviorKind,
}
