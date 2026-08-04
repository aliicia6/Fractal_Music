use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::discretization::{ContinuousRange, UniformSubdivision};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub max_terms: usize,
    pub tolerance: f64,
    pub divergence_limit: f64,
    pub attractor_window: Option<usize>,
    pub max_period: usize,
    pub min_repetitions: usize,
    pub transient_terms: usize,
    pub stability_ratio: f64,
    pub cluster_tolerance: Option<f64>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_terms: 1000,
            tolerance: 1e-7,
            divergence_limit: 1e6,
            attractor_window: None,
            max_period: 16,
            min_repetitions: 5,
            transient_terms: 100,
            stability_ratio: 0.98,
            cluster_tolerance: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisRangeConfig {
    pub minimum: f64,
    pub maximum: f64,
    pub steps: usize,
}

impl AxisRangeConfig {
    pub fn to_subdivision(&self) -> anyhow::Result<UniformSubdivision> {
        let range = ContinuousRange::new(self.minimum, self.maximum)?;
        UniformSubdivision::new(range, self.steps)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrenceConfig {
    pub name: String,
    pub parameters: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscretizationConfig {
    pub coordinate_system: String,
    pub ranges: std::collections::BTreeMap<String, AxisRangeConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorerConfig {
    pub recurrence: RecurrenceConfig,
    pub analysis: AnalysisConfig,
    pub discretization: DiscretizationConfig,
}

pub fn default_concept_config() -> ExplorerConfig {
    ExplorerConfig {
        recurrence: RecurrenceConfig {
            name: String::from("quadratic_complex"),
            parameters: std::collections::BTreeMap::from([
                (String::from("c_real"), -0.123),
                (String::from("c_imag"), 0.745),
                (String::from("z0_real"), 0.0),
                (String::from("z0_imag"), 0.0),
            ]),
        },
        analysis: AnalysisConfig::default(),
        discretization: DiscretizationConfig {
            coordinate_system: String::from("polar"),
            ranges: std::collections::BTreeMap::from([
                (
                    String::from("r"),
                    AxisRangeConfig {
                        minimum: 0.0,
                        maximum: 2.0,
                        steps: 12,
                    },
                ),
                (
                    String::from("theta"),
                    AxisRangeConfig {
                        minimum: 0.0,
                        maximum: TAU,
                        steps: 24,
                    },
                ),
            ]),
        },
    }
}
