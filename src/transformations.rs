use std::sync::Arc;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::coordinates::CoordinateSystem;
use crate::discretization::DiscretizationGrid;
use crate::models::{Attractor, Number, SequenceAnalysisResult, SequenceTerm};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscretePoint {
    pub value: Number,
    pub coordinates: IndexMap<String, f64>,
    pub indices: Option<IndexMap<String, usize>>,
    pub discrete_coordinates: Option<IndexMap<String, f64>>,
    pub cell_id: Option<usize>,
    pub inside_domain: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscreteTerm {
    pub term: SequenceTerm,
    pub point: DiscretePoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscreteAttractor {
    pub attractor: Attractor,
    pub points: Vec<DiscretePoint>,
}

#[derive(Clone)]
pub struct DiscreteTransformer {
    pub coordinate_system: Arc<dyn CoordinateSystem>,
    pub grid: DiscretizationGrid,
}

impl DiscreteTransformer {
    pub fn new(coordinate_system: Arc<dyn CoordinateSystem>, grid: DiscretizationGrid) -> Result<Self> {
        let missing: Vec<String> = grid
            .axes()
            .into_iter()
            .filter(|axis| !coordinate_system.axes().contains(&axis.as_str()))
            .collect();
        if !missing.is_empty() {
            bail!(
                "coordinate system does not provide axes: {}",
                missing.join(", ")
            );
        }
        Ok(Self {
            coordinate_system,
            grid,
        })
    }

    pub fn transform_value(&self, value: Number) -> Result<DiscretePoint> {
        let coordinates = self.coordinate_system.coordinates_from(value)?;
        let indices = self.grid.indices_of(&coordinates)?;
        if let Some(indices) = indices {
            let discrete_coordinates = self.grid.centers_of(&indices)?;
            let cell_id = self.grid.cell_id(&indices)?;
            Ok(DiscretePoint {
                value,
                coordinates,
                indices: Some(indices),
                discrete_coordinates: Some(discrete_coordinates),
                cell_id: Some(cell_id),
                inside_domain: true,
            })
        } else {
            Ok(DiscretePoint {
                value,
                coordinates,
                indices: None,
                discrete_coordinates: None,
                cell_id: None,
                inside_domain: false,
            })
        }
    }

    pub fn transform_terms(&self, terms: &[SequenceTerm]) -> Result<Vec<DiscreteTerm>> {
        terms
            .iter()
            .map(|term| {
                Ok(DiscreteTerm {
                    term: term.clone(),
                    point: self.transform_value(term.value)?,
                })
            })
            .collect()
    }

    pub fn transform_attractor(&self, attractor: &Attractor) -> Result<DiscreteAttractor> {
        let points = attractor
            .values
            .iter()
            .map(|value| self.transform_value(*value))
            .collect::<Result<Vec<_>>>()?;
        Ok(DiscreteAttractor {
            attractor: attractor.clone(),
            points,
        })
    }

    pub fn transform_attractors(&self, attractors: &[Attractor]) -> Result<Vec<DiscreteAttractor>> {
        attractors
            .iter()
            .map(|attractor| self.transform_attractor(attractor))
            .collect()
    }
}

#[derive(Clone)]
pub struct DiscreteAnalysisResult {
    pub sequence_result: SequenceAnalysisResult,
    pub transformer: DiscreteTransformer,
    pub terms: Vec<DiscreteTerm>,
    pub attractors: Vec<DiscreteAttractor>,
    pub config: Option<crate::config::ExplorerConfig>,
}

impl DiscreteAnalysisResult {
    pub fn from_analysis(
        sequence_result: SequenceAnalysisResult,
        transformer: DiscreteTransformer,
        config: Option<crate::config::ExplorerConfig>,
    ) -> Result<Self> {
        let terms = transformer.transform_terms(&sequence_result.terms)?;
        let attractors = transformer.transform_attractors(&sequence_result.attractors)?;
        Ok(Self {
            sequence_result,
            transformer,
            terms,
            attractors,
            config,
        })
    }
}

pub fn transform_terms(terms: &[SequenceTerm], transformer: &DiscreteTransformer) -> Result<Vec<DiscreteTerm>> {
    transformer.transform_terms(terms)
}

pub fn transform_attractors(
    attractors: &[Attractor],
    transformer: &DiscreteTransformer,
) -> Result<Vec<DiscreteAttractor>> {
    transformer.transform_attractors(attractors)
}
