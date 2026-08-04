use std::sync::Arc;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use num_complex::Complex64;

use crate::analyzer::analyze_recurrence;
use crate::config::{AnalysisConfig, DiscretizationConfig, ExplorerConfig, RecurrenceConfig};
use crate::coordinates::{
    AngleProjectionCoordinates, ComplexCartesianCoordinates, ComplexPolarCoordinates,
    CoordinateSystem, ProjectionXCoordinates, ProjectionYCoordinates, RadiusProjectionCoordinates,
};
use crate::discretization::DiscretizationGrid;
use crate::examples::{cosine_fixed_point, logistic_map, quadratic_complex_map};
use crate::recurrence::Recurrence;
use crate::transformations::{DiscreteAnalysisResult, DiscreteTransformer};

pub fn run_exploration(config: &ExplorerConfig) -> Result<DiscreteAnalysisResult> {
    let recurrence = build_recurrence(&config.recurrence)?;
    let coordinate_system = build_coordinate_system(&config.discretization.coordinate_system)?;
    let grid = build_grid(&config.discretization)?;
    let transformer = DiscreteTransformer::new(coordinate_system, grid)?;
    let analysis = &config.analysis;
    let result = analyze_recurrence(
        &recurrence,
        analysis.max_terms,
        analysis.tolerance,
        analysis.divergence_limit,
        attractor_window(analysis),
        analysis.max_period,
        analysis.min_repetitions,
        analysis.transient_terms,
        analysis.stability_ratio,
        analysis.cluster_tolerance,
    )?;
    DiscreteAnalysisResult::from_analysis(result, transformer, Some(config.clone()))
}

pub fn build_recurrence(config: &RecurrenceConfig) -> Result<Recurrence> {
    let params = &config.parameters;
    match config.name.as_str() {
        "cosine" => Ok(cosine_fixed_point(*params.get("x0").unwrap_or(&1.0))),
        "logistic" => Ok(logistic_map(
            *params.get("r").unwrap_or(&3.2),
            *params.get("x0").unwrap_or(&0.2),
        )),
        "quadratic_complex" => {
            let c = Complex64::new(
                *params.get("c_real").unwrap_or(&-0.123),
                *params.get("c_imag").unwrap_or(&0.745),
            );
            let z0 = Complex64::new(
                *params.get("z0_real").unwrap_or(&0.0),
                *params.get("z0_imag").unwrap_or(&0.0),
            );
            Ok(quadratic_complex_map(c, z0))
        }
        _ => bail!("unknown recurrence config: {}", config.name),
    }
}

pub fn build_coordinate_system(name: &str) -> Result<Arc<dyn CoordinateSystem>> {
    match name {
        "x" => Ok(Arc::new(ProjectionXCoordinates)),
        "cartesian" => Ok(Arc::new(ComplexCartesianCoordinates)),
        "polar" => Ok(Arc::new(ComplexPolarCoordinates)),
        "y" => Ok(Arc::new(ProjectionYCoordinates)),
        "radius" => Ok(Arc::new(RadiusProjectionCoordinates)),
        "angle" => Ok(Arc::new(AngleProjectionCoordinates)),
        _ => bail!("unknown coordinate system config: {name}"),
    }
}

pub fn build_grid(config: &DiscretizationConfig) -> Result<DiscretizationGrid> {
    let coordinate_system = build_coordinate_system(&config.coordinate_system)?;
    let required_axes = coordinate_system.axes();
    for axis in required_axes {
        if !config.ranges.contains_key(*axis) {
            bail!("missing discretization ranges for axes: {axis}");
        }
    }
    let mut subdivisions = IndexMap::new();
    for axis in required_axes {
        let range_config = config
            .ranges
            .get(*axis)
            .ok_or_else(|| anyhow::anyhow!("missing discretization range: {axis}"))?;
        subdivisions.insert(String::from(*axis), range_config.to_subdivision()?);
    }
    DiscretizationGrid::new(subdivisions)
}

fn attractor_window(config: &AnalysisConfig) -> usize {
    config.attractor_window.unwrap_or((config.max_terms / 3).max(50))
}
