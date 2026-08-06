use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{AnalysisConfig, DiscretizationConfig, ExplorerConfig, RecurrenceConfig};
use crate::models::{Attractor, SequenceTerm};
use crate::transformations::{DiscreteAnalysisResult, DiscretePoint};

pub const SESSION_SCHEMA: &str = "fractal_sequences.visual_session";
pub const SESSION_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationCase {
    pub identifier: String,
    pub label: String,
    pub recurrence: RecurrenceConfig,
    pub source: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedSession {
    pub config: ExplorerConfig,
    pub cases: Vec<ExplorationCase>,
    pub active_case_id: Option<String>,
    pub table_mode: String,
    pub map_limits: BTreeMap<String, (f64, f64, f64, f64)>,
    pub music: BTreeMap<String, String>,
}

pub fn config_to_data(config: &ExplorerConfig) -> Value {
    json!({
        "recurrence": {
            "name": config.recurrence.name,
            "parameters": config.recurrence.parameters,
        },
        "analysis": config.analysis,
        "discretization": {
            "coordinate_system": config.discretization.coordinate_system,
            "ranges": config.discretization.ranges,
        }
    })
}

pub fn config_from_data(data: &Value) -> Result<ExplorerConfig> {
    let object = data
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Configuracion de sesion invalida"))?;
    let recurrence_data = object
        .get("recurrence")
        .ok_or_else(|| anyhow::anyhow!("Configuracion de sesion incompleta"))?;
    let analysis_data = object
        .get("analysis")
        .ok_or_else(|| anyhow::anyhow!("Configuracion de sesion incompleta"))?;
    let discretization_data = object
        .get("discretization")
        .ok_or_else(|| anyhow::anyhow!("Configuracion de sesion incompleta"))?;

    let recurrence = serde_json::from_value::<RecurrenceConfig>(recurrence_data.clone())?;
    let analysis = serde_json::from_value::<AnalysisConfig>(analysis_data.clone())?;
    let discretization = serde_json::from_value::<DiscretizationConfig>(discretization_data.clone())?;
    Ok(ExplorerConfig {
        recurrence,
        analysis,
        discretization,
    })
}

pub fn case_to_data(case: &ExplorationCase) -> Value {
    json!({
        "identifier": case.identifier,
        "label": case.label,
        "recurrence": case.recurrence,
        "source": case.source,
        "metadata": case.metadata,
    })
}

pub fn case_from_data(data: &Value) -> Result<ExplorationCase> {
    let case = serde_json::from_value::<ExplorationCase>(data.clone())?;
    if case.identifier.trim().is_empty() {
        bail!("Caso de sesion sin identificador");
    }
    Ok(case)
}

pub fn result_to_data(result: &DiscreteAnalysisResult) -> Value {
    json!({
        "config": result.config.as_ref().map(config_to_data),
        "behavior": result.sequence_result.behavior,
        "reason": result.sequence_result.reason,
        "diverged": result.sequence_result.diverged,
        "terms": result.sequence_result.terms.iter().map(term_to_data).collect::<Vec<_>>(),
        "attractors": result.sequence_result.attractors.iter().map(attractor_to_data).collect::<Vec<_>>(),
        "discrete_terms": result.terms.iter().map(|item| {
            json!({
                "index": item.term.index,
                "point": point_to_data(&item.point),
            })
        }).collect::<Vec<_>>(),
        "discrete_attractors": result.attractors.iter().map(|item| {
            json!({
                "kind": item.attractor.kind,
                "period": item.attractor.period,
                "points": item.points.iter().map(point_to_data).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
        "coordinate_system": result.transformer.coordinate_system.name(),
        "grid": result.transformer.grid.subdivisions.iter().map(|(axis, subdivision)| {
            (axis.clone(), json!({
                "minimum": subdivision.range.minimum,
                "maximum": subdivision.range.maximum,
                "steps": subdivision.steps,
            }))
        }).collect::<BTreeMap<_, _>>(),
    })
}

pub fn build_session_data(
    config: &ExplorerConfig,
    cases: &[ExplorationCase],
    results: &[(ExplorationCase, DiscreteAnalysisResult)],
    active_case_id: Option<String>,
    table_mode: &str,
    map_limits: &BTreeMap<String, (f64, f64, f64, f64)>,
    music: &BTreeMap<String, String>,
) -> Value {
    json!({
        "schema": SESSION_SCHEMA,
        "version": SESSION_VERSION,
        "view": {
            "config": config_to_data(config),
            "active_case_id": active_case_id,
            "table_mode": table_mode,
            "map_limits": map_limits.iter().map(|(key, value)| {
                (key.clone(), vec![value.0, value.1, value.2, value.3])
            }).collect::<BTreeMap<_, _>>(),
            "music": music,
        },
        "cases": cases.iter().map(case_to_data).collect::<Vec<_>>(),
        "results": results.iter().map(|(case, result)| {
            json!({
                "case_id": case.identifier,
                "case_label": case.label,
                "result": result_to_data(result),
            })
        }).collect::<Vec<_>>(),
    })
}

pub fn parse_session_data(data: &Value) -> Result<ImportedSession> {
    let object = data
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("La raiz de la sesion debe ser un objeto JSON"))?;
    if object.get("schema").and_then(Value::as_str) != Some(SESSION_SCHEMA) {
        bail!("El archivo no es una sesion de fractal_sequences");
    }
    if object.get("version").and_then(Value::as_u64) != Some(SESSION_VERSION as u64) {
        bail!("Version de sesion no compatible");
    }
    let view = object
        .get("view")
        .ok_or_else(|| anyhow::anyhow!("Sesion incompleta"))?;
    let view_object = view
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Sesion incompleta"))?;
    let config = config_from_data(
        view_object
            .get("config")
            .ok_or_else(|| anyhow::anyhow!("Sesion incompleta"))?,
    )?;
    let cases_value = object
        .get("cases")
        .ok_or_else(|| anyhow::anyhow!("Sesion incompleta"))?;
    let case_items = cases_value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Sesion incompleta"))?;
    let cases = case_items
        .iter()
        .map(case_from_data)
        .collect::<Result<Vec<_>>>()?;

    let mut map_limits = BTreeMap::new();
    if let Some(raw_limits) = view_object.get("map_limits").and_then(Value::as_object) {
        for (key, value) in raw_limits {
            if let Some(items) = value.as_array()
                && items.len() == 4
            {
                let parsed = (
                    items[0].as_f64().unwrap_or(0.0),
                    items[1].as_f64().unwrap_or(0.0),
                    items[2].as_f64().unwrap_or(0.0),
                    items[3].as_f64().unwrap_or(0.0),
                );
                map_limits.insert(key.clone(), parsed);
            }
        }
    }

    let table_mode = view_object
        .get("table_mode")
        .and_then(Value::as_str)
        .unwrap_or("todo")
        .to_string();
    let active_case_id = view_object
        .get("active_case_id")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let music = if let Some(music_object) = view_object.get("music").and_then(Value::as_object) {
        music_object
            .iter()
            .map(|(key, value)| (key.clone(), value.as_str().unwrap_or("").to_string()))
            .collect()
    } else {
        BTreeMap::new()
    };

    Ok(ImportedSession {
        config,
        cases,
        active_case_id,
        table_mode,
        map_limits,
        music,
    })
}

pub fn write_session(path: impl AsRef<Path>, data: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(data)?)?;
    Ok(())
}

pub fn read_session(path: impl AsRef<Path>) -> Result<ImportedSession> {
    let raw = std::fs::read_to_string(path)?;
    let data: Value = serde_json::from_str(&raw)?;
    parse_session_data(&data)
}

pub fn write_results_csv(
    path: impl AsRef<Path>,
    results: &[(ExplorationCase, DiscreteAnalysisResult)],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "case_id",
        "case_label",
        "source",
        "row_type",
        "attractor_index",
        "term_index",
        "value_real",
        "value_imag",
        "coordinates",
        "indices",
        "discrete_coordinates",
        "cell_id",
        "inside_domain",
        "attractor_kind",
        "attractor_period",
    ])?;

    for (case, result) in results {
        for item in &result.terms {
            write_csv_row(&mut writer, case, "sequence", &item.point, None, Some(item.term.index), None, None)?;
        }
        for (attractor_index, attractor) in result.attractors.iter().enumerate() {
            for point in &attractor.points {
                write_csv_row(
                    &mut writer,
                    case,
                    "attractor",
                    point,
                    Some(attractor_index + 1),
                    None,
                    Some(format!("{:?}", attractor.attractor.kind).to_lowercase()),
                    Some(attractor.attractor.period),
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_csv_row(
    writer: &mut csv::Writer<std::fs::File>,
    case: &ExplorationCase,
    row_type: &str,
    point: &DiscretePoint,
    attractor_index: Option<usize>,
    term_index: Option<usize>,
    attractor_kind: Option<String>,
    attractor_period: Option<usize>,
) -> Result<()> {
    let value = point.value;
    writer.write_record([
        case.identifier.clone(),
        case.label.clone(),
        case.source.clone(),
        row_type.to_string(),
        attractor_index.map_or_else(String::new, |value| value.to_string()),
        term_index.map_or_else(String::new, |value| value.to_string()),
        value.re.to_string(),
        value.im.to_string(),
        serde_json::to_string(&point.coordinates)?,
        serde_json::to_string(&point.indices)?,
        serde_json::to_string(&point.discrete_coordinates)?,
        point.cell_id.map_or_else(String::new, |value| value.to_string()),
        point.inside_domain.to_string(),
        attractor_kind.unwrap_or_default(),
        attractor_period.map_or_else(String::new, |value| value.to_string()),
    ])?;
    Ok(())
}

fn term_to_data(term: &SequenceTerm) -> Value {
    json!({
        "index": term.index,
        "value": number_to_data(term.value),
    })
}

fn attractor_to_data(attractor: &Attractor) -> Value {
    json!({
        "values": attractor.values.iter().map(|value| number_to_data(*value)).collect::<Vec<_>>(),
        "kind": attractor.kind,
        "period": attractor.period,
        "tolerance": attractor.tolerance,
        "start_index": attractor.start_index,
    })
}

fn point_to_data(point: &DiscretePoint) -> Value {
    json!({
        "value": number_to_data(point.value),
        "coordinates": point.coordinates,
        "indices": point.indices,
        "discrete_coordinates": point.discrete_coordinates,
        "cell_id": point.cell_id,
        "inside_domain": point.inside_domain,
    })
}

fn number_to_data(value: crate::models::Number) -> Value {
    json!({
        "real": value.re,
        "imag": value.im,
    })
}
