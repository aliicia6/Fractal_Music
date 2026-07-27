"""Portable session data for the interactive sequence explorer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import (
    AnalysisConfig,
    AxisRangeConfig,
    DiscretizationConfig,
    ExplorerConfig,
    RecurrenceConfig,
)
from .models import Attractor, Number, SequenceAnalysisResult, SequenceTerm
from .transformations import DiscreteAnalysisResult, DiscretePoint


SESSION_SCHEMA = "fractal_sequences.visual_session"
SESSION_VERSION = 1


@dataclass(frozen=True)
class ExplorationCase:
    """A named recurrence kept in the explorer's multi-case collection."""

    identifier: str
    label: str
    recurrence: RecurrenceConfig
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportedSession:
    """Validated state restored from an exported JSON session."""

    config: ExplorerConfig
    cases: list[ExplorationCase]
    active_case_id: str | None = None
    table_mode: str = "todo"
    map_limits: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    music: dict[str, str] = field(default_factory=dict)


def config_to_data(config: ExplorerConfig) -> dict[str, Any]:
    """Encode a pipeline configuration using JSON-native values."""

    return {
        "recurrence": {
            "name": config.recurrence.name,
            "parameters": dict(config.recurrence.parameters),
        },
        "analysis": asdict(config.analysis),
        "discretization": {
            "coordinate_system": config.discretization.coordinate_system,
            "ranges": {
                axis: asdict(axis_range)
                for axis, axis_range in config.discretization.ranges.items()
            },
        },
    }


def config_from_data(data: dict[str, Any]) -> ExplorerConfig:
    """Decode and validate a JSON-native pipeline configuration."""

    try:
        recurrence_data = data["recurrence"]
        analysis_data = data["analysis"]
        discretization_data = data["discretization"]
        ranges_data = discretization_data["ranges"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Configuracion de sesion incompleta") from exc

    if not isinstance(recurrence_data, dict) or not isinstance(analysis_data, dict):
        raise ValueError("Configuracion de sesion invalida")
    if not isinstance(discretization_data, dict) or not isinstance(ranges_data, dict):
        raise ValueError("Discretizacion de sesion invalida")

    return ExplorerConfig(
        recurrence=RecurrenceConfig(
            name=str(recurrence_data["name"]),
            parameters={
                str(key): float(value)
                for key, value in dict(recurrence_data.get("parameters", {})).items()
            },
        ),
        analysis=AnalysisConfig(**analysis_data),
        discretization=DiscretizationConfig(
            coordinate_system=str(discretization_data["coordinate_system"]),
            ranges={
                str(axis): AxisRangeConfig(**axis_data)
                for axis, axis_data in ranges_data.items()
            },
        ),
    )


def case_to_data(case: ExplorationCase) -> dict[str, Any]:
    """Encode one stored case."""

    return {
        "identifier": case.identifier,
        "label": case.label,
        "recurrence": {
            "name": case.recurrence.name,
            "parameters": dict(case.recurrence.parameters),
        },
        "source": case.source,
        "metadata": case.metadata,
    }


def case_from_data(data: dict[str, Any]) -> ExplorationCase:
    """Decode one stored case."""

    recurrence_data = data.get("recurrence")
    if not isinstance(recurrence_data, dict):
        raise ValueError("Caso de sesion sin recurrencia")

    identifier = str(data.get("identifier", "")).strip()
    if not identifier:
        raise ValueError("Caso de sesion sin identificador")

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Metadatos de caso invalidos")

    return ExplorationCase(
        identifier=identifier,
        label=str(data.get("label", identifier)),
        recurrence=RecurrenceConfig(
            name=str(recurrence_data["name"]),
            parameters={
                str(key): float(value)
                for key, value in dict(recurrence_data.get("parameters", {})).items()
            },
        ),
        source=str(data.get("source", "manual")),
        metadata=metadata,
    )


def result_to_data(result: DiscreteAnalysisResult) -> dict[str, Any]:
    """Encode all calculated values, including their discrete projections."""

    sequence = result.sequence_result
    return {
        "config": config_to_data(result.config) if isinstance(result.config, ExplorerConfig) else None,
        "behavior": sequence.behavior,
        "reason": sequence.reason,
        "diverged": sequence.diverged,
        "terms": [term_to_data(term) for term in sequence.terms],
        "attractors": [attractor_to_data(attractor) for attractor in sequence.attractors],
        "discrete_terms": [
            {
                "index": item.term.index,
                "point": point_to_data(item.point),
            }
            for item in result.terms
        ],
        "discrete_attractors": [
            {
                "kind": item.attractor.kind,
                "period": item.attractor.period,
                "points": [point_to_data(point) for point in item.points],
            }
            for item in result.attractors
        ],
        "coordinate_system": result.transformer.coordinate_system.name,
        "grid": {
            axis: {
                "minimum": subdivision.range.minimum,
                "maximum": subdivision.range.maximum,
                "steps": subdivision.steps,
            }
            for axis, subdivision in result.transformer.grid.subdivisions.items()
        },
    }


def build_session_data(
    config: ExplorerConfig,
    cases: Iterable[ExplorationCase],
    results: Iterable[tuple[ExplorationCase, DiscreteAnalysisResult]],
    *,
    active_case_id: str | None,
    table_mode: str,
    map_limits: dict[str, tuple[float, float, float, float]],
    music: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a complete JSON document without writing it to disk."""

    return {
        "schema": SESSION_SCHEMA,
        "version": SESSION_VERSION,
        "view": {
            "config": config_to_data(config),
            "active_case_id": active_case_id,
            "table_mode": table_mode,
            "map_limits": {
                key: list(value)
                for key, value in map_limits.items()
            },
            "music": dict(music or {}),
        },
        "cases": [case_to_data(case) for case in cases],
        "results": [
            {
                "case_id": case.identifier,
                "case_label": case.label,
                "result": result_to_data(result),
            }
            for case, result in results
        ],
    }


def parse_session_data(data: dict[str, Any]) -> ImportedSession:
    """Validate an exported JSON document and restore its editable state."""

    if data.get("schema") != SESSION_SCHEMA:
        raise ValueError("El archivo no es una sesion de fractal_sequences")
    if data.get("version") != SESSION_VERSION:
        raise ValueError("Version de sesion no compatible")

    view = data.get("view")
    case_data = data.get("cases")
    if not isinstance(view, dict) or not isinstance(case_data, list):
        raise ValueError("Sesion incompleta")

    config = config_from_data(view.get("config", {}))
    cases = [case_from_data(item) for item in case_data]
    identifiers = [case.identifier for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("La sesion contiene identificadores duplicados")

    raw_limits = view.get("map_limits", {})
    map_limits: dict[str, tuple[float, float, float, float]] = {}
    if isinstance(raw_limits, dict):
        for key, values in raw_limits.items():
            if isinstance(values, list) and len(values) == 4:
                map_limits[str(key)] = tuple(float(value) for value in values)

    active_case_id = view.get("active_case_id")
    if active_case_id is not None:
        active_case_id = str(active_case_id)
        if active_case_id not in identifiers:
            active_case_id = None

    raw_music = view.get("music", {})
    music = (
        {str(key): str(value) for key, value in raw_music.items()}
        if isinstance(raw_music, dict)
        else {}
    )

    return ImportedSession(
        config=config,
        cases=cases,
        active_case_id=active_case_id,
        table_mode=str(view.get("table_mode", "todo")),
        map_limits=map_limits,
        music=music,
    )


def write_session(path: str | Path, data: dict[str, Any]) -> None:
    """Write a session document using UTF-8 JSON."""

    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_session(path: str | Path) -> ImportedSession:
    """Read and validate a session document."""

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("El archivo JSON no es valido") from exc
    if not isinstance(data, dict):
        raise ValueError("La raiz de la sesion debe ser un objeto JSON")
    return parse_session_data(data)


def write_results_csv(
    path: str | Path,
    results: Iterable[tuple[ExplorationCase, DiscreteAnalysisResult]],
) -> None:
    """Export sequence and attractor values for all cases to one CSV file."""

    columns = [
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
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for case, result in results:
            for item in result.terms:
                writer.writerow(
                    _csv_row(
                        case,
                        "sequence",
                        item.point,
                        term_index=item.term.index,
                    )
                )
            for attractor_index, attractor in enumerate(result.attractors, start=1):
                for point in attractor.points:
                    writer.writerow(
                        _csv_row(
                            case,
                            "attractor",
                            point,
                            attractor_index=attractor_index,
                            attractor_kind=attractor.attractor.kind,
                            attractor_period=attractor.attractor.period,
                        )
                    )


def term_to_data(term: SequenceTerm) -> dict[str, Any]:
    return {"index": term.index, "value": number_to_data(term.value)}


def attractor_to_data(attractor: Attractor) -> dict[str, Any]:
    return {
        "values": [number_to_data(value) for value in attractor.values],
        "kind": attractor.kind,
        "period": attractor.period,
        "tolerance": attractor.tolerance,
        "start_index": attractor.start_index,
    }


def point_to_data(point: DiscretePoint) -> dict[str, Any]:
    return {
        "value": number_to_data(point.value),
        "coordinates": point.coordinates,
        "indices": point.indices,
        "discrete_coordinates": point.discrete_coordinates,
        "cell_id": point.cell_id,
        "inside_domain": point.inside_domain,
    }


def number_to_data(value: Number) -> dict[str, float]:
    """Represent a real or complex number without losing its imaginary part."""

    complex_value = complex(value)
    return {"real": complex_value.real, "imag": complex_value.imag}


def _csv_row(
    case: ExplorationCase,
    row_type: str,
    point: DiscretePoint,
    *,
    attractor_index: int | None = None,
    term_index: int | None = None,
    attractor_kind: str | None = None,
    attractor_period: int | None = None,
) -> dict[str, Any]:
    value = complex(point.value)
    return {
        "case_id": case.identifier,
        "case_label": case.label,
        "source": case.source,
        "row_type": row_type,
        "attractor_index": attractor_index,
        "term_index": term_index,
        "value_real": value.real,
        "value_imag": value.imag,
        "coordinates": json.dumps(point.coordinates, ensure_ascii=False),
        "indices": json.dumps(point.indices, ensure_ascii=False),
        "discrete_coordinates": json.dumps(
            point.discrete_coordinates,
            ensure_ascii=False,
        ),
        "cell_id": point.cell_id,
        "inside_domain": point.inside_domain,
        "attractor_kind": attractor_kind,
        "attractor_period": attractor_period,
    }
