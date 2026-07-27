"""Translate discretized attractors into musical patterns for Strudel.

This module deliberately works on :class:`DiscreteAnalysisResult` rather than
on a particular visual widget.  The same musical mapping can therefore be used
from a script, a future MIDI exporter, or the interactive explorer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .transformations import DiscreteAnalysisResult, DiscretePoint


MusicCriterion = Literal["melody", "harmony", "rhythm", "texture"]
MusicSegmentation = Literal["bands", "positions", "sectors"]

MUSIC_CRITERIA: tuple[MusicCriterion, ...] = (
    "melody",
    "harmony",
    "rhythm",
    "texture",
)
MUSIC_SEGMENTATIONS: tuple[MusicSegmentation, ...] = (
    "bands",
    "positions",
    "sectors",
)
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic": (0, 3, 5, 7, 10),
    "chromatic": tuple(range(12)),
}

_PITCH_CLASSES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
_ROOT_OFFSETS = {
    "c": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
}


@dataclass(frozen=True)
class MusicMappingConfig:
    """Musical interpretation of the cells occupied by attractors."""

    criterion: MusicCriterion = "melody"
    segmentation: MusicSegmentation = "bands"
    scale: str = "minor"
    root: str = "C4"
    octaves: int = 2

    def __post_init__(self) -> None:
        if self.criterion not in MUSIC_CRITERIA:
            raise ValueError(f"Unknown music criterion: {self.criterion}")
        if self.segmentation not in MUSIC_SEGMENTATIONS:
            raise ValueError(f"Unknown segmentation: {self.segmentation}")
        if self.scale not in SCALES:
            raise ValueError(f"Unknown scale: {self.scale}")
        if self.octaves < 1:
            raise ValueError("octaves must be at least 1")
        _root_midi(self.root)


@dataclass(frozen=True)
class MusicalAttractor:
    """One attractor represented as notes and its source discrete cells."""

    case_label: str
    attractor_index: int
    period: int
    notes: tuple[str, ...]
    cells: tuple[int | None, ...]
    source_values: tuple[complex, ...]


@dataclass(frozen=True)
class StrudelPattern:
    """A complete Strudel command plus the musical parameters that produced it."""

    config: MusicMappingConfig
    attractors: tuple[MusicalAttractor, ...]
    code: str

    @property
    def description(self) -> str:
        return (
            f"{self.config.criterion}, {self.config.segmentation}, "
            f"{self.config.root}:{self.config.scale}, "
            f"atractores={len(self.attractors)}"
        )


def generate_strudel_pattern(
    results: Iterable[tuple[str, DiscreteAnalysisResult]],
    config: MusicMappingConfig | None = None,
) -> StrudelPattern:
    """Generate a directly pasteable Strudel command from all attractors.

    ``bands`` uses the most expressive spatial axis (``y``, ``r`` or ``x``),
    ``positions`` uses the discrete cell identifier, and ``sectors`` uses the
    polar angle of each point.  Every choice is normalized to scale degrees
    over the selected number of octaves.
    """

    if config is None:
        config = MusicMappingConfig()

    musical_attractors: list[MusicalAttractor] = []
    for case_label, result in results:
        for attractor_index, discrete_attractor in enumerate(result.attractors, start=1):
            notes: list[str] = []
            cells: list[int | None] = []
            values: list[complex] = []
            for point in discrete_attractor.points:
                normalized = _normalized_position(
                    point,
                    result,
                    config.segmentation,
                )
                if normalized is None:
                    continue
                notes.append(_note_from_position(normalized, config))
                cells.append(point.cell_id)
                values.append(complex(point.value))
            if notes:
                musical_attractors.append(
                    MusicalAttractor(
                        case_label=case_label,
                        attractor_index=attractor_index,
                        period=discrete_attractor.attractor.period,
                        notes=tuple(notes),
                        cells=tuple(cells),
                        source_values=tuple(values),
                    )
                )

    attractors = tuple(musical_attractors)
    return StrudelPattern(
        config=config,
        attractors=attractors,
        code=_strudel_code(attractors, config),
    )


def _normalized_position(
    point: DiscretePoint,
    result: DiscreteAnalysisResult,
    segmentation: MusicSegmentation,
) -> float | None:
    grid = result.transformer.grid
    if segmentation == "sectors":
        if point.indices is not None and "theta" in point.indices:
            subdivision = grid.subdivisions["theta"]
            return _index_fraction(point.indices["theta"], subdivision.steps)
        value = complex(point.value)
        return (math.atan2(value.imag, value.real) % math.tau) / math.tau

    if point.indices is None:
        return None
    if segmentation == "positions":
        total_cells = math.prod(
            subdivision.steps for subdivision in grid.subdivisions.values()
        )
        if point.cell_id is None or total_cells <= 0:
            return None
        return _index_fraction(point.cell_id, total_cells)

    axis = _band_axis(tuple(grid.axes))
    subdivision = grid.subdivisions[axis]
    return _index_fraction(point.indices[axis], subdivision.steps)


def _band_axis(axes: tuple[str, ...]) -> str:
    for axis in ("y", "r", "x", "theta"):
        if axis in axes:
            return axis
    return axes[0]


def _index_fraction(index: int, steps: int) -> float:
    if steps <= 0:
        return 0.0
    return min(1.0 - 1e-12, max(0.0, (index + 0.5) / steps))


def _note_from_position(position: float, config: MusicMappingConfig) -> str:
    degrees_per_octave = len(SCALES[config.scale])
    total_degrees = degrees_per_octave * config.octaves
    degree = min(total_degrees - 1, int(position * total_degrees))
    octave_offset, scale_degree = divmod(degree, degrees_per_octave)
    midi = _root_midi(config.root) + SCALES[config.scale][scale_degree] + 12 * octave_offset
    pitch_class = _PITCH_CLASSES[midi % 12]
    octave = midi // 12 - 1
    return f"{pitch_class}{octave}"


def _root_midi(root: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", root.strip())
    if match is None:
        raise ValueError("root must use a note such as C4, D#3 or Bb2")
    name = (match.group(1) + match.group(2)).lower()
    if name not in _ROOT_OFFSETS:
        raise ValueError("root note is not valid")
    octave = int(match.group(3))
    return (octave + 1) * 12 + _ROOT_OFFSETS[name]


def _strudel_code(
    attractors: tuple[MusicalAttractor, ...],
    config: MusicMappingConfig,
) -> str:
    if not attractors:
        return "silence"

    groups = [" ".join(attractor.notes) for attractor in attractors]
    melody = " ".join(groups)
    if config.criterion == "melody":
        return f'note("{melody}").sound("piano").slow(2)'
    if config.criterion == "harmony":
        chords = " ".join(f"[{','.join(attractor.notes)}]" for attractor in attractors)
        return f'note("{chords}").sound("gm_epiano1").slow(2)'
    if config.criterion == "rhythm":
        rhythm = " ".join("x" if index % 2 == 0 else "~" for index in range(len(melody.split())))
        return (
            "stack("
            f'note("{melody}").sound("gm_acoustic_bass"), '
            f's("bd").struct("{rhythm}")'
            ")"
        )
    return (
        "stack("
        f'note("{melody}").sound("piano"), '
        f'note("{melody}").sound("sawtooth").slow(2).gain(0.35)'
        ")"
    )
