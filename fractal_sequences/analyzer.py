"""Sequence calculation and attractor detection."""

from __future__ import annotations

import math

from .models import Attractor, Number, SequenceAnalysisResult, SequenceTerm
from .recurrence import Recurrence


def analyze_recurrence(
    recurrence: Recurrence,
    max_terms: int = 1000,
    tolerance: float = 1e-12,
    divergence_limit: float = 1e6,
    attractor_window: int = 100,
    max_period: int = 20,
    min_repetitions: int = 5,
    transient_terms: int = 0,
    stability_ratio: float = 1.0,
    cluster_tolerance: float | None = None,
) -> SequenceAnalysisResult:
    """Calculate a recurrence and try to detect fixed points or cycles."""

    _validate_analysis_parameters(
        max_terms=max_terms,
        tolerance=tolerance,
        divergence_limit=divergence_limit,
        attractor_window=attractor_window,
        max_period=max_period,
        min_repetitions=min_repetitions,
        transient_terms=transient_terms,
        stability_ratio=stability_ratio,
        cluster_tolerance=cluster_tolerance,
    )

    terms = [SequenceTerm(index=0, value=recurrence.initial_value)]
    current = recurrence.initial_value
    diverged = False
    reason = "max_terms reached"
    behavior = "insufficient_data"

    for index in range(max_terms - 1):
        try:
            current = recurrence.next_value(current, index)
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            diverged = True
            reason = f"calculation stopped: {exc}"
            behavior = "calculation_error"
            break

        terms.append(SequenceTerm(index=index + 1, value=current))

        if _is_invalid_number(current) or abs(current) > divergence_limit:
            diverged = True
            reason = "divergence detected"
            behavior = "divergent"
            break

    attractors = []
    if not diverged:
        attractors = detect_attractors(
            terms=terms,
            tolerance=tolerance,
            attractor_window=attractor_window,
            max_period=max_period,
            min_repetitions=min_repetitions,
            transient_terms=transient_terms,
            stability_ratio=stability_ratio,
            cluster_tolerance=cluster_tolerance,
        )
        behavior, reason = classify_behavior(terms, attractors, attractor_window)

    return SequenceAnalysisResult(
        terms=terms,
        attractors=attractors,
        diverged=diverged,
        reason=reason,
        behavior=behavior,
    )


def detect_attractor(
    terms: list[SequenceTerm],
    tolerance: float,
    attractor_window: int,
    max_period: int,
    min_repetitions: int,
    transient_terms: int = 0,
    stability_ratio: float = 1.0,
    cluster_tolerance: float | None = None,
) -> Attractor | None:
    """Detect the first stable attractor in the final part of the sequence."""

    attractors = detect_attractors(
        terms=terms,
        tolerance=tolerance,
        attractor_window=attractor_window,
        max_period=max_period,
        min_repetitions=min_repetitions,
        transient_terms=transient_terms,
        stability_ratio=stability_ratio,
        cluster_tolerance=cluster_tolerance,
    )
    return attractors[0] if attractors else None


def detect_attractors(
    terms: list[SequenceTerm],
    tolerance: float,
    attractor_window: int,
    max_period: int,
    min_repetitions: int,
    transient_terms: int = 0,
    stability_ratio: float = 1.0,
    cluster_tolerance: float | None = None,
) -> list[Attractor]:
    """Detect stable fixed points or cycles after the transient section.

    The detector accepts approximate cycles: a period is considered stable when
    at least ``stability_ratio`` of comparable terms satisfy
    ``abs(x[n] - x[n - period]) <= tolerance``.
    """

    _validate_detection_parameters(
        tolerance=tolerance,
        attractor_window=attractor_window,
        max_period=max_period,
        min_repetitions=min_repetitions,
        transient_terms=transient_terms,
        stability_ratio=stability_ratio,
        cluster_tolerance=cluster_tolerance,
    )

    values = [term.value for term in terms]
    if len(values) < 2:
        return []

    start = max(transient_terms, len(values) - attractor_window)
    window_size = len(values) - start
    if window_size < 2:
        return []

    period_limit = min(max_period, window_size // 2)
    attractors: list[Attractor] = []
    representative_tolerance = cluster_tolerance or tolerance

    for period in range(1, period_limit + 1):
        total_checks = window_size - period
        required_checks = period * min_repetitions
        if total_checks < required_checks:
            continue

        matches = 0
        for index in range(start + period, len(values)):
            if abs(values[index] - values[index - period]) <= tolerance:
                matches += 1

        if matches / total_checks >= stability_ratio:
            cycle_values = _cycle_representatives(values, start, period)
            cycle_values = _reduce_cycle_period(cycle_values, representative_tolerance)
            period = len(cycle_values)
            kind = "fixed_point" if period == 1 else "cycle"
            attractor = Attractor(
                values=cycle_values,
                kind=kind,
                period=period,
                tolerance=tolerance,
                start_index=start,
            )
            if _is_distinct_attractor(attractor, attractors, representative_tolerance):
                attractors.append(attractor)

    return attractors


def classify_behavior(
    terms: list[SequenceTerm],
    attractors: list[Attractor],
    attractor_window: int,
) -> tuple[str, str]:
    """Classify the final behavior using the detected attractors."""

    if len(terms) < 2:
        return "insufficient_data", "insufficient data"
    if attractors:
        first = attractors[0]
        behavior = "fixed_point" if first.period == 1 else "periodic"
        return behavior, f"{first.kind} detected"
    if len(terms) < attractor_window:
        return "insufficient_data", "insufficient data for attractor detection"
    return "chaotic_or_unresolved", "no stable attractor detected"


def _cycle_representatives(
    values: list[Number],
    start: int,
    period: int,
) -> list[Number]:
    representatives: list[Number] = []
    cycle_start = len(values) - period

    for phase in range(period):
        phase_values = [
            values[index]
            for index in range(cycle_start + phase, start - 1, -period)
        ]
        representatives.append(_mean_number(phase_values))

    return representatives


def _mean_number(values: list[Number]) -> Number:
    if not values:
        raise ValueError("cannot average an empty list")

    total = sum(complex(value) for value in values)
    mean = total / len(values)
    if all(not isinstance(value, complex) for value in values):
        return mean.real
    return mean


def _reduce_cycle_period(values: list[Number], tolerance: float) -> list[Number]:
    period = len(values)
    for candidate_period in range(1, period):
        if period % candidate_period != 0:
            continue
        if all(
            abs(values[index] - values[index % candidate_period]) <= tolerance
            for index in range(period)
        ):
            return values[:candidate_period]
    return values


def _is_distinct_attractor(
    candidate: Attractor,
    existing: list[Attractor],
    tolerance: float,
) -> bool:
    for attractor in existing:
        if candidate.period != attractor.period:
            continue
        if _cycles_are_close(candidate.values, attractor.values, tolerance):
            return False
    return True


def _cycles_are_close(
    first: list[Number],
    second: list[Number],
    tolerance: float,
) -> bool:
    if len(first) != len(second):
        return False

    period = len(first)
    for shift in range(period):
        if all(
            abs(first[index] - second[(index + shift) % period]) <= tolerance
            for index in range(period)
        ):
            return True
    return False


def _validate_analysis_parameters(
    max_terms: int,
    tolerance: float,
    divergence_limit: float,
    attractor_window: int,
    max_period: int,
    min_repetitions: int,
    transient_terms: int,
    stability_ratio: float,
    cluster_tolerance: float | None,
) -> None:
    if max_terms < 2:
        raise ValueError("max_terms must be at least 2")
    if divergence_limit <= 0:
        raise ValueError("divergence_limit must be positive")

    _validate_detection_parameters(
        tolerance=tolerance,
        attractor_window=attractor_window,
        max_period=max_period,
        min_repetitions=min_repetitions,
        transient_terms=transient_terms,
        stability_ratio=stability_ratio,
        cluster_tolerance=cluster_tolerance,
    )


def _validate_detection_parameters(
    tolerance: float,
    attractor_window: int,
    max_period: int,
    min_repetitions: int,
    transient_terms: int,
    stability_ratio: float,
    cluster_tolerance: float | None,
) -> None:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if attractor_window < 2:
        raise ValueError("attractor_window must be at least 2")
    if max_period < 1:
        raise ValueError("max_period must be at least 1")
    if min_repetitions < 1:
        raise ValueError("min_repetitions must be at least 1")
    if transient_terms < 0:
        raise ValueError("transient_terms cannot be negative")
    if stability_ratio <= 0 or stability_ratio > 1:
        raise ValueError("stability_ratio must be in the interval (0, 1]")
    if cluster_tolerance is not None and cluster_tolerance <= 0:
        raise ValueError("cluster_tolerance must be positive")


def _is_invalid_number(value: Number) -> bool:
    if isinstance(value, complex):
        return (
            math.isnan(value.real)
            or math.isnan(value.imag)
            or math.isinf(value.real)
            or math.isinf(value.imag)
        )

    return math.isnan(value) or math.isinf(value)
