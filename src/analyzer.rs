use anyhow::{Result, bail};

use crate::models::{
    Attractor, AttractorKind, BehaviorKind, Number, SequenceAnalysisResult, SequenceTerm,
};
use crate::recurrence::Recurrence;

pub fn analyze_recurrence(
    recurrence: &Recurrence,
    max_terms: usize,
    tolerance: f64,
    divergence_limit: f64,
    attractor_window: usize,
    max_period: usize,
    min_repetitions: usize,
    transient_terms: usize,
    stability_ratio: f64,
    cluster_tolerance: Option<f64>,
) -> Result<SequenceAnalysisResult> {
    validate_analysis_parameters(
        max_terms,
        tolerance,
        divergence_limit,
        attractor_window,
        max_period,
        min_repetitions,
        transient_terms,
        stability_ratio,
        cluster_tolerance,
    )?;

    let mut terms = vec![SequenceTerm {
        index: 0,
        value: recurrence.initial_value,
    }];
    let mut current = recurrence.initial_value;
    let mut diverged = false;
    let mut reason = String::from("max_terms reached");
    let mut behavior = BehaviorKind::InsufficientData;

    for index in 0..(max_terms - 1) {
        current = recurrence.next_value(current, index);
        terms.push(SequenceTerm {
            index: index + 1,
            value: current,
        });

        if is_invalid_number(current) || current.norm() > divergence_limit {
            diverged = true;
            reason = String::from("divergence detected");
            behavior = BehaviorKind::Divergent;
            break;
        }
    }

    let mut attractors = Vec::new();
    if !diverged {
        attractors = detect_attractors(
            &terms,
            tolerance,
            attractor_window,
            max_period,
            min_repetitions,
            transient_terms,
            stability_ratio,
            cluster_tolerance,
        )?;
        let (classified_behavior, classified_reason) =
            classify_behavior(&terms, &attractors, attractor_window);
        behavior = classified_behavior;
        reason = classified_reason;
    }

    Ok(SequenceAnalysisResult {
        terms,
        attractors,
        diverged,
        reason,
        behavior,
    })
}

pub fn detect_attractor(
    terms: &[SequenceTerm],
    tolerance: f64,
    attractor_window: usize,
    max_period: usize,
    min_repetitions: usize,
    transient_terms: usize,
    stability_ratio: f64,
    cluster_tolerance: Option<f64>,
) -> Result<Option<Attractor>> {
    let attractors = detect_attractors(
        terms,
        tolerance,
        attractor_window,
        max_period,
        min_repetitions,
        transient_terms,
        stability_ratio,
        cluster_tolerance,
    )?;
    Ok(attractors.into_iter().next())
}

#[allow(clippy::too_many_arguments)]
pub fn detect_attractors(
    terms: &[SequenceTerm],
    tolerance: f64,
    attractor_window: usize,
    max_period: usize,
    min_repetitions: usize,
    transient_terms: usize,
    stability_ratio: f64,
    cluster_tolerance: Option<f64>,
) -> Result<Vec<Attractor>> {
    validate_detection_parameters(
        tolerance,
        attractor_window,
        max_period,
        min_repetitions,
        transient_terms,
        stability_ratio,
        cluster_tolerance,
    )?;

    let values: Vec<Number> = terms.iter().map(|term| term.value).collect();
    if values.len() < 2 {
        return Ok(Vec::new());
    }

    let start = transient_terms.max(values.len().saturating_sub(attractor_window));
    let window_size = values.len() - start;
    if window_size < 2 {
        return Ok(Vec::new());
    }

    let period_limit = max_period.min(window_size / 2);
    let representative_tolerance = cluster_tolerance.unwrap_or(tolerance);
    let mut attractors = Vec::new();

    for period in 1..=period_limit {
        let total_checks = window_size - period;
        let required_checks = period * min_repetitions;
        if total_checks < required_checks {
            continue;
        }

        let mut matches = 0usize;
        for index in (start + period)..values.len() {
            if (values[index] - values[index - period]).norm() <= tolerance {
                matches += 1;
            }
        }
        if (matches as f64 / total_checks as f64) >= stability_ratio {
            let cycle_values = cycle_representatives(&values, start, period);
            let reduced = reduce_cycle_period(cycle_values, representative_tolerance);
            let reduced_period = reduced.len();
            let kind = if reduced_period == 1 {
                AttractorKind::FixedPoint
            } else {
                AttractorKind::Cycle
            };
            let candidate = Attractor {
                values: reduced,
                kind,
                period: reduced_period,
                tolerance,
                start_index: Some(start),
            };
            if is_distinct_attractor(&candidate, &attractors, representative_tolerance) {
                attractors.push(candidate);
            }
        }
    }

    Ok(attractors)
}

pub fn classify_behavior(
    terms: &[SequenceTerm],
    attractors: &[Attractor],
    attractor_window: usize,
) -> (BehaviorKind, String) {
    if terms.len() < 2 {
        return (BehaviorKind::InsufficientData, String::from("insufficient data"));
    }
    if let Some(first) = attractors.first() {
        let behavior = if first.period == 1 {
            BehaviorKind::FixedPoint
        } else {
            BehaviorKind::Periodic
        };
        return (
            behavior,
            format!(
                "{} detected",
                match first.kind {
                    AttractorKind::FixedPoint => "fixed_point",
                    AttractorKind::Cycle => "cycle",
                    AttractorKind::Unknown => "unknown",
                }
            ),
        );
    }
    if terms.len() < attractor_window {
        return (
            BehaviorKind::InsufficientData,
            String::from("insufficient data for attractor detection"),
        );
    }
    (
        BehaviorKind::ChaoticOrUnresolved,
        String::from("no stable attractor detected"),
    )
}

fn cycle_representatives(values: &[Number], start: usize, period: usize) -> Vec<Number> {
    let cycle_start = values.len() - period;
    let mut representatives = Vec::with_capacity(period);
    for phase in 0..period {
        let mut phase_values = Vec::new();
        let mut index = cycle_start + phase;
        loop {
            phase_values.push(values[index]);
            if index < start + period {
                break;
            }
            index -= period;
        }
        representatives.push(mean_number(&phase_values));
    }
    representatives
}

fn mean_number(values: &[Number]) -> Number {
    let total = values.iter().copied().sum::<Number>();
    total / values.len() as f64
}

fn reduce_cycle_period(values: Vec<Number>, tolerance: f64) -> Vec<Number> {
    let period = values.len();
    for candidate_period in 1..period {
        if !period.is_multiple_of(candidate_period) {
            continue;
        }
        if (0..period).all(|index| (values[index] - values[index % candidate_period]).norm() <= tolerance)
        {
            return values[0..candidate_period].to_vec();
        }
    }
    values
}

fn is_distinct_attractor(candidate: &Attractor, existing: &[Attractor], tolerance: f64) -> bool {
    for attractor in existing {
        if candidate.period != attractor.period {
            continue;
        }
        if cycles_are_close(&candidate.values, &attractor.values, tolerance) {
            return false;
        }
    }
    true
}

fn cycles_are_close(first: &[Number], second: &[Number], tolerance: f64) -> bool {
    if first.len() != second.len() {
        return false;
    }
    let period = first.len();
    for shift in 0..period {
        if (0..period).all(|index| (first[index] - second[(index + shift) % period]).norm() <= tolerance)
        {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn validate_analysis_parameters(
    max_terms: usize,
    tolerance: f64,
    divergence_limit: f64,
    attractor_window: usize,
    max_period: usize,
    min_repetitions: usize,
    transient_terms: usize,
    stability_ratio: f64,
    cluster_tolerance: Option<f64>,
) -> Result<()> {
    if max_terms < 2 {
        bail!("max_terms must be at least 2");
    }
    if divergence_limit <= 0.0 {
        bail!("divergence_limit must be positive");
    }
    validate_detection_parameters(
        tolerance,
        attractor_window,
        max_period,
        min_repetitions,
        transient_terms,
        stability_ratio,
        cluster_tolerance,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_detection_parameters(
    tolerance: f64,
    attractor_window: usize,
    max_period: usize,
    min_repetitions: usize,
    transient_terms: usize,
    stability_ratio: f64,
    cluster_tolerance: Option<f64>,
) -> Result<()> {
    if tolerance <= 0.0 {
        bail!("tolerance must be positive");
    }
    if attractor_window < 2 {
        bail!("attractor_window must be at least 2");
    }
    if max_period < 1 {
        bail!("max_period must be at least 1");
    }
    if min_repetitions < 1 {
        bail!("min_repetitions must be at least 1");
    }
    if !(0.0 < stability_ratio && stability_ratio <= 1.0) {
        bail!("stability_ratio must be in the interval (0, 1]");
    }
    if cluster_tolerance.is_some_and(|value| value <= 0.0) {
        bail!("cluster_tolerance must be positive");
    }
    if transient_terms > usize::MAX / 2 {
        bail!("transient_terms out of range");
    }
    Ok(())
}

fn is_invalid_number(value: Number) -> bool {
    value.re.is_nan() || value.im.is_nan() || value.re.is_infinite() || value.im.is_infinite()
}
