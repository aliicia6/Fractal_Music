use std::collections::BTreeMap;

use num_complex::Complex64;
use visualizador_mandelbrot::{
    AttractorKind, ContinuousRange, MusicMappingConfig, Recurrence, RecurrenceConfig, SequenceTerm,
    UniformSubdivision, analyze_recurrence, default_concept_config, detect_attractor,
    detect_attractors, generate_strudel_pattern, logistic_map, logistic_to_quadratic,
    quadratic_complex_map, quadratic_to_logistic, run_exploration,
};

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

#[test]
fn cosine_fixed_point_is_detected() {
    let recurrence = visualizador_mandelbrot::cosine_fixed_point(1.0);
    let result = analyze_recurrence(&recurrence, 500, 1e-7, 1e6, 100, 20, 5, 0, 1.0, None).unwrap();
    assert!(!result.diverged);
    assert!(!result.attractors.is_empty());
    assert_eq!(result.attractors[0].kind, AttractorKind::FixedPoint);
    assert!(close(result.attractors[0].values[0].re, 0.7390851332, 1e-6));
}

#[test]
fn logistic_period_two_is_detected() {
    let recurrence = logistic_map(3.2, 0.2);
    let result = analyze_recurrence(&recurrence, 800, 1e-7, 1e6, 120, 8, 5, 0, 1.0, None).unwrap();
    assert!(!result.diverged);
    assert_eq!(result.attractors[0].kind, AttractorKind::Cycle);
    assert_eq!(result.attractors[0].period, 2);
}

#[test]
fn complex_period_three_is_detected() {
    let recurrence = quadratic_complex_map(Complex64::new(-0.123, 0.745), Complex64::new(0.0, 0.0));
    let result = analyze_recurrence(&recurrence, 800, 1e-7, 1e6, 120, 8, 5, 0, 1.0, None).unwrap();
    assert!(!result.diverged);
    assert_eq!(result.attractors[0].period, 3);
}

#[test]
fn divergence_is_detected() {
    let recurrence = Recurrence::new(
        "divergent_square",
        Complex64::new(2.0, 0.0),
        std::sync::Arc::new(|x, _| x * x),
    );
    let result = analyze_recurrence(&recurrence, 50, 1e-12, 1e6, 20, 5, 2, 0, 1.0, None).unwrap();
    assert!(result.diverged);
    assert_eq!(result.reason, "divergence detected");
}

#[test]
fn direct_attractor_detection() {
    let terms: Vec<SequenceTerm> = (0..40)
        .map(|index| SequenceTerm {
            index,
            value: if index % 2 == 0 {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(-1.0, 0.0)
            },
        })
        .collect();
    let attractor = detect_attractor(&terms, 1e-8, 20, 6, 5, 0, 1.0, None).unwrap();
    assert!(attractor.is_some());
    let attractor = attractor.unwrap();
    assert_eq!(attractor.kind, AttractorKind::Cycle);
    assert_eq!(attractor.period, 2);
}

#[test]
fn approximate_cycle_detection() {
    let terms: Vec<SequenceTerm> = (0..80)
        .map(|index| {
            let noise = if index % 11 == 0 { 1e-4 } else { 0.0 };
            let base = if index % 2 == 0 { 0.2 } else { 0.8 };
            SequenceTerm {
                index,
                value: Complex64::new(base + noise, 0.0),
            }
        })
        .collect();
    let attractors = detect_attractors(&terms, 1e-5, 60, 6, 5, 0, 0.8, Some(1e-3)).unwrap();
    assert!(!attractors.is_empty());
    assert_eq!(attractors[0].period, 2);
}

#[test]
fn unresolved_behavior_for_r4() {
    let recurrence = logistic_map(4.0, 0.123456);
    let result = analyze_recurrence(&recurrence, 500, 1e-10, 1e6, 200, 12, 5, 0, 0.98, None).unwrap();
    assert!(!result.diverged);
    assert!(result.attractors.is_empty());
}

#[test]
fn uniform_subdivision_and_transform() {
    let subdivision = UniformSubdivision::new(ContinuousRange::new(-2.0, 2.0).unwrap(), 8).unwrap();
    assert!(close(subdivision.step_size(), 0.5, 1e-12));
    assert_eq!(subdivision.index_of(-2.0), Some(0));
    assert_eq!(subdivision.index_of(2.0), Some(7));
    assert_eq!(subdivision.index_of(0.3), Some(4));
    assert_eq!(subdivision.index_of(3.0), None);
}

#[test]
fn reproducible_pipeline_works() {
    let config = default_concept_config();
    let result = run_exploration(&config).unwrap();
    assert_eq!(result.terms.len(), result.sequence_result.terms.len());
    assert!(!result.terms.is_empty());
}

#[test]
fn conjugacy_round_trip_works() {
    let mapped = logistic_to_quadratic(3.2, 0.2);
    let recovered = quadratic_to_logistic(mapped.c, mapped.z0, 1e-9).unwrap();
    assert!(close(recovered.r, 3.2, 1e-12));
    assert!(close(recovered.x0, 0.2, 1e-12));
    assert!(quadratic_to_logistic(Complex64::new(-0.2, 0.1), Complex64::new(0.0, 0.0), 1e-9).is_none());
}

#[test]
fn strudel_generation_works() {
    let config = default_concept_config();
    let result = run_exploration(&config).unwrap();
    let pattern = generate_strudel_pattern(
        &[(String::from("ejemplo"), result)],
        Some(MusicMappingConfig {
            criterion: String::from("harmony"),
            segmentation: String::from("sectors"),
            scale: String::from("minor"),
            root: String::from("C4"),
            octaves: 2,
        }),
    )
    .unwrap();
    assert!(!pattern.attractors.is_empty());
    assert!(pattern.code.contains("note(\""));

    let silent = generate_strudel_pattern(&[], None).unwrap();
    assert_eq!(silent.code, "silence");
}

#[test]
fn session_payload_roundtrip() {
    let config = default_concept_config();
    let case = visualizador_mandelbrot::ExplorationCase {
        identifier: String::from("case-001"),
        label: String::from("Q1: ejemplo"),
        recurrence: RecurrenceConfig {
            name: String::from("quadratic_complex"),
            parameters: BTreeMap::from([
                (String::from("c_real"), -0.123),
                (String::from("c_imag"), 0.745),
                (String::from("z0_real"), 0.0),
                (String::from("z0_imag"), 0.0),
            ]),
        },
        source: String::from("manual"),
        metadata: BTreeMap::new(),
    };
    let result = run_exploration(&config).unwrap();
    let data = visualizador_mandelbrot::build_session_data(
        &config,
        std::slice::from_ref(&case),
        &[(case.clone(), result)],
        Some(case.identifier.clone()),
        "todo",
        &BTreeMap::from([(String::from("z^2+c"), (-2.0, 1.0, -1.35, 1.35))]),
        &BTreeMap::from([
            (String::from("criterion"), String::from("armonia")),
            (String::from("segmentation"), String::from("sectores")),
            (String::from("scale"), String::from("menor")),
        ]),
    );
    let restored = visualizador_mandelbrot::parse_session_data(&data).unwrap();
    assert_eq!(restored.config, config);
    assert_eq!(restored.cases, vec![case]);
    assert_eq!(restored.active_case_id, Some(String::from("case-001")));
}
