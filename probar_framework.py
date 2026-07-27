"""Smoke tests for the fractal sequence framework.

Run with:
    python probar_framework.py
"""

import math

from fractal_sequences import (
    ComplexCartesianCoordinates,
    ComplexPolarCoordinates,
    ContinuousRange,
    DiscreteTransformer,
    DiscretizationGrid,
    ExplorationCase,
    MusicMappingConfig,
    ProjectionXCoordinates,
    Recurrence,
    UniformSubdivision,
    analyze_recurrence,
    build_session_data,
    default_concept_config,
    detect_attractors,
    logistic_to_quadratic,
    generate_strudel_pattern,
    parse_session_data,
    quadratic_to_logistic,
    run_exploration,
    transform_attractors,
    transform_terms,
)
from fractal_sequences.config import RecurrenceConfig
from fractal_sequences.analyzer import detect_attractor
from fractal_sequences.examples import (
    cosine_fixed_point,
    logistic_map,
    quadratic_complex_map,
)
from fractal_sequences.models import Attractor, SequenceAnalysisResult, SequenceTerm


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_public_imports_and_models():
    term = SequenceTerm(index=0, value=1.0)
    attractor = Attractor(
        values=[1.0],
        kind="fixed_point",
        period=1,
        tolerance=1e-8,
    )
    result = SequenceAnalysisResult(
        terms=[term],
        attractors=[attractor],
        diverged=False,
        reason="manual test",
    )

    assert_true(result.terms[0].value == 1.0, "SequenceTerm no conserva el valor")
    assert_true(result.attractors[0].kind == "fixed_point", "Attractor incorrecto")


def test_cosine_fixed_point():
    recurrence = cosine_fixed_point(x0=1.0)
    result = analyze_recurrence(recurrence, max_terms=500, tolerance=1e-7)

    assert_true(not result.diverged, "cos(x) no deberia divergir")
    assert_true(result.attractors, "No se detecto el punto fijo de cos(x)")
    assert_true(result.attractors[0].kind == "fixed_point", "cos(x) debe dar punto fijo")
    assert_true(
        abs(result.attractors[0].values[0] - 0.7390851332) < 1e-6,
        "El punto fijo de cos(x) no coincide con el esperado",
    )


def test_logistic_cycle():
    recurrence = logistic_map(r=3.2, x0=0.2)
    result = analyze_recurrence(
        recurrence,
        max_terms=800,
        tolerance=1e-7,
        attractor_window=120,
        max_period=8,
    )

    assert_true(not result.diverged, "La logistica r=3.2 no deberia divergir")
    assert_true(result.attractors, "No se detecto ciclo en la logistica")
    assert_true(result.attractors[0].kind == "cycle", "La logistica debe dar ciclo")
    assert_true(result.attractors[0].period == 2, "La logistica r=3.2 debe dar periodo 2")


def test_complex_cycle():
    recurrence = quadratic_complex_map(c=-0.123 + 0.745j)
    result = analyze_recurrence(
        recurrence,
        max_terms=800,
        tolerance=1e-7,
        attractor_window=120,
        max_period=8,
    )

    assert_true(not result.diverged, "El ejemplo complejo no deberia divergir")
    assert_true(result.attractors, "No se detecto atractor complejo")
    assert_true(result.attractors[0].period == 3, "El ejemplo complejo debe dar periodo 3")


def test_divergence():
    recurrence = Recurrence(
        name="divergent_square",
        initial_value=2.0,
        step=lambda x, _n: x * x,
    )
    result = analyze_recurrence(recurrence, max_terms=50, divergence_limit=1e6)

    assert_true(result.diverged, "La sucesion x[n+1]=x[n]^2 desde 2 debe divergir")
    assert_true(result.reason == "divergence detected", "Motivo de divergencia inesperado")


def test_direct_attractor_detection():
    terms = [
        SequenceTerm(index=i, value=1.0 if i % 2 == 0 else -1.0)
        for i in range(40)
    ]
    attractor = detect_attractor(
        terms=terms,
        tolerance=1e-8,
        attractor_window=20,
        max_period=6,
        min_repetitions=5,
    )

    assert_true(attractor is not None, "No se detecto el ciclo manual")
    assert_true(attractor.kind == "cycle", "El ciclo manual debe clasificarse como cycle")
    assert_true(attractor.period == 2, "El ciclo manual debe tener periodo 2")


def test_transient_terms_control():
    terms = [
        SequenceTerm(index=i, value=float(i))
        for i in range(10)
    ]
    terms.extend(
        SequenceTerm(index=i + 10, value=1.0 if i % 2 == 0 else -1.0)
        for i in range(40)
    )

    attractors = detect_attractors(
        terms=terms,
        tolerance=1e-8,
        attractor_window=50,
        max_period=6,
        min_repetitions=5,
        transient_terms=10,
    )

    assert_true(attractors, "No se detecto ciclo despues del transitorio")
    assert_true(attractors[0].period == 2, "El transitorio debe ignorarse al detectar")
    assert_true(attractors[0].start_index >= 10, "El atractor debe empezar tras el transitorio")


def test_approximate_cycle_detection():
    terms = []
    base_cycle = [0.2, 0.8]
    for index in range(80):
        noise = 1e-4 if index % 11 == 0 else 0.0
        terms.append(
            SequenceTerm(
                index=index,
                value=base_cycle[index % 2] + noise,
            )
        )

    attractors = detect_attractors(
        terms=terms,
        tolerance=1e-5,
        attractor_window=60,
        max_period=6,
        min_repetitions=5,
        stability_ratio=0.8,
        cluster_tolerance=1e-3,
    )

    assert_true(attractors, "No se detecto ciclo aproximado")
    assert_true(attractors[0].period == 2, "El ciclo aproximado debe tener periodo 2")


def test_unresolved_behavior_classification():
    recurrence = logistic_map(r=4.0, x0=0.123456)
    result = analyze_recurrence(
        recurrence,
        max_terms=500,
        tolerance=1e-10,
        attractor_window=200,
        max_period=12,
        stability_ratio=0.98,
    )

    assert_true(not result.diverged, "La logistica r=4 debe quedar acotada")
    assert_true(
        result.behavior == "chaotic_or_unresolved",
        "La logistica r=4 debe clasificarse como no resuelta/caotica",
    )
    assert_true(not result.attractors, "No debe detectar atractor estable espurio")


def test_uniform_subdivision():
    subdivision = UniformSubdivision(ContinuousRange(-2.0, 2.0), steps=8)

    assert_true(subdivision.step_size == 0.5, "Tamano de celda inesperado")
    assert_true(subdivision.index_of(-2.0) == 0, "Borde inferior incorrecto")
    assert_true(subdivision.index_of(2.0) == 7, "Borde superior incorrecto")
    assert_true(subdivision.index_of(0.3) == 4, "Indice interno incorrecto")
    assert_true(subdivision.index_of(3.0) is None, "Fuera de dominio debe ser None")
    assert_true(abs(subdivision.center_of(4) - 0.25) < 1e-12, "Centro incorrecto")


def test_cartesian_discrete_transform():
    grid = DiscretizationGrid(
        {
            "x": UniformSubdivision(ContinuousRange(-2.0, 2.0), steps=4),
            "y": UniformSubdivision(ContinuousRange(-2.0, 2.0), steps=4),
        }
    )
    transformer = DiscreteTransformer(ComplexCartesianCoordinates(), grid)

    point = transformer.transform_value(0.4 + 1.2j)

    assert_true(point.inside_domain, "El punto cartesiano debe caer dentro del dominio")
    assert_true(point.indices == {"x": 2, "y": 3}, "Indices cartesianos incorrectos")
    assert_true(point.cell_id == 11, "cell_id cartesiano incorrecto")
    assert_true(
        point.discrete_coordinates == {"x": 0.5, "y": 1.5},
        "Coordenadas discretas cartesianas incorrectas",
    )


def test_polar_discrete_transform():
    grid = DiscretizationGrid(
        {
            "r": UniformSubdivision(ContinuousRange(0.0, 2.0), steps=4),
            "theta": UniformSubdivision(ContinuousRange(0.0, 2 * math.pi), steps=8),
        }
    )
    transformer = DiscreteTransformer(ComplexPolarCoordinates(), grid)

    point = transformer.transform_value(1.0 + 1.0j)

    assert_true(point.inside_domain, "El punto polar debe caer dentro del dominio")
    assert_true(point.indices == {"r": 2, "theta": 1}, "Indices polares incorrectos")
    assert_true(point.cell_id == 17, "cell_id polar incorrecto")


def test_projection_and_sequence_transform():
    recurrence = logistic_map(r=3.2, x0=0.2)
    result = analyze_recurrence(recurrence, max_terms=80, tolerance=1e-7)
    transformer = DiscreteTransformer(
        ProjectionXCoordinates(),
        DiscretizationGrid(
            {"x": UniformSubdivision(ContinuousRange(0.0, 1.0), steps=10)}
        ),
    )

    discrete_terms = transform_terms(result.terms, transformer)

    assert_true(len(discrete_terms) == len(result.terms), "Faltan terminos discretizados")
    assert_true(
        all(item.point.inside_domain for item in discrete_terms),
        "La logistica debe permanecer dentro de [0, 1]",
    )


def test_attractor_transform():
    attractor = Attractor(
        values=[0.25, 0.75],
        kind="cycle",
        period=2,
        tolerance=1e-8,
    )
    transformer = DiscreteTransformer(
        ProjectionXCoordinates(),
        DiscretizationGrid(
            {"x": UniformSubdivision(ContinuousRange(0.0, 1.0), steps=4)}
        ),
    )

    discrete_attractors = transform_attractors([attractor], transformer)
    indices = [point.indices["x"] for point in discrete_attractors[0].points]

    assert_true(indices == [1, 3], "Indices discretos del atractor incorrectos")


def test_reproducible_concept_pipeline():
    config = default_concept_config()
    result = run_exploration(config)

    assert_true(result.config == config, "El resultado debe conservar la configuracion")
    assert_true(result.terms, "El pipeline debe devolver terminos discretizados")
    assert_true(
        len(result.terms) == len(result.sequence_result.terms),
        "El resultado discreto debe conservar todos los terminos",
    )


def test_logistic_quadratic_conjugacy():
    r_value = 3.2
    x_value = 0.2
    mapped = logistic_to_quadratic(r_value, x_value)
    recovered = quadratic_to_logistic(mapped.c, mapped.z0)

    assert_true(recovered is not None, "La conversion inversa debe existir en el eje real")
    assert_true(abs(recovered.r - r_value) < 1e-12, "r no se recupera correctamente")
    assert_true(abs(recovered.x0 - x_value) < 1e-12, "x0 no se recupera correctamente")

    z_value = mapped.z0
    for _ in range(30):
        x_value = r_value * x_value * (1.0 - x_value)
        z_value = z_value * z_value + mapped.c
        expected = r_value * (0.5 - x_value)
        assert_true(
            abs(z_value - expected) < 1e-10,
            "Las orbitas conjugadas deben coincidir en cada iteracion",
        )

    assert_true(
        quadratic_to_logistic(-0.2 + 0.1j) is None,
        "Un parametro complejo no real no debe convertirse a logistica real",
    )


def test_session_payload_round_trip():
    config = default_concept_config()
    case = ExplorationCase(
        identifier="case-001",
        label="Q1: ejemplo",
        recurrence=RecurrenceConfig(
            name="quadratic_complex",
            parameters={
                "c_real": -0.123,
                "c_imag": 0.745,
                "z0_real": 0.0,
                "z0_imag": 0.0,
            },
        ),
        source="manual",
    )
    result = run_exploration(config)
    payload = build_session_data(
        config,
        [case],
        [(case, result)],
        active_case_id=case.identifier,
        table_mode="todo",
        map_limits={"z^2+c": (-2.0, 1.0, -1.35, 1.35)},
        music={
            "criterion": "armonia",
            "segmentation": "sectores",
            "scale": "menor",
        },
    )
    restored = parse_session_data(payload)

    assert_true(restored.config == config, "La configuracion de sesion debe conservarse")
    assert_true(restored.cases == [case], "Los casos de sesion deben conservarse")
    assert_true(restored.active_case_id == case.identifier, "El caso activo debe conservarse")
    assert_true(
        restored.music["criterion"] == "armonia",
        "La configuracion musical debe conservarse",
    )
    assert_true(payload["results"][0]["result"]["terms"], "La sesion debe exportar terminos")
    assert_true(
        payload["results"][0]["result"]["discrete_terms"],
        "La sesion debe exportar terminos discretizados",
    )


def test_strudel_generation_from_attractors():
    config = default_concept_config()
    result = run_exploration(config)
    pattern = generate_strudel_pattern(
        [("ejemplo", result)],
        MusicMappingConfig(
            criterion="harmony",
            segmentation="sectors",
            scale="minor",
            root="C4",
        ),
    )

    assert_true(pattern.attractors, "Los atractores deben producir eventos musicales")
    assert_true("note(" in pattern.code, "La armonia debe generar un patron note de Strudel")
    assert_true(
        all(event.notes for event in pattern.attractors),
        "Cada atractor musical debe contener notas",
    )

    silent = generate_strudel_pattern([], MusicMappingConfig())
    assert_true(silent.code == "silence", "Sin atractores el patron debe ser silence")
    assert_true(
        result.transformer.coordinate_system.name == "complex_polar",
        "El caso base debe usar coordenadas polares",
    )
    assert_true(
        result.sequence_result.behavior in {
            "fixed_point",
            "periodic",
            "chaotic_or_unresolved",
            "insufficient_data",
        },
        "Comportamiento del caso base inesperado",
    )


def run_test(name, function):
    function()
    print(f"[OK] {name}")


def main():
    tests = [
        ("imports publicos y modelos", test_public_imports_and_models),
        ("punto fijo coseno", test_cosine_fixed_point),
        ("ciclo logistico", test_logistic_cycle),
        ("ciclo complejo", test_complex_cycle),
        ("divergencia", test_divergence),
        ("deteccion directa de atractor", test_direct_attractor_detection),
        ("control de transitorio", test_transient_terms_control),
        ("ciclo aproximado", test_approximate_cycle_detection),
        ("clasificacion no resuelta", test_unresolved_behavior_classification),
        ("subdivision uniforme", test_uniform_subdivision),
        ("transformacion cartesiana", test_cartesian_discrete_transform),
        ("transformacion polar", test_polar_discrete_transform),
        ("proyeccion y terminos discretos", test_projection_and_sequence_transform),
        ("atractores discretos", test_attractor_transform),
        ("pipeline reproducible", test_reproducible_concept_pipeline),
        ("conjugacion logistica-cuadratica", test_logistic_quadratic_conjugacy),
        ("sesion JSON completa", test_session_payload_round_trip),
        ("generacion Strudel desde atractores", test_strudel_generation_from_attractors),
    ]

    print("Probando framework de sucesiones fractales...\n")
    for name, function in tests:
        run_test(name, function)

    print("\nTodas las pruebas pasaron correctamente.")


if __name__ == "__main__":
    main()
