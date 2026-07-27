"""Run the main concept case without opening the visual interface.

Run with:
    python demo_concepto.py
"""

from fractal_sequences import default_concept_config, run_exploration


def format_value(value):
    if isinstance(value, complex):
        return f"{value.real:.8g}{value.imag:+.8g}j"
    return f"{value:.8g}"


def main():
    config = default_concept_config()
    result = run_exploration(config)
    sequence = result.sequence_result

    print("Demo de concepto: sucesion compleja z[n+1] = z[n]^2 + c")
    print(f"recurrencia: {config.recurrence}")
    print(f"coordenadas: {config.discretization.coordinate_system}")
    print(f"terminos: {len(sequence.terms)}")
    print(f"comportamiento: {sequence.behavior}")
    print(f"motivo: {sequence.reason}")
    print()

    if sequence.attractors:
        print("Atractores:")
        for index, discrete_attractor in enumerate(result.attractors, start=1):
            attractor = discrete_attractor.attractor
            print(f"  A{index}: {attractor.kind}, periodo={attractor.period}")
            for point_index, point in enumerate(discrete_attractor.points, start=1):
                print(
                    "    "
                    f"{point_index}: valor={format_value(point.value)}, "
                    f"coord={point.coordinates}, "
                    f"idx={point.indices}, "
                    f"centro={point.discrete_coordinates}, "
                    f"celda={point.cell_id}"
                )
    else:
        print("Atractores: no detectados")

    print()
    print("Ultimos terminos discretizados:")
    for discrete_term in result.terms[-8:]:
        point = discrete_term.point
        print(
            f"  n={discrete_term.term.index}: "
            f"valor={format_value(point.value)}, "
            f"idx={point.indices}, celda={point.cell_id}"
        )


if __name__ == "__main__":
    main()
