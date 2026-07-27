"""Small manual test for the recurrence framework."""

from fractal_sequences.analyzer import analyze_recurrence
from fractal_sequences.examples import (
    cosine_fixed_point,
    logistic_map,
    quadratic_complex_map,
)


def print_result(title, result, tail=8):
    print(f"\n{title}")
    print(f"reason: {result.reason}")
    print(f"diverged: {result.diverged}")
    print("last terms:")
    for term in result.terms[-tail:]:
        print(f"  {term.index}: {term.value}")

    if result.attractors:
        print("attractors:")
        for attractor in result.attractors:
            print(
                f"  {attractor.kind}, period={attractor.period}, "
                f"values={attractor.values}"
            )
    else:
        print("attractors: none detected")


def main():
    examples = [
        cosine_fixed_point(x0=1.0),
        logistic_map(r=3.2, x0=0.2),
        quadratic_complex_map(c=-0.123 + 0.745j),
    ]

    for recurrence in examples:
        result = analyze_recurrence(
            recurrence,
            max_terms=1000,
            tolerance=1e-7,
            divergence_limit=1e6,
            attractor_window=120,
            max_period=16,
        )
        print_result(recurrence.name, result)


if __name__ == "__main__":
    main()
