"""Reusable example recurrences."""

from __future__ import annotations

import math

from .recurrence import Recurrence


def logistic_map(r: float, x0: float) -> Recurrence:
    """x[n + 1] = r * x[n] * (1 - x[n])."""

    return Recurrence(
        name=f"logistic_map(r={r}, x0={x0})",
        initial_value=x0,
        step=lambda x, _n: r * x.real * (1 - x.real),
    )


def quadratic_complex_map(c: complex, z0: complex = 0j) -> Recurrence:
    """z[n + 1] = z[n] ** 2 + c."""

    return Recurrence(
        name=f"quadratic_complex_map(c={c}, z0={z0})",
        initial_value=z0,
        step=lambda z, _n: z * z + c,
    )


def cosine_fixed_point(x0: float) -> Recurrence:
    """x[n + 1] = cos(x[n])."""

    return Recurrence(
        name=f"cosine_fixed_point(x0={x0})",
        initial_value=x0,
        step=lambda x, _n: math.cos(x.real),
    )
