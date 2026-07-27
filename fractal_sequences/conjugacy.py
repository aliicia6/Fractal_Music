"""Affine conjugacy between the logistic and quadratic complex maps.

The logistic map ``x -> r*x*(1-x)`` is conjugate to
``z -> z**2 + c`` through ``z = r*(1/2 - x)``.  This module keeps that
relationship explicit so visual tools do not need to duplicate the formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuadraticParameters:
    """Parameter and initial value for a quadratic complex recurrence."""

    c: complex
    z0: complex


@dataclass(frozen=True)
class LogisticParameters:
    """Parameter and initial value for a logistic recurrence."""

    r: float
    x0: float


def logistic_to_quadratic(r: float, x0: float) -> QuadraticParameters:
    """Map a logistic orbit to its affine-conjugate quadratic orbit.

    ``c`` always belongs to the real axis of the Mandelbrot parameter plane.
    No range restriction is imposed here, which also makes the function useful
    when studying logistic maps outside the customary ``[0, 4]`` interval.
    """

    r_value = float(r)
    x_value = float(x0)
    return QuadraticParameters(
        c=complex(r_value * (2.0 - r_value) / 4.0, 0.0),
        z0=complex(r_value * (0.5 - x_value), 0.0),
    )


def quadratic_to_logistic(
    c: complex,
    z0: complex = 0j,
    *,
    tolerance: float = 1e-9,
) -> LogisticParameters | None:
    """Map a real quadratic parameter back to the principal logistic branch.

    A genuinely complex ``c`` or ``z0`` has no real logistic counterpart, so
    the function returns ``None``.  For ``0 < c < 1/4`` two logistic parameters
    exist; the principal branch ``r = 1 + sqrt(1 - 4c)`` is selected.
    """

    c_value = complex(c)
    z_value = complex(z0)
    if abs(c_value.imag) > tolerance or abs(z_value.imag) > tolerance:
        return None

    discriminant = 1.0 - 4.0 * c_value.real
    if discriminant < -tolerance:
        return None

    r_value = 1.0 + math.sqrt(max(0.0, discriminant))
    if abs(r_value) <= tolerance:
        return None

    return LogisticParameters(
        r=r_value,
        x0=0.5 - z_value.real / r_value,
    )
