"""Coordinate systems and projections for sequence values."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .models import Number


CoordinateMap = dict[str, float]


class CoordinateSystem(ABC):
    """Strategy that projects a real or complex value into coordinates."""

    name: str
    axes: tuple[str, ...]

    @abstractmethod
    def coordinates_from(self, value: Number) -> CoordinateMap:
        """Return the coordinates associated with a sequence value."""


class RealLineCoordinates(CoordinateSystem):
    name = "real_line"
    axes = ("x",)

    def coordinates_from(self, value: Number) -> CoordinateMap:
        return {"x": _real_part(value)}


class ComplexCartesianCoordinates(CoordinateSystem):
    name = "complex_cartesian"
    axes = ("x", "y")

    def coordinates_from(self, value: Number) -> CoordinateMap:
        z = complex(value)
        return {"x": z.real, "y": z.imag}


class ComplexPolarCoordinates(CoordinateSystem):
    name = "complex_polar"
    axes = ("r", "theta")

    def coordinates_from(self, value: Number) -> CoordinateMap:
        z = complex(value)
        theta = math.atan2(z.imag, z.real)
        if theta < 0:
            theta += 2 * math.pi
        return {"r": abs(z), "theta": theta}


class ProjectionXCoordinates(CoordinateSystem):
    name = "projection_x"
    axes = ("x",)

    def coordinates_from(self, value: Number) -> CoordinateMap:
        return {"x": complex(value).real}


class ProjectionYCoordinates(CoordinateSystem):
    name = "projection_y"
    axes = ("y",)

    def coordinates_from(self, value: Number) -> CoordinateMap:
        return {"y": complex(value).imag}


class RadiusProjectionCoordinates(CoordinateSystem):
    name = "radius_projection"
    axes = ("r",)

    def coordinates_from(self, value: Number) -> CoordinateMap:
        return {"r": abs(value)}


class AngleProjectionCoordinates(CoordinateSystem):
    name = "angle_projection"
    axes = ("theta",)

    def coordinates_from(self, value: Number) -> CoordinateMap:
        z = complex(value)
        theta = math.atan2(z.imag, z.real)
        if theta < 0:
            theta += 2 * math.pi
        return {"theta": theta}


def _real_part(value: Number) -> float:
    z = complex(value)
    if abs(z.imag) > 0:
        raise ValueError("real-line coordinates require a real value")
    return z.real
