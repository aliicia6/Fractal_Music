"""Generic recurrence definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .models import Number

StepFunction = Callable[[Number, int], Number]


@dataclass(frozen=True)
class Recurrence:
    """Definition of x[n + 1] = f(x[n], n)."""

    name: str
    initial_value: Number
    step: StepFunction

    def next_value(self, current_value: Number, index: int) -> Number:
        return self.step(current_value, index)
