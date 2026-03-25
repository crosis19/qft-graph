"""Boundary condition implementations for lattice geometries."""

from __future__ import annotations

from enum import Enum


class BoundaryCondition(Enum):
    """Supported boundary conditions."""

    PERIODIC = "periodic"
    OPEN = "open"
    ANTIPERIODIC = "antiperiodic"

    @staticmethod
    def from_string(s: str) -> BoundaryCondition:
        return BoundaryCondition(s.lower())
