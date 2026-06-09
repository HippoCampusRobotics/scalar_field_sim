"""Geometry container types for scalar field simulation scenarios.

These dataclasses describe the static geometric parts of a scenario:
- field bounds,
- wall segments,
- source positions.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


Point2D = tuple[float, float]


def _validate_point2d(name: str, point: Point2D) -> None:
    """Validate that a 2D point contains two finite coordinates."""
    if len(point) != 2:
        raise ValueError(f"{name} must have length 2, got {point}.")
    if not all(isfinite(v) for v in point):
        raise ValueError(f"{name} must contain only finite values, got {point}.")


def _validate_range(name: str, value_range: tuple[float, float]) -> None:
    """Validate that a numeric interval is finite and ordered."""
    if len(value_range) != 2:
        raise ValueError(f"{name} must have length 2, got {value_range}.")
    v_min, v_max = value_range
    if not isfinite(v_min) or not isfinite(v_max):
        raise ValueError(f"{name} must contain finite values, got {value_range}.")
    if v_min > v_max:
        raise ValueError(f"{name} must satisfy min <= max, got {value_range}.")


@dataclass(frozen=True, slots=True)
class WallSegment:
    """One wall segment in 2D.

    Attributes
    ----------
    start
        Start point of the wall segment as (x, y).
    end
        End point of the wall segment as (x, y).
    """

    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        _validate_point2d("WallSegment.start", self.start)
        _validate_point2d("WallSegment.end", self.end)

    def as_tuple(self) -> tuple[Point2D, Point2D]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class SourceGeometry:
    """Geometric description of one source.

    Attributes
    ----------
    position
        Source center position as (x, y).
    name
        Optional source name for visualization or debugging. Currently not used.
    """

    position: Point2D
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_point2d("SourceGeometry.position", self.position)


@dataclass(frozen=True, slots=True)
class ScenarioGeometry:
    """Static geometric description of one full scenario.

    Attributes
    ----------
    name
        Scenario name.
    x_range
        Inclusive field bounds in x as (x_min, x_max).
    y_range
        Inclusive field bounds in y as (y_min, y_max).
    walls
        All wall segments in the scenario.
    sources
        All source positions in the scenario.
    """

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[WallSegment, ...]
    sources: tuple[SourceGeometry, ...]

    def __post_init__(self) -> None:
        _validate_range("ScenarioGeometry.x_range", self.x_range)
        _validate_range("ScenarioGeometry.y_range", self.y_range)

    @property
    def num_walls(self) -> int:
        return len(self.walls)

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def wall_tuples(self) -> list[tuple[Point2D, Point2D]]:
        return [wall.as_tuple() for wall in self.walls]

    def source_centers(self) -> list[Point2D]:
        return [src.position for src in self.sources]


@dataclass(frozen=True, slots=True)
class WallGeometry:
    """Reduced geometry view containing only bounds and walls.

    This is useful for code that only needs obstacle geometry and does not care
    about source positions.
    """

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[WallSegment, ...]

    def __post_init__(self) -> None:
        _validate_range("WallGeometry.x_range", self.x_range)
        _validate_range("WallGeometry.y_range", self.y_range)

    @classmethod
    def from_scenario(cls, scenario: ScenarioGeometry) -> "WallGeometry":
        return cls(
            name=scenario.name,
            x_range=scenario.x_range,
            y_range=scenario.y_range,
            walls=scenario.walls,
        )
