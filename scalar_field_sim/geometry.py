from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]

    def as_tuple(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class SourceGeometry:
    position: tuple[float, float]
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioGeometry:
    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[WallSegment, ...]
    sources: tuple[SourceGeometry, ...]

    @property
    def num_walls(self) -> int:
        return len(self.walls)

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def wall_tuples(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return [wall.as_tuple() for wall in self.walls]

    def source_centers(self) -> list[tuple[float, float]]:
        return [src.position for src in self.sources]


@dataclass(frozen=True, slots=True)
class WallGeometry:
    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    walls: tuple[WallSegment, ...]

    @classmethod
    def from_scenario(cls, scenario: ScenarioGeometry) -> "WallGeometry":
        return cls(
            name=scenario.name,
            x_range=scenario.x_range,
            y_range=scenario.y_range,
            walls=scenario.walls,
        )
