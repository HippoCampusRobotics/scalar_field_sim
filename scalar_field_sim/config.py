from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from scalar_field_sim.geometry import (
    ScenarioGeometry,
    SourceGeometry,
    WallSegment,
)
from scalar_field_sim.field import (
    SimulationSourceSpec,
    WallAwareGaussianField2D,
)


@dataclass(frozen=True, slots=True)
class FieldSimConfig:
    name: str
    frame_id: str
    geometry: ScenarioGeometry
    source_specs: tuple[SimulationSourceSpec, ...]
    background_floor: float
    measurement_noise_std: float
    clip_range: tuple[float, float] | None
    seed: int | None
    visualization_grid_step: float | None = None


def load_field_sim_config(path: str | Path) -> FieldSimConfig:
    path = Path(path)

    with path.open("rb") as f:
        cfg = tomllib.load(f)

    name = cfg.get("general", {}).get("name", path.stem)
    frame_id = cfg.get("frame", {}).get("frame_id", "map")

    geometry = _parse_geometry(cfg, default_name=name)
    source_specs = _parse_source_specs(cfg)

    sim_cfg = cfg.get("simulation", {})
    background_floor = float(sim_cfg.get("background_floor", 0.0))
    measurement_noise_std = float(sim_cfg.get("measurement_noise_std", 0.0))
    clip_range = _parse_clip_range(sim_cfg)
    seed = sim_cfg.get("seed", None)
    if seed is not None:
        seed = int(seed)

    vis_cfg = cfg.get("visualization", {})
    visualization_grid_step = vis_cfg.get("grid_step", None)
    if visualization_grid_step is not None:
        visualization_grid_step = float(visualization_grid_step)

    return FieldSimConfig(
        name=name,
        frame_id=frame_id,
        geometry=geometry,
        source_specs=source_specs,
        background_floor=background_floor,
        measurement_noise_std=measurement_noise_std,
        clip_range=clip_range,
        seed=seed,
        visualization_grid_step=visualization_grid_step,
    )


def build_field_from_config(config: FieldSimConfig) -> WallAwareGaussianField2D:
    return WallAwareGaussianField2D(
        sources=list(config.source_specs),
        walls=list(config.geometry.walls),
        background_floor=config.background_floor,
        measurement_noise_std=config.measurement_noise_std,
        clip_range=config.clip_range,
        x_range=config.geometry.x_range,
        y_range=config.geometry.y_range,
    )


def load_field_from_toml(
    path: str | Path,
) -> tuple[FieldSimConfig, WallAwareGaussianField2D]:
    config = load_field_sim_config(path)
    field = build_field_from_config(config)
    return config, field


def _parse_geometry(cfg: dict, default_name: str) -> ScenarioGeometry:
    bounds_cfg = cfg["bounds"]
    x_range = (float(bounds_cfg["x_min"]), float(bounds_cfg["x_max"]))
    y_range = (float(bounds_cfg["y_min"]), float(bounds_cfg["y_max"]))

    objects_cfg = cfg.get("objects", {})

    walls = _parse_walls(objects_cfg.get("walls", {}))
    sources = _parse_source_geometry(objects_cfg.get("sources", {}))

    return ScenarioGeometry(
        name=default_name,
        x_range=x_range,
        y_range=y_range,
        walls=walls,
        sources=sources,
    )


def _parse_walls(walls_cfg: dict) -> tuple[WallSegment, ...]:
    wall_vertices = walls_cfg.get("wall_vertices", [])
    walls: list[WallSegment] = []

    for i, wall in enumerate(wall_vertices):
        if len(wall) != 4:
            raise ValueError(f"Wall {i} must have format [x1, y1, x2, y2], got {wall}")
        x1, y1, x2, y2 = map(float, wall)
        walls.append(WallSegment(start=(x1, y1), end=(x2, y2)))

    return tuple(walls)


def _parse_source_geometry(sources_cfg: dict) -> tuple[SourceGeometry, ...]:
    sources: list[SourceGeometry] = []

    if "source_specs" in sources_cfg:
        for i, spec in enumerate(sources_cfg["source_specs"]):
            pos = spec.get("position", spec.get("center"))
            if pos is None or len(pos) != 2:
                raise ValueError(
                    f"Source spec {i} must define position=[x,y] or center=[x,y]."
                )
            x, y = map(float, pos)
            name = spec.get("name", f"source_{i}")
            sources.append(SourceGeometry(position=(x, y), name=name))
        return tuple(sources)

    for i, pos in enumerate(sources_cfg.get("source_positions", [])):
        if len(pos) != 2:
            raise ValueError(f"Source {i} must have format [x, y], got {pos}")
        x, y = map(float, pos)
        sources.append(SourceGeometry(position=(x, y), name=f"source_{i}"))

    return tuple(sources)


def _parse_source_specs(cfg: dict) -> tuple[SimulationSourceSpec, ...]:
    sources_cfg = cfg.get("objects", {}).get("sources", {})

    if "source_specs" in sources_cfg:
        specs: list[SimulationSourceSpec] = []
        for i, spec in enumerate(sources_cfg["source_specs"]):
            pos = spec.get("position", spec.get("center"))
            if pos is None or len(pos) != 2:
                raise ValueError(
                    f"Source spec {i} must define position=[x,y] or center=[x,y]."
                )

            specs.append(
                SimulationSourceSpec(
                    center=(float(pos[0]), float(pos[1])),
                    amplitude=float(spec["amplitude"]),
                    sigma_x=float(spec["sigma_x"]),
                    sigma_y=float(spec["sigma_y"]),
                    blocking_strength=float(spec.get("blocking_strength", 1.0)),
                    name=spec.get("name", f"source_{i}"),
                )
            )
        return tuple(specs)

    source_positions = sources_cfg.get("source_positions", [])
    defaults = cfg.get("simulation", {}).get("source_defaults", {})

    required = ("amplitude", "sigma_x", "sigma_y")
    missing = [key for key in required if key not in defaults]
    if source_positions and missing:
        raise ValueError(
            "TOML uses objects.sources.source_positions, but simulation.source_defaults "
            f"is missing required keys: {missing}"
        )

    specs = []
    for i, pos in enumerate(source_positions):
        if len(pos) != 2:
            raise ValueError(f"Source {i} must have format [x, y], got {pos}")

        specs.append(
            SimulationSourceSpec(
                center=(float(pos[0]), float(pos[1])),
                amplitude=float(defaults["amplitude"]),
                sigma_x=float(defaults["sigma_x"]),
                sigma_y=float(defaults["sigma_y"]),
                blocking_strength=float(defaults.get("blocking_strength", 1.0)),
                name=f"source_{i}",
            )
        )

    return tuple(specs)


def _parse_clip_range(sim_cfg: dict) -> tuple[float, float] | None:
    clip_min = sim_cfg.get("clip_min", None)
    clip_max = sim_cfg.get("clip_max", None)

    if clip_min is None and clip_max is None:
        return None
    if clip_min is None or clip_max is None:
        raise ValueError(
            "clip_min and clip_max must either both be set or both be omitted."
        )

    clip_range = (float(clip_min), float(clip_max))
    if clip_range[1] < clip_range[0]:
        raise ValueError("clip_max must be >= clip_min.")

    return clip_range
