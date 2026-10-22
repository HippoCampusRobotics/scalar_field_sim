"""Load scalar field simulation scenarios from TOML files.

This module parses a TOML scenario description into a typed
`FieldSimConfig` and builds the corresponding `WallAwareGaussianField2D`.

A scenario file either defines all tables inline or references one shared
field parameter file via the top-level key `field_file`. In the latter case
the scenario file may only contain `field_file`, `[general]`, and
`[objects.walls]`; everything else comes from the shared file. References
are resolved relative to the scenario file and are not recursive.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scalar_field_sim.field import (
    SimulationSourceSpec,
    WallAwareGaussianField2D,
)
from scalar_field_sim.geometry import (
    ScenarioGeometry,
    SourceGeometry,
    WallSegment,
)


@dataclass(frozen=True, slots=True)
class FieldSimConfig:
    """Configuration for one scalar field simulation scenario.

    Attributes
    ----------
    name
        Human-readable scenario name.
    frame_id
        ROS frame in which the field is defined. Should be "map".
    geometry
        Rectangular bounds, wall segments, and source positions.
        See definition in geometry.py.
    source_specs
        Source parameters used to construct the simulated field.
        See definition in field.py.
    background_level
        Constant additive background level of the latent field.
    measurement_noise_std
        Standard deviation of additive measurement noise.
    clip_range
        Optional clipping range applied to sampled measurements.
    seed
        Optional random seed for reproducible noisy sampling.
    visualization_grid_step
        Optional suggested grid spacing for ground-truth visualization.
    field_value_unit
        Optional unit of the field values, from `[units] field_value`.
    description
        Optional scenario description, from `[general] description`.
    field_file
        Resolved path of the shared field parameter file, if referenced.
    """

    name: str
    frame_id: str
    geometry: ScenarioGeometry
    source_specs: tuple[SimulationSourceSpec, ...]
    background_level: float
    measurement_noise_std: float
    clip_range: tuple[float, float] | None
    seed: int | None
    visualization_grid_step: float | None = None
    field_value_unit: str | None = None
    description: str | None = None
    field_file: Path | None = None


def load_field_sim_config(path: str | Path) -> FieldSimConfig:
    """Parse one TOML scenario file into a `FieldSimConfig`.

    The TOML file defines:
    - general scenario metadata,
    - frame id,
    - rectangular field bounds,
    - optional walls,
    - source positions or full source specifications,
    - simulation parameters such as noise, clipping, and random seed.

    If the top-level key `field_file` is present, all tables except
    `[general]` and `[objects.walls]` are read from that shared file instead,
    see `_combine_with_field_file`.

    Raises
    ------
    ValueError
        If inline and referenced field definitions conflict, if `field_file`
        references are nested, or if the shared file is not valid TOML.
    FileNotFoundError
        If the referenced shared field file does not exist.
    """
    path = Path(path)

    with path.open('rb') as f:
        cfg = tomllib.load(f)

    field_file = None
    if 'field_file' in cfg:
        field_file, cfg = _combine_with_field_file(path, cfg)

    general_cfg = cfg.get('general', {})
    name = general_cfg.get('name', path.stem)
    description = general_cfg.get('description', None)
    if description is not None:
        description = str(description)
    frame_id = cfg.get('frame', {}).get('frame_id', 'map')

    geometry = _parse_geometry(cfg, default_name=name)
    source_specs = _parse_source_specs(cfg)

    sim_cfg = cfg.get('simulation', {})
    background_level = float(sim_cfg.get('background_level', 0.0))
    measurement_noise_std = float(sim_cfg.get('measurement_noise_std', 0.0))
    clip_range = _parse_clip_range(sim_cfg)
    seed = sim_cfg.get('seed', None)
    if seed is not None:
        seed = int(seed)

    vis_cfg = cfg.get('visualization', {})
    visualization_grid_step = vis_cfg.get('grid_step', None)
    if visualization_grid_step is not None:
        visualization_grid_step = float(visualization_grid_step)

    field_value_unit = cfg.get('units', {}).get('field_value', None)
    if field_value_unit is not None:
        field_value_unit = str(field_value_unit)

    return FieldSimConfig(
        name=name,
        frame_id=frame_id,
        geometry=geometry,
        source_specs=source_specs,
        background_level=background_level,
        measurement_noise_std=measurement_noise_std,
        clip_range=clip_range,
        seed=seed,
        visualization_grid_step=visualization_grid_step,
        field_value_unit=field_value_unit,
        description=description,
        field_file=field_file,
    )


def build_field_from_config(config: FieldSimConfig) -> WallAwareGaussianField2D:
    """Construct a scalar field from a parsed config.
    See field.py for definition of field object WallAwareGaussianField2D."""
    rng = np.random.default_rng(config.seed) if config.seed is not None else None
    return WallAwareGaussianField2D(
        sources=list(config.source_specs),
        walls=list(config.geometry.walls),
        background_level=config.background_level,
        measurement_noise_std=config.measurement_noise_std,
        clip_range=config.clip_range,
        rng=rng,
        x_range=config.geometry.x_range,
        y_range=config.geometry.y_range,
    )


def load_field_from_toml(
    path: str | Path,
) -> tuple[FieldSimConfig, WallAwareGaussianField2D]:
    """Convenience helper returning both parsed config and constructed field."""
    config = load_field_sim_config(path)
    field = build_field_from_config(config)
    return config, field


def _combine_with_field_file(path: Path, cfg: dict) -> tuple[Path, dict]:
    """Combine a scenario file with its referenced shared field file.

    Parameters
    ----------
    path
        Path of the scenario file containing `field_file`.
    cfg
        Parsed scenario file contents.

    Returns
    -------
    field_file : Path
        Resolved path of the shared field file. Relative references are
        resolved against the scenario file's directory.
    combined : dict
        Shared file tables plus the scenario `[general]` table and the
        scenario `[objects.walls]` table.

    Raises
    ------
    ValueError
        If the scenario defines anything besides `field_file`, `[general]`,
        and `[objects.walls]`, if the shared file defines `field_file`,
        `[general]`, or `[objects.walls]`, if `field_file` references the
        scenario itself, or if the shared file is not valid TOML.
    FileNotFoundError
        If the shared field file does not exist.
    """
    ref = cfg['field_file']
    if not isinstance(ref, str) or not ref:
        raise ValueError(
            f'{path}: field_file must be a non-empty string, got {ref!r}.'
        )
    field_file = Path(ref)
    if not field_file.is_absolute():
        field_file = path.parent / field_file
    field_file = field_file.resolve()
    if field_file == path.resolve():
        raise ValueError(
            f'Scenario {path} references itself via field_file; nested '
            'field_file references are not supported.'
        )

    objects_cfg = cfg.get('objects', {})
    conflicts = sorted(set(cfg) - {'field_file', 'general', 'objects'})
    extra_objects = sorted(set(objects_cfg) - {'walls'})
    conflicts += [f'objects.{key}' for key in extra_objects]
    if conflicts:
        raise ValueError(
            f'Scenario {path} references shared field file {field_file} but '
            f'also defines {conflicts} inline. With field_file, the scenario '
            'may only define [general] and [objects.walls]; define all other '
            'field parameters in the shared file.'
        )

    shared = _load_shared_field_file(field_file, path)
    shared_objects = shared.get('objects', {})

    combined = dict(shared)
    if 'general' in cfg:
        combined['general'] = cfg['general']
    combined['objects'] = {
        **shared_objects,
        'walls': objects_cfg.get('walls', {}),
    }
    return field_file, combined


def _load_shared_field_file(field_file: Path, path: Path) -> dict:
    """Load and validate a shared field file referenced by scenario `path`.

    Raises
    ------
    FileNotFoundError
        If `field_file` does not exist.
    ValueError
        If `field_file` is not valid TOML or defines `field_file`,
        `[general]`, or `[objects.walls]`.
    """
    if not field_file.is_file():
        raise FileNotFoundError(
            f'Shared field file {field_file} referenced by {path} not found.'
        )
    try:
        with field_file.open('rb') as f:
            shared = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f'Invalid TOML in shared field file {field_file} referenced by '
            f'{path}: {exc}'
        ) from exc

    if 'field_file' in shared:
        raise ValueError(
            f'Shared field file {field_file} referenced by {path} defines '
            'field_file itself; nested field_file references are not '
            'supported.'
        )

    forbidden = ['general'] if 'general' in shared else []
    if 'walls' in shared.get('objects', {}):
        forbidden.append('objects.walls')
    if forbidden:
        raise ValueError(
            f'Shared field file {field_file} referenced by {path} must not '
            f'define {forbidden}; [general] and [objects.walls] belong in '
            'the scenario file.'
        )
    return shared


def _parse_geometry(cfg: dict, default_name: str) -> ScenarioGeometry:
    """Parse bounds, walls, and source positions into `ScenarioGeometry`."""
    bounds_cfg = cfg['bounds']
    x_range = (float(bounds_cfg['x_min']), float(bounds_cfg['x_max']))
    y_range = (float(bounds_cfg['y_min']), float(bounds_cfg['y_max']))

    objects_cfg = cfg.get('objects', {})

    walls = _parse_walls(objects_cfg.get('walls', {}))
    sources = _parse_source_geometry(objects_cfg.get('sources', {}))

    return ScenarioGeometry(
        name=default_name,
        x_range=x_range,
        y_range=y_range,
        walls=walls,
        sources=sources,
    )


def _parse_walls(walls_cfg: dict) -> tuple[WallSegment, ...]:
    """Parse wall segments from `objects.walls.wall_vertices`."""
    wall_vertices = walls_cfg.get('wall_vertices', [])
    walls: list[WallSegment] = []

    for i, wall in enumerate(wall_vertices):
        if len(wall) != 4:
            raise ValueError(f'Wall {i} must have format [x1, y1, x2, y2], got {wall}')
        x1, y1, x2, y2 = map(float, wall)
        walls.append(WallSegment(start=(x1, y1), end=(x2, y2)))

    return tuple(walls)


def _parse_source_geometry(sources_cfg: dict) -> tuple[SourceGeometry, ...]:
    """Parse source positions used for geometry/visualization only."""
    sources: list[SourceGeometry] = []

    if 'source_specs' in sources_cfg:
        for i, spec in enumerate(sources_cfg['source_specs']):
            pos = spec.get('position', spec.get('center'))
            if pos is None or len(pos) != 2:
                raise ValueError(
                    f'Source spec {i} must define position=[x,y] or center=[x,y].'
                )
            x, y = map(float, pos)
            name = spec.get('name', f'source_{i}')
            sources.append(SourceGeometry(position=(x, y), name=name))
        return tuple(sources)

    for i, pos in enumerate(sources_cfg.get('source_positions', [])):
        if len(pos) != 2:
            raise ValueError(f'Source {i} must have format [x, y], got {pos}')
        x, y = map(float, pos)
        sources.append(SourceGeometry(position=(x, y), name=f'source_{i}'))

    return tuple(sources)


def _parse_source_specs(cfg: dict) -> tuple[SimulationSourceSpec, ...]:
    """Parse source parameters used to construct the simulated field."""
    sources_cfg = cfg.get('objects', {}).get('sources', {})

    if 'source_specs' in sources_cfg:
        specs: list[SimulationSourceSpec] = []
        for i, spec in enumerate(sources_cfg['source_specs']):
            pos = spec.get('position', spec.get('center'))
            if pos is None or len(pos) != 2:
                raise ValueError(
                    f'Source spec {i} must define position=[x,y] or center=[x,y].'
                )

            specs.append(
                SimulationSourceSpec(
                    center=(float(pos[0]), float(pos[1])),
                    amplitude=float(spec['amplitude']),
                    sigma_x=float(spec['sigma_x']),
                    sigma_y=float(spec['sigma_y']),
                    blocking_strength=float(spec.get('blocking_strength', 1.0)),
                    name=spec.get('name', f'source_{i}'),
                )
            )
        return tuple(specs)

    source_positions = sources_cfg.get('source_positions', [])
    defaults = cfg.get('simulation', {}).get('source_defaults', {})

    required = ('amplitude', 'sigma_x', 'sigma_y')
    missing = [key for key in required if key not in defaults]
    if source_positions and missing:
        raise ValueError(
            'TOML uses objects.sources.source_positions, but simulation.source_defaults '
            f'is missing required keys: {missing}'
        )

    specs = []
    for i, pos in enumerate(source_positions):
        if len(pos) != 2:
            raise ValueError(f'Source {i} must have format [x, y], got {pos}')

        specs.append(
            SimulationSourceSpec(
                center=(float(pos[0]), float(pos[1])),
                amplitude=float(defaults['amplitude']),
                sigma_x=float(defaults['sigma_x']),
                sigma_y=float(defaults['sigma_y']),
                blocking_strength=float(defaults.get('blocking_strength', 1.0)),
                name=f'source_{i}',
            )
        )

    return tuple(specs)


def _parse_clip_range(sim_cfg: dict) -> tuple[float, float] | None:
    """Parse optional measurement clipping limits."""
    clip_min = sim_cfg.get('clip_min', None)
    clip_max = sim_cfg.get('clip_max', None)

    if clip_min is None and clip_max is None:
        return None
    if clip_min is None or clip_max is None:
        raise ValueError(
            'clip_min and clip_max must either both be set or both be omitted.'
        )

    clip_range = (float(clip_min), float(clip_max))
    if clip_range[1] < clip_range[0]:
        raise ValueError('clip_max must be >= clip_min.')

    return clip_range
