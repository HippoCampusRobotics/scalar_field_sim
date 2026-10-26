"""Load scalar field simulation scenarios from TOML files.

This module parses a TOML scenario description into a typed
`FieldSimConfig` and builds the corresponding `WallAwareGaussianField2D`.

A scenario file either defines all tables inline (self-contained form) or
references one geometry file and one field file via the top-level keys
`geometry_file` and `field_file` (reference form):

- geometry file: `[general]`, `[frame]`, `[bounds]`, `[objects.walls]`, and
  optionally `[units] length`;
- field file: `[units]`, `[objects.sources]`, `[simulation]` (including
  `[simulation.source_defaults]`), and `[visualization]`;
- scenario: `geometry_file`, `field_file`, and `[general]` only.

References are resolved relative to the scenario file's directory and are
not recursive.
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
    geometry_file
        Resolved path of the referenced geometry file, `None` for
        self-contained scenarios.
    field_file
        Resolved path of the referenced field file, `None` for
        self-contained scenarios.
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
    geometry_file: Path | None = None
    field_file: Path | None = None


def resolve_package_path(
    path: str | Path, share_dir: str | Path | None = None
) -> Path:
    """Resolve a scenario, geometry, or field path of this package.

    Parameters
    ----------
    path
        Absolute path, or path relative to the scalar_field_sim share
        directory, e.g. `scenarios/tests/fields/01_single_wall.toml`.
    share_dir
        Directory that relative paths are resolved against. Defaults to the
        installed share directory of scalar_field_sim, looked up with
        `ament_index_python` (imported only in that case).

    Returns
    -------
    Path
        `path` unchanged if it is absolute, otherwise `share_dir / path`.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    if share_dir is None:
        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory('scalar_field_sim')
    return Path(share_dir) / path


def load_field_sim_config(path: str | Path) -> FieldSimConfig:
    """Parse one TOML scenario file into a `FieldSimConfig`.

    The scenario defines:
    - general scenario metadata,
    - frame id,
    - rectangular field bounds,
    - optional walls,
    - source positions or full source specifications,
    - simulation parameters such as noise, clipping, and random seed.

    In the reference form, the scenario contains only the top-level keys
    `geometry_file` and `field_file` and an optional `[general]` table. Frame,
    bounds, and walls are then read from the geometry file, and units,
    sources, simulation, and visualization settings from the field file, see
    `_combine_referenced_files`. A scenario with neither key is
    self-contained and defines all tables inline.

    Raises
    ------
    ValueError
        If only one reference is given, if the scenario or a referenced file
        contains keys that are not allowed there, if references are nested,
        or if a referenced file is not valid TOML.
    FileNotFoundError
        If a referenced file does not exist.
    """
    path = Path(path)

    with path.open('rb') as f:
        cfg = tomllib.load(f)

    geometry_file = None
    field_file = None
    if 'geometry_file' in cfg or 'field_file' in cfg:
        geometry_file, field_file, cfg = _combine_referenced_files(path, cfg)

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
        geometry_file=geometry_file,
        field_file=field_file,
    )


def build_field_from_config(config: FieldSimConfig) -> WallAwareGaussianField2D:
    """Construct a scalar field from a parsed config.
    See field.py for definition of field object WallAwareGaussianField2D."""
    rng = (
        np.random.default_rng(config.seed) if config.seed is not None else None
    )
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


# Top-level keys allowed in a scenario that references other files.
_SCENARIO_KEYS = {'geometry_file', 'field_file', 'general'}
# Allowed top-level keys and allowed keys of their subtables.
_GEOMETRY_KEYS = {
    'general': None,
    'frame': None,
    'bounds': None,
    'objects': {'walls'},
    'units': {'length'},
}
_FIELD_KEYS = {
    'units': {'field_value', 'length'},
    'objects': {'sources'},
    'simulation': None,
    'visualization': None,
}
_BOUNDS_KEYS = ('x_min', 'x_max', 'y_min', 'y_max')


def _combine_referenced_files(path: Path, cfg: dict) -> tuple[Path, Path, dict]:
    """Combine a scenario with its referenced geometry and field files.

    Parameters
    ----------
    path
        Path of the scenario file containing `geometry_file` or `field_file`.
    cfg
        Parsed scenario file contents.

    Returns
    -------
    geometry_file, field_file : Path
        Resolved paths of the referenced files. Relative references are
        resolved against the scenario file's directory.
    combined : dict
        One configuration in the self-contained layout: `[general]` from the
        scenario; `[frame]`, `[bounds]`, and `[objects.walls]` from the
        geometry file; `[objects.sources]`, `[simulation]`, and
        `[visualization]` from the field file; `[units]` from both files.

    Raises
    ------
    ValueError
        If only one of the two references is given, if a reference is not a
        non-empty string, if the scenario or a referenced file contains keys
        that are not allowed there, if a referenced file contains a reference
        itself, if a referenced file is not valid TOML, or if both files
        define different `units.length` values.
    FileNotFoundError
        If a referenced file does not exist.
    """
    missing = [key for key in ('geometry_file', 'field_file') if key not in cfg]
    if missing:
        raise ValueError(
            f'Scenario {path} must define both geometry_file and field_file; '
            f'missing {missing}.'
        )
    extra = sorted(set(cfg) - _SCENARIO_KEYS)
    if extra:
        raise ValueError(
            f'Scenario {path} references a geometry and a field file but '
            f'also defines {extra}. It may only define geometry_file, '
            'field_file, and [general].'
        )

    geometry_file = _resolve_reference(path, cfg, 'geometry_file')
    field_file = _resolve_reference(path, cfg, 'field_file')
    geometry = _load_referenced_file(
        geometry_file, path, 'Geometry', _GEOMETRY_KEYS
    )
    field = _load_referenced_file(field_file, path, 'Field', _FIELD_KEYS)

    missing_bounds = [
        key for key in _BOUNDS_KEYS if key not in geometry.get('bounds', {})
    ]
    if missing_bounds:
        raise ValueError(
            f'Geometry file {geometry_file} referenced by {path} is missing '
            f'bounds {missing_bounds}.'
        )

    geometry_units = geometry.get('units', {})
    field_units = field.get('units', {})
    if (
        'length' in geometry_units
        and 'length' in field_units
        and geometry_units['length'] != field_units['length']
    ):
        raise ValueError(
            f'Geometry file {geometry_file} and field file {field_file} '
            f'referenced by {path} define different units.length values '
            f'({geometry_units["length"]!r}, {field_units["length"]!r}).'
        )

    combined = {
        key: geometry[key] for key in ('frame', 'bounds') if key in geometry
    }
    combined.update(
        {
            key: field[key]
            for key in ('simulation', 'visualization')
            if key in field
        }
    )
    if 'general' in cfg:
        combined['general'] = cfg['general']
    combined['units'] = {**geometry_units, **field_units}
    combined['objects'] = {
        **geometry.get('objects', {}),
        **field.get('objects', {}),
    }
    return geometry_file, field_file, combined


def _resolve_reference(path: Path, cfg: dict, key: str) -> Path:
    """Resolve reference `key` of scenario `path` to an absolute path."""
    ref = cfg[key]
    if not isinstance(ref, str) or not ref:
        raise ValueError(
            f'{path}: {key} must be a non-empty string, got {ref!r}.'
        )
    resolved = Path(ref)
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    return resolved.resolve()


def _load_referenced_file(
    ref_path: Path,
    path: Path,
    kind: str,
    allowed: dict[str, set[str] | None],
) -> dict:
    """Load a geometry or field file referenced by scenario `path`.

    Parameters
    ----------
    ref_path
        Resolved path of the referenced file.
    path
        Path of the referencing scenario file, used in error messages.
    kind
        `'Geometry'` or `'Field'`, used in error messages.
    allowed
        Allowed top-level keys. A set value restricts the keys of that
        table; `None` allows any content.

    Raises
    ------
    FileNotFoundError
        If `ref_path` does not exist.
    ValueError
        If `ref_path` is not valid TOML, references another file, or
        contains keys outside `allowed`.
    """
    if not ref_path.is_file():
        raise FileNotFoundError(
            f'{kind} file {ref_path} referenced by {path} not found.'
        )
    try:
        with ref_path.open('rb') as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f'Invalid TOML in {kind.lower()} file {ref_path} referenced by '
            f'{path}: {exc}'
        ) from exc

    nested = sorted({'geometry_file', 'field_file'} & set(data))
    if nested:
        raise ValueError(
            f'{kind} file {ref_path} referenced by {path} defines {nested}; '
            'referenced files must not reference other files.'
        )

    forbidden = sorted(set(data) - set(allowed))
    for key, subkeys in allowed.items():
        table = data.get(key)
        if subkeys is not None and isinstance(table, dict):
            forbidden += [
                f'{key}.{sub}' for sub in sorted(set(table) - subkeys)
            ]
    if forbidden:
        allowed_text = ', '.join(
            key if subkeys is None else f'{key} ({", ".join(sorted(subkeys))})'
            for key, subkeys in allowed.items()
        )
        raise ValueError(
            f'{kind} file {ref_path} referenced by {path} must not define '
            f'{forbidden}; allowed are: {allowed_text}.'
        )
    return data


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
            raise ValueError(
                f'Wall {i} must have format [x1, y1, x2, y2], got {wall}'
            )
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
