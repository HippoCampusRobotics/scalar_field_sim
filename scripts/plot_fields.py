#!/usr/bin/env python3
"""Plot the noiseless true field of one or more scenario TOML files.

    ros2 run scalar_field_sim plot_fields.py SCENARIO [SCENARIO ...]

A SCENARIO is resolved in this order:

1. a path to a TOML file, absolute or relative to the current directory,
2. a path relative to the scenario folder of scalar_field_sim (installed
   `share/scalar_field_sim/scenarios`, or the source `scenarios/` folder when
   run from the source tree), with or without `.toml`,
   e.g. `tests/fields/01_single_wall`,
3. a path relative to the workspace root (the directory containing
   `install/`), e.g. `src/scalar_field_sim/scenarios/simple_field_1.toml`,
4. a unique file name (without `.toml`) below the scenario folder,
   e.g. `01_single_wall`.

Each scenario is loaded with `scalar_field_sim.config.load_field_from_toml`
(including shared `field_file` resolution) and evaluated with the field's own
`evaluate_on_grid`. Walls and sources are drawn from the loaded geometry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scalar_field_sim.config import load_field_from_toml
except ImportError as exc:
    sys.exit(
        f'Cannot import scalar_field_sim ({exc}).\n'
        'Source the workspace overlay (source install/setup.bash) or set '
        'PYTHONPATH to the scalar_field_sim package source directory.'
    )

DEFAULT_GRID_STEP = 0.01


def _package_prefix() -> Path | None:
    """Install prefix of scalar_field_sim from the ament index, if any."""
    try:
        from ament_index_python.packages import get_package_prefix
    except ImportError:
        return None
    try:
        return Path(get_package_prefix('scalar_field_sim'))
    except Exception:  # noqa: BLE001 - not installed / no ament index
        return None


def scenario_roots() -> list[Path]:
    """Scenario folders searched for scenario names (installed, then source)."""
    roots = []
    prefix = _package_prefix()
    if prefix is not None:
        roots.append(prefix / 'share' / 'scalar_field_sim' / 'scenarios')
    import scalar_field_sim

    for package_dir in scalar_field_sim.__path__:
        roots.append(Path(package_dir).resolve().parent / 'scenarios')
    return [root for root in dict.fromkeys(roots) if root.is_dir()]


def workspace_root() -> Path | None:
    """Workspace root (the parent of `install/`), if it can be determined."""
    prefix = _package_prefix()
    if prefix is None:
        return None
    for parent in (prefix, *prefix.parents):
        if parent.name == 'install':
            return parent.parent
    return None


def _with_toml(path: Path) -> list[Path]:
    return [path] if path.suffix == '.toml' else [path, Path(f'{path}.toml')]


def resolve_scenario(spec: str) -> Path:
    """Resolve a scenario argument to a TOML file (see module docstring)."""
    candidates = _with_toml(Path(spec).expanduser())
    roots = scenario_roots()
    for root in roots:
        candidates += _with_toml(root / spec)
    ws = workspace_root()
    if ws is not None:
        candidates += _with_toml(ws / spec)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    name = Path(spec).name.removesuffix('.toml')
    matches = sorted(
        {p.resolve() for root in roots for p in root.rglob(f'{name}.toml')}
    )
    if len(matches) == 1:
        return matches[0]
    searched = ', '.join(str(root) for root in roots) or 'no scenario folder'
    if matches:
        listed = '\n  '.join(str(m) for m in matches)
        raise SystemExit(
            f'Scenario name {spec!r} is ambiguous; give a relative path:'
            f'\n  {listed}'
        )
    raise SystemExit(
        f'Scenario {spec!r} not found (current directory, {searched}'
        f'{", workspace " + str(ws) if ws else ""}). '
        'Use --list to show available scenarios.'
    )


def list_scenarios() -> None:
    for root in scenario_roots():
        print(f'{root}:')
        for path in sorted(root.rglob('*.toml')):
            print(f'  {path.relative_to(root).with_suffix("")}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'scenarios',
        nargs='*',
        help='Scenario TOML files or names (see above).',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List the scenarios in the scenario folder(s) and exit.',
    )
    parser.add_argument(
        '--grid-step',
        type=float,
        default=None,
        help='Grid spacing in m (default: scenario [visualization] '
        f'grid_step, else {DEFAULT_GRID_STEP}).',
    )
    parser.add_argument(
        '--save',
        type=Path,
        default=None,
        metavar='PNG',
        help='Also save the figure to this PNG file.',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not open a window (useful with --save).',
    )
    args = parser.parse_args()
    if not args.list and not args.scenarios:
        parser.error('give at least one scenario (or --list)')
    return args


def evaluate_scenario(path: Path, grid_step: float | None):
    """Load one scenario and evaluate its noiseless field on a grid.

    Returns
    -------
    config
        Parsed `FieldSimConfig`.
    x, y
        Grid axes in m.
    values
        Field values of shape `(len(y), len(x))`.
    """
    config, field = load_field_from_toml(path)
    step = grid_step
    if step is None:
        step = config.visualization_grid_step or DEFAULT_GRID_STEP
    positions, values = field.evaluate_on_grid(step)
    # make_grid_positions uses meshgrid(x, y) raveled in C order.
    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    return config, x, y, values.reshape(len(y), len(x))


def display_scale(unit: str | None) -> tuple[float, str]:
    """Return the factor and label for displayed field values."""
    if unit == 'V':
        return 1e3, 'Field value [mV]'
    return 1.0, f'Field value [{unit or "a.u."}]'


def main() -> int:
    args = parse_args()
    if args.list:
        list_scenarios()
        return 0

    results = []
    for spec in args.scenarios:
        path = resolve_scenario(spec)
        config, x, y, values = evaluate_scenario(path, args.grid_step)
        shared = config.field_file or 'none (inline definition)'
        print(f'{config.name}: {path.resolve()}')
        print(f'  shared parameter file: {shared}')
        results.append((config, x, y, values))

    units = {config.field_value_unit for config, *_ in results}
    if len(units) > 1:
        print(
            f'Error: selected scenarios use different field value units '
            f'{sorted(units, key=str)}; cannot share one color scale.',
            file=sys.stderr,
        )
        return 1
    scale, cbar_label = display_scale(results[0][0].field_value_unit)

    vmin = min(values.min() for *_, values in results) * scale
    vmax = max(values.max() for *_, values in results) * scale

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(3.2 * len(results) + 1.5, 5.6),
        squeeze=False,
    )
    fig.subplots_adjust(bottom=0.15, wspace=0.35)
    axes = axes[0]
    mesh = None
    for ax, (config, x, y, values) in zip(axes, results, strict=True):
        mesh = ax.pcolormesh(
            x, y, values * scale, shading='nearest', vmin=vmin, vmax=vmax
        )
        for i, wall in enumerate(config.geometry.walls):
            ax.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                color='0.1',
                linewidth=3,
                label='wall' if i == 0 else None,
            )
        sources = np.array([s.position for s in config.geometry.sources])
        if len(sources):
            ax.plot(
                sources[:, 0],
                sources[:, 1],
                linestyle='none',
                marker='x',
                color='red',
                markersize=6,
                label='source (truth)',
            )
        ax.set_xlim(config.geometry.x_range)
        ax.set_ylim(config.geometry.y_range)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(config.name)

    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=2)
    fig.colorbar(mesh, ax=list(axes), label=cbar_label, shrink=0.8)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'Saved figure: {args.save.resolve()}')

    if not args.no_show:
        plt.show(block=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
