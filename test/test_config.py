"""Tests of scenario loading in scalar_field_sim.config.

Run from the package root:  python3 -m pytest test
"""

import pytest
from scalar_field_sim.config import (
    load_field_sim_config,
    resolve_package_path,
)

GEOMETRY = """
[general]
name = "tank"

[frame]
frame_id = "map"

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [[1.0, 0.4, 1.0, 3.6]]
"""

FIELD = """
[units]
field_value = "V"

[objects.sources]
source_positions = [[0.5, 1.0], [1.5, 3.0]]

[simulation.source_defaults]
amplitude = 0.011
sigma_x = 0.35
sigma_y = 0.35

[simulation]
seed = 7
"""

SCENARIO = """
geometry_file = "geometry.toml"
field_file = "field.toml"

[general]
name = "single_wall"
description = "One wall."
"""

SELF_CONTAINED = """
[general]
name = "inline"

[bounds]
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 2.0

[objects.walls]
wall_vertices = [[0.5, 0.2, 0.5, 1.8]]

[objects.sources]
source_positions = [[0.2, 1.0]]

[simulation.source_defaults]
amplitude = 1.0
sigma_x = 0.1
sigma_y = 0.1
"""


def write_files(tmp_path, scenario=SCENARIO, geometry=GEOMETRY, field=FIELD):
    """Write the three files to tmp_path and return the scenario path."""
    (tmp_path / 'geometry.toml').write_text(geometry)
    (tmp_path / 'field.toml').write_text(field)
    path = tmp_path / 'scenario.toml'
    path.write_text(scenario)
    return path


def test_reference_scenario(tmp_path):
    config = load_field_sim_config(write_files(tmp_path))
    assert config.name == 'single_wall'
    assert config.description == 'One wall.'
    assert config.frame_id == 'map'
    assert config.geometry.x_range == (0.0, 2.0)
    assert config.geometry.y_range == (0.0, 4.0)
    walls = [(w.start, w.end) for w in config.geometry.walls]
    assert walls == [((1.0, 0.4), (1.0, 3.6))]
    assert [s.center for s in config.source_specs] == [(0.5, 1.0), (1.5, 3.0)]
    assert config.seed == 7
    assert config.field_value_unit == 'V'
    assert config.geometry_file == (tmp_path / 'geometry.toml').resolve()
    assert config.field_file == (tmp_path / 'field.toml').resolve()


def test_reference_paths_resolve_against_scenario_directory(tmp_path):
    (tmp_path / 'scenarios').mkdir()
    write_files(tmp_path)
    path = tmp_path / 'scenarios' / 'scenario.toml'
    path.write_text(
        'geometry_file = "../geometry.toml"\n'
        f'field_file = "{tmp_path / "field.toml"}"\n'
    )
    config = load_field_sim_config(path)
    assert config.name == 'scenario'
    assert len(config.geometry.walls) == 1


def test_self_contained_scenario(tmp_path):
    path = tmp_path / 'inline.toml'
    path.write_text(SELF_CONTAINED)
    config = load_field_sim_config(path)
    assert config.name == 'inline'
    assert config.geometry.y_range == (0.0, 2.0)
    assert len(config.geometry.walls) == 1
    assert len(config.source_specs) == 1
    assert config.geometry_file is None and config.field_file is None


@pytest.mark.parametrize(
    'files',
    [
        {'scenario': SCENARIO + '[bounds]\nx_min = 0.0\n'},
        {'scenario': SCENARIO + '[objects.walls]\nwall_vertices = []\n'},
        {'geometry': GEOMETRY + '[objects.sources]\nsource_positions = []\n'},
        {'geometry': GEOMETRY + '[simulation]\nseed = 1\n'},
        {'field': FIELD + '[bounds]\nx_min = 0.0\n'},
        {'field': FIELD + '[frame]\nframe_id = "map"\n'},
        {'field': FIELD + '[objects.walls]\nwall_vertices = []\n'},
        {'field': FIELD + '[general]\nname = "field"\n'},
    ],
    ids=[
        'bounds in scenario',
        'walls in scenario',
        'sources in geometry',
        'simulation in geometry',
        'bounds in field',
        'frame in field',
        'walls in field',
        'general in field',
    ],
)
def test_forbidden_keys_rejected(tmp_path, files):
    with pytest.raises(ValueError):
        load_field_sim_config(write_files(tmp_path, **files))


@pytest.mark.parametrize('key', ['geometry_file', 'field_file'])
def test_single_reference_rejected(tmp_path, key):
    scenario = '\n'.join(
        line for line in SCENARIO.splitlines() if not line.startswith(key)
    )
    with pytest.raises(ValueError, match=key):
        load_field_sim_config(write_files(tmp_path, scenario=scenario))


@pytest.mark.parametrize(
    'files',
    [
        {'geometry': 'field_file = "field.toml"\n' + GEOMETRY},
        {'field': 'geometry_file = "geometry.toml"\n' + FIELD},
    ],
    ids=['in geometry', 'in field'],
)
def test_nested_reference_rejected(tmp_path, files):
    with pytest.raises(ValueError, match='must not reference'):
        load_field_sim_config(write_files(tmp_path, **files))


def test_missing_referenced_file(tmp_path):
    path = write_files(tmp_path)
    (tmp_path / 'field.toml').unlink()
    with pytest.raises(FileNotFoundError, match='field.toml'):
        load_field_sim_config(path)


def test_invalid_toml_in_referenced_file(tmp_path):
    path = write_files(tmp_path, geometry='[bounds\n')
    with pytest.raises(ValueError, match='geometry.toml'):
        load_field_sim_config(path)


def test_resolve_package_path_keeps_absolute_path(tmp_path):
    path = tmp_path / 'scenario.toml'
    assert resolve_package_path(path, share_dir='/elsewhere') == path


def test_resolve_package_path_joins_relative_path_with_share_dir(tmp_path):
    resolved = resolve_package_path('scenarios/a.toml', share_dir=tmp_path)
    assert resolved == tmp_path / 'scenarios' / 'a.toml'
