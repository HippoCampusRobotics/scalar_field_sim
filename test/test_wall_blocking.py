"""Tests of the wall line-of-sight blocking in scalar_field_sim.field.

Run from the package root:  python3 -m pytest test
"""

from pathlib import Path

import numpy as np
import pytest
from scalar_field_sim.config import load_field_from_toml
from scalar_field_sim.field import _segment_intersection_mask
from scalar_field_sim.geometry import WallSegment

# One vertical wall from (1, 0) to (1, 2).
WALL = ((1.0, 0.0), (1.0, 2.0))

# (description, source (x, y), query (x, y), expected: is the ray blocked?)
CASES = [
    ('crossing', (0.5, 1.0), (1.5, 1.0), True),
    ('oblique crossing', (0.5, 0.5), (1.5, 1.5), True),
    ('miss, same side', (0.2, 1.0), (0.8, 1.5), False),
    ('miss beyond wall end', (0.5, 2.5), (1.5, 2.5), False),
    ('ray through wall end', (0.5, 2.0), (1.5, 2.0), True),
    ('query on wall', (0.5, 1.0), (1.0, 1.0), True),
    ('query on wall end', (0.5, 1.0), (1.0, 2.0), True),
    ('query 1e-6 m before wall', (0.5, 1.0), (1.0 - 1e-6, 1.0), False),
    ('query on wall line beyond end', (0.5, 1.0), (1.0, 2.5), False),
    ('ray passes 1e-6 m beyond end', (0.5, 2 + 1e-6), (1.5, 2 + 1e-6), False),
    (
        'ray passes 0.5e-9 m beyond end',
        (0.5, 2 + 5e-10),
        (1.5, 2 + 5e-10),
        True,
    ),
    ('parallel, 1e-6 m off', (1 + 1e-6, 0.2), (1 + 1e-6, 1.8), False),
    ('collinear, runs into wall', (1.0, -0.5), (1.0, 0.5), True),
    ('source on wall, leaves sideways', (1.0, 1.0), (1.5, 1.0), False),
    ('source on wall, runs along wall', (1.0, 1.0), (1.0, 1.3), True),
    ('zero-length ray', (0.5, 1.0), (0.5, 1.0), False),
]


def is_blocked(source, query, wall) -> bool:
    """Blocking result for a single ray (the function works on arrays)."""
    mask = _segment_intersection_mask([source], [query], *wall)
    return bool(mask[0])


@pytest.mark.parametrize('reverse_wall', [False, True])
@pytest.mark.parametrize(
    'source, query, expected',
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_blocking(source, query, expected, reverse_wall):
    wall = WALL[::-1] if reverse_wall else WALL
    assert is_blocked(source, query, wall) == expected


@pytest.mark.parametrize('tol', [0.0, -1e-9, np.nan, np.inf])
def test_invalid_contact_tolerance_rejected(tol):
    with pytest.raises(ValueError):
        _segment_intersection_mask([(0, 0)], [(1, 1)], *WALL, contact_tol=tol)


def test_zero_length_wall_rejected():
    with pytest.raises(ValueError):
        WallSegment(start=(1.0, 2.0), end=(1.0, 2.0))


def test_field_is_zero_on_wall():
    # Every source is blocked at a query point on the wall, and the
    # background level of this scenario is 0.
    scenario = (
        Path(__file__).parents[1] / 'scenarios/tests/fields/01_single_wall.toml'
    )
    _, field = load_field_from_toml(scenario)
    on_wall = np.array([[1.0, 0.4], [1.0, 2.0], [1.0, 3.6]])
    assert np.all(field.evaluate(on_wall) == 0.0)
