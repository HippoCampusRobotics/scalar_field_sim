"""Scalar field model for simulation scenarios.

This module defines:
- source parameter containers,
- a continuous 2D scalar field with optional wall blocking,
- helper functions for grid generation and the ray-wall blocking test
  (`_segment_intersection_mask`: closed walls with a contact tolerance of
  `WALL_CONTACT_TOL` metres, round-off-certified orientation tests),
- a helper to build simulation source specifications from scenario geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scalar_field_sim.geometry import ScenarioGeometry, WallSegment

# Geometric contact tolerance (m): points within this distance of a wall
# segment count as on the wall (see `_segment_intersection_mask`).
WALL_CONTACT_TOL = 1e-9

# Round-off filter of the orientation test (dimensionless): Shewchuk's
# ccwerrboundA = (3 + 16 u) u with float64 unit round-off u = 2**-53,
# https://www.cs.cmu.edu/~quake/robust.html
_UNIT_ROUNDOFF = 2.0**-53
ORIENT_ERRBOUND = (3.0 + 16.0 * _UNIT_ROUNDOFF) * _UNIT_ROUNDOFF


@dataclass(frozen=True, slots=True)
class SimulationSourceSpec:
    """Parameters of one simulated scalar field source.

    Attributes
    ----------
    center
        Source center position as (x, y).
    amplitude
        Peak amplitude of the source contribution.
    sigma_x
        Standard deviation in x direction.
    sigma_y
        Standard deviation in y direction.
    blocking_strength
        Attenuation factor applied when a wall blocks line of sight.
        - 0.0 means no attenuation
        - 1.0 means full blocking
    name
        Optional source name for debugging or visualization.
        Currently not used.
    """

    center: tuple[float, float]
    amplitude: float
    sigma_x: float
    sigma_y: float
    blocking_strength: float = 1.0
    name: str | None = None

    def __post_init__(self):
        if len(self.center) != 2:
            raise ValueError('Source center must have length 2.')
        if not np.all(np.isfinite(self.center)):
            raise ValueError('Source center must contain finite coordinates.')
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            raise ValueError('sigma_x and sigma_y must be positive.')
        if self.amplitude < 0.0:
            raise ValueError('Source amplitude must be non-negative.')
        if not (0.0 <= self.blocking_strength <= 1.0):
            raise ValueError('blocking_strength must be between 0.0 and 1.0.')
        if not np.isfinite(self.amplitude):
            raise ValueError('Source amplitude must be finite.')
        if not np.isfinite(self.sigma_x) or not np.isfinite(self.sigma_y):
            raise ValueError('sigma_x and sigma_y must be finite.')
        if not np.isfinite(self.blocking_strength):
            raise ValueError('blocking_strength must be finite.')


class WallAwareGaussianField2D:
    """Continuous 2D scalar field with Gaussian-like sources and wall blocking.

    The latent field is the sum of source contributions plus a constant background
    level. A wall can attenuate a source contribution if the line segment from the
    source center to the query position intersects the wall segment (closed
    segment with contact tolerance `WALL_CONTACT_TOL`, see
    `_segment_intersection_mask`).

    Notes
    -----
    - `evaluate(...)` returns latent, deterministic field values.
    - `sample(...)` adds measurement noise and optional clipping.
    - Query positions may be shaped `(N, 2)`, `(N, 3)`, `(2,)`, or `(3,)`.
      Only the first two coordinates (assumed to be x and y) are used.
    """

    def __init__(
        self,
        sources: Sequence[SimulationSourceSpec],
        walls: Sequence[WallSegment] | None = None,
        background_level: float = 0.0,
        measurement_noise_std: float = 0.0,
        clip_range: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
        x_range: tuple[float, float] = (0.0, 2.0),
        y_range: tuple[float, float] = (0.0, 4.0),
    ) -> None:

        _validate_range(x_range, name='x_range')
        _validate_range(y_range, name='y_range')

        if clip_range is not None:
            if len(clip_range) != 2:
                raise ValueError('clip_range must be a tuple (min, max).')
            if not np.isfinite(clip_range[0]) or not np.isfinite(clip_range[1]):
                raise ValueError('clip_range bounds must be finite.')
            if clip_range[1] < clip_range[0]:
                raise ValueError('clip_range must satisfy max >= min.')
        if not np.isfinite(background_level):
            raise ValueError('background_level must be finite.')
        if not np.isfinite(measurement_noise_std):
            raise ValueError('measurement_noise_std must be finite.')
        if measurement_noise_std < 0.0:
            raise ValueError('measurement_noise_std must be non-negative.')

        self.sources = tuple(sources)
        self.walls = tuple(walls) if walls is not None else tuple()
        self.background_level = float(background_level)
        self.measurement_noise_std = float(measurement_noise_std)
        self.clip_range = clip_range
        self.rng = np.random.default_rng() if rng is None else rng
        self.x_range = x_range
        self.y_range = y_range

    def evaluate(self, query_positions: ArrayLike) -> np.ndarray:
        """
        Evaluate the deterministic latent field at arbitrary continuous positions.
        No clipping applied.
        """
        xy = self._as_xy_array(query_positions)

        latent = np.full(len(xy), self.background_level, dtype=float)

        for src in self.sources:
            center, contribution = self._compute_source_contribution(xy, src)

            if self.walls:
                blocked = self._compute_blocked_mask(
                    xy=xy,
                    center=center,
                )
                contribution = np.where(
                    blocked,
                    (1.0 - float(src.blocking_strength)) * contribution,
                    contribution,
                )

            latent += contribution

        return latent

    def evaluate_at_position(self, query_position) -> float:
        """
        Convenience wrapper for evaluating field at a single position.
        """
        latent = self.evaluate(
            np.asarray(query_position, dtype=float).reshape(1, -1)
        )
        return float(latent[0])

    def sample(
        self, query_positions: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample noisy measurements at arbitrary positions.

        Parameters
        ----------
        query_positions
            Positions shaped `(N, 2)`, `(N, 3)`, `(2,)`, or `(3,)`.

        Returns
        -------
        latent : FloatArray
            Deterministic latent field values of shape `(N,)`.
        noisy : FloatArray
            Noisy, optionally clipped measurements of shape `(N,)`.
        clipped : BoolArray
            Boolean mask of shape `(N,)`. `True` where clipping changed the
            noisy value.

        Raises
        ------
        ValueError
            If `query_positions` does not have a valid shape or contains
            non-finite values.
        """
        latent = self.evaluate(query_positions)
        noisy = latent.copy()

        if self.measurement_noise_std > 0.0:
            noisy += self.rng.normal(
                0.0, self.measurement_noise_std, size=len(noisy)
            )

        unclipped = noisy.copy()
        if self.clip_range is not None:
            noisy = np.clip(noisy, self.clip_range[0], self.clip_range[1])
        clipped = ~np.isclose(noisy, unclipped)

        return latent, noisy, clipped

    def sample_at_position(
        self, query_position: ArrayLike
    ) -> tuple[float, float, bool]:
        """
        Convenience wrapper for a single position.
        """
        latent, noisy, clipped = self.sample(
            np.asarray(query_position, dtype=float).reshape(1, -1)
        )
        return float(latent[0]), float(noisy[0]), bool(clipped[0])

    def make_grid_positions(
        self,
        grid_step: float,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        Create a regular rectangular grid of evaluation positions from a physical step size.

        Returns
        -------
        grid_positions : (N, 2)
            Flattened regular grid positions, compatible with GPFieldVisualizer.
        """
        x_range = self.x_range if x_range is None else x_range
        y_range = self.y_range if y_range is None else y_range

        x = _make_axis(x_range[0], x_range[1], grid_step)
        y = _make_axis(y_range[0], y_range[1], grid_step)
        X, Y = np.meshgrid(x, y)

        grid_positions = np.column_stack([X.ravel(), Y.ravel()])
        return grid_positions

    def evaluate_on_grid(
        self,
        grid_step: float,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the latent field on a regular grid defined by physical step size.
        """
        grid_positions = self.make_grid_positions(
            grid_step=grid_step,
            x_range=x_range,
            y_range=y_range,
        )
        latent_values = self.evaluate(grid_positions)
        return grid_positions, latent_values

    @staticmethod
    def _as_xy_array(query_positions: ArrayLike) -> NDArray[np.float64]:
        """
        Convert input positions to an (N, 2) float array.

        Accepts (N, 2), (N, 3), (2,), or (3,). Only the first two
        coordinates are kept.

        Raises
        ------
        ValueError
            If the input shape is invalid or contains non-finite values.
        """
        xy = np.asarray(query_positions, dtype=float)

        if xy.ndim == 1:
            xy = xy.reshape(1, -1)

        if xy.ndim != 2 or xy.shape[1] not in (2, 3):
            raise ValueError(
                'query_positions must have shape (N,2) or (N,3), (2,) or (3,).'
            )
        if not np.all(np.isfinite(xy[:, :2])):
            raise ValueError(
                'query_positions must contain only finite x/y coordinates.'
            )
        return xy[:, :2]

    @staticmethod
    def _compute_source_contribution(
        xy: np.ndarray, src: SimulationSourceSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the unblocked Gaussian source contribution at positions xy.

        Parameters
        ----------
        xy
            Query positions of shape (N, 2).
        src
            Source specification.


        Returns
        -------
        center : (2,) array
            Source center as xy coordinates.
        contribution : (N,) array
            Unblocked source contribution at each query point.
        """
        center = np.asarray(src.center, dtype=float)
        dx = xy[:, 0] - center[0]
        dy = xy[:, 1] - center[1]

        contribution = src.amplitude * np.exp(
            -0.5 * ((dx / src.sigma_x) ** 2 + (dy / src.sigma_y) ** 2)
        )
        return center, contribution

    def _compute_blocked_mask(
        self,
        xy: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Compute which query positions are blocked from the source center
        by any walls.

        Parameters
        ----------
        xy
            Query positions of shape (N, 2).
        center
            Source center of shape (2,).

        Returns
        -------
        BoolArray
            Boolean mask of shape (N,), `True` where any wall blocks the
            line segment from `center` to the query position.
        """
        blocked = np.zeros(len(xy), dtype=bool)
        a = np.broadcast_to(center, xy.shape)

        for wall in self.walls:
            wall_start = np.asarray(wall.start, dtype=float)
            wall_end = np.asarray(wall.end, dtype=float)
            blocked |= _segment_intersection_mask(a, xy, wall_start, wall_end)

        return blocked


def _validate_range(value_range: tuple[float, float], name: str) -> None:
    """Validate a numeric interval (min, max)."""
    if len(value_range) != 2:
        raise ValueError(f'{name} must be a tuple (min, max).')
    lo, hi = value_range
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f'{name} bounds must be finite.')
    if hi <= lo:
        raise ValueError(f'{name} must satisfy max > min.')


def _make_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Create a 1D axis including the end point."""
    if stop <= start:
        raise ValueError('Axis stop must be larger than start.')
    if step <= 0:
        raise ValueError('Step size must be positive.')

    n_full_steps = int(np.floor((stop - start) / step))
    axis = start + step * np.arange(n_full_steps + 1)

    if not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)

    return axis


def _cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dot2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _orient_sign(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Certified sign of the orientation of the points p, q, x.

    Returns +1 (counter-clockwise), -1 (clockwise) or 0 where the float64
    sign is uncertain. Uses Shewchuk's static error filter of `orient2d`
    (https://www.cs.cmu.edu/~quake/robust.html): the differences must be
    formed directly from the input coordinates, as done here. A nonzero
    result is the exact sign for the float64 inputs; exactly collinear
    points (det == 0) always give 0.
    """
    detleft = (q[..., 0] - p[..., 0]) * (x[..., 1] - p[..., 1])
    detright = (q[..., 1] - p[..., 1]) * (x[..., 0] - p[..., 0])
    det = detleft - detright
    errbound = ORIENT_ERRBOUND * (np.abs(detleft) + np.abs(detright))
    return np.where(np.abs(det) > errbound, np.sign(det), 0.0)


def _point_segment_dist2(
    p: np.ndarray, u: np.ndarray, w: np.ndarray, ww: np.ndarray
) -> np.ndarray:
    """Squared distance from p to the closed segment [u, u + w].

    `ww = dot(w, w)`. Where `ww == 0` the segment is the point u.
    """
    pu = p - u
    num = _dot2d(pu, w)
    shape = np.broadcast_shapes(np.shape(num), np.shape(ww))
    num, ww = np.broadcast_to(num, shape), np.broadcast_to(ww, shape)
    t = np.zeros(shape)
    np.divide(num, ww, out=t, where=ww > 0.0)
    t = np.clip(t, 0.0, 1.0)
    ex = pu[..., 0] - t * w[..., 0]
    ey = pu[..., 1] - t * w[..., 1]
    return ex * ex + ey * ey


def _segment_intersection_mask(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    contact_tol: float = WALL_CONTACT_TOL,
) -> np.ndarray:
    """
    Vectorized line-of-sight test of rays a->b against one wall c->d.

    The wall is a closed segment thickened by the contact tolerance
    delta = `contact_tol` (metres): points within distance delta of the wall
    segment count as on the wall. Policy for each ray segment [a, b]:

    A. a == b (zero-length ray, no travelled path): not blocked.
    B. dist(a, wall) > delta: blocked iff dist([a, b], wall) <= delta. This
       blocks crossings at any nonzero angle, contact at a wall endpoint or
       joint, queries on the wall (any wall direction) and collinear
       overlap or touching. Parallel or collinear segments separated by
       more than delta, and queries on the infinite wall line beyond the
       finite wall, are not blocked.
    C. dist(a, wall) <= delta (source touching the wall): contact at the
       source alone is excluded. Blocked iff the ray continues along the
       wall: dist(b, wall line) <= delta and the projections of [a, b] and
       the wall onto the wall direction overlap by more than delta.

    Computation: dist([a, b], wall) is 0 if the segments properly cross with
    all four orientation signs certified by Shewchuk's orient2d error filter
    (`ORIENT_ERRBOUND`, dimensionless), else the minimum of the four
    endpoint-to-segment distances. There is no division by the cross
    product of the directions and no angle threshold. A certified proper
    crossing is a true crossing. If the segments truly intersect but an
    orientation is uncertain, an endpoint lies within about
    `ORIENT_ERRBOUND * |coordinate differences|` (~1e-15 m at 10 m) of the
    other segment's line and, by linearity of the signed distance along a
    segment, some endpoint lies within that distance of the other segment,
    so the endpoint-distance term blocks it. The result is therefore exact
    for the float64 inputs except for configurations within the round-off
    of the distance computations (a few 1e-16 times the coordinate
    differences) of a delta boundary. This requires delta >> 1e-16 * scene
    size; with delta = 1e-9 m, coordinates up to |x| ~ 1e4 m keep that
    round-off below ~1e-11 m (1 % of delta).

    Limitations: decisions within that round-off of a delta boundary (e.g.
    a query almost exactly delta from a wall) may differ from exact
    arithmetic. For sources farther than delta from a wall, reversing or
    splitting the wall into touching collinear pieces cannot change the
    result (the distance to a union is the minimum of the distances). For a
    source within delta of a wall, rule C applies per wall segment, so
    splitting can matter: a ray leaving such a source at a grazing angle
    (query more than delta off the wall line) is not blocked by that
    segment even if it crosses the wall line farther along, but it is
    blocked by another piece it crosses; and a collinear ray of length
    about 2 delta or less across a joint can overlap each piece by <= delta.

    Parameters
    ----------
    a, b : (N, 2)
        Start (source) and end (query) points of N ray segments.
    c, d : (2,)
        Start/end points of one wall segment; must differ.
    contact_tol
        Contact tolerance delta in metres; positive and finite.

    Returns
    -------
    (N,) boolean mask
        True where the wall blocks the ray a->b.

    Raises
    ------
    ValueError
        If shapes are incompatible, the wall has zero length, or
        contact_tol is not positive and finite.
    """
    if not np.isfinite(contact_tol) or contact_tol <= 0.0:
        raise ValueError('contact_tol must be a positive finite number.')

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    if a.shape != b.shape:
        raise ValueError('a and b must have the same shape.')
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('a and b must have shape (N, 2).')
    if c.shape != (2,) or d.shape != (2,):
        raise ValueError('c and d must have shape (2,).')

    s = d - c
    ss = float(_dot2d(s, s))
    if ss == 0.0:
        raise ValueError('Wall segment must have nonzero length.')
    r = b - a
    rr = _dot2d(r, r)
    tol2 = contact_tol * contact_tol

    # A: zero-length rays have no travelled path.
    has_path = np.any(a != b, axis=1)

    # B: closed, delta-thickened segment distance.
    dist2_a_wall = _point_segment_dist2(a, c, s, ss)
    near = (
        (dist2_a_wall <= tol2)
        | (_point_segment_dist2(b, c, s, ss) <= tol2)
        | (_point_segment_dist2(c, a, r, rr) <= tol2)
        | (_point_segment_dist2(d, a, r, rr) <= tol2)
    )
    crosses = (_orient_sign(a, b, c) * _orient_sign(a, b, d) < 0.0) & (
        _orient_sign(c, d, a) * _orient_sign(c, d, b) < 0.0
    )
    blocked_free_source = crosses | near

    source_contact = dist2_a_wall <= tol2
    if not np.any(source_contact):
        return has_path & blocked_free_source

    # C: source touches the wall; only continuation along the wall blocks.
    wall_length = np.sqrt(ss)
    b_on_line = np.abs(_cross2d(s, b - c)) <= contact_tol * wall_length
    pa = _dot2d(a - c, s) / wall_length
    pb = _dot2d(b - c, s) / wall_length
    overlap = np.minimum(np.maximum(pa, pb), wall_length) - np.maximum(
        np.minimum(pa, pb), 0.0
    )
    continues_along_wall = b_on_line & (overlap > contact_tol)

    return has_path & np.where(
        source_contact, continues_along_wall, blocked_free_source
    )


def build_simulation_sources(
    scenario: ScenarioGeometry,
    amplitude: float,
    sigma_x: float,
    sigma_y: float,
    blocking_strength: float = 1.0,
) -> list[SimulationSourceSpec]:
    """Build one `SimulationSourceSpec` per scenario source."""
    if not np.isfinite(amplitude):
        raise ValueError('amplitude must be finite.')
    if amplitude < 0.0:
        raise ValueError('amplitude must be non-negative.')
    if not np.isfinite(sigma_x) or not np.isfinite(sigma_y):
        raise ValueError('sigma_x and sigma_y must be finite.')
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError('sigma_x and sigma_y must be positive.')
    if not np.isfinite(blocking_strength):
        raise ValueError('blocking_strength must be finite.')
    if not (0.0 <= blocking_strength <= 1.0):
        raise ValueError('blocking_strength must lie in [0.0, 1.0].')

    return [
        SimulationSourceSpec(
            center=src.position,
            amplitude=amplitude,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            blocking_strength=blocking_strength,
            name=src.name,
        )
        for src in scenario.sources
    ]
