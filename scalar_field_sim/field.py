from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray

from scalar_field_sim.geometry import WallSegment, ScenarioGeometry


@dataclass(frozen=True, slots=True)
class SimulationSourceSpec:
    center: tuple[float, float]
    amplitude: float
    sigma_x: float
    sigma_y: float
    blocking_strength: float = 1.0
    name: str | None = None

    def __post_init__(self):
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            raise ValueError("sigma_x and sigma_y must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("Source amplitude must be non-negative.")
        if not (0.0 <= self.blocking_strength <= 1.0):
            raise ValueError("blocking_strength must be between 0.0 and 1.0.")
        if not np.isfinite(self.amplitude):
            raise ValueError("Source amplitude must be finite.")
        if not np.isfinite(self.sigma_x) or not np.isfinite(self.sigma_y):
            raise ValueError("sigma_x and sigma_y must be finite.")
        if not np.isfinite(self.blocking_strength):
            raise ValueError("blocking_strength must be finite.")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("Source center must contain finite coordinates.")


class WallAwareGaussianField2D:
    """Continuous 2D scalar field with Gaussian-like sources and wall blocking.

    The latent field is the sum of source contributions plus a constant background
    floor. A wall can attenuate a source contribution if the line segment from the
    source center to the query position intersects the wall segment.

    Notes
    -----
    - `evaluate(...)` returns latent, deterministic field values.
    - `sample(...)` adds measurement noise and optional clipping.
    - Query positions may be shaped `(N, 2)`, `(N, 3)`, `(2,)`, or `(3,)`.
      Only the first two coordinates are used.
    """

    def __init__(
        self,
        sources: Sequence[SimulationSourceSpec],
        walls: Sequence[WallSegment] | None = None,
        background_floor: float = 0.0,
        measurement_noise_std: float = 0.0,
        clip_range: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
        x_range: tuple[float, float] = (0.0, 2.0),
        y_range: tuple[float, float] = (0.0, 4.0),
    ) -> None:

        _validate_range(x_range, name="x_range")
        _validate_range(y_range, name="y_range")

        if clip_range is not None:
            if len(clip_range) != 2:
                raise ValueError("clip_range must be a tuple (min, max).")
            if not np.isfinite(clip_range[0]) or not np.isfinite(clip_range[1]):
                raise ValueError("clip_range bounds must be finite.")
            if clip_range[1] < clip_range[0]:
                raise ValueError("clip_range must satisfy max >= min.")
        if not np.isfinite(background_floor):
            raise ValueError("background_floor must be finite.")
        if not np.isfinite(measurement_noise_std):
            raise ValueError("measurement_noise_std must be finite.")
        if measurement_noise_std < 0.0:
            raise ValueError("measurement_noise_std must be non-negative.")

        self.sources = tuple(sources)
        self.walls = tuple(walls) if walls is not None else tuple()
        self.background_floor = float(background_floor)
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

        latent = np.full(len(xy), self.background_floor, dtype=float)

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
        Convenience wrapper for a single position.
        """
        latent = self.evaluate(np.asarray(query_position, dtype=float).reshape(1, -1))
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
            noisy += self.rng.normal(0.0, self.measurement_noise_std, size=len(noisy))

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
                "query_positions must have shape (N,2) or (N,3), (2,) or (3,)."
            )
        if not np.all(np.isfinite(xy[:, :2])):
            raise ValueError(
                "query_positions must contain only finite x/y coordinates."
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
        raise ValueError(f"{name} must be a tuple (min, max).")
    lo, hi = value_range
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} bounds must be finite.")
    if hi <= lo:
        raise ValueError(f"{name} must satisfy max > min.")


def _make_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Create a 1D axis including the end point."""
    if stop <= start:
        raise ValueError("Axis stop must be larger than start.")
    if step <= 0:
        raise ValueError("Step size must be positive.")

    n_full_steps = int(np.floor((stop - start) / step))
    axis = start + step * np.arange(n_full_steps + 1)

    if not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)

    return axis


def _cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_intersection_mask(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized segment-segment intersection test.

    Parameters
    ----------
    a, b : (N, 2)
        Start/end points of N segments.
    c, d : (2,)
        Start/end points of one wall segment.

    Returns
    -------
    (N,) boolean mask
        True where segment a->b intersects c->d.

    Raises
    ------
    ValueError
        If shapes are incompatible or eps <= 0
    """
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be a positive finite number.")

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("a and b must have shape (N, 2).")
    if c.shape != (2,) or d.shape != (2,):
        raise ValueError("c and d must have shape (2,).")

    r = b - a
    s = d - c
    denom = _cross2d(r, s)

    q_minus_p = c - a
    numer_t = _cross2d(q_minus_p, s)
    numer_u = _cross2d(q_minus_p, r)

    t = np.full(denom.shape, np.nan)
    u = np.full(denom.shape, np.nan)

    mask = np.abs(denom) > eps
    t[mask] = numer_t[mask] / denom[mask]
    u[mask] = numer_u[mask] / denom[mask]

    # For line-of-sight blocking, it is safer to exclude the exact ray start point.
    return mask & (t > eps) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def build_simulation_sources(
    scenario: ScenarioGeometry,
    amplitude: float,
    sigma_x: float,
    sigma_y: float,
    blocking_strength: float = 1.0,
) -> list[SimulationSourceSpec]:
    """Build one `SimulationSourceSpec` per scenario source."""
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite.")
    if amplitude < 0.0:
        raise ValueError("amplitude must be non-negative.")
    if not np.isfinite(sigma_x) or not np.isfinite(sigma_y):
        raise ValueError("sigma_x and sigma_y must be finite.")
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("sigma_x and sigma_y must be positive.")
    if not np.isfinite(blocking_strength):
        raise ValueError("blocking_strength must be finite.")
    if not (0.0 <= blocking_strength <= 1.0):
        raise ValueError("blocking_strength must lie in [0.0, 1.0].")

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
