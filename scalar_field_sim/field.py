from dataclasses import dataclass
import numpy as np

from geometry import WallSegment, ScenarioGeometry


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
        if not (0.0 <= self.blocking_strength <= 1.0):
            raise ValueError("blocking_strength must be between 0.0 and 1.0.")


class WallAwareGaussianField2D:
    def __init__(
        self,
        sources: list[SimulationSourceSpec],
        walls: list[WallSegment] | None = None,
        background_floor: float = 0.0,
        measurement_noise_std: float = 0.0,
        clip_range: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
        x_range: tuple[float, float] = (0.0, 2.0),
        y_range: tuple[float, float] = (0.0, 4.0),
    ):
        self.sources = sources
        self.walls = walls or []
        self.background_floor = float(background_floor)
        self.measurement_noise_std = float(measurement_noise_std)
        self.clip_range = clip_range
        self.rng = np.random.default_rng() if rng is None else rng
        self.x_range = x_range
        self.y_range = y_range

    def evaluate(self, query_positions):
        """
        Evaluate the deterministic latent field at arbitrary continuous positions.
        No clipping applied.
        """
        xy = self._as_xy_array(query_positions)

        latent = np.full(len(xy), self.background_floor, dtype=float)

        for src in self.sources:
            center, contribution = self._compute_source_contribution(xy, src)

            blocking_strength = float(src.blocking_strength)

            if len(self.walls) > 0:
                blocked = self._compute_blocked_mask(
                    xy=xy,
                    center=center,
                )
                contribution = np.where(
                    blocked,
                    (1.0 - blocking_strength) * contribution,
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

    def sample(self, query_positions):
        """
        Sample noisy measurements at arbitrary continuous positions.
        Clipping is applied.
        """
        latent = self.evaluate(query_positions)
        noisy = latent.copy()

        if self.measurement_noise_std > 0.0:
            noisy += self.rng.normal(0.0, self.measurement_noise_std, size=len(noisy))

        if self.clip_range is not None:
            noisy = np.clip(noisy, self.clip_range[0], self.clip_range[1])

        return latent, noisy

    def sample_at_position(self, query_position):
        """
        Convenience wrapper for a single position.
        """
        latent, noisy = self.sample(
            np.asarray(query_position, dtype=float).reshape(1, -1)
        )
        return float(latent[0]), float(noisy[0])

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
    def _as_xy_array(query_positions):
        query_positions = np.asarray(query_positions, dtype=float)

        if query_positions.ndim == 1:
            query_positions = query_positions.reshape(1, -1)

        if query_positions.ndim != 2 or query_positions.shape[1] < 2:
            raise ValueError("query_positions must have shape (M,2) or (M,3).")

        return query_positions[:, :2]

    @staticmethod
    def _compute_source_contribution(
        xy: np.ndarray, src: SimulationSourceSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the unblocked Gaussian source contribution at positions xy.

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
        """
        blocked = np.zeros(len(xy), dtype=bool)
        a = np.broadcast_to(center, xy.shape)

        for wall in self.walls:
            wall_start = np.asarray(wall.start, dtype=float)
            wall_end = np.asarray(wall.end, dtype=float)
            blocked |= _segment_intersection_mask(a, xy, wall_start, wall_end)

        return blocked


def _make_axis(start: float, stop: float, step: float) -> np.ndarray:
    if stop <= start:
        raise ValueError("Axis stop must be larger than start.")
    if step <= 0:
        raise ValueError("Step size must be positive.")

    n_full_steps = int(np.floor((stop - start) / step))
    axis = start + step * np.arange(n_full_steps + 1)

    if not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)

    return axis


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_intersection_mask(a, b, c, d, eps=1e-12):
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
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

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
