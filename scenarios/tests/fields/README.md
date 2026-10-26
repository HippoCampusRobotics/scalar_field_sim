# Test fields

Manually inspectable field scenarios for later belief/planning evaluation.
All four use the same field file; only the geometry differs.

| File | Geometry file | Content |
|---|---|---|
| `00_open.toml` | `geometry/tank_2x4_open.toml` | No internal walls |
| `01_single_wall.toml` | `geometry/tank_2x4_single_wall.toml` | Wall at x = 1 m, y = 0.4 to 3.6 m |
| `02_wall_with_gap.toml` | `geometry/tank_2x4_wall_with_gap.toml` | Same wall with an opening for y = 1.75 to 2.25 m |
| `03_better_wall.toml` | `geometry/tank_2x4_two_walls.toml` | Wall at x = 1 m, y = 0.6 to 3.4 m, and a wall at y = 2.1 m, x = 1 to 1.5 m (scenario name `simple_wall_case`) |

The field file `fields/test_four_sources.toml` defines the units, hidden
sources, source parameters, simulation and visualization settings. Geometry
and field files are in the package folders `geometry/` and `fields/`.

## Schema

Each scenario references one geometry file and one field file:

```toml
geometry_file = "../../../geometry/tank_2x4_single_wall.toml"
field_file = "../../../fields/test_four_sources.toml"

[general]
name = "01_single_wall"
description = "..."
```

- References are resolved relative to the scenario file's directory
  (independent of the working directory); absolute paths are allowed.
- The scenario may contain only `geometry_file`, `field_file`, and
  `[general]`. The geometry file may only contain `[general]`, `[frame]`,
  `[bounds]`, `[objects.walls]`, and `[units] length`; the field file only
  `[units]`, `[objects.sources]`, `[simulation]`, and `[visualization]`.
  Other keys raise a `ValueError` naming the keys and files. There are no
  overrides.
- Both references are required. Referenced files must not reference other
  files; a nested reference raises a `ValueError`.
- A missing referenced file raises `FileNotFoundError`; invalid TOML in a
  referenced file raises a `ValueError` naming that file.

The full description is in "Geometry, field, and scenario files" of the
package `readme.md`. The loader is
`scalar_field_sim.config.load_field_sim_config`. The field server and
visualization nodes use the same loader, so these files also work with the
existing launch files, e.g.
`ros2 launch scalar_field_sim field_sim.launch.py scenario_path:=scenarios/tests/fields/01_single_wall.toml`
(relative paths are resolved against the installed `share/scalar_field_sim`
directory).

## Units and frame

- Lengths in m, frame `map`, domain x in [0, 2] m, y in [0, 4] m.
- Field values in V (IR-derived scalar measurement value, latent field
  before noise), as declared in `[units] field_value`. The plot shows mV
  (explicit factor 1e3).
- `sigma_x = sigma_y = 0.35` m is chosen deliberately for these test fields
  (older scenarios use 0.18 m) so that appreciable signal reaches across the
  wall position; the amplitude 0.011 V is unchanged.
- Noise and clipping in `[simulation]` only affect `field.sample()`; the
  plot uses the noiseless `field.evaluate()`. Measurement noise is 0 here.

## Shielding model

Each source adds an isotropic Gaussian bump
`amplitude * exp(-0.5 * (dx^2 / sigma_x^2 + dy^2 / sigma_y^2))`.
Walls are zero-thickness line segments. For each source and query point,
if the straight segment from the source center to the query point
intersects any wall, that source's contribution is multiplied by
`1 - blocking_strength` (binary line of sight, no diffraction or
distance-dependent attenuation). With `blocking_strength = 1.0` the
contribution is removed completely. The domain bounds are not walls and do
not shield.

Walls are closed segments with a contact tolerance of 1e-9 m
(`WALL_CONTACT_TOL` in `field.py`): a ray that crosses a wall at any
nonzero angle, passes through a wall end or a joint of adjoining
segments, or ends on (or within 1e-9 m of) a wall is blocked. Queries on
the infinite wall line beyond a wall end, and rays passing a wall end or
running parallel at 1e-6 m or more, are not blocked. Contact at the source
position alone does not block. For these scenarios (sources well away from
the walls) splitting a wall or reversing its endpoints does not change the
field, and grid points on the walls are 0. Policy, numerical range and
limitations are in "Wall attenuation" of the package `readme.md`.

Source positions are hidden ground truth used only by the field generator.
They are drawn in the plot for inspection only and must not be passed to a
belief.

## Visualization

Plot the noiseless fields with the package tool (see "Plotting scenarios" in
the package `readme.md`), e.g.

```bash
# all four fields in one window (shared color scale)
ros2 run scalar_field_sim plot_fields.py 00_open 01_single_wall 02_wall_with_gap 03_better_wall

# one field, coarser grid, also saved as PNG
ros2 run scalar_field_sim plot_fields.py tests/fields/01_single_wall \
    --grid-step 0.05 --save single_wall.png
```

Selected scenarios must declare the same field value unit; otherwise the
tool exits with an error.

## Not driven by these files

- Robot collision obstacles in `ipp` (`ipp/nodes/obstacle_publisher.py`,
  reads `ipp/config/<scenario>.yaml`) are a separate representation.
- The belief geometry is chosen independently of the simulated field: with
  `belief_geometry: scenario`, the `scalar_field_belief` nodes read the walls
  and bounds of the geometry file referenced by `scenario_path`; otherwise
  they read the geometry or scenario file given in `belief_geometry`.
- No Gazebo world geometry is generated from these files.

A matching plot does not imply matching collision or belief geometry.
