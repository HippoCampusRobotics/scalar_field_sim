# Shared-parameter test fields

Manually inspectable field scenarios for later belief/planning evaluation.
All three share the same field parameters; only the walls differ.

| File | Content |
|---|---|
| `general_params.toml` | Shared frame, units, bounds, hidden sources, source parameters, simulation and visualization settings |
| `00_open.toml` | No internal walls |
| `01_single_wall.toml` | Wall at x = 1 m, y = 0.4 to 3.6 m |
| `02_wall_with_gap.toml` | Same wall with an opening for y = 1.75 to 2.25 m |

## Schema

A scenario file references the shared file with a top-level key:

```toml
field_file = "general_params.toml"

[general]
name = "01_single_wall"
description = "..."

[objects.walls]
wall_vertices = [[1.0, 0.4, 1.0, 3.6]]  # [x1, y1, x2, y2] per segment
```

- The reference is resolved relative to the scenario file's directory
  (independent of the working directory); absolute paths are allowed.
- With `field_file`, the scenario may contain only `field_file`, `[general]`,
  and `[objects.walls]`. The shared file must not contain `field_file`,
  `[general]`, or `[objects.walls]`. Any overlap raises a `ValueError`
  naming the conflicting keys and both files. There are no overrides.
- Only one level of reference is supported; a nested or self-referencing
  `field_file` raises a `ValueError`.
- A missing shared file raises `FileNotFoundError`; invalid TOML in the
  shared file raises a `ValueError` naming the shared file.
- Scenario files without `field_file` are parsed exactly as before.

The loader is `scalar_field_sim.config.load_field_sim_config`. The field
server and visualization nodes use the same loader, so these files also work
with the existing launch files, e.g.
`ros2 launch scalar_field_sim field_sim.launch.py scenario_path:=<absolute path to 01_single_wall.toml>`.

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
# all three fields in one window (shared color scale)
ros2 run scalar_field_sim plot_fields.py 00_open 01_single_wall 02_wall_with_gap

# one field, coarser grid, also saved as PNG
ros2 run scalar_field_sim plot_fields.py tests/fields/01_single_wall \
    --grid-step 0.05 --save single_wall.png
```

Selected scenarios must declare the same field value unit; otherwise the
tool exits with an error.

## Not driven by these files

- Robot collision obstacles in `ipp` (`ipp/nodes/obstacle_publisher.py`,
  reads `ipp/config/<scenario>.yaml`) are a separate representation.
- Wall files of `scalar_field_belief` (`config/walls/*.toml`) are separate
  and need inline `[bounds]`; they cannot reference `field_file`.
- No Gazebo world geometry is generated from these files.

A matching plot does not imply matching collision or belief geometry.
