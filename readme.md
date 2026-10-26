# scalar_field_sim

`scalar_field_sim` is a ROS 2 package for simulating continuous 2D scalar-field measurements for active sensing and informative path planning.

The package currently provides:
- a **field sampling service** for querying simulated measurements at arbitrary poses,
- a **static field visualization** published as `PointCloud2`,
- optional wall and source markers for RViz.

The intended use is to provide a lightweight simulated measurement backend that can later be swapped with the real physical measurement setup while keeping the planner-side interface similar.


## Overview

The simulator models a stationary 2D scalar field over a bounded rectangular workspace.

The field consists of:
- one or more Gaussian-like source contributions,
- an optional constant background level,
- optional attenuation/shielding caused by wall segments,
- optional additive Gaussian measurement noise,
- optional clipping of noisy measurements.

Only the **x** and **y** coordinates are used for field evaluation.


## Mathematical model

Let the query position be

$$
\mathbf{x} = (x, y)^T.
$$

For source $j$, with center

$$
\mathbf{c}_j = (c_{x,j}, c_{y,j})^T,
$$

amplitude $A_j$, and spreads $\sigma_{x,j}, \sigma_{y,j}$, the unblocked source contribution is


```math
\begin{aligned}
g_j(\mathbf{x})
&=
A_j
\exp\left(
-\frac{1}{2}
\left[
\left(
\frac{x-c_{x,j}}{\sigma_{x,j}}
\right)^2
+
\left(
\frac{y-c_{y,j}}{\sigma_{y,j}}
\right)^2
\right]
\right).
\end{aligned}
```

### Wall attenuation

Each source has a blocking strength

$$
b_j \in [0, 1].
$$

If the line segment from the source center $\mathbf{c}_j$ to the query point $\mathbf{x}$ intersects any wall segment, the contribution is attenuated by

$$
1 - b_j.
$$

Walls are closed segments with a contact tolerance $\delta$ = `WALL_CONTACT_TOL` = $10^{-9}$ m (`scalar_field_sim/field.py::_segment_intersection_mask`): points within $\delta$ of a wall segment count as on the wall, so every wall has a contact band of about 1 nm on each side.
For a source $\mathbf{a}$ and query $\mathbf{b}$:

- A. $\mathbf{a} = \mathbf{b}$ (no travelled path): not blocked.
- B. Source farther than $\delta$ from the wall: blocked iff the segment $[\mathbf{a}, \mathbf{b}]$ comes within $\delta$ of the finite wall. This blocks crossings at any nonzero angle, rays through a wall endpoint or a joint of adjoining walls, queries on the wall (any direction, including diagonal walls), and collinear overlap or touching. Parallel or collinear segments separated by more than $\delta$, and queries on the infinite wall line beyond the finite wall, are not blocked.
- C. Source within $\delta$ of the wall: contact at the source alone does not block. Blocked iff the ray continues along the wall: the query is within $\delta$ of the wall line and the overlap of ray and wall along the wall direction exceeds $\delta$.

Crossings are detected with orientation (cross-product) signs certified by Shewchuk's orient2d error filter (dimensionless bound $(3 + 16u)u$, $u = 2^{-53}$; <https://www.cs.cmu.edu/~quake/robust.html>); all other cases are decided by endpoint-to-segment distances compared with $\delta$.
There is no division by the cross product of the directions and no angle threshold, so shallow crossings are blocked like steep ones.
The decision is exact for the float64 inputs except for configurations within round-off (a few $10^{-16}$ times the coordinate differences) of a $\delta$ boundary; this needs $\delta \gg 10^{-16}\times$ scene size, which holds for coordinates up to about $10^4$ m.
Reversing a wall or splitting it into touching collinear pieces leaves the result unchanged for sources farther than $\delta$ from the wall; for sources within $\delta$ of a wall (rule C) splitting can matter, see the docstring.
Zero-length walls are rejected by `WallSegment`.

So the effective source contribution is
```math
\tilde{g}_j(\mathbf{x})
=
\begin{cases}
(1-b_j)\, g_j(\mathbf{x}), & \text{if blocked}, \\
g_j(\mathbf{x}),           & \text{otherwise}.
\end{cases}
```

### Latent field

The deterministic latent field is

$$
f(\mathbf{x}) = f_{\mathrm{bg}} + \sum_{j=1}^{N_s} \tilde g_j(\mathbf{x}),
$$

where $f_{\mathrm{bg}}$ is the constant background level.

### Measurement model

A noisy measurement is generated as

$$
y(\mathbf{x}) = f(\mathbf{x}) + \varepsilon,
\qquad
\varepsilon \sim \mathcal N(0, \sigma_n^2),
$$

where $\sigma_n$ is the configured measurement noise standard deviation.

If clipping is enabled with bounds $[y_{\min}, y_{\max}]$, then the published measurement is

$$
y_{\mathrm{clip}}(\mathbf{x}) =
\min\bigl(\max(y(\mathbf{x}), y_{\min}), y_{\max}\bigr).
$$

The simulator also tracks whether clipping changed the noisy value.


### Parameterization of source terms

In the current simulator, each source is modeled by an anisotropic Gaussian contribution with amplitude $A_j$ and spreads $\sigma_{x,j}, \sigma_{y,j}$.

In particular:

- $A_j$ controls the peak source strength,
- $\sigma_{x,j}$ and $\sigma_{y,j}$ control the spatial spread of the source contribution in x- and y-direction.

For the present scenarios, shared default values are used for the source amplitude and spread unless source-specific parameters are explicitly given in the scenario file. These values were selected to produce fields with a spatial extent and intensity range that are qualitatively similar to the physical IR setup (at a specific depth).



## ROS interfaces

### Service

The field server exposes a sampling service:

- `/sample_scalar_field`

using:

- `scalar_field_interfaces/srv/SampleScalarField`

The request contains a `geometry_msgs/PoseStamped` query pose.  
The response contains:
- `success`
- `measurement`
- `status_message`

The returned measurement is a `ScalarMeasurement` containing:
- header
- pose
- scalar value
- clipping flag

### Visualization topics

The visualization node publishes:
- `/field/ground_truth_cloud` (`sensor_msgs/msg/PointCloud2`)
- `/field/walls` (`visualization_msgs/msg/MarkerArray`)
- `/field/sources` (`visualization_msgs/msg/MarkerArray`)

The field cloud is published with transient-local durability. RViz may still require matching QoS settings to receive the retained sample if started after the publisher.


## Scenario configuration

A scenario defines:
- workspace bounds,
- wall geometry,
- source positions or source specifications,
- simulation parameters,
- visualization settings.

It is either self-contained (all tables in one file) or references a geometry
file and a field file (see below). A minimal self-contained example:

```toml
[general]
name = "simple_wall_case"

[frame]
frame_id = "map"

[bounds]
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]
wall_vertices = [
  [0.8, 1.2, 1.4, 1.2],
]

[objects.sources]
source_positions = [
  [0.4, 0.8],
  [1.6, 3.0],
]

[simulation.source_defaults]
amplitude = 0.011
sigma_x = 0.18
sigma_y = 0.18
blocking_strength = 1.0

[simulation]
background_level = 0.0
measurement_noise_std = 0.0005
clip_min = 0.0
clip_max = 0.02
seed = 42
```

Optional metadata:
- `[general] description`: free-text scenario description,
- `[units] field_value`: unit of the field values (e.g. `"V"`), available as
  `FieldSimConfig.field_value_unit`.

### Geometry, field, and scenario files

A scenario can be split into a geometry file (domain layout) and a field file
(scalar field definition). The scenario file then only references both:

```text
geometry/   geometry files: frame, bounds, walls
fields/     field files: units, sources, simulation and visualization settings
scenarios/  scenario files: one geometry file + one field file
```

Geometry file, e.g. `geometry/tank_2x4_single_wall.toml`:

```toml
[general]
name = "tank_2x4_single_wall"     # optional, default: file stem

[frame]
frame_id = "map"                  # optional, default: "map"

[units]
length = "m"                      # optional

[bounds]                          # required
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 4.0

[objects.walls]                   # optional
wall_vertices = [[1.0, 0.4, 1.0, 3.6]]  # [x1, y1, x2, y2] per segment
```

A geometry file may only contain `[general]`, `[frame]`, `[bounds]`,
`[objects.walls]`, and `[units] length`. It uses the same keys as a
self-contained scenario, so the wall loaders of `scalar_field_belief` (GP and
GMRF) read it directly.

Field file, e.g. `fields/test_four_sources.toml`: `[units]` (`field_value`,
optional `length`), `[objects.sources]`, `[simulation]` including
`[simulation.source_defaults]`, and `[visualization]`. It must not contain
`[bounds]`, `[frame]`, `[objects.walls]`, or `[general]`.

Scenario file, e.g. `scenarios/tests/fields/01_single_wall.toml`:

```toml
geometry_file = "../../../geometry/tank_2x4_single_wall.toml"
field_file = "../../../fields/test_four_sources.toml"

[general]
name = "01_single_wall"
description = "One finite wall at x = 1 m from y = 0.4 m to y = 3.6 m."
```

Rules:
- A scenario in reference form needs both `geometry_file` and `field_file`
  and may contain nothing else except `[general]` (`name`, `description`).
  The scenario name and description come from this `[general]` table; the
  name defaults to the scenario file stem.
- Relative paths are resolved against the directory of the scenario file,
  independent of the working directory; absolute paths are allowed.
- Referenced files must not contain `geometry_file` or `field_file` (only one
  level of reference).
- Keys outside the allowed sets above, a single reference, or a nested
  reference raise a `ValueError` naming the keys and files. A missing
  referenced file raises `FileNotFoundError`; invalid TOML in a referenced
  file raises a `ValueError` naming that file.
- The resolved paths are available as `FieldSimConfig.geometry_file` and
  `FieldSimConfig.field_file`.

Scenario files without `geometry_file` and `field_file` are self-contained
(see the example above) and are parsed as before; both paths are then `None`.

The geometry and field folders are installed next to `scenarios/` but are
not scenario folders: tools that list scenarios (e.g. `plot_fields.py --list`)
only search `scenarios/`.

The loader is `scalar_field_sim.config.load_field_sim_config`, which is also
used by the field server and visualization nodes, so both forms work with the
existing launch files.

The `scenario_path` parameter of the nodes (and the `scenario_path` launch
argument) is used as given if it is absolute; a relative path is resolved
against the installed `share/scalar_field_sim` directory, e.g.
`scenario_path:=scenarios/tests/fields/01_single_wall.toml`
(`scalar_field_sim.config.resolve_package_path`). The nodes log the resolved
path. The `scalar_field_belief` nodes resolve their `scenario_path` and
`belief_geometry` with the same rule. `plot_fields.py` has its own lookup (see
below).

### Plotting scenarios

`plot_fields.py` plots the noiseless field, walls, and sources of one or more
scenarios in a Matplotlib window (several scenarios share one color scale):

```bash
ros2 run scalar_field_sim plot_fields.py 01_single_wall
ros2 run scalar_field_sim plot_fields.py tests/fields/00_open tests/fields/02_wall_with_gap
ros2 run scalar_field_sim plot_fields.py ./my_scenario.toml --grid-step 0.05 --save my_scenario.png
ros2 run scalar_field_sim plot_fields.py --list
```

A scenario argument is resolved, in this order, as a path relative to the
current directory (or absolute), a path relative to the installed scenario
folder (`.toml` optional), a path relative to the workspace root (e.g.
`src/scalar_field_sim/scenarios/simple_field_1.toml`), or a unique file name
below the scenario folder. Scenario files added to `scenarios/` are only found
by name after rebuilding the package; any file can always be given by path.
Options: `--grid-step` (default: the scenario's `[visualization] grid_step`,
else 0.01 m), `--save PNG`, `--no-show`.
