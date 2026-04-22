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

$$
g_j(\mathbf{x}) =
A_j \exp\left(
-\frac{1}{2}
\left[
\left(\frac{x-c_{x,j}}{\sigma_{x,j}}\right)^2
+
\left(\frac{y-c_{y,j}}{\sigma_{y,j}}\right)^2
\right]
\right).
$$

### Wall attenuation

Each source has a blocking strength

$$
b_j \in [0, 1].
$$

If the line segment from the source center $\mathbf{c}_j$ to the query point $\mathbf{x}$ intersects any wall segment, the contribution is attenuated by

$$
1 - b_j.
$$

So the effective source contribution is

$$
\tilde g_j(\mathbf{x}) =
\begin{cases}
(1-b_j)\, g_j(\mathbf{x}), & \text{if blocked} \\
g_j(\mathbf{x}), & \text{otherwise.}
\end{cases}
$$

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

The field cloud is published with transient-local durability so RViz can display it even when started after the node.


## Scenario configuration

A scenario TOML file defines:
- workspace bounds,
- wall geometry,
- source positions or source specifications,
- simulation parameters,
- visualization settings.

A minimal example:

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
background_floor = 0.0
measurement_noise_std = 0.0005
clip_min = 0.0
clip_max = 0.02
seed = 1

[visualization]
grid_step = 0.05
