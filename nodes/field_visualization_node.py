#!/usr/bin/env python3
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from math import isfinite

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

from scalar_field_sim.config import load_field_from_toml
from scalar_field_sim.pointcloud import make_field_pointcloud2
from scalar_field_sim.markers import make_wall_markers, make_source_markers


class FieldVisualizationNode(Node):
    """Publishes a ground-truth visualization of the configured scalar field.

    The node loads one field scenario from a TOML file, evaluates the latent field
    on a regular grid, and publishes:
    - a PointCloud2 for the ground-truth field values,
    - wall markers,
    - optional source markers.

    Unlike the field server node, this node does not provide noisy measurements.
    It visualizes the latent field directly.
    """

    def __init__(self):
        super().__init__("field_visualization")

        # Path to the scenario TOML file.
        # Note that this node needs to use the same scenario definition as the
        # field_server_node!
        self.declare_parameter("scenario_path", "")

        # Grid spacing (m) used to sample the latent field for the point cloud.
        # Smaller values increase visual resolution but also increase the number of
        # published points.
        self.declare_parameter("grid_step", 0.05)

        # Visualization mode for the z coordinate of the point cloud:
        # - "flat": all points lie on one plane
        # - "height": scalar value is also shown as height -> helpful for debugging
        self.declare_parameter("z_mode", "flat")

        # Scale factor applied when z_mode == "height".
        self.declare_parameter("height_scale", 0.5)

        # Whether to also publish source markers in RViz.
        self.declare_parameter("publish_sources", True)

        # Fixed scalar range used for color mapping of the ground-truth cloud.
        self.declare_parameter("field_color_min", 0.0)
        self.declare_parameter("field_color_max", 0.015)

        # Constant vertical shift applied to the full cloud for easier separation
        # from other visualizations such as the belief mean/variance clouds.
        self.declare_parameter("z_offset", -1.0)

        scenario_path = (
            self.get_parameter("scenario_path").get_parameter_value().string_value
        )
        if not scenario_path:
            raise ValueError("Parameter 'scenario_path' must be set.")

        self.config, self.field = load_field_from_toml(Path(scenario_path))

        # Use transient-local durability so RViz can receive the last published
        # visualization even if the display is added later.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2, "field/ground_truth_cloud", qos
        )
        self.wall_pub = self.create_publisher(MarkerArray, "field/walls", qos)
        self.source_pub = self.create_publisher(MarkerArray, "field/sources", qos)

        self._grid_step = (
            self.get_parameter("grid_step").get_parameter_value().double_value
        )
        self.get_logger().info(
            f"Using grid_step = {self._grid_step} for visualization of ground truth"
        )

        self._z_mode = self.get_parameter("z_mode").get_parameter_value().string_value
        self._height_scale = (
            self.get_parameter("height_scale").get_parameter_value().double_value
        )
        self._publish_sources = (
            self.get_parameter("publish_sources").get_parameter_value().bool_value
        )
        self._field_color_min = (
            self.get_parameter("field_color_min").get_parameter_value().double_value
        )
        self._field_color_max = (
            self.get_parameter("field_color_max").get_parameter_value().double_value
        )
        self._z_offset = (
            self.get_parameter("z_offset").get_parameter_value().double_value
        )

        self._publish_field_visualization()

        # handle updated visualization parameters
        self._param_cb_handle = self.add_on_set_parameters_callback(
            self._on_set_parameters
        )

    def _on_set_parameters(self, params) -> SetParametersResult:
        """Validate and apply runtime updates of visualization parameters.

        Currently supported runtime updates:
        - field_color_min
        - field_color_max
        - z_offset

        If a valid update is accepted, the node republishes the ground-truth cloud.
        """
        new_field_color_min = self._field_color_min
        new_field_color_max = self._field_color_max
        new_z_offset = self._z_offset

        color_updated = False
        z_offset_updated = False
        updated = False

        for param in params:
            if param.name == "field_color_min":
                new_field_color_min = float(param.value)
                color_updated = True
                updated = True

            elif param.name == "field_color_max":
                new_field_color_max = float(param.value)
                color_updated = True
                updated = True

            elif param.name == "z_offset":
                new_z_offset = float(param.value)
                z_offset_updated = True
                updated = True

        if not isfinite(new_field_color_min):
            return SetParametersResult(
                successful=False,
                reason="field_color_min must be finite.",
            )

        if not isfinite(new_field_color_max):
            return SetParametersResult(
                successful=False,
                reason="field_color_max must be finite.",
            )

        if new_field_color_min >= new_field_color_max:
            return SetParametersResult(
                successful=False,
                reason="field_color_min must be smaller than field_color_max.",
            )

        if not isfinite(new_z_offset):
            return SetParametersResult(
                successful=False,
                reason="z_offset must be finite.",
            )

        self._field_color_min = new_field_color_min
        self._field_color_max = new_field_color_max
        self._z_offset = new_z_offset

        if color_updated:
            self.get_logger().info(
                f"Updated field color range to "
                f"[{self._field_color_min:.6f}, {self._field_color_max:.6f}]."
            )

        if z_offset_updated:
            self.get_logger().info(
                f"Updated z_offset value for ground truth pointcloud to "
                f"{self._z_offset:.6f}"
            )

        if updated:
            self._publish_field_visualization()

        return SetParametersResult(successful=True)

    def _publish_field_visualization(self) -> None:
        """Evaluate the latent field on the visualization grid and publish it.

        Publishes:
        - `/field/ground_truth_cloud`
        - `/field/walls`
        - optionally `/field/sources`
        """
        positions, values = self.field.evaluate_on_grid(grid_step=self._grid_step)
        stamp = self.get_clock().now().to_msg()

        cloud = make_field_pointcloud2(
            positions_xy=positions,
            values=values,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=self._z_mode,
            z_offset=self._z_offset,
            height_scale=self._height_scale,
            colormap_min=self._field_color_min,
            colormap_max=self._field_color_max,
        )
        walls = make_wall_markers(self.config.geometry, self.config.frame_id, stamp)

        self.cloud_pub.publish(cloud)
        self.wall_pub.publish(walls)

        if self._publish_sources:
            sources = make_source_markers(
                self.config.geometry, self.config.frame_id, stamp
            )
            self.source_pub.publish(sources)

        self.get_logger().info(
            f"Published field visualization for '{self.config.name}' "
            f"with {len(positions)} grid points."
        )


def main(args=None):
    rclpy.init(args=args)
    node = FieldVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
