#!/usr/bin/env python3
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

from scalar_field_sim.config import load_field_from_toml
from scalar_field_sim.pointcloud import make_field_pointcloud2
from scalar_field_sim.markers import make_wall_markers, make_source_markers


class FieldVisualizationNode(Node):
    def __init__(self):
        super().__init__("field_visualization")

        self.declare_parameter("scenario_path", "")
        self.declare_parameter("grid_step", 0.05)
        self.declare_parameter("z_mode", "flat")
        self.declare_parameter("height_scale", 0.5)
        self.declare_parameter("publish_sources", True)

        scenario_path = (
            self.get_parameter("scenario_path").get_parameter_value().string_value
        )
        if not scenario_path:
            raise ValueError("Parameter 'scenario_path' must be set.")

        self.config, self.field = load_field_from_toml(Path(scenario_path))

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

        grid_step = self.get_parameter("grid_step").get_parameter_value().double_value
        self.get_logger().info(f"Using grid_step = {grid_step}")
        
        z_mode = self.get_parameter("z_mode").get_parameter_value().string_value
        height_scale = (
            self.get_parameter("height_scale").get_parameter_value().double_value
        )
        publish_sources = (
            self.get_parameter("publish_sources").get_parameter_value().bool_value
        )

        positions, values = self.field.evaluate_on_grid(grid_step=grid_step)
        stamp = self.get_clock().now().to_msg()

        cloud = make_field_pointcloud2(
            positions_xy=positions,
            values=values,
            frame_id=self.config.frame_id,
            stamp=stamp,
            z_mode=z_mode,
            height_scale=height_scale,
        )
        walls = make_wall_markers(self.config.geometry, self.config.frame_id, stamp)

        self.cloud_pub.publish(cloud)
        self.wall_pub.publish(walls)

        if publish_sources:
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
