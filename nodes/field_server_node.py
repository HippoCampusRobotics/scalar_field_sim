#!/usr/bin/env python3
from pathlib import Path

import rclpy
from rclpy.node import Node

from scalar_field_interfaces.srv import SampleScalarField
from scalar_field_sim.config import load_field_from_toml


class FieldServerNode(Node):
    def __init__(self):
        super().__init__("field_server")

        self.declare_parameter("scenario_path", "")
        self.declare_parameter("service_name", "sample_scalar_field")

        scenario_path = (
            self.get_parameter("scenario_path").get_parameter_value().string_value
        )
        if not scenario_path:
            raise ValueError("Parameter 'scenario_path' must be set.")

        self.config, self.field = load_field_from_toml(Path(scenario_path))

        service_name = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )
        self.srv = self.create_service(
            SampleScalarField, service_name, self._handle_sample
        )

        self.get_logger().info(
            f"Loaded scenario '{self.config.name}' from {scenario_path}. "
            f"Serving on '{service_name}'."
        )

    def _handle_sample(self, request, response):
        frame_id = request.query.header.frame_id or self.config.frame_id
        if frame_id != self.config.frame_id:
            response.success = False
            response.status_message = (
                f"Expected frame '{self.config.frame_id}', got '{frame_id}'."
            )
            return response

        x = request.query.pose.position.x
        y = request.query.pose.position.y

        if not (
            self.config.geometry.x_range[0] <= x <= self.config.geometry.x_range[1]
        ):
            response.success = False
            response.status_message = "Query x outside field bounds."
            return response
        if not (
            self.config.geometry.y_range[0] <= y <= self.config.geometry.y_range[1]
        ):
            response.success = False
            response.status_message = "Query y outside field bounds."
            return response

        _, noisy, clipped = self.field.sample_at_position((x, y))

        response.success = True
        response.status_message = "ok"
        response.measurement.header.stamp = self.get_clock().now().to_msg()
        response.measurement.header.frame_id = self.config.frame_id
        response.measurement.pose = request.query.pose
        response.measurement.value = float(noisy)
        response.measurement.clipped = bool(clipped)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FieldServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
