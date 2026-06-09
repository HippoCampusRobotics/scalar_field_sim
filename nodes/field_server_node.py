#!/usr/bin/env python3
from pathlib import Path

import rclpy
from rclpy.node import Node

from scalar_field_interfaces.srv import SampleScalarField
from scalar_field_sim.config import load_field_from_toml


class FieldServerNode(Node):
    """ROS service node providing scalar field samples via a ROS service.

    The node loads a scenario definition from a TOML file and serves the
    `SampleScalarField` service. Each request contains one query pose.
    The node checks that the query lies in the configured field frame and within
    the field bounds, then returns one sampled scalar measurement.

    Notes
    -----
    - The field is evaluated in 2D using only x and y from the query pose.
    - The returned value is the sampled noisy measurement, not the latent field.
    - Queries outside the configured rectangular field bounds are rejected.
    """

    def __init__(self):
        super().__init__("field_server")

        # Path to the scenario TOML file.
        # Note that this node needs to use the same scenario definition as the
        # field_visualization_node!
        self.declare_parameter("scenario_path", "")

        # Name of the ROS service providing scalar field samples.
        self.declare_parameter("service_name", "sample_scalar_field")

        service_name = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )
        scenario_path = (
            self.get_parameter("scenario_path").get_parameter_value().string_value
        )
        if not scenario_path:
            raise ValueError("Parameter 'scenario_path' must be set.")

        # load scenario config, define field object
        self.config, self.field = load_field_from_toml(Path(scenario_path))

        self.srv = self.create_service(
            SampleScalarField, service_name, self._handle_sample
        )

        self.get_logger().info(
            f"Loaded scenario '{self.config.name}' from {scenario_path}. "
            f"Serving on '{service_name}'."
        )

    def _handle_sample(self, request, response):
        """Handle one scalar field sampling request.

        The request pose is interpreted in the configured field frame. If the
        query frame or query position is invalid, the response is returned with
        `success = False`. Otherwise, the node samples the field at the given
        x/y position and returns one `ScalarMeasurement`.

        The returned measurement contains:
        - current node timestamp,
        - configured field frame,
        - the original query pose,
        - sampled noisy scalar value,
        - clipping flag.
        """
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
        response.measurement.header.stamp = request.query.header.stamp
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
