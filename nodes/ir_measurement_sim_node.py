#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scalar_field_interfaces.msg import ScalarMeasurement
from scalar_field_interfaces.srv import SampleScalarField
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class IrMeasurementSimNode(Node):
    """
    Adapter from simulator ground-truth service to planner-facing IR measurements.

    The node subscribes to the latest vehicle odometry, exposes a Trigger service,
    and on each trigger queries the scalar field simulator at the current pose.
    The returned measurement is then published on the ``ir_measurement`` topic.
    """

    def __init__(self) -> None:
        super().__init__("ir_measurement_sim")

        self.declare_parameter("odometry_topic", "odometry")
        self.declare_parameter("measurement_topic", "ir_measurement")
        self.declare_parameter("trigger_service_name", "trigger_ir_measurement")
        self.declare_parameter("sample_service_name", "/sample_scalar_field")
        self.declare_parameter("frame_id_fallback", "map")
        self.declare_parameter("sample_service_timeout_sec", 2.0)

        self._odometry_cb_group = ReentrantCallbackGroup()
        self._client_cb_group = ReentrantCallbackGroup()
        self._service_cb_group = ReentrantCallbackGroup()

        self._odometry_topic = str(self.get_parameter("odometry_topic").value)
        self._measurement_topic = str(self.get_parameter("measurement_topic").value)
        self._trigger_service_name = str(
            self.get_parameter("trigger_service_name").value
        )
        self._sample_service_name = str(self.get_parameter("sample_service_name").value)
        self._frame_id_fallback = str(self.get_parameter("frame_id_fallback").value)
        self._sample_service_timeout_sec = float(
            self.get_parameter("sample_service_timeout_sec").value
        )

        self._latest_odometry: Optional[Odometry] = None

        self._odometry_sub = self.create_subscription(
            Odometry,
            self._odometry_topic,
            self._on_odometry,
            qos_profile=10,
            callback_group=self._odometry_cb_group,
        )
        self._measurement_pub = self.create_publisher(
            ScalarMeasurement,
            self._measurement_topic,
            qos_profile=10,
        )
        self._sample_client = self.create_client(
            SampleScalarField,
            self._sample_service_name,
            callback_group=self._client_cb_group,
        )
        self._trigger_srv = self.create_service(
            Trigger,
            self._trigger_service_name,
            self._handle_trigger,
            callback_group=self._service_cb_group,
        )

        self.get_logger().info(
            "ir_measurement_sim node started. "
            f"odometry='{self._odometry_topic}', "
            f"measurement_topic='{self._measurement_topic}', "
            f"trigger_service='{self._trigger_service_name}', "
            f"sample_service='{self._sample_service_name}'."
        )

    def _on_odometry(self, msg: Odometry) -> None:
        self._latest_odometry = msg

    async def _handle_trigger(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request

        if self._latest_odometry is None:
            response.success = False
            response.message = "No odometry received yet."
            return response

        if not self._sample_client.wait_for_service(
            timeout_sec=self._sample_service_timeout_sec
        ):
            response.success = False
            response.message = (
                f"Sample service '{self._sample_service_name}' not available."
            )
            return response

        odom = self._latest_odometry
        req = SampleScalarField.Request()
        req.query.header.stamp = odom.header.stamp
        req.query.header.frame_id = odom.header.frame_id or self._frame_id_fallback
        req.query.pose = odom.pose.pose

        future = self._sample_client.call_async(req)
        sample_response = await future

        if sample_response is None:
            response.success = False
            response.message = "Sampling call failed or returned no response."
            return response

        if not sample_response.success:
            response.success = False
            response.message = sample_response.status_message
            return response

        measurement = sample_response.measurement
        if not measurement.header.frame_id:
            measurement.header.frame_id = req.query.header.frame_id
        if measurement.header.stamp.sec == 0 and measurement.header.stamp.nanosec == 0:
            measurement.header.stamp = self.get_clock().now().to_msg()

        self._measurement_pub.publish(measurement)

        response.success = True
        response.message = (
            "Published one simulated IR measurement at current odometry pose."
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IrMeasurementSimNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
