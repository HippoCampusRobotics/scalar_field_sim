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
    """Publishes simulated IR measurements at the current vehicle pose.

    This node acts as an adapter between:
    - the vehicle odometry topic,
    - the scalar field simulator service,
    - and the `ir_measurement` topic used by the belief/planning stack.
    by exposing a Trigger service `trigger_ir_measurement`.


    On each `trigger_ir_measurement` request, the node:
    1. takes the most recent odometry pose,
    2. queries the scalar field simulator at that pose,
    3. publishes the returned `ScalarMeasurement` on `ir_measurement`.

    Note that sampling a single measurement is a design choice. On real hardware,
    multiple measurements at a high frequency could be averaged.
    """

    def __init__(self) -> None:
        super().__init__("ir_measurement_sim")

        # Topic providing the latest vehicle odometry.
        self.declare_parameter("odometry_topic", "odometry")

        # Output topic for simulated IR measurements.
        self.declare_parameter("measurement_topic", "ir_measurement")

        # Service name used to trigger one measurement.
        self.declare_parameter("trigger_service_name", "trigger_ir_measurement")

        # Simulator service providing one scalar field sample at a queried pose.
        self.declare_parameter("sample_service_name", "/sample_scalar_field")

        # Fallback frame used if incoming odometry does not provide a frame id.
        self.declare_parameter("frame_id_fallback", "map")

        # Timeout while waiting for the sample service.
        self.declare_parameter("sample_service_timeout_sec", 2.0)

        # Separate callback groups are used because this node handles a service
        # request and, inside that callback, also calls another service.
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

        # Latest odometry message. A measurement can only be taken once odometry
        # has been received at least once.
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
            f"odometry_topic: '{self._odometry_topic}', "
            f"measurement_topic: '{self._measurement_topic}', "
            f"trigger_service: '{self._trigger_service_name}', "
            f"sample_service: '{self._sample_service_name}'."
        )

    def _on_odometry(self, msg: Odometry) -> None:
        self._latest_odometry = msg

    async def _handle_trigger(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Take one simulated measurement at the latest odometry pose.

        The trigger request does not contain a pose. Instead, the node uses the
        most recent odometry message, queries the scalar field simulator at that
        pose, and publishes the returned `ScalarMeasurement`.

        For debugging, this callback also logs simple timing diagnostics:
        - age of the odometry used for sampling,
        - round-trip time of the sample service call,
        - estimated robot motion during the measurement request.

        The thresholds used for warning logs are heuristic values for the current
        setup. They are not yet tuned from a formal timing analysis.
        """
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

        # Age of the odometry pose used for this sample.
        # This is a simple measure of how stale the pose already was when the
        # trigger request was processed.
        now = self.get_clock().now()
        odom_time = rclpy.time.Time.from_msg(odom.header.stamp)
        odom_age_s = (now - odom_time).nanoseconds * 1e-9

        self.get_logger().debug(
            f"Trigger using odom pose "
            f"({odom.pose.pose.position.x:.3f}, {odom.pose.pose.position.y:.3f}, {odom.pose.pose.position.z:.3f}) "
            f"stamp={odom.header.stamp.sec}.{odom.header.stamp.nanosec:09d}, "
            f"age={odom_age_s:.6f}s"
        )

        req = SampleScalarField.Request()
        req.query.header.stamp = odom.header.stamp
        req.query.header.frame_id = odom.header.frame_id or self._frame_id_fallback
        req.query.pose = odom.pose.pose

        # Measure how long the scalar-field service call takes.
        # This is only a coarse runtime diagnostic for the current implementation.
        request_start = self.get_clock().now()
        # actually call the sample scalar field service
        future = self._sample_client.call_async(req)
        sample_response = await future
        request_dt_s = (self.get_clock().now() - request_start).nanoseconds * 1e-9

        # Compare the pose used for sampling with the most recent odometry pose
        # after the service call has returned. This gives a practical estimate of
        # how far the robot may have moved while the request was in flight.
        latest_odom_after = self._latest_odometry
        moved_dist_m = 0.0
        if latest_odom_after is not None:
            dx = latest_odom_after.pose.pose.position.x - odom.pose.pose.position.x
            dy = latest_odom_after.pose.pose.position.y - odom.pose.pose.position.y
            dz = latest_odom_after.pose.pose.position.z - odom.pose.pose.position.z
            moved_dist_m = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Heuristic warning thresholds for the current setup.
        # These are not physically derived limits. They are only meant to flag
        # suspicious cases during debugging:
        # - stale odometry at trigger time,
        # - unusually slow service responses,
        # - noticeable robot motion while the request is in flight.
        stale_odom_threshold_s = 0.05
        moved_dist_threshold_m = 0.03
        slow_service_threshold_s = 0.03

        if odom_age_s > stale_odom_threshold_s:
            self.get_logger().warning(
                f"Measurement used stale odometry: age={odom_age_s:.3f}s "
                f"(pose=({odom.pose.pose.position.x:.3f}, "
                f"{odom.pose.pose.position.y:.3f}, "
                f"{odom.pose.pose.position.z:.3f}))"
            )

        if request_dt_s > slow_service_threshold_s:
            self.get_logger().warning(
                f"Sample service was slow: round_trip={request_dt_s:.3f}s"
            )

        if moved_dist_m > moved_dist_threshold_m:
            self.get_logger().warning(
                f"Robot moved {moved_dist_m:.3f} m during measurement request."
            )

        self.get_logger().debug(
            f"Measurement timing: odom_age={odom_age_s:.3f}s, "
            f"service_dt={request_dt_s:.3f}s, move_dist={moved_dist_m:.3f}m"
        )

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
        self.get_logger().debug(
            f"Published measurement pose "
            f"({measurement.pose.position.x:.3f}, {measurement.pose.position.y:.3f}, {measurement.pose.position.z:.3f}) "
            f"stamp={measurement.header.stamp.sec}.{measurement.header.stamp.nanosec:09d}"
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IrMeasurementSimNode()

    # A multithreaded executor is used because this node handles one service call
    # while also waiting for the response of another service call.
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
