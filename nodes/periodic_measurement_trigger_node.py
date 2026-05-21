#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class PeriodicMeasurementTriggerNode(Node):
    """
    Periodically call the trigger_ir_measurement service.

    This is only a simple helper. In the final system,
    the planner node should call the trigger service instead.
    """

    def __init__(self) -> None:
        super().__init__("periodic_measurement_trigger")

        self.declare_parameter("trigger_service_name", "trigger_ir_measurement")
        self.declare_parameter("period_sec", 2.0)
        self.declare_parameter("wait_for_service_timeout_sec", 1.0)
        self.declare_parameter("start_immediately", True)

        self._trigger_service_name = str(
            self.get_parameter("trigger_service_name").value
        )
        self._period_sec = float(self.get_parameter("period_sec").value)
        self._wait_for_service_timeout_sec = float(
            self.get_parameter("wait_for_service_timeout_sec").value
        )
        self._start_immediately = bool(self.get_parameter("start_immediately").value)

        if self._period_sec <= 0.0:
            raise ValueError("period_sec must be positive.")
        if self._wait_for_service_timeout_sec <= 0.0:
            raise ValueError("wait_for_service_timeout_sec must be positive.")

        self._client = self.create_client(Trigger, self._trigger_service_name)
        self._request_in_flight = False
        self._timer: Optional[rclpy.timer.Timer] = None

        self._timer = self.create_timer(self._period_sec, self._on_timer)

        self.get_logger().info(
            "periodic_measurement_trigger node started. "
            f"trigger_service='{self._trigger_service_name}', "
            f"period_sec={self._period_sec:.3f}."
        )

        if self._start_immediately:
            self._on_timer()

    def _on_timer(self) -> None:
        if self._request_in_flight:
            self.get_logger().warning(
                "Previous trigger request still in flight. Skipping this cycle."
            )
            return

        if not self._client.wait_for_service(
            timeout_sec=self._wait_for_service_timeout_sec
        ):
            self.get_logger().warning(
                f"Trigger service '{self._trigger_service_name}' not available yet."
            )
            return

        req = Trigger.Request()
        future = self._client.call_async(req)
        self._request_in_flight = True
        future.add_done_callback(self._handle_trigger_response)

    def _handle_trigger_response(self, future) -> None:
        self._request_in_flight = False

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Trigger service call failed: {exc}")
            return

        if response.success:
            self.get_logger().info(f"Measurement trigger succeeded: {response.message}")
        else:
            self.get_logger().warning(
                f"Measurement trigger returned failure: {response.message}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PeriodicMeasurementTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
