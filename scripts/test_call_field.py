#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node

from scalar_field_interfaces.srv import SampleScalarField


class FieldClient(Node):
    def __init__(self):
        super().__init__("field_client_helper")
        self.cli = self.create_client(SampleScalarField, "/sample_scalar_field")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for /sample_scalar_field ...")

    def call(self, x: float, y: float, frame_id: str = "map"):
        req = SampleScalarField.Request()
        req.query.header.frame_id = frame_id
        req.query.pose.position.x = x
        req.query.pose.position.y = y
        req.query.pose.position.z = 0.0
        req.query.pose.orientation.w = 1.0

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done():
            print("Timed out waiting for service response.")
            return None

        resp = future.result()
        if resp is None:
            raise RuntimeError("Service call failed.")
        return resp


def main():
    if len(sys.argv) < 3:
        print("usage: call_field.py X Y [FRAME_ID]")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    frame_id = sys.argv[3] if len(sys.argv) > 3 else "map"

    rclpy.init()
    node = FieldClient()
    resp = node.call(x, y, frame_id)

    print(f"success={resp.success}")
    print(f"value={resp.measurement.value}")
    print(f"clipped={resp.measurement.clipped}")
    print(f"status={resp.status_message}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
