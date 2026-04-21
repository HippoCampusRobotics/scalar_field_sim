from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from scalar_field_sim.geometry import ScenarioGeometry


def make_wall_markers(scenario: ScenarioGeometry, frame_id: str, stamp) -> MarkerArray:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "walls"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.03
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    for wall in scenario.walls:
        p0 = Point(x=float(wall.start[0]), y=float(wall.start[1]), z=0.02)
        p1 = Point(x=float(wall.end[0]), y=float(wall.end[1]), z=0.02)
        marker.points.append(p0)
        marker.points.append(p1)

    out = MarkerArray()
    out.markers.append(marker)
    return out


def make_source_markers(
    scenario: ScenarioGeometry, frame_id: str, stamp
) -> MarkerArray:
    out = MarkerArray()
    for i, src in enumerate(scenario.sources):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "sources"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(src.position[0])
        marker.pose.position.y = float(src.position[1])
        marker.pose.position.z = 0.02
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        out.markers.append(marker)
    return out
